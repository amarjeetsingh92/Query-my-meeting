"""
convert_to_md.py
----------------
Production Word (.docx) Transcript to Markdown Converter.

Parses meeting transcripts from Microsoft Teams, Zoom, Otter.ai, Webex,
and custom corporate recording formats. Extracts header metadata:
  - Title & Meeting ID
  - Date & Duration
  - Department & Security Classification
  - Full Participant Roster (from headers, transcription markers, & speech turns)
and structures speaker turns into standardized Markdown with YAML frontmatter.
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import docx
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("convert_to_md")


@dataclass
class SpeakerTurn:
    """Represents a single atomic speaker turn in a meeting transcript."""
    initials: Optional[str]
    speaker: str
    timestamp: str
    dialogue: str

    def to_markdown(self) -> str:
        """Renders the turn into clean, standardized Markdown."""
        ts_clean = self.timestamp.strip("()[] ")
        speaker_clean = self.speaker.strip()
        dialogue_clean = self.dialogue.strip()
        return f"**[{ts_clean}] {speaker_clean}:** {dialogue_clean}"


@dataclass
class MeetingMetadata:
    """Metadata extracted from document headers, tables, or transcription events."""
    meeting_id: str
    title: str = "Untitled Meeting"
    date: str = ""
    duration: str = ""
    department: str = "General"
    min_role_level: str = "member"  # Base ABAC permission level
    participants: List[str] = field(default_factory=list)
    source_file: str = ""
    extra_fields: Dict[str, Any] = field(default_factory=dict)

    def to_yaml_frontmatter(self) -> str:
        """Serializes metadata to standard YAML frontmatter block."""
        data = {
            "meeting_id": self.meeting_id,
            "title": self.title,
            "date": self.date,
            "duration": self.duration,
            "department": self.department,
            "min_role_level": self.min_role_level,
            "participants": sorted(list(set(self.participants))),
            "source_file": self.source_file,
            **self.extra_fields,
        }
        yaml_content = yaml.dump(data, sort_keys=False, allow_unicode=True)
        return f"---\n{yaml_content}---\n"


class TranscriptDocxParser:
    """
    High-performance parser for Word-based meeting transcripts.
    Supports:
      1. Microsoft Teams transcript format:
         - Header: Title, Date, Duration
         - Event markers: '<User Name> started/stopped transcription'
         - Speaker turns: '<Speaker Name>   <Timestamp>\n<Dialogue>'
      2. Single-line formats:
         - '[INITIALS] <Speaker Name> 00:01:23: <Dialogue>' or '<Speaker Name> 00:01:23: <Dialogue>'
      3. Word tables with Speaker / Timestamp / Dialogue columns.
    """

    # Matches MS Teams speaker line: Name followed by 2+ spaces or tab and timestamp
    # e.g., "<Speaker Name>   0:04" or "<Speaker Name>\t12:34"
    TEAMS_SPEAKER_RE = re.compile(
        r"^(?P<speaker>[A-Za-z0-9\.\'\-\s]+?)(?:\s{2,}|\t+)(?P<ts>\(?\d{1,2}:\d{2}(?::\d{2})?\)?)$"
    )

    # Matches inline single-line format:
    # [INITIALS] <Speaker Name> 00:01:23: <Dialogue>
    INLINE_TURN_RE = re.compile(
        r"^(?:\[(?P<initials>[A-Za-z0-9]{1,4})\]\s*)?"
        r"(?P<speaker>[A-Za-z0-9\.\'\-\s]+?)\s+"
        r"(?:\[(?P<ts_b>\d{1,2}:\d{2}(?::\d{2})?)\]|\((?P<ts_p>\d{1,2}:\d{2}(?::\d{2})?)\)|(?P<ts_bare>\d{1,2}:\d{2}(?::\d{2})?))"
        r"\s*[:\-–—]\s*(?P<dialogue>.+)$"
    )

    # Key-value header regex
    METADATA_KV_RE = re.compile(
        r"^(?P<key>Title|Date|Duration|Time|Meeting\s*ID|Department|Participants|Attendees|Role|Classification|Security)\s*[:=]\s*(?P<val>.+)$",
        re.IGNORECASE
    )

    # Date pattern heuristic: e.g. "October 28, 2026, 4:32PM" or "2026-10-28"
    DATE_HEURISTIC_RE = re.compile(
        r"^(?:(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s*\d{4}|\d{4}-\d{2}-\d{2})",
        re.IGNORECASE
    )

    # Duration pattern heuristic: e.g. "24m 52s" or "01:24:00"
    DURATION_HEURISTIC_RE = re.compile(
        r"^(?:\d+h\s*)?\d+m\s*\d+s$|^\d{1,2}:\d{2}:\d{2}$"
    )

    # Transcription marker: e.g. "<Participant Name> started transcription"
    TRANSCRIPTION_MARKER_RE = re.compile(
        r"^(?P<name>[A-Za-z0-9\.\'\-\s]+?)\s+(?:started|stopped)\s+transcription$",
        re.IGNORECASE
    )

    def __init__(self, docx_path: str | Path):
        self.docx_path = Path(docx_path)
        if not self.docx_path.exists():
            raise FileNotFoundError(f"Transcript document not found: {self.docx_path}")
        self.doc = docx.Document(str(self.docx_path))
        self.metadata = MeetingMetadata(
            meeting_id=self._generate_meeting_id(),
            source_file=self.docx_path.name
        )
        self.turns: List[SpeakerTurn] = []
        self._discovered_participants: Set[str] = set()

    def _generate_meeting_id(self) -> str:
        """Derives a clean meeting_id from filename."""
        stem = self.docx_path.stem.upper()
        cleaned = re.sub(r"[^A-Z0-9_\-]", "_", stem)
        return f"MEET-{cleaned}"

    def parse(self) -> Tuple[MeetingMetadata, List[SpeakerTurn]]:
        """Parses tables and paragraphs into metadata and turns."""
        self._parse_tables()
        self._parse_paragraphs()

        # Combine participants from header, transcription events, and speaker turns
        for turn in self.turns:
            if turn.speaker and turn.speaker.strip():
                self._discovered_participants.add(turn.speaker.strip())

        combined_participants = set(self.metadata.participants).union(self._discovered_participants)
        self.metadata.participants = sorted(list(combined_participants))

        # Default fallback title & date
        if self.metadata.title == "Untitled Meeting":
            self.metadata.title = self.docx_path.stem.replace("_", " ").title()

        if not self.metadata.date:
            self.metadata.date = datetime.now().strftime("%Y-%m-%d")

        return self.metadata, self.turns

    def _parse_tables(self) -> None:
        """Extracts metadata or tabular speaker turns from Word tables."""
        for table in self.doc.tables:
            col_count = len(table.columns)
            if col_count == 2:
                # Key-Value metadata table
                for row in table.rows:
                    k = row.cells[0].text.strip().rstrip(":")
                    v = row.cells[1].text.strip()
                    if k and v:
                        self._ingest_metadata_kv(k, v)
            elif col_count >= 3:
                # 3-column turn table: [Time/Initials] [Speaker] [Dialogue]
                for row in table.rows:
                    c0 = row.cells[0].text.strip()
                    c1 = row.cells[1].text.strip()
                    c2 = row.cells[2].text.strip()
                    if not c2 or any(h in c0.lower() for h in ["time", "speaker", "header"]):
                        continue
                    ts_m = re.search(r"(\d{1,2}:\d{2}(?::\d{2})?)", c0)
                    if ts_m and c1:
                        self.turns.append(SpeakerTurn(
                            initials=None,
                            speaker=c1.strip(),
                            timestamp=ts_m.group(1),
                            dialogue=c2.strip()
                        ))

    def _parse_paragraphs(self) -> None:
        """Parses paragraphs handling MS Teams, single-line, and multi-line layouts."""
        current_turn: Optional[SpeakerTurn] = None
        in_header_zone = True

        for p in self.doc.paragraphs:
            raw_text = p.text.strip()
            if not raw_text:
                continue

            # Check if this paragraph contains a transcription system marker
            marker_match = self.TRANSCRIPTION_MARKER_RE.match(raw_text)
            if marker_match:
                name = marker_match.group("name").strip()
                self._discovered_participants.add(name)
                continue

            # Check explicit key-value metadata
            kv_match = self.METADATA_KV_RE.match(raw_text)
            if kv_match and in_header_zone:
                self._ingest_metadata_kv(kv_match.group("key"), kv_match.group("val"))
                continue

            # Check MS Teams header heuristics (first few lines)
            if in_header_zone:
                if self.metadata.title == "Untitled Meeting" and ("Meeting Recording" in raw_text or "Status call" in raw_text or "Scrum" in raw_text):
                    self.metadata.title = raw_text
                    continue
                if not self.metadata.date and self.DATE_HEURISTIC_RE.match(raw_text):
                    self.metadata.date = raw_text
                    continue
                if not self.metadata.duration and self.DURATION_HEURISTIC_RE.match(raw_text):
                    self.metadata.duration = raw_text
                    continue

            # Check MS Teams multi-line speaker turn (speaker header on line 1, dialogue on line 2+)
            lines = [l.strip() for l in raw_text.split("\n") if l.strip()]
            if lines:
                teams_match = self.TEAMS_SPEAKER_RE.match(lines[0])
                if teams_match:
                    in_header_zone = False
                    speaker = teams_match.group("speaker").strip()
                    ts = teams_match.group("ts").strip("()[] ")
                    dialogue = "\n".join(lines[1:]) if len(lines) > 1 else ""

                    current_turn = SpeakerTurn(
                        initials=None,
                        speaker=speaker,
                        timestamp=ts,
                        dialogue=dialogue
                    )
                    self.turns.append(current_turn)
                    continue

            # Check inline turn layout: [JD] Speaker 00:01:23: Dialogue
            inline_match = self.INLINE_TURN_RE.match(raw_text)
            if inline_match:
                in_header_zone = False
                ts = (
                    inline_match.group("ts_b")
                    or inline_match.group("ts_p")
                    or inline_match.group("ts_bare")
                )
                current_turn = SpeakerTurn(
                    initials=inline_match.group("initials"),
                    speaker=inline_match.group("speaker").strip(),
                    timestamp=ts.strip("()[] "),
                    dialogue=inline_match.group("dialogue").strip()
                )
                self.turns.append(current_turn)
                continue

            # If an active turn exists, append text as continuation dialogue
            if current_turn:
                if current_turn.dialogue:
                    current_turn.dialogue += f"\n{raw_text}"
                else:
                    current_turn.dialogue = raw_text
                continue

    def _ingest_metadata_kv(self, key: str, val: str) -> None:
        """Normalizes and maps key-values to MeetingMetadata."""
        k = key.lower().replace(" ", "_")
        val = val.strip()
        if not val:
            return

        if "title" in k or "subject" in k:
            self.metadata.title = val
        elif "date" in k:
            self.metadata.date = val
        elif "duration" in k:
            self.metadata.duration = val
        elif "meeting_id" in k or "id" == k:
            self.metadata.meeting_id = val
        elif "department" in k or "team" in k:
            self.metadata.department = val
        elif "role" in k or "security" in k or "classification" in k:
            self.metadata.min_role_level = val.lower()
        elif "participant" in k or "attendee" in k or "present" in k:
            parts = [p.strip() for p in re.split(r"[,;\n]", val) if p.strip()]
            self.metadata.participants.extend(parts)
        else:
            self.metadata.extra_fields[k] = val

    def convert_to_markdown(self) -> str:
        """Renders complete structured Markdown with frontmatter."""
        self.parse()
        frontmatter = self.metadata.to_yaml_frontmatter()

        md_body = [
            f"# {self.metadata.title}\n",
            f"**Date:** {self.metadata.date} | **Duration:** {self.metadata.duration} | **Department:** {self.metadata.department}\n",
            "### Meeting Participants\n"
        ]
        for p in self.metadata.participants:
            md_body.append(f"- {p}")
        md_body.append("\n---\n")

        md_body.append("### Transcript\n")
        if not self.turns:
            md_body.append("_No structured speaker turns could be extracted._\n")
        else:
            for turn in self.turns:
                md_body.append(turn.to_markdown() + "\n")

        return frontmatter + "\n".join(md_body)


def convert_docx_file(input_path: Path, output_path: Optional[Path] = None) -> Path:
    """Converts a single docx file to markdown and writes it to disk."""
    if output_path is None:
        output_path = input_path.with_suffix(".md")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    parser = TranscriptDocxParser(input_path)
    markdown_content = parser.convert_to_markdown()

    output_path.write_text(markdown_content, encoding="utf-8")
    logger.info(
        f"Converted '{input_path.name}' -> '{output_path.name}' "
        f"({len(parser.turns)} turns, {len(parser.metadata.participants)} participants)"
    )
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Convert Word (.docx) meeting transcripts to structured Markdown with frontmatter."
    )
    parser.add_argument("--input", "-i", type=str, help="Path to input .docx transcript file.")
    parser.add_argument("--output", "-o", type=str, help="Path to output .md file.")
    parser.add_argument("--input-dir", type=str, help="Batch directory containing .docx files.")
    parser.add_argument("--output-dir", type=str, help="Destination directory for .md files.")

    args = parser.parse_args()

    if args.input:
        in_p = Path(args.input)
        out_p = Path(args.output) if args.output else in_p.with_suffix(".md")
        convert_docx_file(in_p, out_p)
    elif args.input_dir:
        in_dir = Path(args.input_dir)
        out_dir = Path(args.output_dir) if args.output_dir else in_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        docx_files = list(in_dir.glob("*.docx"))
        if not docx_files:
            logger.warning(f"No .docx files found in directory: {in_dir}")
            sys.exit(0)
        logger.info(f"Processing {len(docx_files)} transcripts from {in_dir}...")
        for docx_file in docx_files:
            if docx_file.name.startswith("~$"):
                continue
            dest = out_dir / f"{docx_file.stem}.md"
            convert_docx_file(docx_file, dest)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
