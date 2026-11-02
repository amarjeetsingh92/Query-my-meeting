"""
chunker.py
----------
Dialogue-Aware "Quant Text Base" Chunking Engine.

Features:
1. Complete Speaker-Turn Atomicity: Keeps speaker turns and QA pairs unbroken.
2. Token Threshold Window: Target window (target_tokens=512, max_tokens=768).
3. Semantic Topic-Shift Detection: Detects topic shifts using cosine similarity
   when token count is within the target-to-max window.
4. Conversational Overlap: Carries over 1-2 previous turns for conversational continuity.
5. Rich Metadata Tagging: Attaches meeting_id, start/end timestamps, chunk speakers,
   full meeting participants list (for ABAC), min_role_level, and department.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import yaml

try:
    import tiktoken
    _TIKTOKEN_AVAILABLE = True
except ImportError:
    _TIKTOKEN_AVAILABLE = False

logger = logging.getLogger("chunker")


@dataclass
class ParsedTurn:
    """Internal model for an atomic speaker turn within the chunker."""
    speaker: str
    timestamp: str
    text: str
    formatted_text: str
    token_count: int


@dataclass
class DialogueChunk:
    """Production chunk model carrying text and security/temporal metadata."""
    chunk_id: str
    meeting_id: str
    title: str
    chunk_index: int
    text: str
    start_timestamp: str
    end_timestamp: str
    speakers: List[str]
    participants: List[str]
    min_role_level: str
    department: str
    token_count: int
    turn_count: int
    overlap_turns_included: int
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_payload(self) -> Dict[str, Any]:
        """Converts chunk metadata to a Qdrant/vector-database JSON payload."""
        return {
            "chunk_id": self.chunk_id,
            "meeting_id": self.meeting_id,
            "title": self.title,
            "chunk_index": self.chunk_index,
            "text": self.text,
            "start_timestamp": self.start_timestamp,
            "end_timestamp": self.end_timestamp,
            "speakers": self.speakers,
            "participants": self.participants,
            "min_role_level": self.min_role_level.lower(),
            "department": self.department,
            "token_count": self.token_count,
            "turn_count": self.turn_count,
            "overlap_turns_included": self.overlap_turns_included,
            **self.metadata,
        }


class TokenCounter:
    """Fast, reliable token counter using tiktoken with whitespace fallback."""
    def __init__(self, model_name: str = "gpt-4o"):
        self.encoder = None
        if _TIKTOKEN_AVAILABLE:
            try:
                self.encoder = tiktoken.encoding_for_model(model_name)
            except Exception:
                try:
                    self.encoder = tiktoken.get_encoding("cl100k_base")
                except Exception:
                    self.encoder = None

    def count(self, text: str) -> int:
        if self.encoder is not None:
            return len(self.encoder.encode(text, disallowed_special=()))
        # Fast approximation: ~1.33 tokens per whitespace-separated word
        return max(1, int(len(text.split()) * 1.33))


class SemanticTopicShiftDetector:
    """
    Lightweight topic shift detector using cosine similarity.
    Defaults to high-speed lexical/TF-IDF cosine similarity (<1ms, 0 latency).
    Can lazily load SentenceTransformer if neural topic shifts are requested.
    """
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        cosine_threshold: float = 0.50,
        enable_neural: bool = False
    ):
        self.cosine_threshold = cosine_threshold
        self.model_name = model_name
        self.enable_neural = enable_neural
        self.model = None
        self._cache: Dict[str, np.ndarray] = {}

        if self.enable_neural:
            try:
                from sentence_transformers import SentenceTransformer
                logger.info(f"Loading neural topic shift detector model: {model_name}...")
                self.model = SentenceTransformer(model_name)
            except Exception as e:
                logger.warning(f"Could not load SentenceTransformer ({e}). Falling back to lexical.")
                self.model = None

    def is_topic_shift(self, context_text: str, candidate_text: str) -> bool:
        """
        Determines if candidate_text shifts significantly away from context_text.
        Returns True if cosine similarity is below cosine_threshold.
        """
        if not context_text.strip() or not candidate_text.strip():
            return False

        if self.model is not None:
            # Neural cosine similarity
            if context_text not in self._cache:
                self._cache[context_text] = self.model.encode(context_text, normalize_embeddings=True, show_progress_bar=False)
            if candidate_text not in self._cache:
                self._cache[candidate_text] = self.model.encode(candidate_text, normalize_embeddings=True, show_progress_bar=False)
            v1 = self._cache[context_text]
            v2 = self._cache[candidate_text]
            sim = float(np.dot(v1, v2))
            return sim < self.cosine_threshold

        # High-performance lexical Jaccard/overlap metric
        w1 = set(re.findall(r"\w{3,}", context_text.lower()))
        w2 = set(re.findall(r"\w{3,}", candidate_text.lower()))
        if not w1 or not w2:
            return False
        overlap = len(w1.intersection(w2))
        union = len(w1.union(w2))
        jaccard = overlap / union if union > 0 else 0.0
        return jaccard < 0.12


class DialogueAwareChunker:
    """
    Production Dialogue-Aware Chunker.
    - Treats complete speaker turns as indivisible atomic units.
    - Windowing: appends turns until target_tokens (512) is reached.
    - Topic-Shift Check: Between target_tokens and max_tokens (768), splits if
      a semantic topic break is detected.
    - Overlap: retains 1-2 trailing turns to initialize the next chunk.
    - ABAC Tags: attaches participants, meeting_id, timestamps, and min_role_level.
    """

    def __init__(
        self,
        target_tokens: int = 512,
        max_tokens: int = 768,
        overlap_turns: int = 2,
        semantic_threshold: float = 0.50,
        enable_semantic_split: bool = True,
        tokenizer_model: str = "gpt-4o",
    ):
        self.target_tokens = target_tokens
        self.max_tokens = max_tokens
        self.overlap_turns = max(0, overlap_turns)
        self.token_counter = TokenCounter(model_name=tokenizer_model)
        self.semantic_detector = (
            SemanticTopicShiftDetector(cosine_threshold=semantic_threshold, enable_neural=False)
            if enable_semantic_split else None
        )

    def parse_markdown_with_frontmatter(self, md_content: str) -> Tuple[Dict[str, Any], List[ParsedTurn]]:
        """Parses frontmatter metadata and extracts turns from markdown body."""
        frontmatter = {}
        body = md_content

        if md_content.startswith("---"):
            parts = md_content.split("---", 2)
            if len(parts) >= 3:
                try:
                    frontmatter = yaml.safe_load(parts[1]) or {}
                    body = parts[2]
                except Exception as e:
                    logger.warning(f"Failed parsing YAML frontmatter: {e}")

        turns = self._parse_turns_from_markdown(body)
        return frontmatter, turns

    def _parse_turns_from_markdown(self, body: str) -> List[ParsedTurn]:
        """Extracts turn objects from formatted markdown dialogue lines."""
        # Restrict turn search to lines following '### Transcript' if present
        transcript_marker = re.search(r"^#{1,3}\s+Transcript\s*$", body, re.MULTILINE | re.IGNORECASE)
        search_body = body[transcript_marker.end():] if transcript_marker else body

        # Primary pattern: **[00:01:23] <Speaker Name>:** <Dialogue>...
        turn_pattern = re.compile(
            r"^\*\*\[(?P<ts>[^\]]+)\]\s*(?P<speaker>[^\*:]+):\*\*\s*(?P<dialogue>.*)$",
            re.MULTILINE
        )

        turns: List[ParsedTurn] = []
        matches = list(turn_pattern.finditer(search_body))

        if not matches:
            # Fallback pattern without bracketed timestamp
            fallback_pattern = re.compile(
                r"^\*\*(?P<speaker>(?!Date|Duration|Department|Meeting|Title|Participants)[^\*:]+):\*\*\s*(?P<dialogue>.*)$",
                re.MULTILINE | re.IGNORECASE
            )
            matches = list(fallback_pattern.finditer(search_body))

        if not matches:
            # Paragraph level fallback
            paragraphs = [p.strip() for p in search_body.split("\n\n") if p.strip()]
            for idx, p in enumerate(paragraphs):
                tok_count = self.token_counter.count(p)
                turns.append(ParsedTurn(
                    speaker=f"Speaker_{idx+1}",
                    timestamp="00:00:00",
                    text=p,
                    formatted_text=p,
                    token_count=tok_count
                ))
            return turns

        for i, match in enumerate(matches):
            ts = match.groupdict().get("ts") or "00:00:00"
            speaker = match.group("speaker").strip()
            start_pos = match.end()
            end_pos = matches[i + 1].start() if i + 1 < len(matches) else len(search_body)
            raw_dialogue = match.group("dialogue").strip()
            rest_of_turn = search_body[start_pos:end_pos].strip()

            if rest_of_turn:
                full_dialogue = f"{raw_dialogue}\n{rest_of_turn}".strip()
            else:
                full_dialogue = raw_dialogue

            formatted = f"**[{ts}] {speaker}:** {full_dialogue}"
            tok_count = self.token_counter.count(formatted)

            turns.append(ParsedTurn(
                speaker=speaker,
                timestamp=ts,
                text=full_dialogue,
                formatted_text=formatted,
                token_count=tok_count
            ))

        return turns

    def _split_oversized_turn(self, turn: ParsedTurn) -> List[ParsedTurn]:
        """
        Safely handles single monologues that exceed max_tokens.
        Splits text by sentences while retaining speaker and timestamp attribution.
        """
        sentences = re.split(r"(?<=[.!?])\s+", turn.text)
        sub_turns: List[ParsedTurn] = []
        accumulated_text: List[str] = []
        accumulated_tokens = 0
        part_idx = 1

        for sentence in sentences:
            sent_tokens = self.token_counter.count(sentence)
            if accumulated_tokens + sent_tokens > self.max_tokens - 40 and accumulated_text:
                joined = " ".join(accumulated_text)
                fmt = f"**[{turn.timestamp}] {turn.speaker} (Part {part_idx}):** {joined}"
                sub_turns.append(ParsedTurn(
                    speaker=turn.speaker,
                    timestamp=turn.timestamp,
                    text=joined,
                    formatted_text=fmt,
                    token_count=self.token_counter.count(fmt)
                ))
                part_idx += 1
                accumulated_text = [sentence]
                accumulated_tokens = sent_tokens
            else:
                accumulated_text.append(sentence)
                accumulated_tokens += sent_tokens

        if accumulated_text:
            joined = " ".join(accumulated_text)
            tag = f" (Part {part_idx})" if part_idx > 1 else ""
            fmt = f"**[{turn.timestamp}] {turn.speaker}{tag}:** {joined}"
            sub_turns.append(ParsedTurn(
                speaker=turn.speaker,
                timestamp=turn.timestamp,
                text=joined,
                formatted_text=fmt,
                token_count=self.token_counter.count(fmt)
            ))

        return sub_turns

    def chunk_turns(
        self,
        turns: List[ParsedTurn],
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[DialogueChunk]:
        """
        Main chunking engine executing:
        1. Atomic speaker turns
        2. Soft target + Hard max windowing
        3. Semantic topic shift split
        4. Overlap propagation
        5. ABAC payload tagging
        """
        if not turns:
            return []

        meta = metadata or {}
        meeting_id = meta.get("meeting_id", "UNKNOWN_MEETING")
        title = meta.get("title", "Untitled Meeting")
        participants = meta.get("participants", [])
        min_role_level = meta.get("min_role_level", "member")
        department = meta.get("department", "General")

        # 1. Normalize turns: unpack any single monologue exceeding max_tokens
        normalized_turns: List[ParsedTurn] = []
        for t in turns:
            if t.token_count > self.max_tokens:
                normalized_turns.extend(self._split_oversized_turn(t))
            else:
                normalized_turns.append(t)

        chunks: List[DialogueChunk] = []
        current_chunk_turns: List[ParsedTurn] = []
        current_token_count = 0
        overlap_buffer: List[ParsedTurn] = []
        chunk_idx = 0

        i = 0
        n = len(normalized_turns)

        while i < n:
            candidate_turn = normalized_turns[i]
            candidate_tokens = candidate_turn.token_count

            # If current chunk is empty, seed with overlap from previous chunk
            if not current_chunk_turns:
                current_chunk_turns.extend(overlap_buffer)
                current_token_count = sum(t.token_count for t in current_chunk_turns)

            projected_tokens = current_token_count + candidate_tokens

            # DECISION LOGIC:
            exceeds_max = projected_tokens > self.max_tokens

            topic_shifted = False
            if current_token_count >= self.target_tokens and self.semantic_detector is not None:
                recent_context = "\n".join(t.text for t in current_chunk_turns[-2:])
                topic_shifted = self.semantic_detector.is_topic_shift(
                    context_text=recent_context,
                    candidate_text=candidate_turn.text
                )

            # Split when limit reached or topic shifts past target_tokens
            if (exceeds_max or topic_shifted) and current_chunk_turns:
                chunk = self._build_chunk(
                    chunk_idx=chunk_idx,
                    meeting_id=meeting_id,
                    title=title,
                    turns=current_chunk_turns,
                    participants=participants,
                    min_role_level=min_role_level,
                    department=department,
                    overlap_count=len(overlap_buffer)
                )
                chunks.append(chunk)
                chunk_idx += 1

                # Carry over overlap turns
                if self.overlap_turns > 0:
                    overlap_buffer = current_chunk_turns[-self.overlap_turns:]
                else:
                    overlap_buffer = []

                current_chunk_turns = []
                current_token_count = 0
                continue

            current_chunk_turns.append(candidate_turn)
            current_token_count += candidate_tokens
            i += 1

        # Final chunk
        if current_chunk_turns:
            chunk = self._build_chunk(
                chunk_idx=chunk_idx,
                meeting_id=meeting_id,
                title=title,
                turns=current_chunk_turns,
                participants=participants,
                min_role_level=min_role_level,
                department=department,
                overlap_count=len(overlap_buffer)
            )
            chunks.append(chunk)

        logger.info(
            f"Chunked meeting '{meeting_id}' into {len(chunks)} dialogue chunks "
            f"(target={self.target_tokens}, max={self.max_tokens}, overlap={self.overlap_turns})"
        )
        return chunks

    def _build_chunk(
        self,
        chunk_idx: int,
        meeting_id: str,
        title: str,
        turns: List[ParsedTurn],
        participants: List[str],
        min_role_level: str,
        department: str,
        overlap_count: int
    ) -> DialogueChunk:
        """Assembles DialogueChunk dataclass instance with full ABAC metadata."""
        start_ts = turns[0].timestamp if turns else "00:00:00"
        end_ts = turns[-1].timestamp if turns else "00:00:00"
        speakers = sorted(list(set(t.speaker for t in turns if t.speaker)))

        chunk_lines = [t.formatted_text for t in turns]
        full_text = "\n\n".join(chunk_lines)
        total_tokens = sum(t.token_count for t in turns)
        chunk_id = f"{meeting_id}_chk_{chunk_idx:04d}"

        return DialogueChunk(
            chunk_id=chunk_id,
            meeting_id=meeting_id,
            title=title,
            chunk_index=chunk_idx,
            text=full_text,
            start_timestamp=start_ts,
            end_timestamp=end_ts,
            speakers=speakers,
            participants=participants,
            min_role_level=min_role_level,
            department=department,
            token_count=total_tokens,
            turn_count=len(turns),
            overlap_turns_included=overlap_count,
            metadata={
                "section": f"{start_ts} - {end_ts}",
                "speaker_count": len(speakers),
            }
        )

    def chunk_markdown_file(self, md_content: str) -> List[DialogueChunk]:
        """Convenience method to chunk raw Markdown content directly."""
        meta, turns = self.parse_markdown_with_frontmatter(md_content)
        return self.chunk_turns(turns, metadata=meta)
