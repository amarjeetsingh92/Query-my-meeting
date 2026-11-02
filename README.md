# Production Meeting Transcript RAG System with ABAC Security

Production-ready Retrieval-Augmented Generation (RAG) platform purpose-built for conversational meeting transcripts (Microsoft Teams, Zoom, Webex, Otter.ai).

## Key Capabilities

1. **Document Parsing & Frontmatter Normalization (`convert_to_md.py`):**
   - Ingests Word `.docx` transcripts.
   - Extracts Meeting Title, Date, Duration, and full Participant Rosters (from headers, transcription start/stop markers, and speech turns).
   - Handles multi-line MS Teams turns (`Speaker   Timestamp\nDialogue`) as well as inline turns and Word tables.
   - Outputs clean Markdown files with YAML frontmatter.

2. **Dialogue-Aware "Quant Text Base" Chunking (`chunker.py`):**
   - **Speaker Turn Atomicity:** Treats complete turns as indivisible units; never cuts dialogue mid-sentence or mid-thought.
   - **Dual-Threshold Window:** Target 512 tokens, hard maximum 768 tokens.
   - **Topic-Shift Boundary Splitting:** Evaluates cosine semantic similarity within the 512-768 token window and cleanly finalizes chunks on topical shifts.
   - **Contextual Overlap:** Retains 2 previous speaker turns to maintain conversational continuity.
   - **ABAC Tagging:** Attaches `meeting_id`, `start_timestamp`, `end_timestamp`, `speakers`, `participants`, `min_role_level`, and `department`.

3. **Vector Ingestion & Payload Indexing (`ingest.py`):**
   - Pluggable embedders: OpenAI `text-embedding-3-small`, HuggingFace `sentence-transformers`, or zero-dependency `FallbackDeterministicEmbedder`.
   - Embedded local Qdrant (`./qdrant_db`) or remote Qdrant cluster.
   - Automatic collection creation with Cosine distance metric and payload indexes on ABAC fields.

4. **Dynamic ABAC Pre-Query Security Layer (`main.py`):**
   - Translates caller claims (`user_id`, `role`, `department`) into native Qdrant filters *before* vector search.
   - Prevents data leaks at the database index level.
   - Accessible if: User is Superuser (Admin/Executive) **OR** User is in `participants` **OR** (`min_role_level <= user.rank` AND `department` matches).

5. **Multi-Chunk LLM Synthesis Engine (`main.py`):**
   - Synthesizes a comprehensive, grounded summary answer across ALL retrieved context chunks.
   - Attributes claims to specific speakers with chronological timestamp citations (`[Speaker @ Timestamp]`).
   - Supports OpenAI, Ollama / local LLMs via `OPENAI_BASE_URL`, and an advanced local multi-chunk synthesis engine with zero external dependencies.

---

## Quickstart Guide

### 1. Installation
```bash
pip install -r requirements.txt
```

### 2. Convert Word Transcripts to Markdown
```bash
python convert_to_md.py --input-dir sample_transcripts --output-dir markdown_transcripts
```

### 3. Ingest Transcripts into Qdrant Vector Store
```bash
python ingest.py --dir markdown_transcripts
```

### 4. Run Verification Test Suite
```bash
# End-to-end pipeline & ABAC authorization test
python test_pipeline.py

# FastAPI endpoints & security inspection test
python test_api.py
```

### 5. Launch FastAPI Service
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```
Interactive Swagger UI available at: `http://localhost:8000/docs`

---

## Detailed Documentation
See [`PROJECT_PIPELINE_DOCUMENTATION.md`](PROJECT_PIPELINE_DOCUMENTATION.md) for complete architecture diagrams, data dictionary, security authorization rules, and benchmark results.
