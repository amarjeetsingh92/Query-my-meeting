# Production Meeting Transcript RAG System with ABAC Security

Production-ready Retrieval-Augmented Generation (RAG) platform purpose-built for conversational meeting transcripts (Microsoft Teams, Zoom, Webex, Otter.ai).

## Key Capabilities

1. **OAuth 2.0 Enterprise Security & JWT Claims (`RFC 6749`, `RFC 7519`):**
   - Cryptographically signed and verified JWT access tokens (`POST /api/v1/auth/token`).
   - Rejects tampered, forged, or expired tokens with standard HTTP `401 Unauthorized`.
   - Binds vector database ABAC filters directly to verified identity claims extracted from the token.
   - Plug-and-play support for enterprise Identity Providers (Azure AD, Okta, Google Workspace).

2. **Document Parsing & Frontmatter Normalization (`convert_to_md.py`):**
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

3. **Vector Ingestion & 100% Free Embeddings (`ingest.py`):**
   - **Google Gemini Free Embeddings:** `text-embedding-004` (768 dim) via Google AI Studio's free tier.
   - **Sentence Transformers (Local CPU):** `all-MiniLM-L6-v2` (384 dim) via open-source Hugging Face.
   - **Ollama Local Embeddings:** `nomic-embed-text` runs 100% offline.
   - **Fallback Deterministic Embedder:** Zero-dependency instant hash projection for $0 offline testing.
   - **Embedded local Qdrant (`./qdrant_db`):** 100% free open-source vector database on disk (0 cloud fees).

4. **Dynamic ABAC Pre-Query Security Layer (`main.py`):**
   - Translates caller claims (`user_id`, `role`, `department`) into native Qdrant filters *before* vector search.
   - Prevents data leaks at the database index level.
   - Accessible if: User is Superuser (Admin/Executive) **OR** User is in `participants` **OR** (`min_role_level <= user.rank` AND `department` matches).

5. **Multi-Chunk LLM Synthesis Engine (`main.py`):**
   - Synthesizes a comprehensive, grounded summary answer across ALL retrieved context chunks.
   - Attributes claims to specific speakers with chronological timestamp citations (`[Speaker @ Timestamp]`).
   - **100% Free Provider Support:**
     - **Groq Cloud (Free Tier):** Llama 3.3 70B & Llama 3.1 8B at 500+ tokens/sec via `GROQ_API_KEY`.
     - **Google Gemini (Free Tier):** Gemini 1.5 Flash & 2.0 Flash (1,500 free requests/day) via `GEMINI_API_KEY`.
     - **Ollama (100% Local Free):** Open-source Llama 3.2, Mistral, Qwen offline with 0 API keys.
     - **Hugging Face (Free Serverless):** Open-source models via `HF_TOKEN`.
     - **Built-in Zero-Cost Local Engine:** Native dialogue synthesis directly in Python with $0 cost.

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

# FastAPI endpoints, OAuth 2.0 token tests, & security inspection test
python test_api.py
```

### 5. Launch FastAPI Service
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```
Interactive Swagger UI available at: `http://localhost:8000/docs`
Interactive Chatbot UI available at: `http://localhost:8000/`

### 6. OAuth 2.0 Authentication Workflow
```bash
# Obtain OAuth 2.0 JWT Access Token
curl -X POST http://localhost:8000/api/v1/auth/token \
  -H "Content-Type: application/json" \
  -d '{"username": "employee_user", "role": "member", "department": "Engineering"}'

# Query Meeting Transcripts with Bearer Token
curl -X POST http://localhost:8000/api/v1/query \
  -H "Authorization: Bearer <ACCESS_TOKEN>" \
  -H "Content-Type: application/json" \
  -d '{"query": "What decisions were made?", "top_k": 5}'
```

---

### 6. 100% Free API Providers & Registration Links

This project requires **zero paid subscriptions**. You can run with any of these free options:

| Service Type | Provider | Model / Dimension | Where to Get Free Access | Cost |
| :--- | :--- | :--- | :--- | :--- |
| **LLM Inference** | **Groq Cloud** | `llama-3.3-70b-versatile` | [console.groq.com/keys](https://console.groq.com/keys) | **$0.00** (Free Tier, 500+ tok/s) |
| **LLM Inference** | **Google Gemini** | `gemini-1.5-flash` | [aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey) | **$0.00** (1,500 req/day, 1M context) |
| **LLM Inference** | **Ollama** | `llama3.2`, `mistral` | [ollama.com/download](https://ollama.com/download) | **$0.00** (100% Local / Offline) |
| **LLM Inference** | **Hugging Face** | `Llama-3.2-3B-Instruct` | [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) | **$0.00** (Free Serverless API) |
| **Embeddings** | **Google Gemini** | `text-embedding-004` (768d) | [aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey) | **$0.00** (Free Tier API) |
| **Embeddings** | **Hugging Face** | `all-MiniLM-L6-v2` (384d) | Built-in (`sentence-transformers`) | **$0.00** (Local CPU) |
| **Embeddings** | **Ollama** | `nomic-embed-text` (768d) | `ollama pull nomic-embed-text` | **$0.00** (Local CPU) |
| **Vector DB** | **Local Qdrant** | On-disk storage (`./qdrant_db`) | Built-in (`qdrant-client`) | **$0.00** (Local Open-Source) |
| **Vector DB** | **Qdrant Cloud** | Managed Cloud Cluster | [cloud.qdrant.io](https://cloud.qdrant.io/) | **$0.00** (1GB RAM Cluster Free Forever) |
| **Authentication**| **Self-Hosted OAuth2**| PyJWT HMAC/RSA | Built-in (`main.py`) | **$0.00** (Self-Hosted RFC 6749) |

---

## Detailed Documentation
See [`PROJECT_PIPELINE_DOCUMENTATION.md`](PROJECT_PIPELINE_DOCUMENTATION.md) for complete architecture diagrams, data dictionary, security authorization rules, and benchmark results.
