"""
ingest.py
---------
Vector Store Ingestion Pipeline with ABAC Metadata Schema.

Features:
1. Pluggable Embedding Backends:
   - OpenAI ('text-embedding-3-small', 1536 dim)
   - Local HuggingFace / SentenceTransformers ('all-MiniLM-L6-v2', 384 dim, loaded lazily)
   - Deterministic Fallback Embedder for ultra-fast, zero-dependency testing
2. Qdrant Vector Store Integration:
   - Embedded local disk mode (zero external server required) or remote Qdrant server
   - Automatic collection creation with Cosine distance metric
   - Payload indexing on critical ABAC fields (participants, min_role_level, department)
3. End-to-end ingestion from structured Markdown files directly to vector database.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import os
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol

import numpy as np

# Qdrant client
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from chunker import DialogueAwareChunker, DialogueChunk

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("ingest")

# Collection & Schema Constants
DEFAULT_COLLECTION_NAME = "meeting_transcripts"
DEFAULT_QDRANT_PATH = "./qdrant_db"


class BaseEmbedder(Protocol):
    """Protocol defining the interface for embedding models."""
    dimension: int
    def embed_texts(self, texts: List[str]) -> List[List[float]]: ...
    def embed_query(self, text: str) -> List[float]: ...


class OpenAIEmbedder:
    """OpenAI Embedding Provider using text-embedding-3-small."""
    def __init__(self, model_name: str = "text-embedding-3-small"):
        from openai import OpenAI
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required for OpenAIEmbedder.")
        self.client = OpenAI(api_key=api_key)
        self.model_name = model_name
        self.dimension = 1536 if "small" in model_name or "ada" in model_name else 3072

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        response = self.client.embeddings.create(
            input=texts,
            model=self.model_name
        )
        return [item.embedding for item in response.data]

    def embed_query(self, text: str) -> List[float]:
        return self.embed_texts([text])[0]


class SentenceTransformerEmbedder:
    """Local embedding provider using sentence-transformers (lazily initialized)."""
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        logger.info(f"Loading local embedding model '{model_name}'...")
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True
        )
        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        return self.embed_texts([text])[0]


class FallbackDeterministicEmbedder:
    """
    Lightweight, deterministic hash-based pseudo-embedding provider.
    Used for instant testing and CI environments when heavy weights or API keys are unavailable.
    """
    def __init__(self, dimension: int = 384):
        self.dimension = dimension

    def _hash_vector(self, text: str) -> List[float]:
        words = text.lower().split()
        vec = np.zeros(self.dimension, dtype=np.float32)
        for w in words:
            h = int(hashlib.md5(w.encode("utf-8")).hexdigest(), 16)
            idx = h % self.dimension
            val = ((h >> 8) % 100) / 100.0 - 0.5
            vec[idx] += val
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec.tolist()

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        return [self._hash_vector(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._hash_vector(text)


def get_default_embedder() -> BaseEmbedder:
    """Auto-selects the best available embedding backend."""
    if os.getenv("OPENAI_API_KEY"):
        try:
            logger.info("Initializing OpenAIEmbedder (text-embedding-3-small)...")
            return OpenAIEmbedder()
        except Exception as e:
            logger.warning(f"Failed to initialize OpenAIEmbedder ({e}), falling back...")

    backend = os.getenv("EMBEDDING_BACKEND", "").lower()
    if backend == "sentence-transformers":
        try:
            return SentenceTransformerEmbedder("all-MiniLM-L6-v2")
        except Exception as e:
            logger.warning(f"SentenceTransformer failed ({e}). Falling back to deterministic.")

    return FallbackDeterministicEmbedder(dimension=384)


class IngestionPipeline:
    """
    End-to-end transcript ingestion pipeline.
    Connects to Qdrant, manages collection schema, builds payload indexes,
    and indexes dialogue chunks.
    """

    def __init__(
        self,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        qdrant_url: Optional[str] = None,
        qdrant_api_key: Optional[str] = None,
        qdrant_path: Optional[str] = DEFAULT_QDRANT_PATH,
        embedder: Optional[BaseEmbedder] = None,
        chunker: Optional[DialogueAwareChunker] = None,
    ):
        self.collection_name = collection_name
        self.embedder = embedder or get_default_embedder()
        self.chunker = chunker or DialogueAwareChunker(
            target_tokens=512,
            max_tokens=768,
            overlap_turns=2,
            enable_semantic_split=True
        )

        if qdrant_url:
            logger.info(f"Connecting to remote Qdrant at {qdrant_url}...")
            self.client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
        elif qdrant_path:
            logger.info(f"Using local embedded Qdrant storage at '{qdrant_path}'...")
            self.client = QdrantClient(path=qdrant_path)
        else:
            logger.info("Using in-memory Qdrant storage...")
            self.client = QdrantClient(location=":memory:")

        self._ensure_collection()
        self._ensure_payload_indexes()

    def _ensure_collection(self) -> None:
        """Creates the Qdrant collection if it does not already exist."""
        collections = [c.name for c in self.client.get_collections().collections]
        if self.collection_name not in collections:
            logger.info(
                f"Creating collection '{self.collection_name}' with vector size {self.embedder.dimension}..."
            )
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=qmodels.VectorParams(
                    size=self.embedder.dimension,
                    distance=qmodels.Distance.COSINE
                )
            )

    def _ensure_payload_indexes(self) -> None:
        """
        Creates Payload Indexes on ABAC security fields.
        Crucial for sub-millisecond pre-filtering during queries!
        """
        abac_fields = [
            ("participants", qmodels.PayloadSchemaType.KEYWORD),
            ("min_role_level", qmodels.PayloadSchemaType.KEYWORD),
            ("department", qmodels.PayloadSchemaType.KEYWORD),
            ("meeting_id", qmodels.PayloadSchemaType.KEYWORD),
            ("speakers", qmodels.PayloadSchemaType.KEYWORD),
        ]
        for field_name, schema_type in abac_fields:
            try:
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name=field_name,
                    field_schema=schema_type,
                )
            except Exception:
                pass

    def ingest_markdown_file(self, md_path: Path | str) -> int:
        """Ingests a single markdown transcript file into Qdrant."""
        p = Path(md_path)
        if not p.exists():
            raise FileNotFoundError(f"Markdown file does not exist: {p}")

        content = p.read_text(encoding="utf-8")
        chunks = self.chunker.chunk_markdown_file(content)

        if not chunks:
            logger.warning(f"No chunks extracted from {p.name}")
            return 0

        self.ingest_chunks(chunks)
        return len(chunks)

    def ingest_chunks(self, chunks: List[DialogueChunk]) -> None:
        """Embeds and uploads a batch of DialogueChunks into Qdrant."""
        if not chunks:
            return

        texts = [c.text for c in chunks]
        logger.info(f"Generating embeddings for {len(texts)} chunks...")
        embeddings = self.embedder.embed_texts(texts)

        points: List[qmodels.PointStruct] = []
        for chunk, vector in zip(chunks, embeddings):
            point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk.chunk_id))
            payload = chunk.to_payload()

            points.append(qmodels.PointStruct(
                id=point_id,
                vector=vector,
                payload=payload
            ))

        self.client.upsert(
            collection_name=self.collection_name,
            points=points,
            wait=True
        )
        logger.info(f"Successfully upserted {len(points)} chunks into '{self.collection_name}'")

    def ingest_directory(self, dir_path: Path | str) -> int:
        """Scans a directory for all .md files and ingests them."""
        folder = Path(dir_path)
        md_files = list(folder.glob("*.md"))
        total = 0
        logger.info(f"Found {len(md_files)} markdown files in {folder}...")
        for mf in md_files:
            try:
                cnt = self.ingest_markdown_file(mf)
                total += cnt
            except Exception as e:
                logger.error(f"Failed to ingest {mf.name}: {e}", exc_info=True)
        return total


def main():
    parser = argparse.ArgumentParser(description="Ingest Markdown transcripts into Qdrant vector database.")
    parser.add_argument("--file", "-f", type=str, help="Path to single .md transcript file.")
    parser.add_argument("--dir", "-d", type=str, help="Directory containing .md files.")
    parser.add_argument("--collection", type=str, default=DEFAULT_COLLECTION_NAME, help="Qdrant collection name.")
    parser.add_argument("--qdrant-path", type=str, default=DEFAULT_QDRANT_PATH, help="Path to local Qdrant disk storage.")
    parser.add_argument("--qdrant-url", type=str, default=None, help="Remote Qdrant URL (optional).")
    parser.add_argument("--qdrant-api-key", type=str, default=None, help="Remote Qdrant API Key (optional).")

    args = parser.parse_args()

    pipeline = IngestionPipeline(
        collection_name=args.collection,
        qdrant_url=args.qdrant_url,
        qdrant_api_key=args.qdrant_api_key,
        qdrant_path=args.qdrant_path
    )

    if args.file:
        pipeline.ingest_markdown_file(args.file)
    elif args.dir:
        pipeline.ingest_directory(args.dir)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
