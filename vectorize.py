"""Load chunked JSONL records into Chroma using local sentence-transformer embeddings."""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, List, Sequence

import chromadb
from chromadb.api import ClientAPI
from chromadb.api.models.Collection import Collection
from sentence_transformers import SentenceTransformer

DEFAULT_CHUNKS_DIR = Path("data/processed/chunks")
DEFAULT_VECTOR_DIR = Path("data/processed/chroma")
DEFAULT_COLLECTION = "financial_statements"
DEFAULT_EMBED_MODEL = "all-mpnet-base-v2"
DEFAULT_BATCH_SIZE = 32


@dataclass
class ChunkRow:
    chunk_id: str
    text: str
    metadata: dict


def iter_chunk_rows(chunks_dir: Path) -> Iterator[ChunkRow]:
    jsonl_paths = sorted(chunks_dir.glob("*.jsonl"))
    if not jsonl_paths:
        logging.warning("No JSONL chunk files found under %s", chunks_dir)
        return

    for path in jsonl_paths:
        with path.open("r", encoding="utf-8") as fh:
            for line_number, line in enumerate(fh, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    logging.error(
                        "Invalid JSON in %s line %d: %s", path, line_number, exc
                    )
                    continue

                chunk_id = record.get("chunk_id")
                text = record.get("text")
                if not chunk_id or not text:
                    logging.warning(
                        "Skipping record without chunk_id/text in %s line %d",
                        path,
                        line_number,
                    )
                    continue

                metadata = {
                    "company": record.get("company"),
                    "report_type": record.get("report_type"),
                    "report_year": record.get("report_year"),
                    "source_file": record.get("source_file") or path.name,
                    "source_url": record.get("source_url"),
                    "page_start": None,
                    "page_end": None,
                    "tokens": record.get("tokens"),
                }
                page_range = record.get("page_range") or []
                if isinstance(page_range, Sequence) and len(page_range) == 2:
                    metadata["page_start"] = page_range[0]
                    metadata["page_end"] = page_range[1]

                yield ChunkRow(chunk_id=chunk_id, text=text, metadata=metadata)


def batched(iterable: Iterable[ChunkRow], batch_size: int) -> Iterator[list[ChunkRow]]:
    batch: list[ChunkRow] = []
    for item in iterable:
        batch.append(item)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def embed_texts(
    embedder: SentenceTransformer, texts: Sequence[str]
) -> List[List[float]]:
    embeddings = embedder.encode(
        list(texts),
        convert_to_numpy=False,
        show_progress_bar=False,
        batch_size=min(16, len(texts) or 1),
    )
    if len(embeddings) != len(texts):
        raise ValueError("Embedding response size mismatch.")
    return [vector.tolist() for vector in embeddings]


def upsert_batch(
    collection: Collection, rows: Sequence[ChunkRow], vectors: Sequence[Sequence[float]]
) -> None:
    collection.upsert(
        ids=[row.chunk_id for row in rows],
        embeddings=list(vectors),
        documents=[row.text for row in rows],
        metadatas=[row.metadata for row in rows],
    )


def init_clients(
    args: argparse.Namespace,
) -> tuple[SentenceTransformer, ClientAPI, Collection]:
    embedder = SentenceTransformer(args.embedding_model)
    chroma_client = chromadb.PersistentClient(path=str(args.vector_dir))
    if args.reset:
        logging.info("Resetting collection '%s'", args.collection)
        try:
            chroma_client.delete_collection(args.collection)
        except ValueError:
            logging.info(
                "Collection '%s' did not exist; skipping delete.", args.collection
            )
    collection = chroma_client.get_or_create_collection(args.collection)
    return embedder, chroma_client, collection


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Embed chunk JSONL files into a Chroma vector store."
    )
    parser.add_argument(
        "--chunks-dir",
        type=Path,
        default=DEFAULT_CHUNKS_DIR,
        help="Folder containing chunked JSONL files.",
    )
    parser.add_argument(
        "--vector-dir",
        type=Path,
        default=DEFAULT_VECTOR_DIR,
        help="Chroma persistence directory.",
    )
    parser.add_argument(
        "--collection",
        type=str,
        default=DEFAULT_COLLECTION,
        help="Collection name inside Chroma.",
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default=DEFAULT_EMBED_MODEL,
        help="SentenceTransformer model name or local path.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Number of chunks per embedding request.",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete any existing collection entries before loading.",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()
    embedder, _, collection = init_clients(args)

    total = 0
    for batch in batched(iter_chunk_rows(args.chunks_dir), args.batch_size):
        texts = [row.text for row in batch]
        vectors = embed_texts(embedder, texts)
        upsert_batch(collection, batch, vectors)
        total += len(batch)
        logging.info("Indexed %d chunks so far", total)

    logging.info(
        "Finished indexing %d chunks into collection '%s'", total, args.collection
    )


if __name__ == "__main__":
    main()
