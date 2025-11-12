"""Utility script for converting raw PDFs into chunked JSONL records."""

from __future__ import annotations

import argparse
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

import pdfplumber

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
CHUNK_DIR = PROCESSED_DIR / "chunks"

CHUNK_TOKEN_TARGET = 800
CHUNK_TOKEN_OVERLAP = 100


@dataclass
class ChunkRecord:
    chunk_id: str
    company: str
    report_type: str
    report_year: int | None
    source_file: str
    source_url: str | None
    page_range: tuple[int, int]
    tokens: int
    text: str

    def to_json(self) -> str:
        payload = {
            "chunk_id": self.chunk_id,
            "company": self.company,
            "report_type": self.report_type,
            "report_year": self.report_year,
            "source_file": self.source_file,
            "source_url": self.source_url,
            "page_range": list(self.page_range),
            "tokens": self.tokens,
            "text": self.text,
        }
        return json.dumps(payload, ensure_ascii=False)


def infer_metadata(path: Path) -> dict:
    stem = path.stem.replace("%20", " ")
    year_match = re.search(r"(20\d{2})", stem)
    report_year = int(year_match.group(1)) if year_match else None

    cleaned = re.sub(r"[_\-]+", " ", stem)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    company = cleaned.title() if cleaned else path.stem

    return {
        "company": company,
        "company_slug": re.sub(r"[^a-z0-9]+", "-", company.lower()).strip("-") or "company",
        "report_type": "Unknown",
        "report_year": report_year,
        "source_url": None,
    }


def extract_pdf_text(pdf_path: Path) -> List[tuple[int, str]]:
    """Returns a list of (page_number, text) tuples."""
    pages: List[tuple[int, str]] = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_number, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            text = text.strip()
            if text:
                pages.append((page_number, text))
    return pages


def chunk_pages(
    pages: Sequence[tuple[int, str]],
    chunk_size: int,
    overlap: int,
) -> Iterable[dict]:
    assert chunk_size > overlap, "chunk_size must be greater than overlap"
    buffer: List[tuple[str, int]] = []

    def flush_buffer(words_with_page: Sequence[tuple[str, int]]) -> dict | None:
        if not words_with_page:
            return None
        words = [word for word, _ in words_with_page]
        pages_used = [page for _, page in words_with_page]
        return {
            "text": " ".join(words),
            "page_start": min(pages_used),
            "page_end": max(pages_used),
            "tokens": len(words),
        }

    for page_number, text in pages:
        words = text.split()
        for word in words:
            buffer.append((word, page_number))
            if len(buffer) == chunk_size:
                chunk = flush_buffer(buffer)
                if chunk:
                    yield chunk
                buffer = buffer[-overlap:]

    if buffer:
        chunk = flush_buffer(buffer)
        if chunk:
            yield chunk


def build_chunks(pdf_path: Path, chunk_size: int, overlap: int) -> List[ChunkRecord]:
    metadata = infer_metadata(pdf_path)
    pages = extract_pdf_text(pdf_path)
    if not pages:
        logging.warning("No text extracted from %s, skipping.", pdf_path.name)
        return []

    slug = metadata["company_slug"] or pdf_path.stem.lower()
    chunks: List[ChunkRecord] = []
    for idx, chunk in enumerate(chunk_pages(pages, chunk_size=chunk_size, overlap=overlap), start=1):
        chunk_id = f"{slug}-{metadata['report_year'] or 'na'}-{idx:04d}"
        chunks.append(
            ChunkRecord(
                chunk_id=chunk_id,
                company=metadata["company"],
                report_type=metadata["report_type"],
                report_year=metadata["report_year"],
                source_file=pdf_path.name,
                source_url=metadata["source_url"],
                page_range=(chunk["page_start"], chunk["page_end"]),
                tokens=chunk["tokens"],
                text=chunk["text"],
            )
        )
    return chunks


def write_chunks(chunks: Sequence[ChunkRecord], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for record in chunks:
            f.write(record.to_json())
            f.write("\n")


def ingest_pdfs(raw_dir: Path, chunk_dir: Path, chunk_size: int, overlap: int) -> None:
    pdf_paths = sorted(raw_dir.glob("*.pdf"))
    if not pdf_paths:
        logging.warning("No PDFs found under %s", raw_dir)
        return

    for pdf in pdf_paths:
        logging.info("Processing %s", pdf.name)
        chunks = build_chunks(pdf, chunk_size=chunk_size, overlap=overlap)
        if not chunks:
            continue
        output_path = chunk_dir / f"{pdf.stem}.jsonl"
        write_chunks(chunks, output_path)
        logging.info("Wrote %s (%d chunks)", output_path, len(chunks))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest PDFs into chunked JSONL files.")
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR, help="Folder containing raw PDFs.")
    parser.add_argument(
        "--chunk-dir",
        type=Path,
        default=CHUNK_DIR,
        help="Destination folder for chunked JSONL files.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=CHUNK_TOKEN_TARGET,
        help="Approximate number of word tokens per chunk.",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=CHUNK_TOKEN_OVERLAP,
        help="Approximate token overlap between consecutive chunks.",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()
    ingest_pdfs(args.raw_dir, args.chunk_dir, chunk_size=args.chunk_size, overlap=args.overlap)


if __name__ == "__main__":
    main()
