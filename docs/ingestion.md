# Ingestion & Chunking

## Folder layout

```
data/
  raw/
  processed/
    chunks/
```

## Running the ingestion script

```bash
uv run python ingest.py --chunk-size 800 --overlap 100
```

- Converts PDFs to JSONL chunks with metadata.
- Stored at `data/processed/chunks/<file>.jsonl`.

## JSONL schema

```json
{
  "chunk_id": "sbk-2025-001",
  "company": "Standard Bank Group",
  "source_file": "SBG2025InterimResultsOverview.pdf",
  "source_url": "https://example.com/results",
  "report_type": "Interim Results",
  "report_year": 2025,
  "page_range": [12, 14],
  "text": "...chunk content...",
  "tokens": 735
}
```
