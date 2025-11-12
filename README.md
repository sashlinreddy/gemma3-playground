# Gemma 3 L&D Assistant POC

This repo holds a learning-and-development assistant prototype powered by Google’s Gemma 3 models. It ingests public financial statements, chunks them, embeds the text into a Chroma vector store, and retrieves grounded context for Gemma 3 to answer banker-style questions with citations.

## Documentation

- [Getting Started](docs/getting_started.md)
- [Ingestion & Chunking](docs/ingestion.md)
- [Vectorization](docs/vectorization.md)
- [Retrieval & Answering](docs/retrieval.md)
- [Roadmap](docs/roadmap.md)

Use the docs for detailed walkthroughs; the sections below summarize the core pipeline.

1. Ingest PDFs with `ingest.py` (writes JSONL chunks).
2. Vectorize chunks with `vectorize.py` (SentenceTransformer → Chroma).
3. Query via `answer.py` (single shot) or `chat_cli.py` (interactive, with memory/logging).
4. Iterate using the roadmap checkpoints to add structured tools, recommendation logic, and guardrails.
