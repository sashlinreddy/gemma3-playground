# Vectorization

## Command

```bash
uv run python vectorize.py --reset
```

- Reads all JSONL chunk files from `data/processed/chunks`.
- Embeds text with a local SentenceTransformer (default `all-mpnet-base-v2`; override with `--embedding-model`).
- Upserts ids/documents/metadata into a persistent Chroma collection located at `data/processed/chroma`.

Re-run without `--reset` to append new PDFs. Use `--reset` whenever you want a clean rebuild.
