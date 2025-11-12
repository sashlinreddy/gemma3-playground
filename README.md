# Gemma 3 L&D Assistant POC

This repo holds a learning-and-development assistant prototype powered by Google’s Gemma 3 models. The POC workflow is:

1. **Curate sample content** – drop raw PDFs into `data/raw/<source>` (already populated with public financial statements).
2. **Ingest & chunk** – convert each PDF into ~800 token chunks with 100 token overlap so retrieval stays specific but coherent.
3. **Embed & store** – encode chunks with an embedding model and push them into a local vector store along with metadata.
4. **Query with Gemma 3** – retrieve the most relevant chunks for a banker question, feed them to Gemma 3 in the prompt, and surface the synthesized answer plus citations.

## Data layout

```
data/
  raw/          # original PDFs (read-only once ingested)
  processed/
    chunks/     # JSONL files: one row per chunk with text + metadata
    embeddings/ # (later) serialized vector store snapshots
```

Each chunk record follows this schema:

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

Saving the intermediate JSONL lets you re-run the embedding step without redoing PDF extraction, which speeds up iteration while experimenting with different vector stores or chunking strategies.

## Running the ingestion script

Install deps once (`uv sync`), then run:

```bash
uv run python ingest.py --chunk-size 800 --overlap 100
```

This scans every PDF in `data/raw`, emits `data/processed/chunks/<file>.jsonl`, and logs how many chunks were created. These JSONL files are the direct inputs for:

- **Embedding jobs** – feed `text` into your embedding model, then persist vectors + metadata.  
- **Retrieval inspection** – quickly spot-check by loading the JSONL in a notebook and verifying that each chunk’s page range matches the original context.  
- **Gemma prompt construction** – select the top-N chunks at query time, drop them into the model context, and include their `chunk_id`/`page_range` for citations.

Once embeddings are stored (e.g., in Chroma or pgvector), wire a small retriever that filters by `company` or `report_year` before scoring similarity; return both the chunk text and metadata so the assistant can cite sources and transition into the recommendation logic.

## Loading chunks into Chroma

Run `uv sync` (to pull in `chromadb` + `sentence-transformers`), then:

```bash
uv run python vectorize.py --reset
```

What it does:

- Reads every JSONL file in `data/processed/chunks`.
- Encodes each chunk with a local SentenceTransformer model (default: `all-mpnet-base-v2`; override via `--embedding-model path/or/name`).
- Upserts the vectors, source text, and metadata into a persistent Chroma collection located at `data/processed/chroma`.

You can rerun the command without `--reset` to append newly ingested PDFs; use the flag when you want a clean rebuild. Once the collection exists, any FastAPI/Gradio prototype can open the same persistence directory, query by similarity (optionally filtering on `company`/`report_year` metadata), and feed the retrieved chunks into Gemma 3 for grounded responses.

## Asking Gemma 3 grounded questions

With the vector store ready and `GEMINI_API_KEY` set, you can retrieve context + call the model via:

```bash
uv run python answer.py "What are Nestlé's key revenue drivers?" --company "Nestle" --top-k 4
```

The script:

- Loads the Chroma collection from `data/processed/chroma`.
- Embeds the question using the same SentenceTransformer model (override via `--embedding-model` if needed).
- Rewrites the question into search-friendly variants (disable via `--no-rewrite`) before embedding.
- Retrieves the top-k chunks (optionally filtered by `--company` / `--year`), printing the final query when `--show-query` is set.
- Builds a context block and sends it to `gemma-3-27b-it`, enforcing citations like `[chunk-id]`.
- Streams the model’s response so you can watch Gemma cite sources in real time.

### Interactive CLI

For rapid experimentation, launch the REPL-style assistant:

```bash
uv run python chat_cli.py --show-context --show-query --history-log logs/session.jsonl
```

Available commands while the session is running:

- `:company <name>` / `:year <YYYY>` / `:topk <n>` – adjust filters without restarting.
- `:rewrite` – toggle the query-rewriting step on/off; `:show` toggles whether the retrieved context is printed.
- `:help` – list commands; `:quit` (or Ctrl+D/Ctrl+C) exits.

Each free-form question runs the same retrieval + Gemma pipeline as `answer.py`, while the CLI now keeps a rolling memory of recent Q&A so simple follow-ups (“What about Nestlé’s margins?”) automatically inherit prior context in the prompt. Pass `--no-rewrite` if you want to disable the rewrite step for the entire session, or use `:rewrite` to toggle it on the fly.

Provide `--history-log path/to/log.jsonl` to append each turn (question, answer, query used, citations) to a JSONL file for debugging or later analysis.
