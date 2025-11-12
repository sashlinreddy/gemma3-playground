# Gemma 3 L&D Assistant POC

This repo holds a learning-and-development assistant prototype powered by Google’s Gemma 3 models. It ingests public financial statements, chunks them, embeds the text into a Chroma vector store, and retrieves grounded context for Gemma 3 to answer banker-style questions with citations. The L&D code now lives under `ld/` to keep it separate from the skin-care playground tools.

## Documentation

- [Getting Started](docs/getting_started.md)
- [Ingestion & Chunking](docs/ingestion.md)
- [Vectorization](docs/vectorization.md)
- [Retrieval & Answering](docs/retrieval.md)
- [Roadmap](docs/roadmap.md)

Use the docs for detailed walkthroughs; the sections below summarize the core pipeline.

1. Ingest PDFs with `ld/ingest.py` (writes JSONL chunks).
2. Vectorize chunks with `ld/vectorize.py` (SentenceTransformer → Chroma).
3. Query via `ld/answer.py` (single shot) or `ld/chat_cli.py` (interactive, with memory/logging).
4. Iterate using the roadmap checkpoints to add structured tools, recommendation logic, and guardrails.

## Skin-care news digest (no RAG)

The skin-focused experiments now live under `skin/`. Use `skin/skin_news.py` when you need a real-time digest without persisting embeddings:

```bash
uv run python skin/skin_news.py \
  --source-file skin/samples/skin_sources_sample.json \
  --limit 3 \
  --cache-file data/cache/skin_news_cache.json \
  --ttl 900
```

- Reads recent source snippets (the sample file under `skin/samples/` mocks YouTube transcripts; swap in real fetcher output).
- Builds a Gemma 3 prompt on the fly and returns a Markdown digest tuned for a fantasy “skin stock” mechanic.
- Saves the latest result under `data/cache/...` so repeated calls within the TTL reuse the summary.
- Pass `--force-refresh` to bypass the cache or `--output digest.md` to capture the rendered digest.

### Live YouTube pulls

Point the same script at real channels by enabling the YouTube Data API (set `YOUTUBE_API_KEY` in `.env`):

```bash
uv run python skin/skin_news.py \
  --youtube-channel-id https://www.youtube.com/@MasterShiny \
  --youtube-channel-id https://www.youtube.com/@SkinomiCS2 \
  --youtube-playlist-id https://www.youtube.com/playlist?list=PLjRE1qx9Gn8jow1mHKumOlmV01Gtc0JCy \
  --youtube-max-results 3 \
  --youtube-playlist-max-results 2 \
  --skip-source-file \
  --limit 5
```

- Fetches the latest videos per channel, grabs transcripts via `youtube-transcript-api`, and feeds them straight to Gemma.
- Accepts channel IDs, handles (`@channel`), or full URLs; playlist URLs/IDs are supported via `--youtube-playlist-id`.
- Reuses the same cache (keyed by video IDs) so repeated calls within the TTL stay instant.
- Keep `--source-file` around for fixtures; add `--skip-source-file` (as above) to ignore it when hitting real data.

Find additional skin-specific notes in `skin/README.md`.
