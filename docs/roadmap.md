# Roadmap

## Progress Checklist

- [x] PDF ingestion & chunking pipeline (`ingest.py`)
- [x] Local embeddings + Chroma vector store (`vectorize.py`)
- [x] Retrieval layer with query rewriting, streaming answers, and interactive CLI memory (`answer.py`, `chat_cli.py`)
- [x] Agentic workflow planner that flags recommendation/tool needs (`ld/workflow.py`)
- [ ] Structured data retrievers (product catalog tool) – *deferred until documentation/pricing data stabilizes*
- [ ] Query refinement loop with self-checks/citation validation
- [ ] Recommendation bridge with `client_profile` and product mapping
- [ ] Guardrails & evaluation harness
- [ ] FastAPI/Gradio deployment shell with auth/logging

## Upcoming Steps

1. **Query refinement loop** – if Gemma lacks citations, automatically rerun retrieval with clarified queries before responding.
2. **Recommendation bridge** – create a `client_profile` object and map needs → product recommendations with rationales.
3. **Guardrails & evaluation** – log prompt/response pairs, enforce citations, and build an eval set of banker questions.
4. **Deployment shell** – wrap the assistant in FastAPI/Gradio with auth, logging, and metrics.
5. *(Deferred)* **Structured data retrievers** – revisit once product/pricing documentation is consistent enough to build a reliable `lookup_product` tool.
