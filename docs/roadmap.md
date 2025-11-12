# Roadmap

1. **Structured data retrievers** – add a product catalog + `lookup_product` tool for pricing/eligibility questions.
2. **Query refinement loop** – if Gemma lacks citations, automatically rerun retrieval with clarified queries before responding.
3. **Recommendation bridge** – create a `client_profile` object and map needs → product recommendations with rationales.
4. **Guardrails & evaluation** – log prompt/response pairs, enforce citations, and build an eval set of banker questions.
5. **Deployment shell** – wrap the assistant in FastAPI/Gradio with auth, logging, and metrics.
