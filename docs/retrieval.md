# Retrieval & Answering

## Single question

```bash
uv run python ld/answer.py "What are Nestlé's key revenue drivers?" --company "Nestle" --top-k 4
```

Features:
- Optional query rewriting (`--no-rewrite` to disable).
- `--show-query` prints the final search text.
- Streams Gemma 3 output with citations.
- Rewriter sanitizes code-fenced JSON (```json [...] ```) before querying so misformatted rewrites no longer break retrieval.
- The system prompt enforces real chunk citations and honors user word-count/format requests instead of replying with generic limitations when context exists.

## Interactive CLI

```bash
uv run python ld/chat_cli.py --show-context --show-query --history-log logs/session.jsonl
```

- Commands: `:company`, `:year`, `:topk`, `:rewrite`, `:show`, `:help`, `:quit`.
- Keeps short conversation memory for follow-up questions.
- `--history-log` writes JSONL records of each turn (question, answer, query, chunk ids).

## Workflow planner & tool awareness

Both `ld/answer.py` and `ld/chat_cli.py` now call `ld/workflow.plan_workflow` before every turn. The planner uses Gemma 3 to:

- classify the request intent (`retrieval`, `recommendation`, etc.).
- decide whether the future structured product catalog tool is required.
- emit a short rationale and 2-3 step plan that gets injected into the model prompt.
- prefer grounded answers even when the user asks for “steps”, “strategies”, or “pretend you are a strategist,” so the model looks for actionable guidance described in the PDFs before declining.

Flags:

- `ld/answer.py --show-plan`
- `ld/chat_cli.py --show-plan`

When the planner marks `needs_structured_tool=true`, the CLI prints a warning and the prompt reminds Gemma to clearly state that the tool is unavailable before providing any context-based guidance. This creates the decision gate for recommendation-style asks without blocking regular retrieval/summarization flows.
