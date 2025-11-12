# Retrieval & Answering

## Single question

```bash
uv run python ld/answer.py "What are Nestlé's key revenue drivers?" --company "Nestle" --top-k 4
```

Features:
- Optional query rewriting (`--no-rewrite` to disable).
- `--show-query` prints the final search text.
- Streams Gemma 3 output with citations.

## Interactive CLI

```bash
uv run python ld/chat_cli.py --show-context --show-query --history-log logs/session.jsonl
```

- Commands: `:company`, `:year`, `:topk`, `:rewrite`, `:show`, `:help`, `:quit`.
- Keeps short conversation memory for follow-up questions.
- `--history-log` writes JSONL records of each turn (question, answer, query, chunk ids).
