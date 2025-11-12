"""Interactive CLI for asking multiple Gemma 3 questions in one session."""

from __future__ import annotations

import argparse
import json
import os
import sys

from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from sentence_transformers import SentenceTransformer

from answer import (
    DEFAULT_COLLECTION,
    DEFAULT_EMBED_MODEL,
    DEFAULT_MODEL,
    DEFAULT_VECTOR_DIR,
    build_context,
    load_collection,
    rewrite_queries,
    retrieve_chunks,
    stream_gemma,
)

DEFAULT_TOP_K = 4
COMMANDS_HELP = """Commands:
  :q / :quit / :exit   Leave the session
  :show                Toggle printing of retrieved context
  :company <name>      Set a company filter (blank to clear)
  :year <YYYY>         Set a report year filter (blank to clear)
  :topk <n>            Change number of chunks to fetch
  :rewrite             Toggle query rewriting on/off
  :help                Show this message
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Interactive retrieval-augmented Gemma 3 CLI."
    )
    parser.add_argument("--vector-dir", type=str, default=DEFAULT_VECTOR_DIR)
    parser.add_argument("--collection", type=str, default=DEFAULT_COLLECTION)
    parser.add_argument("--embedding-model", type=str, default=DEFAULT_EMBED_MODEL)
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--company", type=str)
    parser.add_argument("--year", type=int)
    parser.add_argument(
        "--show-context",
        action="store_true",
        help="Print retrieved snippets before answers.",
    )
    parser.add_argument(
        "--no-rewrite",
        action="store_true",
        help="Disable query rewriting before retrieval.",
    )
    parser.add_argument(
        "--show-query", action="store_true", help="Print the query used for retrieval."
    )
    parser.add_argument(
        "--history-log",
        type=Path,
        help="Optional path to append chat history as JSONL for debugging.",
    )
    return parser.parse_args()


def should_exit(text: str) -> bool:
    return text.lower() in {":q", ":quit", ":exit"}


def handle_meta_command(
    raw: str,
    state: dict,
) -> bool:
    parts = raw.split(maxsplit=1)
    cmd = parts[0].lower()

    if cmd in {":help", ":h"}:
        print(COMMANDS_HELP)
        return True
    if cmd == ":show":
        state["show_context"] = not state["show_context"]
        print(f"Show context: {state['show_context']}")
        return True
    if cmd == ":company":
        state["company"] = parts[1].strip() if len(parts) > 1 else None
        print(f"Company filter set to: {state['company'] or 'None'}")
        return True
    if cmd == ":year":
        value = parts[1].strip() if len(parts) > 1 else ""
        if not value:
            state["year"] = None
        else:
            state["year"] = int(value)
        print(f"Year filter set to: {state['year'] or 'None'}")
        return True
    if cmd == ":topk":
        if len(parts) == 1:
            raise ValueError("Usage: :topk <number>")
        value = int(parts[1])
        if value < 1:
            raise ValueError("topk must be >= 1")
        state["top_k"] = value
        print(f"Top-k now {state['top_k']}")
        return True
    if cmd == ":rewrite":
        state["rewrite"] = not state["rewrite"]
        print(f"Query rewriting: {state['rewrite']}")
        return True
    return False


def append_history_log(
    path: Path,
    *,
    question: str,
    answer: str,
    query: str | None,
    company: str | None,
    year: int | None,
    chunk_ids: list[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "question": question,
        "answer": answer,
        "query_used": query,
        "company_filter": company,
        "year_filter": year,
        "chunk_ids": chunk_ids,
    }
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False))
        fh.write("\n")


def interactive_loop(
    client: genai.Client,
    embedder: SentenceTransformer,
    state: dict,
) -> None:
    collection = load_collection(
        argparse.Namespace(
            vector_dir=state["vector_dir"], collection=state["collection"]
        )
    )
    print("Interactive L&D assistant. Type :help for commands, :quit to exit.")
    while True:
        try:
            user_input = input("\nQuestion> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not user_input:
            continue
        if user_input.startswith(":"):
            if should_exit(user_input):
                print("Goodbye!")
                break
            try:
                handled = handle_meta_command(user_input, state)
            except Exception as exc:
                print(f"Command error: {exc}")
            else:
                if not handled:
                    print("Unknown command. Type :help for options.")
            continue

        candidate_queries: list[str] = []
        if state["rewrite"]:
            rewrites = rewrite_queries(
                client=client,
                question=user_input,
                company=state["company"],
                year=state["year"],
                model=state["model"],
                history=state["history"],
            )
            candidate_queries.extend(rewrites)
        candidate_queries.append(user_input)

        ordered_queries: list[str] = []
        seen: set[str] = set()
        for query in candidate_queries:
            q = query.strip()
            if not q or q in seen:
                continue
            ordered_queries.append(q)
            seen.add(q)

        chunks = []
        used_query = None
        for query in ordered_queries:
            chunks = retrieve_chunks(
                collection=collection,
                embedder=embedder,
                question=query,
                top_k=state["top_k"],
                company=state["company"],
                year=state["year"],
            )
            if chunks:
                used_query = query
                break

        if not chunks:
            print("No relevant context found. Try adjusting filters.")
            continue

        context = build_context(chunks)
        if state["show_query"] and used_query:
            print(f"\nQuery used: {used_query}")
        if state["show_context"]:
            print("\n--- Context ---")
            print(context)
            print("---------------")

        print("\nAnswer:")
        answer_chunks: list[str] = []
        for text in stream_gemma(
            client, user_input, context, state["model"], history=state["history"]
        ):
            print(text, end="", flush=True)
            answer_chunks.append(text)
        print()

        answer_text = "".join(answer_chunks).strip()
        state["history"].append((user_input, answer_text))

        if state["history_log"]:
            append_history_log(
                state["history_log"],
                question=user_input,
                answer=answer_text,
                query=used_query,
                company=state["company"],
                year=state["year"],
                chunk_ids=[chunk.chunk_id for chunk in chunks],
            )


def main() -> None:
    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("Set GEMINI_API_KEY in your environment or .env file.")

    args = parse_args()
    embedder = SentenceTransformer(args.embedding_model)
    client = genai.Client(api_key=api_key)

    state = {
        "vector_dir": args.vector_dir,
        "collection": args.collection,
        "top_k": args.top_k,
        "company": args.company,
        "year": args.year,
        "model": args.model,
        "show_context": args.show_context,
        "rewrite": not args.no_rewrite,
        "show_query": args.show_query,
        "history": [],
        "history_log": args.history_log,
    }

    interactive_loop(client, embedder, state)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
