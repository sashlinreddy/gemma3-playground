"""Query the Chroma store and ask Gemma 3 for a grounded answer."""

from __future__ import annotations

import argparse
import json
import os
import textwrap
from dataclasses import dataclass
from typing import Iterator, List, Sequence, Tuple

import chromadb
from chromadb.api.models.Collection import Collection
from dotenv import load_dotenv
from google import genai
from sentence_transformers import SentenceTransformer

from workflow import WorkflowDecision, plan_workflow

DEFAULT_VECTOR_DIR = "data/processed/chroma"
DEFAULT_COLLECTION = "financial_statements"
DEFAULT_EMBED_MODEL = "all-mpnet-base-v2"
DEFAULT_MODEL = "gemma-3-27b-it"

SYSTEM_PROMPT = """You are a banking learning-and-development assistant.
Answer questions using ONLY the supplied context snippets from official financial statements.
Instructions:
- Always cite the exact chunk_id(s) shown in the context headers, e.g. [2025-interim-results-analyst-presentation-nedbank-2025-0006]. Never use placeholders like [chunk-id(s): all] or [source].
- Obey any requested length or format constraints (e.g., “40 words”) while still including citations.
- When relevant context exists, synthesize the best possible answer instead of responding with generic limitations.
- If the context truly lacks the required information, say so clearly and explain what is missing without inventing facts or citing nonexistent chunks."""

REWRITE_PROMPT = """You rewrite banker questions into precise search queries for financial documents.
Guidelines:
- Focus on explicit entities, metrics, periods, and product names.
- Include synonyms or alternative phrasing if helpful.
- Do NOT answer the question.
- Return a JSON list of 1-2 strings. Example: ["loan growth Standard Bank 2025", "Nedbank interim results net interest income"].

User question: {question}
Company filter: {company}
Report year filter: {year}
Recent conversation: {history}
"""


@dataclass
class RetrievedChunk:
    chunk_id: str
    text: str
    metadata: dict


ConversationTurn = Tuple[str, str]


def load_collection(args: argparse.Namespace) -> Collection:
    client = chromadb.PersistentClient(path=args.vector_dir)
    return client.get_or_create_collection(args.collection)


def format_history(history: Sequence[ConversationTurn], limit: int = 5) -> str:
    if not history:
        return "None."
    recent = history[-limit:]
    segments = []
    for question, answer in recent:
        segments.append(f"User: {question}\nAssistant: {answer}")
    return "\n\n".join(segments)


def _parse_candidate_list(raw: str) -> List[str] | None:
    if not raw:
        return None
    try:
        data = json.loads(raw)
        if isinstance(data, list):
            return [str(item).strip() for item in data]
    except json.JSONDecodeError:
        pass

    start = raw.find("[")
    end = raw.rfind("]")
    if 0 <= start < end:
        snippet = raw[start : end + 1]
        try:
            data = json.loads(snippet)
            if isinstance(data, list):
                return [str(item).strip() for item in data]
        except json.JSONDecodeError:
            pass
    return None


def rewrite_queries(
    client: genai.Client,
    question: str,
    company: str | None,
    year: int | None,
    model: str,
    history: Sequence[ConversationTurn] | None = None,
) -> List[str]:
    prompt = REWRITE_PROMPT.format(
        question=question.strip(),
        company=company or "None",
        year=year or "None",
        history=format_history(history or [], limit=3),
    )
    response = client.models.generate_content(model=model, contents=prompt)
    raw = (response.text or "").strip()
    if not raw:
        return []

    candidates = _parse_candidate_list(raw)
    if candidates is None:
        candidates = []
        for piece in raw.splitlines():
            cleaned = piece.strip(" -*•`\"[],")
            lowered = cleaned.lower()
            if not cleaned or lowered in {"json", "`", "```"}:
                continue
            candidates.append(cleaned)

    cleaned = []
    for candidate in candidates:
        if not candidate:
            continue
        if candidate not in cleaned:
            cleaned.append(candidate)
    return cleaned[:2]


def retrieve_chunks(
    collection: Collection,
    embedder: SentenceTransformer,
    question: str,
    top_k: int,
    company: str | None,
    year: int | None,
) -> List[RetrievedChunk]:
    vector = embedder.encode(question, convert_to_numpy=False).tolist()
    where = {}
    if company:
        where["company"] = company
    if year:
        where["report_year"] = year

    results = collection.query(
        query_embeddings=[vector],
        n_results=top_k,
        where=where or None,
        include=["documents", "metadatas", "distances"],
    )

    ids = results.get("ids", [[]])[0]
    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]

    chunks: List[RetrievedChunk] = []
    for chunk_id, text, metadata in zip(ids, documents, metadatas):
        if not chunk_id or not text:
            continue
        chunks.append(
            RetrievedChunk(
                chunk_id=chunk_id,
                text=text,
                metadata=metadata or {},
            )
        )
    return chunks


def build_context(chunks: Sequence[RetrievedChunk]) -> str:
    sections = []
    for chunk in chunks:
        page_info = ""
        if chunk.metadata.get("page_start") and chunk.metadata.get("page_end"):
            page_info = (
                f"Pages {chunk.metadata['page_start']}-{chunk.metadata['page_end']}"
            )
        header = f"[{chunk.chunk_id}] {chunk.metadata.get('company', 'Unknown company')} {page_info}".strip()
        body = textwrap.shorten(chunk.text, width=1500, placeholder=" ...")
        sections.append(f"{header}\n{body}")
    return "\n\n".join(sections)


def _format_workflow_block(workflow: WorkflowDecision | None) -> str:
    if not workflow:
        return ""

    plan_lines = "\n".join(f"- {step}" for step in workflow.plan)
    limitation = ""
    if workflow.needs_structured_tool:
        limitation = (
            "\nNOTE: The workflow planner indicates that answering requires the "
            "structured product lookup tool, which is not available. Explain this "
            "limitation clearly before providing any interim guidance."
        )

    return (
        "Workflow guidance:\n"
        f"Intent: {workflow.intent}\n"
        f"Reasoning: {workflow.reasoning}\n"
        f"Needs structured tool: {workflow.needs_structured_tool}\n"
        f"Plan:\n{plan_lines or '- (none)'}{limitation}\n\n"
    )


def _build_prompt(
    question: str,
    context: str,
    history: Sequence[ConversationTurn] | None = None,
    workflow: WorkflowDecision | None = None,
) -> str:
    history_block = format_history(history or [], limit=5)
    workflow_block = _format_workflow_block(workflow)
    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"{workflow_block}"
        f"Conversation so far:\n{history_block}\n\n"
        f"Context:\n{context or 'N/A'}\n\n"
        f"Question: {question}\nAnswer:"
    )


def stream_gemma(
    client: genai.Client,
    question: str,
    context: str,
    model: str,
    history: Sequence[ConversationTurn] | None = None,
    workflow: WorkflowDecision | None = None,
) -> Iterator[str]:
    prompt = _build_prompt(question, context, history=history, workflow=workflow)
    stream = client.models.generate_content_stream(model=model, contents=prompt)
    for chunk in stream:
        text = getattr(chunk, "text", None)
        if text:
            yield text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ask Gemma 3 grounded questions over the Chroma store."
    )
    parser.add_argument("question", type=str, help="User question to answer.")
    parser.add_argument(
        "--vector-dir",
        type=str,
        default=DEFAULT_VECTOR_DIR,
        help="Path to the Chroma persistence directory.",
    )
    parser.add_argument(
        "--collection",
        type=str,
        default=DEFAULT_COLLECTION,
        help="Chroma collection name.",
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default=DEFAULT_EMBED_MODEL,
        help="SentenceTransformer model name/path used for queries.",
    )
    parser.add_argument(
        "--model", type=str, default=DEFAULT_MODEL, help="Gemma model to call."
    )
    parser.add_argument(
        "--top-k", type=int, default=4, help="Number of chunks to retrieve."
    )
    parser.add_argument(
        "--company",
        type=str,
        help="Optional company filter that must match chunk metadata.",
    )
    parser.add_argument("--year", type=int, help="Optional report year filter.")
    parser.add_argument(
        "--no-rewrite",
        action="store_true",
        help="Disable query rewriting before hitting the vector store.",
    )
    parser.add_argument(
        "--show-query",
        action="store_true",
        help="Print the actual query text used for retrieval.",
    )
    parser.add_argument(
        "--show-plan",
        action="store_true",
        help="Print the workflow planner decision before answering.",
    )
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("Set GEMINI_API_KEY in your environment or .env file.")

    args = parse_args()
    client = genai.Client(api_key=api_key)
    collection = load_collection(args)
    embedder = SentenceTransformer(args.embedding_model)

    workflow = plan_workflow(
        client,
        question=args.question,
        history=None,
        model=args.model,
    )
    if args.show_plan:
        print(workflow.pretty())
    if workflow.needs_structured_tool:
        print(
            "⚠️ Planner flagged this question as requiring the structured product "
            "catalog tool, which is not available yet. Proceeding with best-effort "
            "retrieval answer."
        )

    candidate_queries: List[str] = []
    if not args.no_rewrite:
        rewrites = rewrite_queries(
            client=client,
            question=args.question,
            company=args.company,
            year=args.year,
            model=args.model,
        )
        candidate_queries.extend(rewrites)
    candidate_queries.append(args.question)

    ordered_queries: List[str] = []
    seen = set()
    for query in candidate_queries:
        q = query.strip()
        if not q or q in seen:
            continue
        ordered_queries.append(q)
        seen.add(q)

    chunks: List[RetrievedChunk] = []
    used_query = args.question
    for query in ordered_queries:
        chunks = retrieve_chunks(
            collection, embedder, query, args.top_k, args.company, args.year
        )
        if chunks:
            used_query = query
            break

    if not chunks:
        print("No relevant chunks found. Try relaxing filters or ingesting more data.")
        return

    if args.show_query:
        print(f"Retrieval query: {used_query}")

    context = build_context(chunks)

    print("=== Retrieved Context ===")
    print(context)
    print("\n=== Answer ===")
    for text in stream_gemma(
        client,
        args.question,
        context,
        args.model,
        workflow=workflow,
    ):
        print(text, end="", flush=True)
    print()


if __name__ == "__main__":
    main()
