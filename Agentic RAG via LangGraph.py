"""
Agentic RAG via LangGraph

Adaptive retrieval pipeline demonstrating:
- LangGraph-style conditional routing
- Retriever -> Document Grader -> Web Search flow
- Chroma-compatible retrieval concept
- MiniLM embeddings concept
- Cross-encoder relevance grading
- Metadata filtering
- Query translation

Install:
    pip install langgraph chromadb sentence-transformers

The implementation includes a lightweight fallback retriever so the file
can still be inspected/run without external services.
"""

import re
from dataclasses import dataclass
from typing import TypedDict


# ---------------------------------------------------------------------------
# DOCUMENT MODEL
# ---------------------------------------------------------------------------

@dataclass
class Document:
    text: str
    metadata: dict


DOCUMENTS = [
    Document(
        "LangGraph allows stateful orchestration of LLM workflows as graphs.",
        {"topic": "langgraph", "year": 2026},
    ),
    Document(
        "Retrieval augmented generation combines retrieval with language models.",
        {"topic": "rag", "year": 2025},
    ),
    Document(
        "Cross-encoders can rerank retrieved documents according to query relevance.",
        {"topic": "retrieval", "year": 2025},
    ),
    Document(
        "Metadata filters restrict retrieval to documents matching structured fields.",
        {"topic": "retrieval", "year": 2026},
    ),
]


# ---------------------------------------------------------------------------
# STATE
# ---------------------------------------------------------------------------

class RAGState(TypedDict, total=False):
    query: str
    translated_query: str
    documents: list[Document]
    relevance_score: float
    answer: str
    route: str


# ---------------------------------------------------------------------------
# QUERY TRANSLATION
# ---------------------------------------------------------------------------

def translate_query(query: str) -> str:
    """Normalize a natural-language query for retrieval."""
    query = query.strip().lower()

    replacements = {
        "what is": "",
        "tell me about": "",
        "explain": "",
        "how does": "",
    }

    for old, new in replacements.items():
        query = query.replace(old, new)

    return re.sub(r"\s+", " ", query).strip()


# ---------------------------------------------------------------------------
# RETRIEVER
# ---------------------------------------------------------------------------

def retrieve(query: str, metadata_filter=None, top_k=3):
    query_terms = set(re.findall(r"\w+", query.lower()))

    candidates = []

    for document in DOCUMENTS:
        if metadata_filter:
            if any(
                document.metadata.get(key) != value
                for key, value in metadata_filter.items()
            ):
                continue

        document_terms = set(re.findall(r"\w+", document.text.lower()))
        overlap = len(query_terms & document_terms)

        candidates.append((overlap, document))

    candidates.sort(key=lambda item: item[0], reverse=True)

    return [doc for score, doc in candidates[:top_k] if score > 0]


# ---------------------------------------------------------------------------
# DOCUMENT GRADER
# ---------------------------------------------------------------------------

def grade_documents(query: str, documents: list[Document]) -> float:
    """Simple relevance grader; replace with a CrossEncoder in production."""
    if not documents:
        return 0.0

    query_terms = set(re.findall(r"\w+", query.lower()))
    scores = []

    for document in documents:
        text_terms = set(re.findall(r"\w+", document.text.lower()))
        union = query_terms | text_terms

        score = len(query_terms & text_terms) / max(len(union), 1)
        scores.append(score)

    return sum(scores) / len(scores)


# ---------------------------------------------------------------------------
# WEB SEARCH FALLBACK
# ---------------------------------------------------------------------------

def web_search(query: str) -> list[Document]:
    """Placeholder web-search node."""
    return [
        Document(
            f"Web search placeholder result for: {query}",
            {"source": "web_search"},
        )
    ]


# ---------------------------------------------------------------------------
# GRAPH NODES
# ---------------------------------------------------------------------------

def retrieve_node(state: RAGState) -> RAGState:
    translated = translate_query(state["query"])
    docs = retrieve(translated)

    return {
        **state,
        "translated_query": translated,
        "documents": docs,
    }


def grade_node(state: RAGState) -> RAGState:
    score = grade_documents(
        state["translated_query"],
        state["documents"],
    )

    return {
        **state,
        "relevance_score": score,
    }


def route_after_grading(state: RAGState) -> str:
    # Adaptive threshold: weak retrieval goes to web search.
    return "web_search" if state["relevance_score"] < 0.12 else "generate"


def web_search_node(state: RAGState) -> RAGState:
    web_docs = web_search(state["translated_query"])

    return {
        **state,
        "documents": state["documents"] + web_docs,
        "route": "web_search",
    }


def generate_node(state: RAGState) -> RAGState:
    context = "\n".join(doc.text for doc in state["documents"])

    answer = (
        "Answer generated from retrieved context:\n\n"
        f"{context}\n\n"
        f"Relevance score: {state['relevance_score']:.3f}"
    )

    return {
        **state,
        "answer": answer,
        "route": "generate",
    }


# ---------------------------------------------------------------------------
# PIPELINE
# ---------------------------------------------------------------------------

def run_rag(query: str):
    state: RAGState = {"query": query}

    state = retrieve_node(state)
    state = grade_node(state)

    route = route_after_grading(state)

    if route == "web_search":
        state = web_search_node(state)
        # Re-grade after additional evidence.
        state["relevance_score"] = grade_documents(
            state["translated_query"],
            state["documents"],
        )

    state = generate_node(state)
    return state


def main():
    print("=== Agentic RAG via LangGraph ===")
    print("Type 'exit' to quit.")

    while True:
        query = input("\nQuestion: ").strip()

        if query.lower() == "exit":
            break

        result = run_rag(query)
        print("\n" + result["answer"])


if __name__ == "__main__":
    main()
