from __future__ import annotations

from collections.abc import Iterable

from .retrieval import SearchResult


def _display_doc_id(doc_id: str) -> str:
    if doc_id.startswith("speech_"):
        suffix = doc_id.removeprefix("speech_")
        if suffix.isdigit():
            return f"speech_{int(suffix):02d}"
    return doc_id


def format_search_results(results: Iterable[SearchResult], alpha_threshold: float) -> str:
    result_list = list(results)
    lines = [f"Top Relevant Documents (cosine similarity > {alpha_threshold:.4f}):", ""]

    for index, result in enumerate(result_list, start=1):
        lines.append(f"{index}. {_display_doc_id(result.doc_id)}   Score: {result.score:.3f}")

    lines.extend(["", f"Total retrieved: {len(result_list)} documents", "---"])
    return "\n".join(lines)