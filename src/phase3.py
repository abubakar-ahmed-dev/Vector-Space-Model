from __future__ import annotations

from dataclasses import dataclass

from .config import AssignmentConfig, default_config
from .retrieval import SearchResult, VSMSearchEngine


@dataclass(frozen=True)
class Phase3Summary:
    query: str
    top_k: int
    result_count: int
    top_result: str | None
    scores: list[tuple[str, float]]


def run_phase3_demo(query: str, config: AssignmentConfig | None = None, top_k: int = 5) -> tuple[list[SearchResult], Phase3Summary]:
    config = config or default_config()
    engine = VSMSearchEngine.from_config(config)
    results = engine.search(query=query, top_k=top_k)
    summary = Phase3Summary(
        query=query,
        top_k=top_k,
        result_count=len(results),
        top_result=results[0].doc_id if results else None,
        scores=[(result.doc_id, result.score) for result in results],
    )
    return results, summary


def main() -> None:
    query = "american jobs trade"
    results, summary = run_phase3_demo(query)
    print("Phase 3 retrieval complete")
    print(summary)
    for result in results:
        print(f"{result.doc_id}\t{result.score:.6f}\t{result.title}")


if __name__ == "__main__":
    main()
