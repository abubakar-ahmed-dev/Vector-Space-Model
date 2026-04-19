from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys

try:
    from .config import AssignmentConfig, default_config
    from .phase2 import build_and_persist_index
    from .retrieval import VSMSearchEngine
except ImportError:
    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from src.config import AssignmentConfig, default_config
    from src.phase2 import build_and_persist_index
    from src.retrieval import VSMSearchEngine


@dataclass(frozen=True)
class CLIResult:
    query: str
    result_count: int


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Vector Space Model search over Trump speeches")
    parser.add_argument("query", nargs="?", help="Free-text query to search")
    parser.add_argument("--top-k", type=int, default=None, help="Optional number of results to display")
    parser.add_argument("--rebuild-index", action="store_true", help="Rebuild the saved corpus index first")
    parser.add_argument("--show-explanation", action="store_true", help="Print tokenization details for the query")
    return parser


def load_engine(config: AssignmentConfig | None = None, rebuild_index: bool = False) -> VSMSearchEngine:
    config = config or default_config()
    if rebuild_index:
        build_and_persist_index(config)
    return VSMSearchEngine.from_config(config)


def _display_doc_id(doc_id: str) -> str:
    if doc_id.startswith("speech_"):
        suffix = doc_id.removeprefix("speech_")
        if suffix.isdigit():
            return f"speech_{int(suffix):02d}"
    return doc_id


def _print_search_results(engine: VSMSearchEngine, query: str, top_k: int | None, show_explanation: bool) -> CLIResult:
    if show_explanation:
        explanation = engine.explain_query(query)
        print("Query explanation:")
        print(explanation)

    results = engine.search(query, top_k=top_k)
    print()
    print(f"Total retrieved: {len(results)} documents")
    print(f"Top Relevant Documents (cosine similarity > {engine.alpha_threshold:.4f}):")
    print()
    for index, result in enumerate(results, start=1):
        print(f"{index}. {_display_doc_id(result.doc_id)}   Score: {result.score:.3f}")
    print()
    print("---")

    return CLIResult(query=query, result_count=len(results))


def run_interactive_cli(engine: VSMSearchEngine, top_k: int | None, show_explanation: bool) -> None:
    print("Interactive VSM Search")
    print("Type your query and press Enter.")
    print("Type 'exit' to quit.")

    while True:
        try:
            query = input("search> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            return

        if not query:
            continue
        if query.lower() == "exit":
            print("Exiting.")
            return

        _print_search_results(engine, query, top_k, show_explanation)
        print()


def run_cli(argv: list[str] | None = None) -> CLIResult | None:
    parser = build_parser()
    args = parser.parse_args(argv)

    engine = load_engine(rebuild_index=args.rebuild_index)
    if args.query:
        return _print_search_results(engine, args.query, args.top_k, args.show_explanation)

    run_interactive_cli(engine, args.top_k, args.show_explanation)
    return None


def main() -> None:
    run_cli()


if __name__ == "__main__":
    main()