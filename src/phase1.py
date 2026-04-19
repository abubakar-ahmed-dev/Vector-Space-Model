from __future__ import annotations

from dataclasses import dataclass

from .config import AssignmentConfig, default_config
from .corpus import CorpusLoader
from .preprocessing import StopwordLoader, TextPreprocessor


@dataclass(frozen=True)
class Phase1Summary:
    corpus_dir: str
    document_count: int
    sample_document_ids: list[str]
    min_term_frequency: int
    min_document_frequency: int
    alpha_threshold: float


def build_phase1_summary(config: AssignmentConfig | None = None) -> Phase1Summary:
    config = config or default_config()
    corpus_loader = CorpusLoader(config.corpus_dir)
    documents = corpus_loader.discover_documents()
    stopwords = StopwordLoader(config.stopword_path).load()
    _ = TextPreprocessor(stopwords)
    sample_ids = [document.doc_id for document in documents[:5]]
    return Phase1Summary(
        corpus_dir=str(config.corpus_dir),
        document_count=len(documents),
        sample_document_ids=sample_ids,
        min_term_frequency=config.min_term_frequency,
        min_document_frequency=config.min_document_frequency,
        alpha_threshold=config.alpha_threshold,
    )


def main() -> None:
    summary = build_phase1_summary()
    print("Phase 1 setup complete")
    print(summary)


if __name__ == "__main__":
    main()
