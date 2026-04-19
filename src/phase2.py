from __future__ import annotations

from dataclasses import dataclass

from .config import AssignmentConfig, default_config
from .corpus import CorpusLoader
from .indexing import CorpusIndex, IndexBuilder
from .preprocessing import StopwordLoader, TextPreprocessor


@dataclass(frozen=True)
class Phase2Summary:
    document_count: int
    vocabulary_size: int
    index_path: str
    sample_terms: list[str]
    round_trip_ok: bool


def build_and_persist_index(config: AssignmentConfig | None = None) -> tuple[CorpusIndex, Phase2Summary]:
    config = config or default_config()
    documents = CorpusLoader(config.corpus_dir).discover_documents()
    stopwords = StopwordLoader(config.stopword_path).load()
    preprocessor = TextPreprocessor(stopwords)
    builder = IndexBuilder(preprocessor)
    corpus_index = builder.build(
        documents=documents,
        min_term_frequency=config.min_term_frequency,
        min_document_frequency=config.min_document_frequency,
    )
    index_path = corpus_index.save(config.index_dir)
    reloaded_index = CorpusIndex.load(config.index_dir)
    round_trip_ok = corpus_index.to_dict() == reloaded_index.to_dict()
    summary = Phase2Summary(
        document_count=corpus_index.document_count,
        vocabulary_size=len(corpus_index.vocabulary),
        index_path=str(index_path),
        sample_terms=corpus_index.vocabulary[:10],
        round_trip_ok=round_trip_ok,
    )
    return corpus_index, summary


def main() -> None:
    _, summary = build_and_persist_index()
    print("Phase 2 indexing complete")
    print(summary)


if __name__ == "__main__":
    main()
