from __future__ import annotations

from .config import AssignmentConfig, default_config
from .corpus import CorpusLoader
from .indexing import CorpusIndex, IndexBuilder
from .preprocessing import StopwordLoader, TextPreprocessor
from .retrieval import VSMSearchEngine


def build_index(config: AssignmentConfig | None = None) -> CorpusIndex:
    config = config or default_config()
    documents = CorpusLoader(config.corpus_dir).discover_documents()
    stopwords = StopwordLoader(config.stopword_path).load()
    preprocessor = TextPreprocessor(stopwords)
    index = IndexBuilder(preprocessor).build(
        documents=documents,
        min_term_frequency=config.min_term_frequency,
        min_document_frequency=config.min_document_frequency,
    )
    index.save(config.index_dir)
    return index


def load_search_engine(config: AssignmentConfig | None = None) -> VSMSearchEngine:
    config = config or default_config()
    index_path = config.index_dir / "corpus_index.json"
    if not index_path.exists():
        build_index(config)
    return VSMSearchEngine.from_config(config)
