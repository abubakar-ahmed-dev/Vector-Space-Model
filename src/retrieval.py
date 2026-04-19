from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from pathlib import Path
from typing import Iterable

from .config import AssignmentConfig, default_config
from .indexing import CorpusIndex
from .preprocessing import StopwordLoader, TextPreprocessor


@dataclass(frozen=True)
class SearchResult:
    doc_id: str
    title: str
    path: str
    score: float
    matched_terms: list[str]


class VSMSearchEngine:
    def __init__(self, corpus_index: CorpusIndex, preprocessor: TextPreprocessor, alpha_threshold: float) -> None:
        self.corpus_index = corpus_index
        self.preprocessor = preprocessor
        self.alpha_threshold = alpha_threshold

    @classmethod
    def from_config(cls, config: AssignmentConfig | None = None) -> VSMSearchEngine:
        config = config or default_config()
        corpus_index = CorpusIndex.load(config.index_dir)
        stopwords = StopwordLoader(config.stopword_path).load()
        preprocessor = TextPreprocessor(stopwords)
        return cls(corpus_index, preprocessor, config.alpha_threshold)

    def search(self, query: str, top_k: int = 10, alpha_threshold: float | None = None) -> list[SearchResult]:
        query_tokens = self.preprocessor.preprocess_tokens(query)
        if not query_tokens:
            return []

        query_vector = self._build_query_vector(query_tokens)
        if not query_vector:
            return []

        query_norm = self._vector_norm(query_vector.values())
        if query_norm == 0:
            return []

        scores: dict[str, float] = {}
        matched_terms: dict[str, set[str]] = {}

        for term, query_weight in query_vector.items():
            postings = self.corpus_index.inverted_index.get(term, [])
            if not postings:
                continue
            for posting in postings:
                document_vector = self.corpus_index.document_vectors.get(posting.doc_id, {})
                document_weight = document_vector.get(term, 0.0)
                if document_weight == 0.0:
                    continue
                scores[posting.doc_id] = scores.get(posting.doc_id, 0.0) + query_weight * document_weight
                matched_terms.setdefault(posting.doc_id, set()).add(term)

        threshold = self.alpha_threshold if alpha_threshold is None else alpha_threshold
        results: list[SearchResult] = []
        for doc_id, dot_product in scores.items():
            document_norm = self.corpus_index.document_norms.get(doc_id, 0.0)
            if document_norm == 0:
                continue
            similarity = dot_product / (query_norm * document_norm)
            if similarity < threshold:
                continue
            document = self.corpus_index.documents[doc_id]
            results.append(
                SearchResult(
                    doc_id=doc_id,
                    title=document.title,
                    path=document.path,
                    score=similarity,
                    matched_terms=sorted(matched_terms.get(doc_id, set())),
                )
            )

        results.sort(key=lambda result: (-result.score, result.doc_id))
        return results[:top_k]

    def explain_query(self, query: str) -> dict[str, object]:
        tokens = self.preprocessor.preprocess_tokens(query)
        vector = self._build_query_vector(tokens)
        return {
            "query": query,
            "tokens": tokens,
            "vector": vector,
            "vocabulary_hits": [term for term in tokens if term in self.corpus_index.vocabulary],
        }

    def _build_query_vector(self, tokens: Iterable[str]) -> dict[str, float]:
        token_counts: dict[str, int] = {}
        for token in tokens:
            if token in self.corpus_index.inverse_document_frequencies:
                token_counts[token] = token_counts.get(token, 0) + 1

        query_vector: dict[str, float] = {}
        for term, term_frequency in token_counts.items():
            query_vector[term] = float(term_frequency) * self.corpus_index.inverse_document_frequencies[term]
        return query_vector

    @staticmethod
    def _vector_norm(values: Iterable[float]) -> float:
        return sqrt(sum(float(value) * float(value) for value in values))
