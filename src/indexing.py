from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from math import log
from pathlib import Path
from typing import Any, Iterable
import json

from .corpus import Document
from .preprocessing import TextPreprocessor


@dataclass(frozen=True)
class Posting:
    doc_id: str
    term_frequency: int
    positions: list[int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "term_frequency": self.term_frequency,
            "positions": self.positions,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Posting:
        return cls(
            doc_id=str(data["doc_id"]),
            term_frequency=int(data["term_frequency"]),
            positions=[int(position) for position in data.get("positions", [])],
        )


@dataclass(frozen=True)
class IndexedDocument:
    doc_id: str
    title: str
    path: str
    length: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "title": self.title,
            "path": self.path,
            "length": self.length,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> IndexedDocument:
        return cls(
            doc_id=str(data["doc_id"]),
            title=str(data["title"]),
            path=str(data["path"]),
            length=int(data["length"]),
        )


@dataclass
class CorpusIndex:
    document_count: int
    documents: dict[str, IndexedDocument]
    vocabulary: list[str]
    document_frequencies: dict[str, int]
    corpus_frequencies: dict[str, int]
    inverse_document_frequencies: dict[str, float]
    inverted_index: dict[str, list[Posting]]
    positional_index: dict[str, dict[str, list[int]]]
    document_vectors: dict[str, dict[str, float]]
    document_norms: dict[str, float]
    min_term_frequency: int
    min_document_frequency: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "document_count": self.document_count,
            "documents": {doc_id: document.to_dict() for doc_id, document in self.documents.items()},
            "vocabulary": self.vocabulary,
            "document_frequencies": self.document_frequencies,
            "corpus_frequencies": self.corpus_frequencies,
            "inverse_document_frequencies": self.inverse_document_frequencies,
            "inverted_index": {
                term: [posting.to_dict() for posting in postings]
                for term, postings in self.inverted_index.items()
            },
            "positional_index": self.positional_index,
            "document_vectors": self.document_vectors,
            "document_norms": self.document_norms,
            "min_term_frequency": self.min_term_frequency,
            "min_document_frequency": self.min_document_frequency,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CorpusIndex:
        return cls(
            document_count=int(data["document_count"]),
            documents={
                doc_id: IndexedDocument.from_dict(document_data)
                for doc_id, document_data in data["documents"].items()
            },
            vocabulary=[str(term) for term in data["vocabulary"]],
            document_frequencies={str(term): int(value) for term, value in data["document_frequencies"].items()},
            corpus_frequencies={str(term): int(value) for term, value in data["corpus_frequencies"].items()},
            inverse_document_frequencies={
                str(term): float(value) for term, value in data["inverse_document_frequencies"].items()
            },
            inverted_index={
                str(term): [Posting.from_dict(posting) for posting in postings]
                for term, postings in data["inverted_index"].items()
            },
            positional_index={
                str(term): {str(doc_id): [int(position) for position in positions] for doc_id, positions in postings.items()}
                for term, postings in data["positional_index"].items()
            },
            document_vectors={
                str(doc_id): {str(term): float(weight) for term, weight in vector.items()}
                for doc_id, vector in data["document_vectors"].items()
            },
            document_norms={str(doc_id): float(norm) for doc_id, norm in data["document_norms"].items()},
            min_term_frequency=int(data["min_term_frequency"]),
            min_document_frequency=int(data["min_document_frequency"]),
        )

    def save(self, directory: Path) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        index_path = directory / "corpus_index.json"
        index_path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
        return index_path

    @classmethod
    def load(cls, directory: Path) -> CorpusIndex:
        index_path = directory / "corpus_index.json"
        data = json.loads(index_path.read_text(encoding="utf-8"))
        return cls.from_dict(data)


class IndexBuilder:
    def __init__(self, preprocessor: TextPreprocessor) -> None:
        self.preprocessor = preprocessor

    def build(
        self,
        documents: list[Document],
        min_term_frequency: int,
        min_document_frequency: int,
    ) -> CorpusIndex:
        per_document_tokens: dict[str, list[str]] = {}
        per_document_positions: dict[str, dict[str, list[int]]] = {}
        corpus_frequencies: Counter[str] = Counter()
        document_frequencies: Counter[str] = Counter()
        indexed_documents: dict[str, IndexedDocument] = {}

        for document in documents:
            tokens = self.preprocessor.preprocess_tokens(document.text)
            per_document_tokens[document.doc_id] = tokens
            indexed_documents[document.doc_id] = IndexedDocument(
                doc_id=document.doc_id,
                title=document.title,
                path=str(document.path),
                length=len(tokens),
            )

            token_counts = Counter(tokens)
            corpus_frequencies.update(token_counts)
            document_frequencies.update(token_counts.keys())

            positions_by_term: dict[str, list[int]] = defaultdict(list)
            for position, token in enumerate(tokens):
                positions_by_term[token].append(position)
            per_document_positions[document.doc_id] = dict(positions_by_term)

        vocabulary = [
            term
            for term, frequency in corpus_frequencies.items()
            if frequency >= min_term_frequency and document_frequencies[term] >= min_document_frequency
        ]
        vocabulary.sort()
        vocabulary_set = set(vocabulary)

        inverted_index: dict[str, list[Posting]] = defaultdict(list)
        positional_index: dict[str, dict[str, list[int]]] = defaultdict(dict)
        document_vectors: dict[str, dict[str, float]] = defaultdict(dict)
        document_norms: dict[str, float] = {}

        document_count = len(documents)
        inverse_document_frequencies = {
            term: log(document_count / document_frequencies[term]) if document_frequencies[term] else 0.0
            for term in vocabulary
        }

        for document in documents:
            tokens = per_document_tokens[document.doc_id]
            token_counts = Counter(token for token in tokens if token in vocabulary_set)
            vector: dict[str, float] = {}
            for term, term_frequency in token_counts.items():
                weight = float(term_frequency) * inverse_document_frequencies[term]
                vector[term] = weight
                inverted_index[term].append(
                    Posting(
                        doc_id=document.doc_id,
                        term_frequency=term_frequency,
                        positions=per_document_positions[document.doc_id][term],
                    )
                )
                positional_index[term][document.doc_id] = per_document_positions[document.doc_id][term]

            document_vectors[document.doc_id] = vector
            document_norms[document.doc_id] = self._vector_norm(vector.values())

        for postings in inverted_index.values():
            postings.sort(key=lambda posting: posting.doc_id)

        return CorpusIndex(
            document_count=document_count,
            documents=indexed_documents,
            vocabulary=vocabulary,
            document_frequencies={term: document_frequencies[term] for term in vocabulary},
            corpus_frequencies={term: corpus_frequencies[term] for term in vocabulary},
            inverse_document_frequencies=inverse_document_frequencies,
            inverted_index={term: postings for term, postings in inverted_index.items()},
            positional_index={term: postings for term, postings in positional_index.items()},
            document_vectors={doc_id: vector for doc_id, vector in document_vectors.items()},
            document_norms=document_norms,
            min_term_frequency=min_term_frequency,
            min_document_frequency=min_document_frequency,
        )

    @staticmethod
    def _vector_norm(values: Iterable[float]) -> float:
        return sum(float(value) * float(value) for value in values) ** 0.5
