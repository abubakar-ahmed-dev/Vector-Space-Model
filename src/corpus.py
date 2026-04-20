from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Document:
    doc_id: str
    path: Path
    title: str
    text: str


class CorpusLoader:
    def __init__(self, corpus_dir: Path) -> None:
        self.corpus_dir = corpus_dir

    def discover_documents(self) -> list[Document]:
        documents: list[Document] = []
        for path in sorted(self.corpus_dir.glob("*.txt")):
            text = path.read_text(encoding="utf-8", errors="ignore")
            title = text.splitlines()[0].strip() if text.splitlines() else path.stem
            documents.append(
                Document(
                    doc_id=path.stem,
                    path=path,
                    title=title,
                    text=text,
                )
            )
        return documents
