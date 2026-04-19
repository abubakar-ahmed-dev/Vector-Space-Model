from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AssignmentConfig:
    corpus_dir: Path
    stopword_path: Path
    index_dir: Path
    min_term_frequency: int = 2
    min_document_frequency: int = 2
    alpha_threshold: float = 0.004


def default_config() -> AssignmentConfig:
    base_dir = Path(__file__).resolve().parent.parent
    return AssignmentConfig(
        corpus_dir=base_dir / "speeches",
        stopword_path=base_dir / "Stopword-List.txt",
        index_dir=base_dir / "index",
    )
