from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

_TOKEN_PATTERN = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)?")


@dataclass(frozen=True)
class PreprocessingResult:
    original_text: str
    tokens: list[str]


class StopwordLoader:
    def __init__(self, stopword_path: Path) -> None:
        self.stopword_path = stopword_path

    def load(self) -> set[str]:
        stopwords: set[str] = set()
        for line in self.stopword_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            word = line.strip().lower()
            if word:
                stopwords.add(word)
        return stopwords


class TextPreprocessor:
    def __init__(self, stopwords: Iterable[str]) -> None:
        self.stopwords = {word.lower() for word in stopwords}

    def preprocess(self, text: str) -> PreprocessingResult:
        normalized = text.lower()
        raw_tokens = _TOKEN_PATTERN.findall(normalized)
        filtered_tokens = [token for token in raw_tokens if token not in self.stopwords]
        lemmatized_tokens = [self._lemmatize(token) for token in filtered_tokens]
        return PreprocessingResult(original_text=text, tokens=lemmatized_tokens)

    def preprocess_tokens(self, text: str) -> list[str]:
        return self.preprocess(text).tokens

    def _lemmatize(self, token: str) -> str:
        if len(token) <= 3:
            return token
        if token.endswith("ies") and len(token) > 4:
            return token[:-3] + "y"
        if token.endswith("ing") and len(token) > 5:
            return token[:-3]
        if token.endswith("ed") and len(token) > 4:
            return token[:-2]
        if token.endswith("s") and not token.endswith("ss") and len(token) > 3:
            return token[:-1]
        return token
