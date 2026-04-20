from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import nltk
from nltk import pos_tag
from nltk.corpus import wordnet
from nltk.data import find
from nltk.stem import WordNetLemmatizer

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
        self._ensure_nltk_resources()
        self.stopwords = {word.lower() for word in stopwords}
        self.lemmatizer = WordNetLemmatizer()

    def preprocess(self, text: str) -> PreprocessingResult:
        normalized = text.lower()
        raw_tokens = _TOKEN_PATTERN.findall(normalized)
        filtered_tokens = [token for token in raw_tokens if token not in self.stopwords]
        tagged_tokens = pos_tag(filtered_tokens)
        lemmatized_tokens = [
            self._lemmatize(token, self._to_wordnet_pos(tag)) for token, tag in tagged_tokens
        ]
        return PreprocessingResult(original_text=text, tokens=lemmatized_tokens)

    def preprocess_tokens(self, text: str) -> list[str]:
        return self.preprocess(text).tokens

    def _lemmatize(self, token: str, pos: str | None) -> str:
        if pos is None:
            return self.lemmatizer.lemmatize(token)
        return self.lemmatizer.lemmatize(token, pos=pos)

    @staticmethod
    def _to_wordnet_pos(tag: str) -> str | None:
        if tag.startswith("J"):
            return wordnet.ADJ
        if tag.startswith("V"):
            return wordnet.VERB
        if tag.startswith("N"):
            return wordnet.NOUN
        if tag.startswith("R"):
            return wordnet.ADV
        return None

    @staticmethod
    def _ensure_nltk_resources() -> None:
        required_resources = (
            ("taggers/averaged_perceptron_tagger_eng", "averaged_perceptron_tagger_eng"),
            ("taggers/averaged_perceptron_tagger", "averaged_perceptron_tagger"),
            ("corpora/wordnet", "wordnet"),
            ("corpora/omw-1.4", "omw-1.4"),
        )

        for resource_path, download_name in required_resources:
            try:
                find(resource_path)
            except LookupError:
                nltk.download(download_name, quiet=True)
