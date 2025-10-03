from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, List, Optional, Sequence

try:  # Optional dependency
    from openai import OpenAI
except ImportError:  # pragma: no cover - optional path
    OpenAI = None  # type: ignore


@dataclass
class SpanRecord:
    span: str
    question: Optional[str] = None
    label: Optional[str] = None
    metadata: Optional[dict] = None


class SimSpanSource:
    """Yields span records from a JSON file used in the demo."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"Span file not found: {self.path}")
        self._records = self._load()

    def _load(self) -> List[SpanRecord]:
        data = json.loads(self.path.read_text())
        if isinstance(data, dict) and "spans" in data:
            items = data["spans"]
        else:
            items = data
        if not isinstance(items, list):
            raise ValueError("Span file must contain a list or a 'spans' list")
        records: List[SpanRecord] = []
        for idx, item in enumerate(items):
            if isinstance(item, str):
                records.append(SpanRecord(span=item))
            elif isinstance(item, dict) and "span" in item:
                records.append(
                    SpanRecord(
                        span=str(item["span"]),
                        question=item.get("question"),
                        label=item.get("label"),
                        metadata=item.get("metadata"),
                    )
                )
            else:
                raise ValueError(f"Invalid span entry at index {idx}: {item}")
        return records

    def __iter__(self) -> Iterator[SpanRecord]:  # pragma: no cover - simple generator
        yield from self._records


class LLMSpanSource:
    """Streams spans by querying an LLM (OpenAI compatible).

    Falls back to a deterministic echo when the OpenAI client is unavailable so
    tests can run without network credentials.
    """

    def __init__(
        self,
        questions: Iterable[str],
        *,
        model: str = "gpt-4o-mini",
        temperature: float = 0.2,
        max_tokens: int = 120,
        system_prompt: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> None:
        self.questions = [q.strip() for q in questions if str(q).strip()]
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.system_prompt = system_prompt or (
            "You are a fact-checking assistant. Answer with a single sentence that "
            "can be validated against the provided knowledge base."
        )
        key = api_key or os.getenv("OPENAI_API_KEY")
        if OpenAI and key:
            self._client = OpenAI(api_key=key)
        else:  # pragma: no cover - deterministic fallback for offline/dev
            self._client = None

    def _call_llm(self, question: str) -> str:
        if self._client is None:
            return question
        response = self._client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": question},
            ],
        )
        content = response.choices[0].message.content or ""
        return content.strip()

    def __iter__(self) -> Iterator[SpanRecord]:
        for question in self.questions:
            span = self._call_llm(question)
            yield SpanRecord(span=span, question=question)


__all__ = ["SpanRecord", "SimSpanSource", "LLMSpanSource"]
