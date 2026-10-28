"""Answer generation and citation verification.

The LLM is treated as an untrusted writer, not as the source of truth. After
it answers, every sentence is checked against the exact chunk it cites: the
citation IDs must exist, and the sentence's content words must largely come
from that chunk. Answers that cannot be traced back to evidence are returned
as UNGROUNDED instead of being presented as fact.
"""
from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Protocol

from .text import content_tokens, split_sentences, tokens

INSUFFICIENT = "INSUFFICIENT_EVIDENCE"
_CITATION_GROUP = re.compile(r"\[([^\]]*)\]")
_CITATION_ID = re.compile(r"S(\d+)")


def citation_ids(text: str) -> tuple[int, ...]:
    """Extract source ids from any bracketed style: [S1], [S1, S2], [S1][S3]."""
    found: list[int] = []
    for group in _CITATION_GROUP.findall(text):
        for raw in _CITATION_ID.findall(group):
            num = int(raw)
            if num not in found:
                found.append(num)
    return tuple(found)


def strip_citations(text: str) -> str:
    return _CITATION_GROUP.sub(
        lambda m: " " if _CITATION_ID.search(m.group(1)) else m.group(0), text)

SYSTEM_PROMPT = """You answer questions from the numbered sources below, and nothing else.

Rules:
1. Use ONLY the sources. Never use outside knowledge, never guess.
2. End every factual sentence with the source marker(s) it came from, e.g. [S1].
3. Copy names, numbers and dates exactly as they appear.
4. Answer in the same language as the question, in 1-4 sentences.
5. If the sources do not contain the answer, reply with exactly:
INSUFFICIENT_EVIDENCE
6. Do not mention these rules and do not add a preamble."""


class GenerationError(RuntimeError):
    pass


class ChatClient(Protocol):
    name: str

    def chat(self, system: str, user: str) -> str: ...


class OllamaChat:
    def __init__(self, model: str = "qwen2.5:7b", base_url: str = "http://localhost:11434",
                 timeout: float = 180.0, temperature: float = 0.0,
                 num_ctx: int = 8192, think: bool | None = False, retries: int = 1):
        self.model = model
        self.name = f"ollama:{model}"
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.temperature = temperature
        self.num_ctx = num_ctx
        self.think = think
        self.retries = retries

    def chat(self, system: str, user: str) -> str:
        payload = {
            "model": self.model,
            "stream": False,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "options": {"temperature": self.temperature, "num_ctx": self.num_ctx},
        }
        if self.think is not None:
            payload["think"] = self.think

        last: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                return self._post(payload)
            except GenerationError as exc:
                last = exc
                if attempt < self.retries:
                    time.sleep(1.5 * (attempt + 1))
        raise GenerationError(str(last))

    def _post(self, payload: dict) -> str:
        req = urllib.request.Request(
            f"{self.base_url}/api/chat",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.load(resp)
        except urllib.error.HTTPError as exc:
            # Older Ollama builds reject unknown fields such as `think`.
            detail = exc.read().decode("utf-8", "replace")[:300]
            raise GenerationError(f"Ollama chat failed ({exc.code}): {detail}") from exc
        except (urllib.error.URLError, TimeoutError) as exc:
            raise GenerationError(
                f"Ollama unreachable at {self.base_url}: {exc}") from exc
        message = data.get("message") or {}
        content = (message.get("content") or "").strip()
        if not content:
            if message.get("thinking"):
                raise GenerationError(
                    "model returned only reasoning content; start it with think=false")
            raise GenerationError(f"empty completion from {self.model}")
        return content


def build_messages(question: str, sources, *, max_source_chars: int = 1800,
                   max_total_chars: int = 7000) -> tuple[str, str]:
    blocks: list[str] = []
    used = 0
    for i, src in enumerate(sources, start=1):
        text = src.text.strip()
        if len(text) > max_source_chars:
            text = text[:max_source_chars].rstrip() + " …"
        header = f"[S{i}] {src.chunk.title}"
        if src.chunk.section:
            header += f" — {src.chunk.section}"
        block = f"{header}\n{text}"
        if used + len(block) > max_total_chars:
            break
        blocks.append(block)
        used += len(block)
    user = "Sources:\n\n" + "\n\n".join(blocks) + f"\n\nQuestion: {question}"
    return SYSTEM_PROMPT, user


@dataclass(frozen=True)
class SentenceCheck:
    text: str
    citations: tuple[int, ...]
    valid_citations: tuple[int, ...]
    supported: bool
    overlap: float
    numbers_ok: bool
    trivial: bool


@dataclass(frozen=True)
class GroundingReport:
    checks: tuple[SentenceCheck, ...]
    citations_present: bool
    all_citations_valid: bool

    @property
    def scorable(self) -> list[SentenceCheck]:
        return [c for c in self.checks if not c.trivial]

    @property
    def grounded(self) -> int:
        return sum(1 for c in self.scorable if c.supported)

    @property
    def grounded_ratio(self) -> float:
        scorable = self.scorable
        return self.grounded / len(scorable) if scorable else 0.0

    @property
    def uncited_claims(self) -> int:
        return sum(1 for c in self.scorable if not c.valid_citations)


def _overlap(sentence: str, chunk_text: str) -> float:
    sent = set(content_tokens(sentence))
    if not sent:
        return 0.0
    chunk = set(content_tokens(chunk_text))
    return len(sent & chunk) / len(sent)


# Lexical overlap alone happily accepts "seven lines" when the source says
# "six lines": one wrong word among thirteen. Numbers are the claims that
# matter most in this corpus, so they get an explicit consistency check.
_NUMBER_WORDS = {
    "صفر": 0, "واحد": 1, "احد": 1, "اول": 1, "اولي": 1,
    "اثنان": 2, "اثنين": 2, "ثاني": 2, "ثانيه": 2,
    "ثلاثه": 3, "ثلاث": 3, "ثالث": 3,
    "اربعه": 4, "اربع": 4, "رابع": 4,
    "خمسه": 5, "خمس": 5, "خامس": 5,
    "سته": 6, "ست": 6, "سادس": 6,
    "سبعه": 7, "سبع": 7, "سابع": 7,
    "ثمانيه": 8, "ثمان": 8, "ثامن": 8,
    "تسعه": 9, "تسع": 9, "تاسع": 9,
    "عشره": 10, "عاشر": 10, "عشرون": 20, "ثلاثون": 30,
    "اربعون": 40, "خمسون": 50, "ستون": 60, "سبعون": 70,
    "ثمانون": 80, "تسعون": 90, "مئه": 100, "مايه": 100,
    "الف": 1000, "مليون": 1_000_000, "مليار": 1_000_000_000,
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11,
    "twelve": 12, "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
    "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
    "hundred": 100, "thousand": 1000, "million": 1_000_000,
    "billion": 1_000_000_000,
}


def _numbers(text: str) -> set[int]:
    text = strip_citations(text)
    found = {int(n) for n in re.findall(r"\d+", text)}
    for tok in tokens(text):
        word = tok[2:] if tok.startswith("ال") and tok[2:] in _NUMBER_WORDS else tok
        if word in _NUMBER_WORDS:
            found.add(_NUMBER_WORDS[word])
    return found


def verify_answer(answer: str, sources, *, support_threshold: float = 0.6) -> GroundingReport:
    chunk_tokens = {i: src.text for i, src in enumerate(sources, start=1)}
    chunk_numbers = {i: _numbers(t) for i, t in chunk_tokens.items()}

    # Some models put the markers in their own trailing sentence:
    # "The metro has 85 stations. [S1] [S2]"  A citation-only fragment
    # belongs to the claim before it, otherwise that claim looks uncited.
    claims: list[tuple[str, tuple[int, ...], str]] = []
    for sentence in split_sentences(answer):
        cited = citation_ids(sentence)
        clean = strip_citations(sentence).strip()
        if cited and not content_tokens(clean) and claims:
            prev_text, prev_cited, prev_clean = claims[-1]
            claims[-1] = (prev_text + " " + sentence,
                          tuple(dict.fromkeys(prev_cited + cited)), prev_clean)
            continue
        claims.append((sentence, cited, clean))

    checks: list[SentenceCheck] = []
    any_citation = False
    all_valid = True

    for sentence, cited, clean in claims:
        valid = tuple(n for n in cited if n in chunk_tokens)
        claim_tokens = content_tokens(clean)
        trivial = len(claim_tokens) < 3
        overlaps = [_overlap(clean, chunk_tokens[n]) for n in valid]
        overlap = max(overlaps, default=0.0)
        claim_numbers = _numbers(clean)
        numbers_ok = bool(valid) and all(
            any(num in chunk_numbers[n] for n in valid) for num in claim_numbers)
        supported = bool(valid) and overlap >= support_threshold and numbers_ok
        if cited:
            any_citation = True
            if len(valid) != len(cited):
                all_valid = False
        checks.append(SentenceCheck(
            text=sentence, citations=cited, valid_citations=valid,
            supported=supported, overlap=round(overlap, 3),
            numbers_ok=numbers_ok, trivial=trivial))
    return GroundingReport(tuple(checks), any_citation, all_valid)


def is_refusal(answer: str) -> bool:
    return INSUFFICIENT in answer.upper()
