"""The RAG pipeline: retrieve -> answer with citations -> verify -> trust or abstain."""
from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum

from .generate import ChatClient, GroundingReport, build_messages, is_refusal, verify_answer
from .retrieve import HybridRetriever, Retrieved


class Status(str, Enum):
    ANSWERED = "answered"
    INSUFFICIENT = "insufficient"
    UNGROUNDED = "ungrounded"


@dataclass(frozen=True)
class AnswerResult:
    question: str
    answer: str
    status: Status
    reason: str
    sources: tuple[Retrieved, ...]
    grounding: GroundingReport | None
    retrieval_ms: float
    generation_ms: float
    model: str

    @property
    def abstained(self) -> bool:
        return self.status is not Status.ANSWERED

    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "answer": self.answer,
            "status": self.status.value,
            "reason": self.reason,
            "sources": [
                {"id": s.chunk.id, "title": s.chunk.title, "lang": s.chunk.lang,
                 "section": s.chunk.section, "score": round(s.score, 6),
                 "bm25": s.bm25_score, "dense": s.dense_score}
                for s in self.sources
            ],
            "grounding": None if self.grounding is None else {
                "grounded_ratio": round(self.grounding.grounded_ratio, 3),
                "all_citations_valid": self.grounding.all_citations_valid,
                "sentences": [
                    {"text": c.text, "citations": list(c.citations),
                     "supported": c.supported, "overlap": c.overlap,
                     "numbers_ok": c.numbers_ok, "trivial": c.trivial}
                    for c in self.grounding.checks
                ],
            },
            "retrieval_ms": round(self.retrieval_ms, 1),
            "generation_ms": round(self.generation_ms, 1),
            "model": self.model,
        }


class RagEngine:
    def __init__(self, retriever: HybridRetriever, chat: ChatClient, *,
                 support_threshold: float = 0.6,
                 dense_min_score: float | None = None,
                 max_source_chars: int = 1800,
                 max_context_chars: int = 7000,
                 repair_attempts: int = 1):
        self.retriever = retriever
        self.chat = chat
        self.support_threshold = support_threshold
        self.dense_min_score = dense_min_score
        self.max_source_chars = max_source_chars
        self.max_context_chars = max_context_chars
        self.repair_attempts = repair_attempts

    def ask(self, question: str, *, top_k: int = 5, lang: str | None = None,
            mode: str = "hybrid") -> AnswerResult:
        question = question.strip()
        if not question:
            raise ValueError("question must not be empty")

        t0 = time.perf_counter()
        sources = self.retriever.search(question, top_k=top_k, lang=lang, mode=mode)
        retrieval_ms = (time.perf_counter() - t0) * 1000

        base = dict(question=question, sources=tuple(sources),
                    retrieval_ms=retrieval_ms, model=getattr(self.chat, "name", "unknown"))

        if not sources:
            return AnswerResult(answer="", status=Status.INSUFFICIENT,
                                reason="no sources retrieved", grounding=None,
                                generation_ms=0.0, **base)

        if self.dense_min_score is not None and mode == "hybrid":
            best_dense = max((s.dense_score or 0.0) for s in sources)
            if best_dense < self.dense_min_score:
                return AnswerResult(
                    answer="", status=Status.INSUFFICIENT,
                    reason=f"best dense similarity {best_dense:.3f} < "
                           f"threshold {self.dense_min_score:.3f}",
                    grounding=None, generation_ms=0.0, **base)

        system, user = build_messages(
            question, sources, max_source_chars=self.max_source_chars,
            max_total_chars=self.max_context_chars)
        t1 = time.perf_counter()
        answer = self.chat.chat(system, user)
        generation_ms = (time.perf_counter() - t1) * 1000

        if is_refusal(answer):
            return AnswerResult(answer=answer, status=Status.INSUFFICIENT,
                                reason="model declined: evidence insufficient",
                                grounding=None, generation_ms=generation_ms, **base)

        grounding = verify_answer(answer, sources, support_threshold=self.support_threshold)
        failure = _failure_reason(grounding)
        if failure is None:
            return AnswerResult(answer=answer, status=Status.ANSWERED, reason="",
                                grounding=grounding, generation_ms=generation_ms, **base)

        original = AnswerResult(
            answer=answer, status=Status.UNGROUNDED, reason=failure,
            grounding=grounding, generation_ms=generation_ms, **base)

        # One corrective round: show the model why the answer failed and
        # ask again. The repair only counts if it passes the same checks;
        # otherwise the original rejection stands.
        for _ in range(self.repair_attempts):
            repair_user = (
                f"{user}\n\nYour previous answer was rejected: {failure}.\n"
                f"Previous answer:\n{answer}\n\n"
                "Rewrite the answer. Use only the sources above, copy names "
                "and numbers exactly as written, and end every factual "
                "sentence with its citation marker.")
            t2 = time.perf_counter()
            repaired = self.chat.chat(system, repair_user)
            generation_ms += (time.perf_counter() - t2) * 1000
            if is_refusal(repaired):
                break
            repaired_grounding = verify_answer(
                repaired, sources, support_threshold=self.support_threshold)
            repaired_failure = _failure_reason(repaired_grounding)
            if repaired_failure is None:
                return AnswerResult(
                    answer=repaired, status=Status.ANSWERED, reason="",
                    grounding=repaired_grounding, generation_ms=generation_ms, **base)
            break  # a second bad answer is not more trustworthy than the first

        return AnswerResult(
            answer=answer, status=Status.UNGROUNDED, reason=failure,
            grounding=grounding, generation_ms=generation_ms, **base)


def _failure_reason(grounding: GroundingReport) -> str | None:
    if not grounding.citations_present:
        return "answer contained no source citations"
    if not grounding.all_citations_valid:
        return "answer cited source IDs that do not exist"
    if grounding.scorable and grounding.grounded == 0:
        return "no sentence was supported by its cited source"
    return None
