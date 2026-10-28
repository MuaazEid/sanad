"""Offline evaluation: retrieval quality, answer correctness, grounding, abstention.

Every number this module prints is computed from a gold-labeled file
(`eval/questions.jsonl`) whose quotes are validated against the corpus
snapshot. Metrics are split into retrieval (no LLM) and end-to-end (real
model), so model regressions and retrieval regressions cannot hide each other.
"""
from __future__ import annotations

import json
import math
import re
import statistics
from dataclasses import dataclass, field
from pathlib import Path

from .corpus import load_manifest
from .pipeline import AnswerResult, RagEngine
from .retrieve import HybridRetriever
from .text import normalize


@dataclass(frozen=True)
class Evidence:
    doc_id: str
    quote: str


@dataclass(frozen=True)
class Question:
    id: str
    split: str
    lang: str
    question: str
    answerable: bool
    gold_answers: tuple[str, ...]
    evidence: tuple[Evidence, ...]


def load_questions(path: str | Path) -> list[Question]:
    questions: list[Question] = []
    for line_no, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
            questions.append(Question(
                id=raw["id"], split=raw["split"], lang=raw["lang"],
                question=raw["question"], answerable=bool(raw["answerable"]),
                gold_answers=tuple(raw.get("gold_answers", [])),
                evidence=tuple(Evidence(e["doc_id"], e["quote"])
                               for e in raw.get("evidence", [])),
            ))
        except (KeyError, TypeError, json.JSONDecodeError) as exc:
            raise ValueError(f"{path}:{line_no}: invalid question record ({exc})") from exc
    if not questions:
        raise ValueError(f"{path}: no questions")
    return questions


def validate_dataset(questions: list[Question], corpus_dir: str | Path) -> list[str]:
    """Return integrity errors. An empty list means the labels are usable."""
    errors: list[str] = []
    docs = {d.id: normalize(d.text) for d in load_manifest(corpus_dir)}
    seen: set[str] = set()
    for q in questions:
        if q.id in seen:
            errors.append(f"{q.id}: duplicate id")
        seen.add(q.id)
        if q.split not in {"dev", "test"}:
            errors.append(f"{q.id}: split must be dev|test, got {q.split!r}")
        if not q.question.strip():
            errors.append(f"{q.id}: empty question")
        if q.answerable:
            if not q.gold_answers:
                errors.append(f"{q.id}: answerable question without gold_answers")
            if not q.evidence:
                errors.append(f"{q.id}: answerable question without evidence")
            for ev in q.evidence:
                if ev.doc_id not in docs:
                    errors.append(f"{q.id}: unknown doc_id {ev.doc_id!r}")
                    continue
                if normalize(ev.quote) not in docs[ev.doc_id]:
                    errors.append(f"{q.id}: quote not found verbatim in {ev.doc_id}: "
                                  f"{ev.quote[:60]!r}")
            quotes = " ".join(normalize(ev.quote) for ev in q.evidence)
            if q.gold_answers and not any(normalize(g) in quotes for g in q.gold_answers):
                errors.append(f"{q.id}: no gold answer appears in its evidence quotes")
        else:
            if q.evidence:
                errors.append(f"{q.id}: unanswerable question must not have evidence")
            if q.gold_answers:
                errors.append(f"{q.id}: unanswerable question must not have gold_answers")
    return errors


# ── metric primitives (pure, individually tested) ──────────────────
def matches(chunk_text: str, quote: str) -> bool:
    return normalize(quote) in normalize(chunk_text)


_MONTHS = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}
_DAY_MONTH_YEAR = re.compile(r"\b(\d{1,2})\s+([a-z]{3,9})\s+(\d{4})\b")
_MONTH_DAY_YEAR = re.compile(r"\b([a-z]{3,9})\s+(\d{1,2})\s+(\d{4})\b")


def canonical_dates(text: str) -> str:
    """Rewrite English dates to ISO so '1 December 2024' == 'December 1, 2024'."""
    def day_month_year(m: re.Match) -> str:
        month = _MONTHS.get(m.group(2)[:3])
        return f"{m.group(3)}-{month:02d}-{int(m.group(1)):02d}" if month else m.group(0)

    def month_day_year(m: re.Match) -> str:
        month = _MONTHS.get(m.group(1)[:3])
        return f"{m.group(3)}-{month:02d}-{int(m.group(2)):02d}" if month else m.group(0)

    text = normalize(text)
    return _MONTH_DAY_YEAR.sub(month_day_year, _DAY_MONTH_YEAR.sub(day_month_year, text))


def answer_contains(answer: str, gold: str) -> bool:
    plain_answer, plain_gold = normalize(answer), normalize(gold)
    if plain_gold in plain_answer:
        return True
    dated_answer, dated_gold = canonical_dates(answer), canonical_dates(gold)
    return dated_gold != plain_gold and dated_gold in dated_answer


def evidence_ranks(texts: list[str], quotes: list[str]) -> list[int | None]:
    """Best rank (1-based) of a chunk matching each evidence quote."""
    norm_texts = [normalize(t) for t in texts]
    ranks: list[int | None] = []
    for quote in quotes:
        nq = normalize(quote)
        ranks.append(next((i for i, t in enumerate(norm_texts, start=1) if nq in t), None))
    return ranks


def first_relevant_rank(texts: list[str], quotes: list[str]) -> int | None:
    found = [r for r in evidence_ranks(texts, quotes) if r is not None]
    return min(found) if found else None


def ndcg_at_k(ranks: list[int | None], k: int) -> float:
    gains = [1.0 / math.log2(r + 1) for r in ranks if r is not None and r <= k]
    ideal_hits = min(len(ranks), k)
    ideal = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
    return sum(gains) / ideal if ideal else 0.0


# ── reports ────────────────────────────────────────────────────────
@dataclass
class RetrievalReport:
    mode: str
    split: str
    k: int
    n_answerable: int
    recall_at_k: float
    mrr: float
    ndcg_at_k: float
    evidence_coverage: float
    per_question: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "mode": self.mode, "split": self.split, "k": self.k,
            "n_answerable": self.n_answerable,
            "recall_at_k": round(self.recall_at_k, 4),
            "mrr": round(self.mrr, 4),
            "ndcg_at_k": round(self.ndcg_at_k, 4),
            "evidence_coverage": round(self.evidence_coverage, 4),
            "per_question": self.per_question,
        }


def evaluate_retrieval(retriever: HybridRetriever, questions: list[Question], *,
                       k: int = 5, mode: str = "hybrid",
                       split: str = "test") -> RetrievalReport:
    rows = [q for q in questions if q.answerable and (split == "all" or q.split == split)]
    if not rows:
        raise ValueError(f"no answerable questions in split '{split}'")
    recalls, rrs, ndcgs, coverage = [], [], [], []
    per_question = []
    for q in rows:
        hits = retriever.search(q.question, top_k=k, mode=mode)
        texts = [h.text for h in hits]
        quotes = [e.quote for e in q.evidence]
        ranks = evidence_ranks(texts, quotes)
        first = min((r for r in ranks if r is not None), default=None)
        recalls.append(1.0 if first is not None else 0.0)
        rrs.append(1.0 / first if first else 0.0)
        ndcgs.append(ndcg_at_k(ranks, k))
        coverage.append(sum(1 for r in ranks if r is not None and r <= k) / len(ranks))
        per_question.append({
            "id": q.id, "first_rank": first,
            "evidence_ranks": ranks,
            "top": [h.chunk.id for h in hits],
        })
    return RetrievalReport(
        mode=mode, split=split, k=k, n_answerable=len(rows),
        recall_at_k=sum(recalls) / len(recalls),
        mrr=sum(rrs) / len(rrs),
        ndcg_at_k=sum(ndcgs) / len(ndcgs),
        evidence_coverage=sum(coverage) / len(coverage),
        per_question=per_question,
    )


@dataclass
class AnswerReport:
    split: str
    k: int
    model: str
    n_answerable: int
    n_unanswerable: int
    answer_accuracy: float
    false_abstention_rate: float
    full_grounding_rate: float
    unanswerable_abstention_rate: float
    citation_validity_rate: float
    mean_grounded_ratio: float
    latency_p50_ms: float
    latency_p95_ms: float
    per_question: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "split": self.split, "k": self.k, "model": self.model,
            "n_answerable": self.n_answerable,
            "n_unanswerable": self.n_unanswerable,
            "answer_accuracy": round(self.answer_accuracy, 4),
            "false_abstention_rate": round(self.false_abstention_rate, 4),
            "full_grounding_rate": round(self.full_grounding_rate, 4),
            "unanswerable_abstention_rate": round(self.unanswerable_abstention_rate, 4),
            "citation_validity_rate": round(self.citation_validity_rate, 4),
            "mean_grounded_ratio": round(self.mean_grounded_ratio, 4),
            "latency_p50_ms": round(self.latency_p50_ms, 1),
            "latency_p95_ms": round(self.latency_p95_ms, 1),
            "per_question": self.per_question,
        }


def _p95(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, math.ceil(0.95 * len(ordered)) - 1)
    return ordered[idx]


def evaluate_answers(engine: RagEngine, questions: list[Question], *,
                     k: int = 5, split: str = "test") -> AnswerReport:
    rows = [q for q in questions if split == "all" or q.split == split]
    if not rows:
        raise ValueError(f"no questions in split '{split}'")
    answerable = [q for q in rows if q.answerable]
    unanswerable = [q for q in rows if not q.answerable]
    if not answerable or not unanswerable:
        raise ValueError("each split needs answerable and unanswerable questions")

    correct = abstained_answerable = answered = fully_grounded = 0
    correct_abs = 0
    cited_answers = valid_citations = 0
    grounded_ratios: list[float] = []
    latencies: list[float] = []
    per_question = []

    for q in rows:
        result: AnswerResult = engine.ask(q.question, top_k=k)
        latencies.append(result.retrieval_ms + result.generation_ms)
        row = {"id": q.id, "answerable": q.answerable, "status": result.status.value,
               "answer": result.answer, "reason": result.reason,
               "grounded_ratio": (round(result.grounding.grounded_ratio, 3)
                                  if result.grounding else None),
               "citations_valid": (result.grounding.all_citations_valid
                                   if result.grounding else None)}
        if result.grounding and result.grounding.citations_present:
            cited_answers += 1
            if result.grounding.all_citations_valid:
                valid_citations += 1
        if q.answerable:
            gold_hit = any(answer_contains(result.answer, g) for g in q.gold_answers)
            row["correct"] = bool(not result.abstained and gold_hit)
            if result.abstained:
                abstained_answerable += 1
            else:
                answered += 1
                if result.grounding:
                    grounded_ratios.append(result.grounding.grounded_ratio)
                    if result.grounding.grounded_ratio >= 1.0:
                        fully_grounded += 1
            if row["correct"]:
                correct += 1
        else:
            row["correct"] = result.abstained
            if result.abstained:
                correct_abs += 1
        per_question.append(row)

    n_ans = len(answerable)
    n_unans = len(unanswerable)
    return AnswerReport(
        split=split, k=k, model=getattr(engine.chat, "name", "unknown"),
        n_answerable=n_ans, n_unanswerable=n_unans,
        answer_accuracy=correct / n_ans,
        false_abstention_rate=abstained_answerable / n_ans,
        full_grounding_rate=fully_grounded / n_ans,
        unanswerable_abstention_rate=correct_abs / n_unans,
        citation_validity_rate=(valid_citations / cited_answers) if cited_answers else 0.0,
        mean_grounded_ratio=(statistics.fmean(grounded_ratios) if grounded_ratios else 0.0),
        latency_p50_ms=statistics.median(latencies) if latencies else 0.0,
        latency_p95_ms=_p95(latencies),
        per_question=per_question,
    )


def report_markdown(*, retrieval: list[RetrievalReport], answers: AnswerReport | None,
                    dataset: dict) -> str:
    lines = ["# Sanad evaluation report", "",
             f"- dataset: {dataset['path']} ({dataset['n_questions']} questions; "
             f"dev {dataset['n_dev']}, test {dataset['n_test']})",
             f"- corpus snapshot: {dataset['corpus_documents']} documents, "
             f"{dataset['corpus_chunks']} chunks",
             f"- embedding model: `{dataset['embedding_model']}`", ""]
    lines += ["## Retrieval", "",
              "| mode | split | k | n | recall@k | MRR | nDCG@k | evidence coverage |",
              "|---|---|---|---|---|---|---|---|"]
    for r in retrieval:
        lines.append(f"| {r.mode} | {r.split} | {r.k} | {r.n_answerable} | "
                     f"{r.recall_at_k:.3f} | {r.mrr:.3f} | {r.ndcg_at_k:.3f} | "
                     f"{r.evidence_coverage:.3f} |")
    if answers:
        lines += ["", "## End-to-end answers", "",
                  f"- model: `{answers.model}`; split: {answers.split}; top-k: {answers.k}",
                  f"- answer accuracy (gold answer present): {answers.answer_accuracy:.3f} "
                  f"({answers.n_answerable} answerable)",
                  f"- false abstention on answerable: {answers.false_abstention_rate:.3f}",
                  f"- fully grounded answers: {answers.full_grounding_rate:.3f}",
                  f"- abstention on unanswerable: {answers.unanswerable_abstention_rate:.3f} "
                  f"({answers.n_unanswerable} unanswerable)",
                  f"- citation validity among answered: {answers.citation_validity_rate:.3f}",
                  f"- mean grounded-sentence ratio: {answers.mean_grounded_ratio:.3f}",
                  f"- latency p50/p95: {answers.latency_p50_ms:.0f} / "
                  f"{answers.latency_p95_ms:.0f} ms", "",
                  "### Per question", "",
                  "| id | answerable | status | correct | grounded | answer |",
                  "|---|---|---|---|---|---|"]
        for row in answers.per_question:
            answer = row["answer"].replace("|", "\\|").replace("\n", " ")
            lines.append(f"| {row['id']} | {row['answerable']} | {row['status']} | "
                         f"{row['correct']} | {row['grounded_ratio']} | {answer[:180]} |")
    lines += ["", "## Method notes", "",
              "- Retrieval hit = a retrieved chunk contains the labeled evidence quote verbatim "
              "after Arabic/Latin normalization; labels are validated against the corpus.",
              "- Answer accuracy uses normalized gold-answer containment; it is strict about "
              "the fact and lenient about surrounding wording.",
              "- \"Unsupported\" = answered without a grounded, valid citation.",
              ""]
    return "\n".join(lines)
