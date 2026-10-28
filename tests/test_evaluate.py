import json
import math

import pytest

from sanad.evaluate import (Evidence, Question, answer_contains,
                            evaluate_answers, evaluate_retrieval, evidence_ranks,
                            first_relevant_rank, load_questions, matches,
                            ndcg_at_k, report_markdown, validate_dataset)
from sanad.generate import INSUFFICIENT
from sanad.pipeline import RagEngine
from sanad.retrieve import Retrieved
from sanad.store import StoredChunk
from tests.conftest import FakeChat

TEXT = (
    "يتكون المترو من ستة خطوط رئيسة تربط مختلف أحياء المدينة على امتداد "
    "176 كيلومتر، ويضم 85 محطة منها 8 محطات رئيسة تربط الخطوط الستة ببعضها."
)


def hit(text=TEXT, cid="c1"):
    chunk = StoredChunk(cid, "ar-metro", "قطار الرياض", "ar", "", text)
    return Retrieved(chunk=chunk, score=0.03, bm25_score=1.0, dense_score=0.8,
                     bm25_rank=1, dense_rank=1)


class MapRetriever:
    def __init__(self, mapping):
        self.mapping = mapping

    def search(self, query, **kwargs):
        return self.mapping.get(query, [])


def q(qid="q1", split="test", answerable=True, question="س؟",
      gold=("85",), quotes=("85 محطة",)):
    return Question(qid, split, "ar", question, answerable, tuple(gold),
                    tuple(Evidence("ar-metro", x) for x in quotes))


def test_matches_is_normalization_insensitive():
    assert matches("محطة، 85", "محطه 85")
    assert not matches("85", "95")


def test_answer_contains_accepts_equivalent_date_orderings():
    answer = ("The metro was inaugurated on 27 November 2024, and opened on "
              "1 December 2024.")
    assert answer_contains(answer, "December 1, 2024")
    assert not answer_contains(answer, "15 December 2024")


def test_answer_contains_does_not_treat_plain_numbers_as_dates():
    assert not answer_contains("The system has 1 line and 2024 stations.", "2024-01")
    assert answer_contains("The fare is 50 percent of the cost.", "50%")


def test_evidence_ranks_and_first_relevant():
    texts = ["nothing", "the answer is 85 station", "85 station again"]
    assert evidence_ranks(texts, ["85 station"]) == [2]
    assert first_relevant_rank(texts, ["85 station"]) == 2
    assert first_relevant_rank(texts, ["missing"]) is None


def test_ndcg_perfect_and_displaced():
    assert ndcg_at_k([1], 5) == 1.0
    assert ndcg_at_k([2], 5) == pytest.approx(1 / math.log2(3), abs=1e-4)
    assert ndcg_at_k([None], 5) == 0.0
    assert ndcg_at_k([6], 5) == 0.0


def test_evaluate_retrieval_known_ranking():
    q1 = q("q1", question="one")
    q2 = q("q2", question="two")
    retriever = MapRetriever({
        "one": [hit()],
        "two": [hit("unrelated")],
    })
    report = evaluate_retrieval(retriever, [q1, q2], k=5, split="test")
    assert report.recall_at_k == 0.5
    assert report.mrr == 0.5
    assert report.n_answerable == 2
    assert report.per_question[0]["first_rank"] == 1
    assert report.per_question[1]["first_rank"] is None


def test_load_questions_roundtrip(tmp_path):
    path = tmp_path / "q.jsonl"
    path.write_text(json.dumps({
        "id": "q1", "split": "dev", "lang": "ar", "question": "س؟",
        "answerable": True, "gold_answers": ["85"],
        "evidence": [{"doc_id": "ar-metro", "quote": "85 محطة"}]}, ensure_ascii=False) + "\n",
        encoding="utf-8")
    rows = load_questions(path)
    assert rows[0].id == "q1" and rows[0].evidence[0].quote == "85 محطة"


def test_load_questions_reports_line_number(tmp_path):
    path = tmp_path / "q.jsonl"
    path.write_text('{"id": "q1"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match=":1:"):
        load_questions(path)


def _corpus(tmp_path, text="جامعة الجزيرة تأسست في 1975."):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "d1.txt").write_text(text, encoding="utf-8")
    (corpus / "manifest.json").write_text(json.dumps({
        "documents": [{"id": "d1", "title": "T", "lang": "ar",
                       "path": "corpus/d1.txt", "source_url": "", "license": "",
                       "revision": "1"}]}, ensure_ascii=False), encoding="utf-8")
    return corpus


def test_validate_dataset_accepts_consistent_labels(tmp_path):
    corpus = _corpus(tmp_path)
    good = Question("q1", "test", "ar", "متى تأسست؟", True, ("1975",),
                    (Evidence("d1", "تأسست في 1975"),))
    assert validate_dataset([good], corpus) == []


def test_validate_dataset_catches_bad_quote(tmp_path):
    corpus = _corpus(tmp_path)
    bad = Question("q1", "test", "ar", "متى تأسست؟", True, ("1975",),
                   (Evidence("d1", "تأسست في 1980"),))
    errors = validate_dataset([bad], corpus)
    assert any("not found verbatim" in e for e in errors)


def test_validate_dataset_catches_answer_not_in_evidence(tmp_path):
    corpus = _corpus(tmp_path)
    bad = Question("q1", "test", "ar", "متى تأسست؟", True, ("1980",),
                   (Evidence("d1", "تأسست في 1975"),))
    assert any("gold answer" in e for e in validate_dataset([bad], corpus))


def test_validate_dataset_rejects_evidence_for_unanswerable(tmp_path):
    corpus = _corpus(tmp_path)
    bad = Question("q1", "test", "ar", "س؟", False, (),
                   (Evidence("d1", "تأسست في 1975"),))
    assert any("must not have evidence" in e for e in validate_dataset([bad], corpus))


def test_evaluate_answers_end_to_end_with_fake_model():
    engine = RagEngine(MapRetriever({"سؤال؟": [hit()], "مجهول؟": [hit()]}),
                       FakeChat("يضم المترو 85 محطة [S1].", INSUFFICIENT))
    rows = [q("q1", question="سؤال؟"), q("q2", question="مجهول؟", answerable=False,
                                         gold=(), quotes=())]
    report = evaluate_answers(engine, rows, k=5, split="test")
    assert report.answer_accuracy == 1.0
    assert report.unanswerable_abstention_rate == 1.0
    assert report.false_abstention_rate == 0.0
    assert report.mean_grounded_ratio == 1.0
    assert report.citation_validity_rate == 1.0


def test_report_markdown_contains_metrics():
    retrieval = evaluate_retrieval(MapRetriever({"one": [hit()]}),
                                   [q("q1", question="one")], k=5, split="test")
    md = report_markdown(retrieval=[retrieval], answers=None, dataset={
        "path": "x", "n_questions": 1, "n_dev": 0, "n_test": 1,
        "corpus_documents": 1, "corpus_chunks": 1, "embedding_model": "x"})
    assert "## Retrieval" in md
    assert "1.000" in md
