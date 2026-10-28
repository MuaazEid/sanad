from pathlib import Path

from sanad.corpus import load_manifest
from sanad.evaluate import load_questions, validate_dataset
from sanad.text import chunk_text

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "corpus"
QUESTIONS = ROOT / "eval" / "questions.jsonl"


def test_manifest_documents_exist():
    docs = load_manifest(CORPUS)
    assert len(docs) == 10
    assert {d.lang for d in docs} == {"ar", "en"}
    for doc in docs:
        assert doc.path.exists(), doc.path
        assert len(doc.text) > 500


def test_corpus_chunks_are_reasonable():
    docs = load_manifest(CORPUS)
    total = 0
    for doc in docs:
        chunks = chunk_text(doc.text)
        total += len(chunks)
        assert chunks
        assert all(c.text.strip() for c in chunks)
    assert 400 <= total <= 900


def test_eval_dataset_has_no_integrity_errors():
    questions = load_questions(QUESTIONS)
    assert validate_dataset(questions, CORPUS) == []


def test_eval_dataset_balance():
    questions = load_questions(QUESTIONS)
    assert len(questions) == 50
    for split in ("dev", "test"):
        rows = [q for q in questions if q.split == split]
        assert len(rows) == 25
        assert sum(1 for q in rows if not q.answerable) == 5
        assert sum(1 for q in rows if q.lang == "ar") >= 8


def test_eval_questions_are_unique_and_nonempty():
    questions = load_questions(QUESTIONS)
    assert len({q.id for q in questions}) == len(questions)
    assert all(q.question.strip() for q in questions)
