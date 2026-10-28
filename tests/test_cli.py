import json
from pathlib import Path

from sanad.cli import main

ROOT = Path(__file__).resolve().parents[1]


def make_corpus(tmp_path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "doc.txt").write_text(
        "جامعة الجزيرة تأسست في 9 نوفمبر 1975 في مدينة ود مدني. "
        "من أشهر كلياتها كلية الطب.", encoding="utf-8")
    (corpus / "manifest.json").write_text(json.dumps({
        "documents": [{"id": "ar-g", "title": "جامعة الجزيرة", "lang": "ar",
                       "path": "corpus/doc.txt", "source_url": "https://example.org",
                       "revision": "1", "license": "CC BY-SA 4.0"}]},
        ensure_ascii=False), encoding="utf-8")
    return corpus


def base(tmp_path):
    return ["--db", str(tmp_path / "sanad.db"), "--embedder", "hashing"]


def test_ingest_search_stats_roundtrip(tmp_path, capsys):
    corpus = make_corpus(tmp_path)
    assert main(base(tmp_path) + ["ingest", "--corpus", str(corpus)]) == 0
    assert "1 docs" in capsys.readouterr().out

    assert main(base(tmp_path) + ["search", "ود مدني", "-k", "1"]) == 0
    out = capsys.readouterr().out
    assert "جامعة الجزيرة" in out

    assert main(base(tmp_path) + ["stats"]) == 0
    stats = json.loads(capsys.readouterr().out)
    assert stats["documents"] == 1 and stats["chunks"] >= 1


def test_search_without_hits_exits_3(tmp_path):
    corpus = make_corpus(tmp_path)
    assert main(base(tmp_path) + ["ingest", "--corpus", str(corpus)]) == 0
    assert main(base(tmp_path) + ["search", "zzzz-unfindable", "--mode", "lexical"]) == 3


def test_validate_real_dataset():
    assert main(["validate"]) == 0


def test_bad_corpus_returns_error(tmp_path):
    assert main(base(tmp_path) + ["ingest", "--corpus", str(tmp_path / "nope")]) == 1


def test_search_json_output(tmp_path, capsys):
    corpus = make_corpus(tmp_path)
    main(base(tmp_path) + ["ingest", "--corpus", str(corpus)])
    capsys.readouterr()
    assert main(base(tmp_path) + ["search", "الطب", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload[0]["title"] == "جامعة الجزيرة"
