"""Live integration test. Requires Ollama + models; skipped by default.

Run with:
    SANAD_LIVE=1 SANAD_LIVE_MODEL=qwen2.5:7b pytest -m live -v

The assertion is the pipeline invariant, not the 7B model's accuracy: an
answer is either verified as grounded or withheld. Model accuracy is measured
separately by `sanad eval --generate` on the gold test split.
"""
import os
from pathlib import Path

import pytest

from sanad.corpus import ingest_corpus
from sanad.embed import OllamaEmbedder
from sanad.generate import OllamaChat
from sanad.pipeline import RagEngine, Status
from sanad.retrieve import HybridRetriever
from sanad.store import Store
from sanad.text import normalize

pytestmark = pytest.mark.live

ROOT = Path(__file__).resolve().parents[1]
LIVE = os.environ.get("SANAD_LIVE") == "1"

@pytest.mark.skipif(not LIVE, reason="set SANAD_LIVE=1 to run against a real model")
def test_live_answer_is_verified_or_withheld(tmp_path):
    model = os.environ.get("SANAD_LIVE_MODEL", "qwen2.5:7b")
    url = os.environ.get("SANAD_OLLAMA_URL", "http://localhost:11434")
    with Store(tmp_path / "live.db") as store:
        ingest_corpus(store, OllamaEmbedder(model="bge-m3", base_url=url), ROOT / "corpus")
        engine = RagEngine(HybridRetriever(store, OllamaEmbedder(model="bge-m3", base_url=url)),
                           OllamaChat(model=model, base_url=url))
        result = engine.ask("كم عدد محطات مترو الرياض؟", top_k=5)
        assert result.status is not Status.INSUFFICIENT, result.reason
        assert result.generation_ms > 0 and result.sources
        if result.status is Status.ANSWERED:
            assert result.grounding.all_citations_valid
            assert result.grounding.grounded_ratio == 1.0
            assert "85" in normalize(result.answer)
        else:
            assert result.status is Status.UNGROUNDED and result.reason
