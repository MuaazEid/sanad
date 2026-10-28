import pytest

from sanad.generate import INSUFFICIENT
from sanad.pipeline import RagEngine, Status
from sanad.retrieve import Retrieved
from sanad.store import StoredChunk
from tests.conftest import FakeChat

TEXT = (
    "يتكون المترو من ستة خطوط رئيسة تربط مختلف أحياء المدينة على امتداد "
    "176 كيلومتر، ويضم 85 محطة منها 8 محطات رئيسة تربط الخطوط الستة ببعضها."
)


def retrieved(text=TEXT, dense=0.8, bm25=1.2):
    chunk = StoredChunk("ar-metro:0000", "ar-metro", "قطار الرياض", "ar", "", text)
    return Retrieved(chunk=chunk, score=0.03, bm25_score=bm25, dense_score=dense,
                     bm25_rank=1, dense_rank=1)


class StubRetriever:
    def __init__(self, hits):
        self.hits = hits
        self.calls = []

    def search(self, query, **kwargs):
        self.calls.append((query, kwargs))
        return self.hits


def engine(hits, *replies, **kwargs):
    retriever = StubRetriever(hits)
    chat = FakeChat(*replies)
    return RagEngine(retriever, chat, **kwargs), retriever, chat


def test_answered_with_grounded_citation():
    eng, retriever, chat = engine([retrieved()], "يضم المترو 85 محطة [S1].")
    res = eng.ask("كم عدد المحطات؟", top_k=5)
    assert res.status is Status.ANSWERED
    assert res.grounding.grounded_ratio == 1.0
    assert retriever.calls[0][1]["top_k"] == 5
    assert len(chat.calls) == 1


def test_model_refusal_maps_to_insufficient():
    eng, _, _ = engine([retrieved()], INSUFFICIENT)
    res = eng.ask("سؤال غير موجود؟")
    assert res.status is Status.INSUFFICIENT
    assert res.abstained
    assert "declined" in res.reason


def test_no_sources_short_circuits_without_llm_call():
    eng, _, chat = engine([], "should not be used")
    res = eng.ask("سؤال؟")
    assert res.status is Status.INSUFFICIENT
    assert chat.calls == []


def test_uncited_answer_is_ungrounded():
    eng, _, _ = engine([retrieved()], "يضم المترو 85 محطة.")
    res = eng.ask("كم عدد المحطات؟")
    assert res.status is Status.UNGROUNDED
    assert "no source citations" in res.reason


def test_invalid_citation_is_ungrounded():
    eng, _, _ = engine([retrieved()], "يضم المترو 85 محطة [S4].")
    res = eng.ask("كم عدد المحطات؟")
    assert res.status is Status.UNGROUNDED
    assert "do not exist" in res.reason


def test_unsupported_citation_is_ungrounded():
    eng, _, _ = engine([retrieved()], "The metro cost 22.5 billion dollars [S1].")
    res = eng.ask("How much did it cost?")
    assert res.status is Status.UNGROUNDED
    assert "no sentence was supported" in res.reason


def test_repair_round_can_rescue_an_invalid_citation():
    eng, _, chat = engine([retrieved()],
                          "يضم المترو 85 محطة [S4].",
                          "يضم المترو 85 محطة [S1].")
    res = eng.ask("كم عدد المحطات؟")
    assert res.status is Status.ANSWERED
    assert len(chat.calls) == 2
    repair_prompt = chat.calls[1][1]
    assert "rejected" in repair_prompt and "do not exist" in repair_prompt
    assert "85" in res.answer


def test_repair_that_stays_wrong_keeps_the_original_rejection():
    eng, _, chat = engine([retrieved()],
                          "The metro cost 22.5 billion dollars [S1].",
                          "The metro cost 30 billion dollars [S1].")
    res = eng.ask("How much did it cost?")
    assert res.status is Status.UNGROUNDED
    assert "no sentence was supported" in res.reason
    assert "22.5" in res.answer
    assert len(chat.calls) == 2


def test_repair_refusal_does_not_turn_into_a_weak_abstention():
    eng, _, _ = engine([retrieved()], "يضم المترو 85 محطة [S4].")
    res = eng.ask("كم عدد المحطات؟")
    assert res.status is Status.UNGROUNDED
    assert "do not exist" in res.reason


def test_repair_can_be_disabled():
    eng, _, chat = engine([retrieved()], "يضم المترو 85 محطة [S4].",
                          repair_attempts=0)
    res = eng.ask("كم عدد المحطات؟")
    assert res.status is Status.UNGROUNDED
    assert len(chat.calls) == 1


def test_dense_threshold_abstains_before_generation():
    eng, _, chat = engine([retrieved(dense=0.10)], "never used", dense_min_score=0.3)
    res = eng.ask("سؤال؟")
    assert res.status is Status.INSUFFICIENT
    assert chat.calls == []


def test_empty_question_rejected():
    eng, _, _ = engine([retrieved()], "x")
    with pytest.raises(ValueError):
        eng.ask("   ")


def test_result_serialization_contains_evidence_and_timings():
    eng, _, _ = engine([retrieved()], "يضم المترو 85 محطة [S1].")
    payload = eng.ask("كم عدد المحطات؟").to_dict()
    assert payload["status"] == "answered"
    assert payload["sources"][0]["id"] == "ar-metro:0000"
    assert payload["grounding"]["sentences"][0]["supported"] is True
    assert payload["retrieval_ms"] >= 0
    assert payload["generation_ms"] >= 0
