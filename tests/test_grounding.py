from types import SimpleNamespace

import pytest

from sanad.generate import (INSUFFICIENT, build_messages, is_refusal,
                            verify_answer)
from sanad.store import StoredChunk

CHUNK = (
    "يتكون المترو من ستة خطوط رئيسة تربط مختلف أحياء المدينة على امتداد "
    "176 كيلومتر وبسرعة 80 كلم في الساعة، ويضم 85 محطة منها 8 محطات "
    "رئيسة تربط الخطوط الستة ببعضها."
)


def source(text, cid="ar-metro:0000", title="قطار الرياض", section=""):
    return SimpleNamespace(
        text=text,
        chunk=StoredChunk(cid, "ar-metro", title, "ar", section, text))


def test_correct_answer_is_grounded():
    report = verify_answer("يضم المترو 85 محطة [S1]." , [source(CHUNK)])
    assert report.citations_present
    assert report.all_citations_valid
    assert report.grounded_ratio == 1.0


def test_low_overlap_is_not_grounded():
    report = verify_answer("The project cost 22.5 billion dollars [S1].",
                           [source(CHUNK)])
    assert report.grounded == 0
    assert not report.checks[0].supported


def test_wrong_digit_is_not_grounded():
    report = verify_answer("يضم المترو 95 محطة [S1].", [source(CHUNK)])
    assert report.checks[0].numbers_ok is False
    assert report.checks[0].supported is False


def test_wrong_number_word_is_not_grounded():
    # Regression: "seven" passed lexical overlap even though the source says six.
    report = verify_answer(
        "يضم المترو 85 محطة، منها 8 محطات رئيسة تربط الخطوط السبعة ببعضها [S1].",
        [source(CHUNK)])
    assert report.checks[0].numbers_ok is False
    assert report.checks[0].supported is False


def test_correct_number_word_is_grounded():
    report = verify_answer(
        "يضم المترو 85 محطة، منها 8 محطات رئيسة تربط الخطوط الستة ببعضها [S1].",
        [source(CHUNK)])
    assert report.checks[0].numbers_ok is True
    assert report.checks[0].supported is True


def test_invalid_citation_id_is_flagged():
    report = verify_answer("يضم المترو 85 محطة [S9].", [source(CHUNK)])
    assert report.citations_present
    assert report.all_citations_valid is False
    assert report.checks[0].valid_citations == ()


def test_answer_without_citations_is_flagged():
    report = verify_answer("يضم المترو 85 محطة.", [source(CHUNK)])
    assert report.citations_present is False
    assert report.grounded == 0


def test_uncited_claims_are_counted():
    report = verify_answer("يضم المترو 85 محطة [S1]. تقع الجامعة في مدينة الرياض بالسودان.",
                           [source(CHUNK)])
    assert report.uncited_claims == 1


def test_trivial_sentences_are_excluded_from_ratio():
    report = verify_answer("نعم [S1]. يضم المترو 85 محطة [S1].", [source(CHUNK)])
    assert len(report.scorable) == 1
    assert report.grounded_ratio == 1.0


def test_multiple_citations_accept_any_supporting_source():
    other = source("تحلية المياه تزيل الأملاح من مياه البحر.", cid="ar-water:0000",
                   title="تحلية")
    report = verify_answer("يضم المترو 85 محطة [S1][S2].", [source(CHUNK), other])
    assert report.all_citations_valid
    assert report.checks[0].supported


@pytest.mark.parametrize("marker, expected", [
    ("[S1, S2]", (1, 2)),
    ("[S1; S2]", (1, 2)),
    ("[S2, S1, S3]", (2, 1, 3)),
])
def test_grouped_citation_markers_are_parsed(marker, expected):
    other = source("تحلية المياه تزيل الأملاح من مياه البحر.", cid="ar-water:0000",
                   title="تحلية")
    sources = [source(CHUNK), other, source(CHUNK, cid="ar-metro:0001")]
    report = verify_answer(f"يضم المترو 85 محطة {marker}.", sources)
    assert report.citations_present
    assert report.all_citations_valid
    assert report.checks[0].valid_citations == expected
    assert report.checks[0].supported


def test_grouped_citation_still_flags_unknown_ids():
    report = verify_answer("يضم المترو 85 محطة [S1, S9].", [source(CHUNK)])
    assert report.all_citations_valid is False
    assert report.checks[0].supported


def test_trailing_citation_only_sentence_is_attached_to_the_claim():
    report = verify_answer("يضم المترو 85 محطة. [S1]", [source(CHUNK)])
    assert report.checks[0].citations == (1,)
    assert report.checks[0].supported
    assert report.grounded_ratio == 1.0


def test_trailing_citation_fragment_keeps_invalid_id_visible():
    report = verify_answer("يضم المترو 85 محطة. [S1] [S9]", [source(CHUNK)])
    assert report.checks[0].citations == (1, 9)
    assert report.all_citations_valid is False
    assert report.checks[0].supported


def test_is_refusal_detects_sentinel_case_insensitively():
    assert is_refusal(INSUFFICIENT)
    assert is_refusal("insufficient_evidence.")
    assert not is_refusal("The answer is 85.")


def test_build_messages_numbers_sources_and_caps_context():
    sources = [source(CHUNK * 3, cid=f"c{i}") for i in range(5)]
    system, user = build_messages("كم عدد المحطات؟", sources,
                                  max_source_chars=100, max_total_chars=450)
    assert "ONLY the sources" in system
    assert "[S1]" in user
    assert "Question: كم عدد المحطات؟" in user
    assert len(user) < 700
    assert "…" in user
