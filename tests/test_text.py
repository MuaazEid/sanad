import pytest

from sanad.text import (Chunk, chunk_text, content_tokens, normalize,
                        split_sentences, tokens)


@pytest.mark.parametrize("raw,expected", [
    ("أحمد", "احمد"),
    ("إبراهيم", "ابراهيم"),
    ("آمنة", "امنه"),
    ("مُحَمَّد", "محمد"),
    ("مـــحمد", "محمد"),
    ("جامعة", "جامعه"),
    ("مصطفى", "مصطفي"),
])
def test_normalize_arabic_folds(raw, expected):
    assert normalize(raw) == expected


def test_arabic_punctuation_is_not_part_of_tokens():
    # Regression: U+060C is inside the Arabic Unicode block and was once
    # treated as a letter, so "محطة،" never matched "محطة".
    assert tokens("محطة، الرياض؛") == ["محطه", "الرياض"]
    assert "السبعه" in tokens("الخطوط السبعة،")


def test_arabic_indic_digits_become_ascii():
    assert normalize("١٧٦ كيلومتر و٨٥ محطة") == "176 كيلومتر و85 محطه"


def test_normalize_english_case_and_punctuation():
    assert normalize("The Riyadh Metro, 2024!") == "the riyadh metro 2024"


def test_normalize_is_idempotent():
    once = normalize("الإعْذَابُ والزَّملحة — ١٩٧٥")
    assert normalize(once) == once


def test_tokens_mixed_script():
    assert tokens("مترو Riyadh 2024") == ["مترو", "riyadh", "2024"]


def test_content_tokens_remove_stopwords_and_single_chars():
    got = content_tokens("The metro has 85 stations in Riyadh")
    assert "the" not in got and "in" not in got
    assert "85" in got and "stations" in got


def test_split_sentences_arabic_and_english():
    text = "تأسست الجامعة في 1975. تقع في ود مدني؟ نعم!"
    assert split_sentences(text) == ["تأسست الجامعة في 1975.", "تقع في ود مدني؟", "نعم!"]
    assert split_sentences("One line.\nTwo line.") == ["One line.", "Two line."]


def test_chunk_text_respects_size_and_covers_content():
    text = " ".join(f"Sentence number {i} carries some content." for i in range(40))
    chunks = chunk_text(text, max_chars=200, overlap_sentences=1)
    assert len(chunks) > 1
    assert all(len(c.text) <= 260 for c in chunks)
    assert "Sentence number 0" in chunks[0].text
    assert "Sentence number 39" in chunks[-1].text


def test_chunk_text_tracks_sections():
    text = "Intro paragraph here.\n\n== History ==\nThe university was founded in 1975.\n\n== Campus ==\nIt sits in Wad Madani."
    chunks = chunk_text(text, max_chars=200)
    assert chunks[0].section == ""
    assert chunks[1].section == "History"
    assert chunks[2].section == "Campus"


def test_chunk_text_splits_overlong_sentence():
    text = "word " * 400
    chunks = chunk_text(text, max_chars=200)
    assert all(len(c.text) <= 200 for c in chunks)


def test_chunk_text_merges_trailing_crumb():
    text = "A short paragraph with enough words to matter. " * 3 + "Tiny."
    chunks = chunk_text(text, max_chars=250, min_chars=80)
    assert len(chunks[-1].text) >= 80


def test_chunk_text_empty_and_whitespace():
    assert chunk_text("") == []
    assert chunk_text("   \n\n  ") == []


def test_chunk_text_deduplicates_identical_blocks():
    text = "Same paragraph repeated.\n\nSame paragraph repeated."
    chunks = chunk_text(text)
    assert len(chunks) == 1


def test_chunk_ordinals_are_contiguous():
    text = " ".join(f"Fact {i} is stated here clearly." for i in range(60))
    chunks = chunk_text(text, max_chars=150)
    assert [c.ordinal for c in chunks] == list(range(len(chunks)))
