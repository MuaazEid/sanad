"""Arabic/English text handling: normalization, tokenization, chunking.

The normalization rules are deliberately *lossy for retrieval only*: the
original text is stored untouched for display and citation, while a folded
copy is indexed. Arabic search fails silently when أ/إ/آ, ة/ه and harakat
are treated as different tokens, so folding them is the difference between
a working index and a bag of near-misses.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

_AR_DIACRITICS = re.compile(r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED]")
_AR_PUNCT = re.compile(r"[\u060C\u061B\u061F\u066A-\u066D\u06D4\u0964\u0965\u00AB\u00BB]")
_TATWEEL = "\u0640"
_PUNCT = re.compile(r"[^\w\u0600-\u06FF]+|_", re.UNICODE)
_WORD = re.compile(r"[a-z0-9]+|[\u0600-\u06FF]+", re.UNICODE)

_DIGIT_MAP = {
    **{ord(c): str(i) for i, c in enumerate("٠١٢٣٤٥٦٧٨٩")},
    **{ord(c): str(i) for i, c in enumerate("۰۱۲۳۴۵۶۷۸۹")},
}
_TRANSLATE = str.maketrans({
    **{c: "ا" for c in "أإآٱ"},
    "ى": "ي",
    "ة": "ه",
    **_DIGIT_MAP,
})

STOPWORDS_AR = frozenset("""
في من على إلى عن أن إن ان التي الذي الذين هذا هذه هذان هو هي هم كان كانت يكون
مع لا ما قد كل بين بعد قبل حتى أو ثم كما لدى منذ نحو وهو وهي ولا وما ومن وفي
علي الي الى انه أنها أنها كانوا لم لن له لها لهم به بها بهذا كذلك أيضا ايضا
اذا إذا غير سوى ضمن خلال حول دون عبر عند فقط وقد وكان وكانت هناك هنالك أي اي
""".split())

STOPWORDS_EN = frozenset("""
a an the and or of to in on at for from by with without is are was were be been
being it its this that these those as into about over under than then there here
what which who whom whose when where why how many much does do did has have had
can could should would may might will shall not no nor but if while during per
""".split())

STOPWORDS = STOPWORDS_AR | STOPWORDS_EN


def normalize(text: str) -> str:
    """Fold text into the retrieval alphabet."""
    text = unicodedata.normalize("NFKC", text)
    text = _AR_DIACRITICS.sub("", text).replace(_TATWEEL, "")
    text = text.translate(_TRANSLATE).lower()
    text = _AR_PUNCT.sub(" ", text)
    text = _PUNCT.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


def tokens(text: str) -> list[str]:
    return _WORD.findall(normalize(text))


def content_tokens(text: str) -> list[str]:
    """Tokens that carry evidence: no stopwords, no single characters."""
    return [t for t in tokens(text) if len(t) > 1 and t not in STOPWORDS]


def split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?؟۔])\s+|\n+", text)
    return [p.strip() for p in parts if p.strip()]


_HEADING = re.compile(r"^\s*(?:={2,6}|#{1,6})\s*(.+?)\s*(?:={2,6}|#*)\s*$")


@dataclass(frozen=True)
class Chunk:
    ordinal: int
    section: str
    text: str

    @property
    def n_tokens(self) -> int:
        return len(tokens(self.text))


def _split_long(sentence: str, max_chars: int) -> list[str]:
    if len(sentence) <= max_chars:
        return [sentence]
    pieces, current = [], ""
    for word in sentence.split():
        candidate = f"{current} {word}".strip()
        if current and len(candidate) > max_chars:
            pieces.append(current)
            current = word
        else:
            current = candidate
    if current:
        pieces.append(current)
    return pieces


def _blocks(text: str):
    """Yield (section, paragraph) while stripping wiki/markdown headings."""
    section = ""
    buffer: list[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            if buffer:
                yield section, " ".join(buffer)
                buffer = []
            continue
        m = _HEADING.match(line)
        if m:
            if buffer:
                yield section, " ".join(buffer)
                buffer = []
            section = m.group(1).strip()
            continue
        buffer.append(line)
    if buffer:
        yield section, " ".join(buffer)


def chunk_text(text: str, *, max_chars: int = 900, min_chars: int = 200,
               overlap_sentences: int = 1) -> list[Chunk]:
    chunks: list[Chunk] = []
    for section, block in _blocks(text):
        sentences: list[str] = []
        for sent in split_sentences(block):
            sentences.extend(_split_long(sent, max_chars))
        current: list[str] = []
        length = 0
        for sent in sentences:
            if current and length + len(sent) + 1 > max_chars:
                chunks.append(Chunk(len(chunks), section, " ".join(current)))
                tail = current[-overlap_sentences:] if overlap_sentences else []
                while tail and len(" ".join(tail)) > max_chars // 3:
                    tail = tail[1:]
                if tail and len(" ".join(tail)) + len(sent) + 1 > max_chars:
                    tail = []
                current = list(tail)
                length = len(" ".join(current))
            current.append(sent)
            length += len(sent) + 1
        if current:
            chunks.append(Chunk(len(chunks), section, " ".join(current)))

    # A trailing crumb is not worth its own embedding — unless merging it
    # would move text into the wrong section.
    if (len(chunks) > 1 and len(chunks[-1].text) < min_chars
            and chunks[-1].section == chunks[-2].section
            and len(chunks[-1].text) + len(chunks[-2].text) + 1 <= max_chars):
        last = chunks.pop()
        prev = chunks[-1]
        chunks[-1] = Chunk(prev.ordinal, prev.section, f"{prev.text} {last.text}".strip())

    seen: set[str] = set()
    unique: list[Chunk] = []
    for c in chunks:
        key = normalize(c.text)
        if key and key not in seen:
            seen.add(key)
            unique.append(Chunk(len(unique), c.section, c.text))
    return unique
