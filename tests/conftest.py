import pytest

from sanad.embed import HashingEmbedder
from sanad.retrieve import HybridRetriever
from sanad.store import Store
from sanad.text import chunk_text

DOCS = {
    "ar-gezira": ("ar", "جامعة الجزيرة", (
        "جامعة الجزيرة جامعة حكومية سودانية تأسست في 9 نوفمبر 1975. "
        "تقع رئاستها في مدينة ود مدني حاضرة ولاية الجزيرة. "
        "من أشهر كلياتها كلية الطب."
        "\n\n"
        "بدأت الدراسة في العام 1978 بأربع كليات. تخرجت الدفعة الأولى في العام 1984."
    )),
    "en-metro": ("en", "Riyadh Metro", (
        "The Riyadh Metro has six lines and 85 stations. "
        "It opened on December 1, 2024. The project cost $22.5 billion to build."
        "\n\n"
        "The metro is the longest fully automated, driverless metro in the world."
    )),
    "en-desal": ("en", "Desalination", (
        "Desalination converts salt water into fresh water and produces brine as a by-product. "
        "Around 95 million cubic meters per day were produced worldwide in 2019."
    )),
}


@pytest.fixture
def embedder():
    return HashingEmbedder(dim=256)


@pytest.fixture
def store(tmp_path, embedder):
    s = Store(tmp_path / "test.db")
    for doc_id, (lang, title, text) in DOCS.items():
        s.ingest(doc_id=doc_id, title=title, lang=lang, text=text,
                 chunks=chunk_text(text, max_chars=220), embedder=embedder,
                 source_url=f"https://example.org/{doc_id}", license="CC BY-SA 4.0")
    yield s
    s.close()


@pytest.fixture
def retriever(store, embedder):
    return HybridRetriever(store, embedder, candidates=10)


class FakeChat:
    name = "fake"

    def __init__(self, *replies):
        self.replies = list(replies)
        self.calls = []

    def chat(self, system, user):
        self.calls.append((system, user))
        if not self.replies:
            return "INSUFFICIENT_EVIDENCE"
        return self.replies.pop(0)
