import sys
from pathlib import Path
from types import SimpleNamespace

# If `openai` is not installed in the test environment, provide a minimal stub
try:
    import openai  # type: ignore
except Exception:
    import types

    openai = types.ModuleType("openai")

    class DummyOpenAI:
        def __init__(self, *a, **k):
            pass

    openai.OpenAI = DummyOpenAI
    sys.modules["openai"] = openai

# Ensure repo root is importable for package imports
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from ragonometrics.main import Settings, top_k_context
from ragonometrics.io_loaders import chunk_words, trim_words


def test_trim_words_truncates():
    text = "".join(["word "] * 10).strip()
    out = trim_words(text, 5)
    assert len(out.split()) == 5


def test_chunk_words_overlap_and_size():
    text = " ".join(str(i) for i in range(20))
    chunks = chunk_words(text, chunk_words=7, overlap_words=2)
    # step = 7 - 2 = 5 -> expected chunks: indices [0:7], [5:12], [10:17], [15:20]
    assert len(chunks) == 4
    assert chunks[0].split()[0] == "0"
    assert chunks[1].split()[0] == "5"


def test_top_k_context_selects_top_chunks():
    chunks = ["chunk_a", "chunk_b", "chunk_c"]
    # embeddings are 2-D for simplicity
    chunk_embeddings = [[1.0, 0.0], [0.0, 1.0], [0.1, 0.1]]

    class DummyClient:
        class _E:
            def __init__(self, emb):
                self.data = [SimpleNamespace(embedding=emb)]

        def __init__(self, emb):
            self._emb = emb

        def embeddings(self):
            raise RuntimeError("not used")

        @property
        def embeddings_create(self):
            return None

        # support the call pattern used in top_k_context
        def embeddings_create(self, model, input):
            return DummyClient._E(self._emb)

    # monkeypatch client to expose .embeddings.create
    class ClientWrapper:
        def __init__(self, emb):
            self._emb = emb
            self.embeddings = SimpleNamespace(create=lambda model, input: SimpleNamespace(data=[SimpleNamespace(embedding=self._emb)]))

    client = ClientWrapper([1.0, 0.0])
    settings = Settings(
        papers_dir=Path("papers"),
        max_papers=1,
        max_words=1000,
        chunk_words=100,
        chunk_overlap=10,
        top_k=2,
        batch_size=64,
        embedding_model="emb-model",
        chat_model="chat-model",
    )

    ctx = top_k_context(chunks, chunk_embeddings, query="test", client=client, settings=settings)
    # should pick chunk_a (closest) and chunk_c (next closest)
    assert "chunk_a" in ctx
    assert "chunk_c" in ctx

