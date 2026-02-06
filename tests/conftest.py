"""Pytest bootstrap stubs for external deps and a sqlite-backed psycopg2 shim."""

import sys
import types
import sqlite3


class SQLiteCursorWrapper:
    def __init__(self, cur):
        self._cur = cur

    def execute(self, sql, params=None):
        if params is None:
            params = ()
        # translate psycopg2 %s params to sqlite ? params
        sql2 = sql.replace("%s", "?")
        return self._cur.execute(sql2, params)

    def fetchone(self):
        return self._cur.fetchone()

    def fetchall(self):
        return self._cur.fetchall()

    def __iter__(self):
        return iter(self._cur)


class SQLiteConnWrapper:
    def __init__(self):
        self._conn = sqlite3.connect(":memory:")

    def cursor(self):
        return SQLiteCursorWrapper(self._conn.cursor())

    def commit(self):
        return self._conn.commit()

    def close(self):
        # keep underlying in-memory DB open for the test process
        return None


_SINGLE_CONN = None


def fake_connect(*args, **kwargs):
    global _SINGLE_CONN
    if _SINGLE_CONN is None:
        _SINGLE_CONN = SQLiteConnWrapper()
    return _SINGLE_CONN


# insert a fake psycopg2 module if real one is absent
if "psycopg2" not in sys.modules:
    fake_psycopg2 = types.SimpleNamespace(connect=fake_connect)
    sys.modules["psycopg2"] = fake_psycopg2


# minimal fake rank_bm25 module
if "rank_bm25" not in sys.modules:
    mod = types.ModuleType("rank_bm25")

    class BM25Okapi:
        def __init__(self, tokenized_corpus):
            self.corpus = tokenized_corpus

        def get_scores(self, tokenized_query):
            scores = []
            qset = set(tokenized_query)
            for doc in self.corpus:
                # simple overlap count
                scores.append(sum(1 for t in doc if t in qset))
            return scores

    mod.BM25Okapi = BM25Okapi
    sys.modules["rank_bm25"] = mod


# fake faiss if missing
if "faiss" not in sys.modules:
    import numpy as _np

    faiss_mod = types.ModuleType("faiss")

    class IndexFlatIP:
        def __init__(self, d):
            self.d = d
            self.vectors = _np.zeros((0, d), dtype=_np.float32)

        def add(self, X):
            self.vectors = _np.vstack([self.vectors, _np.array(X, dtype=_np.float32)])

        @property
        def ntotal(self):
            return self.vectors.shape[0]

        def search(self, q, k):
            # q: (n, d)
            q = _np.array(q, dtype=_np.float32)
            # normalize
            qn = q / ( _np.linalg.norm(q, axis=1, keepdims=True) + 1e-9)
            vn = self.vectors / (_np.linalg.norm(self.vectors, axis=1, keepdims=True) + 1e-9)
            sims = _np.dot(qn, vn.T)
            idx = _np.argsort(-sims, axis=1)[:, :k]
            dist = _np.take_along_axis(sims, idx, axis=1)
            return dist, idx

    def write_index(idx, path):
        _np.savez(path, vectors=idx.vectors, d=idx.d)

    def read_index(path):
        data = _np.load(path + '.npz' if not str(path).endswith('.npz') else path)
        idx = IndexFlatIP(int(data['d']))
        idx.vectors = data['vectors'].astype(_np.float32)
        return idx

    faiss_mod.IndexFlatIP = IndexFlatIP
    faiss_mod.write_index = write_index
    faiss_mod.read_index = read_index
    sys.modules['faiss'] = faiss_mod


# fake openai if missing
if "openai" not in sys.modules:
    mod = types.ModuleType("openai")

    class FakeOpenAI:
        class embeddings:
            @staticmethod
            def create(model, input):
                class Item:
                    def __init__(self, embedding):
                        self.embedding = embedding

                # simple deterministic small vector
                return types.SimpleNamespace(data=[Item([0.01] * 8)])

        class chat:
            @staticmethod
            def completions():
                return None

    mod.OpenAI = FakeOpenAI
    sys.modules['openai'] = mod
