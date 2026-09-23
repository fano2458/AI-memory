import hashlib
import json
import time
from pathlib import Path

import numpy as np

CACHE = Path("data/cache")


def key(text):
    return hashlib.md5(text.encode()).hexdigest()


class Embedder:
    def __init__(self, model="text-embedding-3-small", batch=512):
        self.model = model
        self.batch = batch
        self.vec_path = CACHE / f"{model}.npy"
        self.idx_path = CACHE / f"{model}.json"
        self.index = json.loads(self.idx_path.read_text()) if self.idx_path.exists() else {}
        self.vectors = np.load(self.vec_path) if self.vec_path.exists() else None
        if self.vectors is not None and self.vectors.dtype == object:
            self.vectors = None
            self.index = {}
        self.pending = []
        self._client = None

    @property
    def client(self):
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI()
        return self._client

    @property
    def count(self):
        n = 0 if self.vectors is None else len(self.vectors)
        return n + sum(len(a) for a in self.pending)

    def missing(self, texts):
        out, seen = [], set()
        for t in texts:
            h = key(t)
            if h not in self.index and h not in seen:
                seen.add(h)
                out.append(t)
        return out

    def _call(self, chunk, tries=5):
        for attempt in range(tries):
            try:
                resp = self.client.embeddings.create(model=self.model, input=chunk)
                return [d.embedding for d in resp.data]
            except Exception:
                if attempt == tries - 1:
                    raise
                time.sleep(2**attempt)

    def add(self, texts, progress=None, checkpoint=20480):
        todo = self.missing(texts)
        if not todo:
            return 0
        since = 0
        for i in range(0, len(todo), self.batch):
            chunk = todo[i : i + self.batch]
            arr = np.asarray(self._call(chunk), dtype=np.float32)
            arr /= np.linalg.norm(arr, axis=1, keepdims=True)
            base = self.count
            self.pending.append(arr.astype(np.float16))
            for j, t in enumerate(chunk):
                self.index[key(t)] = base + j
            since += len(chunk)
            if progress:
                progress(min(i + self.batch, len(todo)), len(todo))
            if since >= checkpoint:
                self.save()
                since = 0
        self.save()
        return len(todo)

    def save(self):
        if self.pending:
            new = np.vstack(self.pending)
            self.vectors = new if self.vectors is None else np.vstack([self.vectors, new])
            self.pending = []
        if self.vectors is None:
            return
        CACHE.mkdir(parents=True, exist_ok=True)
        np.save(self.vec_path, self.vectors)
        self.idx_path.write_text(json.dumps(self.index))

    def get(self, texts):
        rows = [self.index[key(t)] for t in texts]
        return self.vectors[rows].astype(np.float32)
