import hashlib
import json
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
        self._client = None

    @property
    def client(self):
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI()
        return self._client

    def missing(self, texts):
        out, seen = [], set()
        for t in texts:
            h = key(t)
            if h not in self.index and h not in seen:
                seen.add(h)
                out.append(t)
        return out

    def add(self, texts, progress=None):
        todo = self.missing(texts)
        if not todo:
            return 0
        new = []
        for i in range(0, len(todo), self.batch):
            chunk = todo[i : i + self.batch]
            resp = self.client.embeddings.create(model=self.model, input=chunk)
            new.extend(d.embedding for d in resp.data)
            if progress:
                progress(min(i + self.batch, len(todo)), len(todo))
        arr = np.asarray(new, dtype=np.float32)
        arr /= np.linalg.norm(arr, axis=1, keepdims=True)
        arr = arr.astype(np.float16)
        base = 0 if self.vectors is None else len(self.vectors)
        self.vectors = arr if self.vectors is None else np.vstack([self.vectors, arr])
        for j, t in enumerate(todo):
            self.index[key(t)] = base + j
        return len(todo)

    def save(self):
        CACHE.mkdir(parents=True, exist_ok=True)
        np.save(self.vec_path, self.vectors)
        self.idx_path.write_text(json.dumps(self.index))

    def get(self, texts):
        rows = [self.index[key(t)] for t in texts]
        return self.vectors[rows].astype(np.float32)
