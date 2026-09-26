import hashlib
import json
import time
from pathlib import Path

from .embed import CACHE, load_env


def key(model, system, user):
    return hashlib.md5(f"{model}\x00{system}\x00{user}".encode()).hexdigest()


class Reader:
    def __init__(self, model="gpt-5.4-mini", max_tokens=300):
        self.model = model
        self.max_tokens = max_tokens
        self.path = CACHE / f"reader_{model}.jsonl"
        self.cache = {}
        self._client = None
        if self.path.exists():
            for line in self.path.read_text().splitlines():
                if line.strip():
                    r = json.loads(line)
                    self.cache[r["key"]] = r

    @property
    def client(self):
        if self._client is None:
            from openai import OpenAI

            load_env()
            self._client = OpenAI()
        return self._client

    def ask(self, system, user):
        k = key(self.model, system, user)
        if k in self.cache:
            return self.cache[k]
        rec = self._call(system, user)
        rec["key"] = k
        self.cache[k] = rec
        CACHE.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as f:
            f.write(json.dumps(rec) + "\n")
        return rec

    def _call(self, system, user, tries=5):
        for attempt in range(tries):
            try:
                t0 = time.time()
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    max_completion_tokens=self.max_tokens,
                )
                return {
                    "text": resp.choices[0].message.content or "",
                    "in_tokens": resp.usage.prompt_tokens,
                    "out_tokens": resp.usage.completion_tokens,
                    "ms": round((time.time() - t0) * 1000),
                }
            except Exception:
                if attempt == tries - 1:
                    raise
                time.sleep(2**attempt)
