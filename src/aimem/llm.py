import hashlib
import json
import os
import threading
import time
from pathlib import Path

CACHE = Path("data/cache")
LOCK = threading.Lock()


def load_env(path=".env"):
    p = Path(path)
    if not p.exists():
        return
    for line in p.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip("\"'"))


def fatal(exc):
    if getattr(exc, "status_code", None) == 400:
        return True
    return "insufficient_quota" in str(exc) or "credit_balance" in str(exc)


class LLM:
    def __init__(self, model="gpt-5.4-mini", max_tokens=300):
        self.model = model
        self.max_tokens = max_tokens
        self.path = CACHE / f"reader_{model}.jsonl"
        self.cache = {}
        self.client = None
        if self.path.exists():
            for line in self.path.read_text().splitlines():
                if line.strip():
                    r = json.loads(line)
                    self.cache[r["key"]] = r

    def ask(self, system, user):
        k = hashlib.md5(f"{self.model}\x00{system}\x00{user}".encode()).hexdigest()
        if k in self.cache:
            return self.cache[k]
        rec = self.call(system, user)
        rec["key"] = k
        with LOCK:
            self.cache[k] = rec
            CACHE.mkdir(parents=True, exist_ok=True)
            with self.path.open("a") as f:
                f.write(json.dumps(rec) + "\n")
        return rec

    def call(self, system, user, tries=5):
        if self.client is None:
            from openai import OpenAI

            load_env()
            self.client = OpenAI()
        for attempt in range(tries):
            try:
                t0 = time.time()
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": user}],
                    max_completion_tokens=self.max_tokens,
                )
                return {
                    "text": resp.choices[0].message.content or "",
                    "in_tokens": resp.usage.prompt_tokens,
                    "out_tokens": resp.usage.completion_tokens,
                    "ms": round((time.time() - t0) * 1000),
                }
            except Exception as exc:
                if attempt == tries - 1 or fatal(exc):
                    raise
                time.sleep(2**attempt)
