import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aimem import stale
from aimem.embed import Embedder


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="text-embedding-3-small")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    xs = stale.load(limit=args.limit)
    texts = [t.content for x in xs for s in x.sessions for t in s.turns]

    emb = Embedder(args.model)
    todo = emb.missing(texts)
    print(f"{len(texts):,} turn slots, {len(todo):,} new to embed")
    if not todo:
        print("cache already complete")
        return

    def progress(done, total):
        print(f"\r  {done:,}/{total:,}", end="", flush=True)
        if done % 20480 == 0:
            emb.save()

    emb.add(texts, progress)
    emb.save()
    print(f"\nsaved {len(emb.index):,} vectors to {emb.vec_path}")


if __name__ == "__main__":
    main()
