"""Vendor the embedding weights into models/.

    python -m scripts.vendor_model
    python -m scripts.vendor_model --verify

Serving a question needs the embedding model in the process (app/retrieve calls
embed_query on every question), so a deployment whose filesystem starts empty has
to fetch it. On a Render free instance that fetch is an ~87 MB download from
huggingface.co, which is Cloudflare-fronted and answers past its quota with HTTP
429 and a "Just a moment..." HTML page. Every cold start and every spin-down races
the same limit from a shared free-tier IP, so the service stays LIVE while its
chat box reports a connection error.

Committing the weights removes the download instead of retrying it. The trade is
87 MB in the repository, and the obligation to refresh them deliberately when
EMBED_MODEL changes - which this script, and --verify, are how you do that.

Deliberately copies file *contents*. The Hugging Face cache stores each snapshot
file as a symlink into ../blobs, and committing symlinks produces a checkout whose
model files are dangling on any other platform.

Run with --verify to prove the vendored weights are the ones the committed index
was built with. That is the check that makes changing the manifest label safe:
it re-embeds a spread of chunks and compares them to vectors.npy.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from app import config

#: Chunks to re-embed under --verify. A spread across the corpus rather than the
#: head, so a partial or mis-merged copy cannot pass on the first few rows.
VERIFY_IDX = (0, 1, 7, 42, 100, 250, 400, 719)


def dest_dir() -> Path:
    return REPO / config.DEFAULT_EMBED_MODEL


def vendor() -> int:
    """Copy the hub snapshot into models/ and report what landed.

    `snapshot_download` is used rather than loading a SentenceTransformer: it
    resolves and fetches the snapshot, and returns the directory, which is the
    only thing needed here. A model's own attributes cannot be used to find it -
    `auto_model.name_or_path` is the repo id ("sentence-transformers/..."), not a
    filesystem path, so globbing relative to it silently finds nothing.
    """
    from huggingface_hub import snapshot_download

    target = dest_dir()
    print(f"fetching {config.HUB_EMBED_MODEL} into the local cache ...")
    snapshot = Path(snapshot_download(config.HUB_EMBED_MODEL))
    print(f"  snapshot: {snapshot}")

    # The whole tree, minus the cache's bookkeeping. 1_Pooling/config.json matters:
    # it is what makes SentenceTransformer build mean pooling, the pooling the
    # vectors were built with. Leave it out and every similarity silently changes.
    copied = 0
    for source in sorted(snapshot.rglob("*")):
        if not source.is_file():
            continue
        relative = source.relative_to(snapshot)
        if relative.parts[0].startswith("."):
            continue  # .no_exist, .gitattributes
        dest = target / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        # copyfile, not copy2 or copytree: the cache stores each file as a symlink
        # into ../blobs, and committing symlinks yields a checkout whose model
        # files dangle on every platform but the one that made it.
        shutil.copyfile(source, dest)
        copied += 1

    total = sum(f.stat().st_size for f in target.rglob("*") if f.is_file())
    print(f"\nwrote {copied} files to {target}  ({total / 1024 / 1024:.1f} MB)")
    print(f"EMBED_MODEL default: {config.DEFAULT_EMBED_MODEL!r}")
    print(
        "\nNext: record the path in the index manifest, or the committed index is "
        "rejected as a model mismatch and rebuilt:\n"
        f'  set "embed_model" to "{config.DEFAULT_EMBED_MODEL}" in '
        f"{config.INDEX_DIR / 'manifest.json'}\n"
        "Then prove the weights match the stored vectors:\n"
        "  python -m scripts.vendor_model --verify"
    )
    return 0


def verify() -> int:
    """Prove the vendored weights reproduce the committed vectors."""
    import numpy as np
    from sentence_transformers import SentenceTransformer

    target = dest_dir()
    if not target.is_dir():
        print(f"ERROR: nothing vendored at {target}", file=sys.stderr)
        return 2

    index_dir = config.INDEX_DIR
    vectors_path = index_dir / "vectors.npy"
    if not vectors_path.exists():
        print(
            f"ERROR: no committed index at {index_dir}, so there is nothing to "
            "verify against. Build it with: python -m scripts.build_index",
            file=sys.stderr,
        )
        return 2

    vectors = np.load(vectors_path, allow_pickle=False)
    records = [
        json.loads(line)
        for line in (index_dir / "chunks.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    idxs = [i for i in VERIFY_IDX if i < len(records)]

    print(f"verifying {target} against {vectors_path}")
    model = SentenceTransformer(str(target))
    got = np.asarray(
        model.encode(
            [records[i]["embed_text"] for i in idxs],
            batch_size=config.EMBED_BATCH,
            normalize_embeddings=True,
            convert_to_numpy=True,
        ),
        dtype="float32",
    )
    got = got / np.linalg.norm(got, axis=1, keepdims=True)

    want = vectors[idxs]
    want = want / np.linalg.norm(want, axis=1, keepdims=True)
    sims = (got * want).sum(axis=1)

    print("\n  cosine     chunk_id")
    for i, s in zip(idxs, sims):
        print(f"  {s:.8f}   {records[i]['chunk_id']}")

    worst = float(sims.min())
    print(f"\nmin cosine = {worst:.8f}")
    if worst <= 0.9999:
        print(
            "VERDICT: MISMATCH - these weights did not build that index. Do not "
            "relabel the manifest; rebuild it with: python -m scripts.build_index",
            file=sys.stderr,
        )
        return 1
    print("VERDICT: IDENTICAL - safe to record this path in the manifest.")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--verify",
        action="store_true",
        help="check the vendored weights reproduce the committed vectors",
    )
    args = ap.parse_args(argv)
    return verify() if args.verify else vendor()


if __name__ == "__main__":
    raise SystemExit(main())