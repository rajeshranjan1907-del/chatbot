"""Build the local vector index from data/raw/.

    python scripts/build_index.py

ChromaDB's write path segfaults on this machine (see app/localindex.py), so the
searchable index is a numpy matrix on disk. Everything upstream is unchanged:
chunks come from app.chunking, vectors from app.embedder, ids are chunk_ids, and
the space is cosine over L2-normalised vectors - the same contract Chroma gives.

The work itself lives in `app.localindex.build_from_raw` so this command and the
app's on-demand fallback produce the same index. This file is the reporting
wrapper: it is what a Render build command or pre-deploy step runs, and what a
developer runs locally.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from app import config
from app.localindex import IndexError_, build_from_raw


def main() -> int:
    if not config.RAW_DIR.exists():
        print(f"ERROR: {config.RAW_DIR} does not exist. Run Phase 1 first.", file=sys.stderr)
        return 2

    try:
        index = build_from_raw()
    except IndexError_ as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(f"\nwrote index to {index.root}")
    print(f"  count       : {index.manifest['count']}")
    print(f"  dim         : {index.manifest['dim']}")
    print(f"  space       : {index.manifest['space']}")
    print(f"  corpus      : {index.manifest['corpus_version']}")
    # build_from_raw already refused to return an index that cannot answer its
    # own query, and logged the self-query result above.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
