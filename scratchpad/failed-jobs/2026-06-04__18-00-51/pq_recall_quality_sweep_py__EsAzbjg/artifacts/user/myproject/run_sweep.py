"""
CLI entrypoint for the IVF_PQ recall sweep.

Usage:
    python3 /home/user/myproject/run_sweep.py

Writes /home/user/myproject/result.json with string-keyed recall values, e.g.:
    {"4": 0.78, "8": 0.91, "16": 0.97}
"""

from __future__ import annotations

import json
import pathlib

from solution import sweep

_OUTPUT_PATH = pathlib.Path("/home/user/myproject/result.json")


def main() -> None:
    recall_by_m = sweep()

    # Serialise with string keys as required.
    output = {str(m): v for m, v in recall_by_m.items()}

    _OUTPUT_PATH.write_text(json.dumps(output, indent=2))
    print(f"Wrote results to {_OUTPUT_PATH}")
    for m, recall in sorted(recall_by_m.items()):
        print(f"  m={m:>2d}  recall@10 = {recall:.4f}")


if __name__ == "__main__":
    main()
