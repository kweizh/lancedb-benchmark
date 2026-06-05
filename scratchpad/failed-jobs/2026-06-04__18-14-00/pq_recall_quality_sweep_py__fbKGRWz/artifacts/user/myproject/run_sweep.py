"""
CLI entrypoint: runs the IVF_PQ recall sweep and writes result.json.

Usage:
    python3 /home/user/myproject/run_sweep.py
"""

import json
import os
import sys

# Allow running from any working directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from solution import sweep

_OUTPUT_PATH = "/home/user/myproject/result.json"


def main() -> None:
    print("Starting IVF_PQ recall sweep …")
    recall_map = sweep()

    # Serialise with string keys as required
    output = {str(m): v for m, v in recall_map.items()}

    with open(_OUTPUT_PATH, "w") as fh:
        json.dump(output, fh, indent=2)

    print(f"\nResults written to {_OUTPUT_PATH}")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
