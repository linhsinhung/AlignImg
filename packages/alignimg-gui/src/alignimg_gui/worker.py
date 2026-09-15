"""Subprocess entry point used by the Qt application."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import signal
import sys
import traceback

from .runner import execute_run
from .spec import RunSpec


def _interrupted(_signum, _frame) -> None:
    raise InterruptedError("alignment was stopped by the user")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    args = parser.parse_args(argv)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, _interrupted)

    log_path = args.spec.parent / "run.log"

    def emit(value: dict) -> None:
        line = json.dumps(value, sort_keys=True)
        with log_path.open("a", encoding="utf-8") as stream:
            stream.write(line + "\n")
        print(line, flush=True)

    try:
        execute_run(RunSpec.read(args.spec), emit=emit)
    except Exception:
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
