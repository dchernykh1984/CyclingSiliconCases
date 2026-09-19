#!/usr/bin/env python3
"""Отформатировать только что записанный python-файл.

pre-commit сделает то же самое на коммите, но тогда он валит первую попытку
и её приходится повторять. Отформатировать сразу после записи дешевле.
"""

from __future__ import annotations

import json
import pathlib
import subprocess
import sys


def main() -> int:
    try:
        payload = json.load(sys.stdin)
    except (json.JSONDecodeError, ValueError):
        return 0

    raw_path = payload.get("tool_input", {}).get("file_path", "")
    if not raw_path:
        return 0

    path = pathlib.Path(raw_path).resolve()
    root = pathlib.Path.cwd().resolve()
    if path.suffix != ".py" or root not in path.parents:
        return 0

    ruff = root / ".venv" / "bin" / "ruff"
    subprocess.run(
        [str(ruff) if ruff.exists() else "ruff", "format", "--quiet", str(path)],
        check=False,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
