#!/usr/bin/env python3
"""Не пропускать коммит со следами соавторства.

Правило репозитория: сообщение коммита — одна строка, без `Co-Authored-By`
и без упоминания Claude. Хук ловит это до того, как коммит уйдёт в историю,
где вычищать его пришлось бы перезаписью веток.
"""

from __future__ import annotations

import json
import re
import sys

FORBIDDEN = (
    re.compile(r"co-authored-by", re.IGNORECASE),
    re.compile(r"generated with .*claude", re.IGNORECASE),
    re.compile(r"🤖", re.IGNORECASE),
)


def main() -> int:
    try:
        payload = json.load(sys.stdin)
    except (json.JSONDecodeError, ValueError):
        return 0

    command = payload.get("tool_input", {}).get("command", "")
    if "git commit" not in command:
        return 0

    for pattern in FORBIDDEN:
        if pattern.search(command):
            print(
                "Коммит отклонён: сообщение должно быть одной строкой без "
                "соавторства и упоминания Claude — см. CLAUDE.md.",
                file=sys.stderr,
            )
            return 2

    return 0


if __name__ == "__main__":
    sys.exit(main())
