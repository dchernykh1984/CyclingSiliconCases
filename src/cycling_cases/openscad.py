"""Запуск OpenSCAD: сборка командной строки и нарезка STL.

Правки к чехлам живут в `.scad`-моделях, которые импортируют исходную сетку
и вычитают из неё надписи или добавляют к ней тело. Здесь только запуск:
геометрия — в `assets/models`.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

OPENSCAD = "openscad"
"""Имя бинаря по умолчанию; путь можно задать переменной OPENSCAD."""

Definition = str | float | int | bool


@dataclass(frozen=True, slots=True)
class RenderTask:
    """Одна нарезка модели в STL."""

    filename: str
    model: Path
    definitions: dict[str, Definition] = field(default_factory=dict)
    comment: str = ""


def scad_literal(value: Definition) -> str:
    """Литерал OpenSCAD для значения, уходящего в `-D`.

    Строки задаются людьми, и кавычка или обратный слеш в надписи превратили
    бы `-D` в синтаксически битый кусок модели.
    """
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int | float):
        return repr(value)
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


@lru_cache(maxsize=8)
def features(executable: str) -> frozenset[str]:
    """Что умеет установленный OpenSCAD.

    Булевы операции с импортированной сеткой на старом движке CGAL считаются
    минутами, а на manifold (сборки 2023 года и новее) — за секунды. Заодно
    бинарный STL ужимает файл в несколько раз против текстового.
    """
    try:
        answer = subprocess.run(
            [executable, "--help"], capture_output=True, text=True, timeout=60, check=False
        )
    except (OSError, subprocess.SubprocessError):  # pragma: no cover - совсем битый бинарь
        return frozenset()

    help_text = (answer.stdout or "") + (answer.stderr or "")
    found = set()
    if "--backend" in help_text:
        found.add("manifold")
    if "--export-format" in help_text:
        found.add("binstl")
    return frozenset(found)


def command(
    output: Path,
    task: RenderTask,
    executable: str = OPENSCAD,
    known: frozenset[str] = frozenset(),
    render: bool = False,
) -> list[str]:
    """Командная строка OpenSCAD для одной нарезки."""
    line = [executable]
    if "manifold" in known:
        line.append("--backend=manifold")
    if "binstl" in known and output.suffix.lower() == ".stl":
        line.append("--export-format=binstl")
    if render:
        # Превью без --render рисует предварительный просмотр OpenCSG:
        # на нём соседние грани мерцают, и гравировка выглядит рваной,
        # хотя в самой модели всё цело.
        line.append("--render")
    line += ["-o", str(output)]
    for name, value in task.definitions.items():
        line += ["-D", f"{name}={scad_literal(value)}"]
    line.append(str(task.model))
    return line


def executable() -> str | None:
    """Путь к OpenSCAD или None, если его нет в системе."""
    return shutil.which(os.environ.get("OPENSCAD", OPENSCAD))


def run(
    output: Path,
    task: RenderTask,
    binary: str | None = None,
    timeout: float = 900,
    render: bool = False,
    extra: list[str] | None = None,
) -> Path:
    """Нарезать STL или отрендерить превью. Требует установленного OpenSCAD."""
    tool = binary or executable()
    if tool is None:
        raise RuntimeError(
            "не найден openscad — поставьте его (https://openscad.org/) "
            "или укажите путь в переменной окружения OPENSCAD"
        )

    known = features(tool)
    if "manifold" not in known:
        print(
            f"openscad {tool} без движка manifold: булевы операции с импортированной "
            "сеткой займут минуты вместо секунд, поставьте сборку 2023 года или новее"
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    line = command(output, task, tool, known, render=render)
    if extra:
        line[1:1] = extra
    subprocess.run(line, check=True, capture_output=True, timeout=timeout)
    return output
