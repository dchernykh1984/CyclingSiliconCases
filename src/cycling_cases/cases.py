"""Каталог чехлов: что за исходник, что мы с ним делаем и что уходит в релиз.

Исходные модели скачаны готовыми и лежат в `input_data/` — пересобрать их
не из чего, поэтому они и есть исходный код. Наши правки живут в моделях
OpenSCAD рядом, в `assets/models`.

Пока у чехла нет своей модели, он уходит в сборку **как есть**: копией файла,
байт в байт. Так первый релиз честно отдаёт то, что лежит в `input_data`,
а следующие — уже правленые чехлы, и разница между ними видна по файлу.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path

from .openscad import Definition, RenderTask, run

PACKAGE_ROOT = Path(__file__).parent
MODELS = PACKAGE_ROOT / "assets" / "models"


def repository_root() -> Path:
    """Корень репозитория — тот каталог, в котором лежит `input_data`.

    Ищем его снизу вверх, а не отсчитываем шаги от пакета: при установке
    в режиме разработки пакет лежит в `src/`, а при установке колесом —
    в `site-packages`, и фиксированное «два шага вверх» там промахивается
    мимо исходников.
    """
    for candidate in (PACKAGE_ROOT, *PACKAGE_ROOT.parents):
        if (candidate / "input_data").is_dir():
            return candidate
    here = Path.cwd()
    for candidate in (here, *here.parents):
        if (candidate / "input_data").is_dir():
            return candidate
    raise RuntimeError(
        "не найден каталог input_data — запускайте из репозитория "
        "или поставьте пакет в режиме разработки (uv sync)"
    )


def sources_dir() -> Path:
    return repository_root() / "input_data"


@dataclass(frozen=True, slots=True)
class Case:
    """Один чехол: исходник, наша модель поверх него и что печатать."""

    slug: str
    title: str
    computer: str
    source_name: str
    model_name: str | None = None
    definitions: dict[str, Definition] = field(default_factory=dict)
    comment: str = ""

    @property
    def source(self) -> Path:
        return sources_dir() / self.source_name

    @property
    def model(self) -> Path | None:
        return MODELS / self.model_name if self.model_name else None

    @property
    def filename(self) -> str:
        return f"{self.slug}.stl"

    @property
    def modified(self) -> bool:
        """Есть ли у чехла своя модель, или он уходит копией исходника."""
        return self.model_name is not None


CASES: tuple[Case, ...] = (
    Case(
        slug="garmin-830",
        title="Чехол Garmin Edge 830",
        computer="Garmin Edge 830",
        source_name="Garmin830.stl",
        comment="Готовый чехол одной деталью; правок пока нет — уходит как есть",
    ),
    Case(
        slug="garmin-840",
        title="Комплект Garmin Edge 840",
        computer="Garmin Edge 840",
        source_name="Garmin840.stl",
        comment="Целая раскладка на стол: чехол, вынос и хомуты; правок пока нет",
    ),
)


def case(slug: str) -> Case:
    """Чехол по короткому имени."""
    for item in CASES:
        if item.slug == slug:
            return item
    known = ", ".join(item.slug for item in CASES)
    raise KeyError(f"нет чехла {slug!r}; есть {known}")


def render_plan() -> tuple[RenderTask, ...]:
    """Что резать в OpenSCAD — только чехлы со своей моделью."""
    tasks = []
    for item in CASES:
        model = item.model
        if model is None:
            continue
        tasks.append(
            RenderTask(
                filename=item.filename,
                model=model,
                definitions=dict(item.definitions),
                comment=item.comment,
            )
        )
    return tuple(tasks)


def build(directory: Path, binary: str | None = None) -> list[Path]:
    """Собрать все чехлы в каталог: копией исходника или нарезкой модели."""
    directory.mkdir(parents=True, exist_ok=True)
    built = []
    for item in CASES:
        target = directory / item.filename
        model = item.model
        if model is None:
            shutil.copyfile(item.source, target)
        else:
            run(
                target,
                RenderTask(
                    filename=item.filename,
                    model=model,
                    definitions=dict(item.definitions),
                    comment=item.comment,
                ),
                binary=binary,
            )
        built.append(target)
    return built
