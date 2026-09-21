"""Каталог чехлов: что за исходник, что мы с ним делаем и что уходит в релиз.

Исходные модели скачаны готовыми и лежат в `input_data/` — пересобрать их
не из чего, поэтому они и есть исходный код. Наши правки живут рядом, в
`recipes.py`: на каждый чехол одна функция, которая берёт исходную сетку
и возвращает готовое тело.
"""

from __future__ import annotations

import shutil
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from manifold3d import Manifold

from . import recipes, solid
from .mesh import Triangles, read_stl, write_stl

PACKAGE_ROOT = Path(__file__).parent

Recipe = Callable[[Triangles], Manifold]


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
    """Одна деталь: исходник, правка поверх него и имя в релизе."""

    slug: str
    title: str
    fits: str
    source_name: str
    recipe: Recipe
    comment: str = ""

    @property
    def source(self) -> Path:
        return sources_dir() / self.source_name

    @property
    def filename(self) -> str:
        return f"{self.slug}.stl"

    def triangles(self) -> Triangles:
        return read_stl(self.source)

    def build(self) -> Manifold:
        return self.recipe(self.triangles())


CASES: tuple[Case, ...] = (
    Case(
        slug="garmin-830",
        title="Чехол Garmin Edge 830",
        fits="Garmin Edge 830",
        source_name="Garmin830.stl",
        recipe=recipes.garmin_830,
        comment="Надпись UBT 8 YEARS спереди, губы удержания по бокам и спереди",
    ),
    Case(
        slug="garmin-840",
        title="Чехол Garmin Edge 840",
        fits="Garmin Edge 840",
        source_name="Garmin840.stl",
        recipe=recipes.garmin_840,
        comment="Только чехол из раскладки: окна под все кнопки и надпись на носу",
    ),
    Case(
        slug="can-lid",
        title="Крышка банки для инструмента",
        fits="банка obj_2_Can 170mm",
        source_name="obj_1_Lid.stl",
        recipe=recipes.can_lid,
        comment="Надпись UBT 8 YEARS на наружном торце; деталь стоит надписью вверх",
    ),
    Case(
        slug="bottle-cap",
        title="Крышка фляжки для инструмента",
        fits="фляжка 水壶3.3",
        source_name="水壶盖3.1无孔.STL",
        recipe=recipes.bottle_cap,
        comment="Надпись UBT 8 YEARS на наружном торце; деталь стоит надписью вверх",
    ),
)


@dataclass(frozen=True, slots=True)
class Companion:
    """Вторая половина пары: сосуд к своей крышке.

    Банку и фляжку мы не правим, но в релиз кладём — **копией байт в
    байт**. Смысл релиза в том, чтобы скачать его целиком и пойти
    печатать; отправлять человека за второй половиной набора в
    `input_data` — значит экономить мегабайт ценой его времени.

    Копия именно побайтовая, а не пересохранённая: прогон чужой сетки
    через наш экспорт переварил бы её (сшивание вершин, float32) и
    выдал бы за то же самое немного другой файл.

    Исключение — сосуд, который лежит в исходнике на боку. Такой мы
    ставим на дно (`stand`): это поворот, а не правка формы, и
    треугольники при этом остаются теми же, у них только переезжают
    координаты. Форму по-прежнему не трогаем — объём до и после
    совпадает, и это проверяется тестом.
    """

    slug: str
    source_name: str
    title: str
    lid_slug: str
    axis: int
    stand: tuple[tuple[float, ...], ...] | None = None
    comment: str = ""

    @property
    def source(self) -> Path:
        return sources_dir() / self.source_name

    @property
    def filename(self) -> str:
        return f"{self.slug}.stl"

    @property
    def copied(self) -> bool:
        """Уходит ли сосуд в релиз побайтовой копией.

        Копией уходит тот, кто в исходнике уже стоит на дне. Лежащий
        на боку мы поворачиваем — иначе слайсер получит стосемидесяти-
        миллиметровый цилиндр, лежащий плашмя, и обложит его
        поддержками по всей длине.
        """
        return self.stand is None


BOTTLE_STAND = ((1.0, 0.0, 0.0, 0.0), (0.0, 0.0, -1.0, 0.0), (0.0, 1.0, 0.0, 0.0))
"""Поворот фляжки на дно: ось Y исходника становится Z.

Поворот на +90° вокруг X, а не на −90°: у фляжки закрытое дно на y=0,
а резьбовое горлышко наверху, и обратный поворот поставил бы её
горлышком в стол. Определитель матрицы +1 — это поворот, не зеркало.
"""


COMPANIONS: tuple[Companion, ...] = (
    Companion(
        slug="can",
        source_name="obj_2_Can 170mm.stl",
        title="Банка для инструмента, 170 мм",
        lid_slug="can-lid",
        axis=2,
        comment="Без правок, копия исходника — под крышку can-lid; стоит вертикально",
    ),
    Companion(
        slug="bottle",
        source_name="水壶3.3.STL",
        title="Фляжка для инструмента, 166 мм",
        lid_slug="bottle-cap",
        axis=1,
        stand=BOTTLE_STAND,
        comment=(
            "Форма без правок — под крышку bottle-cap; "
            "в исходнике лежит на боку, в релизе поставлена на дно"
        ),
    ),
)


def companion(slug: str) -> Companion:
    """Сосуд по короткому имени."""
    for item in COMPANIONS:
        if item.slug == slug:
            return item
    known = ", ".join(item.slug for item in COMPANIONS)
    raise KeyError(f"нет сосуда {slug!r}; есть {known}")


def part(slug: str) -> Case | Companion:
    """Что угодно из релиза по короткому имени — деталь или сосуд.

    В ошибке перечисляем всё сразу: человек, опечатавшийся в имени
    чехла, не должен читать, что «нет такого сосуда» и видеть список
    из одних сосудов.
    """
    for item in CASES:
        if item.slug == slug:
            return item
    for vessel in COMPANIONS:
        if vessel.slug == slug:
            return vessel
    known = ", ".join([item.slug for item in CASES] + [vessel.slug for vessel in COMPANIONS])
    raise KeyError(f"нет детали {slug!r}; есть {known}")


def case(slug: str) -> Case:
    """Чехол по короткому имени."""
    for item in CASES:
        if item.slug == slug:
            return item
    known = ", ".join(item.slug for item in CASES)
    raise KeyError(f"нет чехла {slug!r}; есть {known}")


def build(directory: Path, only: str | None = None) -> list[Path]:
    """Собрать в каталог всё, что уходит в релиз.

    Детали со своим рецептом режутся, сосуды копируются байт в байт.
    Опечатка в имени — не пустая сборка, а ошибка: молча собрать ноль
    файлов и выйти с нулём хуже, чем сказать, что такого имени нет.
    """
    if only is not None:
        part(only)
    directory.mkdir(parents=True, exist_ok=True)
    built = []
    for item in CASES:
        if only and item.slug != only:
            continue
        built.append(solid.save(item.build(), directory / item.filename))
    for vessel in COMPANIONS:
        if only and vessel.slug != only:
            continue
        built.append(stand_up(vessel, directory / vessel.filename))
    return built


def stand_up(vessel: Companion, target: Path) -> Path:
    """Положить сосуд в каталог сборки: копией или повёрнутым на дно.

    Поворот делаем прямо по треугольникам, не прогоняя сетку через
    булев движок: так у чужой формы не меняется ни один треугольник,
    у них только переезжают координаты.
    """
    if vessel.stand is None:
        shutil.copyfile(vessel.source, target)
        return target
    matrix = np.asarray(vessel.stand, dtype=np.float64)
    moved = read_stl(vessel.source) @ matrix[:, :3].T + matrix[:, 3]
    flat = moved.reshape(-1, 3)
    low, high = flat.min(axis=0), flat.max(axis=0)
    moved = moved - [(low[0] + high[0]) / 2, (low[1] + high[1]) / 2, low[2]]
    return write_stl(target, moved)
