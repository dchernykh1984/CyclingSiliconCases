"""Приёмка исходных моделей.

Модели скачаны готовыми, и всё, что мы дальше делаем, опирается на их
размеры. Если файл подменят или перекачают исправленным, тест скажет об этом
прямо — с цифрами, а не «что-то поехало».
"""

from __future__ import annotations

import pytest

from cycling_cases.cases import CASES, COMPANIONS, sources_dir
from cycling_cases.devices import DEVICES
from cycling_cases.mesh import edge_counts, is_watertight, parts, read_stl
from cycling_cases.recipes import E840_BUTTON_POCKETS, E840_PART, pick_case
from walls import thin_wall_runs, wall_thickness

TOLERANCE = 0.05
"""Допуск на габариты, мм. Сетка лежит во float32, отсюда шум в сотых."""


def test_every_case_has_its_source() -> None:
    for item in CASES:
        assert item.source.is_file(), f"нет исходной модели {item.source_name}"


def test_sources_are_binary_stl() -> None:
    # Текстовый STL весит вдесятеро больше и читается другим кодом:
    # проверяем формат явно, а не надеемся на расширение файла.
    for item in CASES:
        assert len(read_stl(item.source)) > 0


RAW = "mesh.stl"
"""Как приезжают сырые сканы.

Скан прибора весит шестнадцать мегабайт — в git такому файлу не место.
Его кладут в `input_data` под именем, кончающимся на `Mesh.stl`, git
его игнорирует, а в репозиторий уходит прореженная копия.
"""


def sources_on_disk() -> set[str]:
    """Имена всех моделей в `input_data`, кроме сырых сканов.

    Расширение сверяем без учёта регистра: у части чужих файлов оно
    записано прописными, и на Linux обычный шаблон `*.stl` их не видит.
    """
    return {
        path.name
        for path in sources_dir().iterdir()
        if path.suffix.lower() == ".stl" and not path.name.lower().endswith(RAW)
    }


def test_every_source_is_registered() -> None:
    # Новый чехол начинается с файла в `input_data` и записи в `CASES`.
    # Если файл положили, а запись забыли, он молча не попадёт ни в сборку,
    # ни в релиз — поэтому тест ловит это сразу.
    on_disk = sources_on_disk()
    registered = (
        {item.source_name for item in CASES}
        | {item.source_name for item in DEVICES}
        | {item.source_name for item in COMPANIONS}
    )
    forgotten = on_disk - registered
    assert not forgotten, f"исходники есть, а записи в CASES нет: {sorted(forgotten)}"
    missing = registered - on_disk
    assert not missing, f"в CASES есть чехлы без исходника: {sorted(missing)}"


def test_case_of_the_830_is_one_closed_body() -> None:
    # Чехол 830 — готовая деталь одним телом: его можно резать булевыми
    # операциями без предварительной починки.
    triangles = read_stl(sources_dir() / "Garmin830.stl")
    assert len(triangles) == 3938
    found = parts(triangles)
    assert len(found) == 1, "в файле больше одного тела — раскладка изменилась"
    width, depth, height = found[0].size
    assert width == pytest.approx(53.61, abs=TOLERANCE)
    assert depth == pytest.approx(86.61, abs=TOLERANCE)
    assert height == pytest.approx(20.06, abs=TOLERANCE)
    assert is_watertight(triangles), "сетка чехла 830 перестала быть замкнутой"


def test_kit_of_the_840_is_a_whole_plate() -> None:
    # Файл 840 — это не чехол, а целая раскладка на стол: вынос, чехол,
    # два хомута и две проставки. Правки начинаются с выбора тела.
    triangles = read_stl(sources_dir() / "Garmin840.stl")
    assert len(triangles) == 25248
    found = parts(triangles)
    assert len(found) == 6, "число тел в раскладке 840 изменилось"

    sizes = [tuple(round(float(value), 2) for value in part.size) for part in found]
    assert sizes[0] == (60.0, 168.85, 35.6), "самое крупное тело — вынос"
    assert sizes[1] == (60.0, 103.59, 17.0), "второе тело — сам чехол"
    assert sizes[2] == sizes[3] == (15.39, 21.17, 29.38), "хомуты идут парой"
    assert sizes[4][0] == sizes[5][0] == 25.49, "проставки идут парой"


def test_the_840_case_body_touches_itself() -> None:
    # У тела чехла 840 четыре ребра принадлежат четырём граням — деталь
    # сама себя касается. Дырок нет, manifold это переваривает, но правки
    # начинать стоит с проверки: если появятся рёбра без пары, сетку надо
    # чинить, иначе булевы операции дадут мусор.
    triangles = read_stl(sources_dir() / "Garmin840.stl")
    counts = edge_counts(triangles)
    assert 1 not in counts, "в раскладке 840 появились дырки"
    assert counts.get(4) == 4, "самокасания в чехле 840 изменились"


def test_the_840_case_has_its_buttons_marked_from_the_inside() -> None:
    """Автор исходника разметил кнопки карманами в полости.

    Точных координат кнопок Edge 840 нет ни в мануале, ни в обзорах, и
    окна мы режем ровно по этим карманам. Если исходник заменят, а
    карманы окажутся в других местах, окна уедут мимо кнопок — поэтому
    разметка проверяется как приёмка, с цифрами.
    """
    case = pick_case(read_stl(sources_dir() / "Garmin840.stl"), E840_PART)
    for sign, expected in E840_BUTTON_POCKETS.items():
        found = tuple(pocket for pocket in thin_wall_runs(case, sign) if pocket[1] - pocket[0] > 3)
        assert len(found) == len(expected), f"бок {sign:+d}: число карманов изменилось"
        for (start, finish), (want_start, want_finish) in zip(found, expected, strict=True):
            assert start == pytest.approx(want_start, abs=TOLERANCE)
            assert finish == pytest.approx(want_finish, abs=TOLERANCE)


def test_the_840_pockets_are_thinner_than_the_ribs() -> None:
    """Карман — это стенка 0.75 мм, ребро между карманами — 1.75 мм."""
    case = pick_case(read_stl(sources_dir() / "Garmin840.stl"), E840_PART)
    assert wall_thickness(case, -1, 60.0) == pytest.approx(0.75, abs=0.1), "середина кармана"
    assert wall_thickness(case, -1, 47.0) == pytest.approx(1.75, abs=0.1), "ребро перед карманом"
