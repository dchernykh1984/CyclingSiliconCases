"""Приёмка крышек для банки и фляжки.

Крышка — плоский диск, и проверять тут надо другое, чем на чехлах: что
надпись выступает ровно на заявленную высоту, что она укладывается в
плоскую площадку и не лезет на скругление кромки, и что деталь стоит
надписью вверх — иначе выпуклые буквы не напечатать.
"""

from __future__ import annotations

import numpy as np
import pytest

from cycling_cases import cases, lettering, panel, recipes
from cycling_cases.cases import COMPANIONS
from cycling_cases.mesh import Triangles, distance_to_outline, edge_counts, is_inside, parts

LIDS = ("can-lid", "bottle-cap")

FLAT_RADIUS = {"can-lid": 38.1, "bottle-cap": 35.3}
"""Радиус плоской площадки наружного торца, мм — дальше идёт скругление.

Снято лучами по исходникам: луч идёт вдоль оси крышки и ищет, с какого
радиуса торец перестаёт быть плоским.
"""

AXIS = {"can-lid": 2, "bottle-cap": 1}
"""Вдоль какой оси лежит ось крышки в исходнике: у фляжки она на боку."""


@pytest.fixture(scope="session")
def lids(built: dict[str, Triangles]) -> dict[str, Triangles]:
    """Крышки из общей сборки: собирать их второй раз незачем."""
    return {slug: built[slug] for slug in LIDS}


@pytest.mark.parametrize("slug", LIDS)
def test_lid_is_one_closed_body(lids: dict[str, Triangles], slug: str) -> None:
    triangles = lids[slug]
    assert len(parts(triangles)) == 1
    assert edge_counts(triangles) == {2: len(triangles) * 3 // 2}, "сетка перестала быть замкнутой"


@pytest.mark.parametrize("slug", LIDS)
def test_lid_stands_with_the_lettering_up(lids: dict[str, Triangles], slug: str) -> None:
    """Деталь лежит в файле надписью вверх и стоит на плоскости z=0.

    Выпуклые буквы нельзя напечатать на грани, которая лежит на столе:
    их пришлось бы печатать в воздухе. Значит, крышка печатается
    открытой стороной вниз, и в файле она уже так и стоит.
    """
    flat = lids[slug].reshape(-1, 3)
    low, high = flat.min(axis=0), flat.max(axis=0)
    assert low[2] == pytest.approx(0.0, abs=1e-6)
    assert (low[0] + high[0]) / 2 == pytest.approx(0.0, abs=0.02)
    assert (low[1] + high[1]) / 2 == pytest.approx(0.0, abs=0.02)

    # Самая верхняя точка — это буквы: диск кончается на 0.8 мм ниже.
    topmost = flat[flat[:, 2] > high[2] - 1e-6]
    assert np.hypot(topmost[:, 0], topmost[:, 1]).max() < FLAT_RADIUS[slug]


@pytest.mark.parametrize("slug", LIDS)
def test_lettering_stands_proud_by_the_whole_relief(lids: dict[str, Triangles], slug: str) -> None:
    """Высота букв над торцом — ровно заявленная.

    Торец плоский, гнуть надпись не по чему, поэтому здесь рельеф
    обязан совпасть с `RELIEF` без всяких скидок на наклон.
    """
    triangles = lids[slug]
    high = triangles.reshape(-1, 3).max(axis=0)
    line = lettering.line(recipes.SLOGAN, recipes.LID_CAP).centred()
    edge = line.extent()
    window = (
        (float(edge[0][0]) - 0.6, float(edge[1][0]) + 0.6),
        (float(edge[0][1]) - 0.6, float(edge[1][1]) + 0.6),
    )
    face = panel.measure(triangles, 2, 1, (0.0, 0.0, 0.0), *window, step=0.3)
    assert face.heights.max() == pytest.approx(high[2], abs=1e-3)
    assert face.relief_span == pytest.approx(recipes.RELIEF, abs=0.02)


@pytest.mark.parametrize("slug", LIDS)
def test_lettering_keeps_clear_of_the_rim(slug: str) -> None:
    """Углы строки не доходят до скругления кромки."""
    line = lettering.line(recipes.SLOGAN, recipes.LID_CAP).centred()
    low, high = line.extent()
    corner = float(np.hypot(max(abs(low[0]), high[0]), max(abs(low[1]), high[1])))
    assert corner < FLAT_RADIUS[slug] - 2.0, f"{slug}: строка подходит к кромке на {corner:.1f} мм"


@pytest.mark.parametrize("slug", LIDS)
def test_lid_grew_only_by_the_lettering(lids: dict[str, Triangles], slug: str) -> None:
    """Крышка стала выше ровно на высоту букв, а в поперечнике не изменилась."""
    source = cases.case(slug).triangles().reshape(-1, 3)
    source_span = source.max(axis=0) - source.min(axis=0)
    thickness = float(source_span[AXIS[slug]])
    across = sorted(float(value) for index, value in enumerate(source_span) if index != AXIS[slug])

    lid = lids[slug].reshape(-1, 3)
    span = lid.max(axis=0) - lid.min(axis=0)
    assert span[2] == pytest.approx(thickness + recipes.RELIEF, abs=0.02)
    assert sorted((float(span[0]), float(span[1]))) == pytest.approx(across, abs=0.02)


def test_companions_are_shipped_as_sources_only() -> None:
    """Банка и фляжка лежат исходниками и в релиз не идут.

    Мы их не правим, а класть чужой файл в релиз байт в байт — значит
    выдавать его за свою работу.
    """
    assert {item.source_name for item in COMPANIONS} == {
        "obj_2_Can 170mm.stl",
        "水壶3.3.STL",
    }
    for item in COMPANIONS:
        assert item.source.is_file(), f"нет исходника {item.source_name}"
        assert cases.case(item.lid_slug), "у сосуда должна быть своя крышка"
    shipped = {item.source_name for item in cases.CASES}
    assert not shipped & {item.source_name for item in COMPANIONS}


@pytest.mark.parametrize("slug", LIDS)
def test_lettering_lands_letter_for_letter(lids: dict[str, Triangles], slug: str) -> None:
    """Буквы стоят ровно там, где их нарисовал шрифт.

    Это проверка на разворот и на зеркало. Крышку приходится
    переворачивать, чтобы надпись смотрела вверх, и поворот не вокруг
    той оси даёт строку вверх ногами, а `mirror` — отражённые буквы.
    И то, и другое проходит мимо всех остальных проверок: рельеф,
    габарит и радиус у перевёрнутой надписи те же самые.

    Обе крышки после разворота кладут строку так, что её собственные
    координаты совпадают с X и Y детали, — поэтому контуры шрифта
    сравниваются с деталью напрямую.
    """
    line = lettering.line(recipes.SLOGAN, recipes.LID_CAP).centred()
    segments = [
        ((float(a[0]), float(a[1])), (float(b[0]), float(b[1])))
        for contour in line.contours
        for a, b in zip(contour, np.roll(contour, -1, axis=0), strict=True)
    ]

    low, high = line.extent()
    face = panel.measure(
        lids[slug],
        2,
        1,
        (0.0, 0.0, 0.0),
        (float(low[0]) - 1.0, float(high[0]) + 1.0),
        (float(low[1]) - 1.0, float(high[1]) + 1.0),
        step=0.5,
    )
    # Сетка берётся у самого замера: у него свой шаг по краям, и
    # угадывать её отдельно — верный способ сравнить не то с тем.
    grid_u, grid_v = np.meshgrid(face.us, face.vs)
    grid = np.column_stack([grid_u.ravel(), grid_v.ravel()])

    # Клетки у самой кромки буквы пропускаем: там ответ зависит от того,
    # с какой стороны от кромки лёг узел сетки.
    clear = distance_to_outline(grid, segments) > 0.4
    inside = is_inside(grid, segments)
    # Замер отсчитывается от начала координат, поэтому «выступает» —
    # это выше самого торца, а не выше нуля.
    surface = face.heights.ravel()
    raised = surface > float(np.median(surface)) + recipes.RELIEF / 2

    assert raised[clear & inside].all(), "внутри буквы рельефа нет — надпись развернуло"
    assert not raised[clear & ~inside].any(), "рельеф там, где буквы нет"
