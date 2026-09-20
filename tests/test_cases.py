"""Каталог чехлов и приёмка собранных деталей.

Тесты меряют готовую сетку, а не повторяют числа из рецепта: рецепт
может ошибиться, замер по детали — нет.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from manifold3d import Manifold

from conftest import depth_to_material, inner_face
from cycling_cases import cases, panel, recipes, solid
from cycling_cases.cases import CASES, build, case, repository_root
from cycling_cases.cli import build_parser, main
from cycling_cases.mesh import Triangles, edge_counts, parts

GRID = 0.3
"""Шаг сетки при замере рельефа надписи, мм."""

THROUGH = 280.0
"""Луч длиннее этого — значит, пролетел сквозь окно, а не упёрся в стенку."""


def relief(
    source: Triangles,
    finished: Triangles,
    axis: int,
    sign: int,
    origin: tuple[float, float, float],
) -> np.ndarray:
    """Насколько поверхность готовой детали выступила над исходной."""
    window = ((-17.8, 17.8), (-2.6, 2.6))
    before = panel.measure(source, axis, sign, origin, *window, step=GRID)
    after = panel.measure(finished, axis, sign, origin, *window, step=GRID)
    assert before.misses == 0, "площадка под надпись вышла за стенку"
    return after.heights - before.heights


# --------------------------------------------------------------------------
# каталог
# --------------------------------------------------------------------------


def test_repository_root_is_found_by_its_sources() -> None:
    # Корень ищется по наличию `input_data`, а не отсчётом шагов от пакета:
    # при установке колесом пакет лежит в site-packages, и шаги промахиваются.
    assert (repository_root() / "input_data").is_dir()


def test_repository_root_falls_back_to_the_working_directory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Пакет, установленный колесом, лежит вдали от исходников: тогда корень
    # ищется от рабочего каталога. Эту ветку легко сломать и не заметить,
    # потому что в режиме разработки она никогда не выполняется.
    real = repository_root()
    monkeypatch.setattr(cases, "PACKAGE_ROOT", tmp_path / "site-packages" / "cycling_cases")
    monkeypatch.chdir(real)
    assert repository_root() == real


def test_slugs_are_unique_and_tidy() -> None:
    slugs = [item.slug for item in CASES]
    assert len(slugs) == len(set(slugs))
    for slug in slugs:
        assert slug == slug.lower()
        assert " " not in slug, "имя уходит в имя файла, пробелам там не место"


def test_case_lookup() -> None:
    assert case("garmin-830").fits == "Garmin Edge 830"
    with pytest.raises(KeyError):
        case("garmin-1030")


def test_filenames_are_named_after_the_case() -> None:
    assert {item.filename for item in CASES} == {
        "garmin-830.stl",
        "garmin-840.stl",
        "can-lid.stl",
        "bottle-cap.stl",
    }


def test_build_writes_every_case(tmp_path: Path) -> None:
    written = build(tmp_path)
    assert {path.name for path in written} == {item.filename for item in CASES}
    for path in written:
        assert path.stat().st_size > 1024


def test_build_can_do_a_single_case(tmp_path: Path) -> None:
    written = build(tmp_path, only="garmin-830")
    assert [path.name for path in written] == ["garmin-830.stl"]


def test_release_files_differ_from_the_sources(tmp_path: Path) -> None:
    """В релиз уходит наша деталь, а не копия скачанного файла.

    Первый релиз проекта отдавал исходники байт в байт — так было
    задумано, пока правок не было. Теперь правки есть, и совпадение
    с исходником означало бы, что рецепт молча не применился.
    """
    for path in build(tmp_path):
        source = case(path.stem).source
        assert path.read_bytes() != source.read_bytes()


def test_cli_knows_its_commands() -> None:
    parser = build_parser()
    for name in ("list", "build", "inspect", "split", "preview"):
        assert parser.parse_args([name, *(["x.stl"] if name in {"inspect", "split"} else [])])


def test_cli_lists_cases(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["list"]) == 0
    printed = capsys.readouterr().out
    for item in CASES:
        assert item.slug in printed


# --------------------------------------------------------------------------
# общее для обеих деталей
# --------------------------------------------------------------------------


@pytest.mark.parametrize("item", CASES, ids=lambda item: item.slug)
def test_build_gives_one_printable_body(item: cases.Case) -> None:
    body = item.build()
    assert body.status().name == "NoError"
    assert len(body.decompose()) == 1, "деталь развалилась на куски"
    triangles = solid.to_triangles(body)
    assert len(parts(triangles)) == 1
    assert 1 not in edge_counts(triangles), "в готовой сетке дырки — слайсер их не закроет"


def test_840_gains_no_new_self_touches(built: dict[str, Triangles]) -> None:
    """Самокасания исходника мы не лечим, но и не плодим.

    Резать ровно по грани кармана нельзя: совпавшие плоскости дают
    новые самокасания. Четыре ребра — ровно столько их в исходнике.
    """
    assert edge_counts(built["garmin-840"]).get(4) == 4
    assert 6 not in edge_counts(built["garmin-840"])


@pytest.mark.parametrize(
    ("slug", "origin"),
    [("garmin-830", (0.0, 0.0, recipes.E830_TEXT_Z)), ("garmin-840", (-90.0, 115.0, 1.0))],
)
def test_slogan_stands_proud_by_the_same_amount_everywhere(
    sources: dict[str, Triangles], slug: str, origin: tuple[float, float, float]
) -> None:
    """Главное свойство надписи: её высота одинакова по всей строке.

    Стенки чехлов не плоские, и надпись кладётся по измеренной
    поверхности. Положенная на плоскость, она в середине торчала бы на
    0.8 мм, а по краям тонула на два-три миллиметра.
    """
    source = sources[slug]
    finished = solid.to_triangles(case(slug).recipe(source))
    if slug == "garmin-840":
        source = recipes.pick_case(source, recipes.E840_PART)
        finished = finished - np.asarray(recipes.E840_CENTRE)
    grown = relief(source, finished, 1, 1, origin)

    letters = grown[grown > 0.4]
    assert letters.size > 0.2 * grown.size, "надпись занимает подозрительно мало места"
    assert letters.min() > 0.7, "в самом пологом месте буква почти пропала"
    assert letters.max() <= recipes.RELIEF + 0.02
    assert letters.mean() == pytest.approx(recipes.RELIEF, abs=0.06)


# --------------------------------------------------------------------------
# Garmin Edge 830
# --------------------------------------------------------------------------

CAVITY_Z = 14.0
"""Высота, на которой стенки 830 уже вертикальны, а полость равна прибору.

Ближе к кромке стенка отходит наружу на полмиллиметра — это заводская
разгрузка исходника, и захват губы надо считать не от неё, а от полости.
"""


def cavity_width(triangles: Triangles, z: float) -> float:
    left = inner_face(triangles, 0, -1, (0.0, 2.0, z))
    right = inner_face(triangles, 0, 1, (0.0, 2.0, z))
    return right - left


def test_830_keeps_the_rim_height(
    sources: dict[str, Triangles], built: dict[str, Triangles]
) -> None:
    """Губы не должны торчать над кромкой — иначе прибор в чехол не сядет."""
    before = sources["garmin-830"].reshape(-1, 3).max(axis=0)
    after = built["garmin-830"].reshape(-1, 3).max(axis=0)
    assert after[2] == pytest.approx(before[2], abs=0.002)


def test_830_grows_only_forward(sources: dict[str, Triangles], built: dict[str, Triangles]) -> None:
    """Единственное, что выросло наружу, — надпись на передней стенке."""
    source = sources["garmin-830"].reshape(-1, 3)
    finished = built["garmin-830"].reshape(-1, 3)
    assert np.allclose(finished.min(axis=0), source.min(axis=0), atol=0.002)
    high_before, high_after = source.max(axis=0), finished.max(axis=0)
    assert high_after[1] - high_before[1] == pytest.approx(recipes.RELIEF, abs=0.01)
    assert np.allclose(high_after[[0, 2]], high_before[[0, 2]], atol=0.002)


@pytest.mark.parametrize(
    ("name", "axis", "sign", "point"),
    [
        ("кнопка питания слева", 0, -1, (0.0, 19.0, 11.0)),
        ("кнопки lap и start/stop сзади", 1, -1, (0.0, 0.0, 11.0)),
        ("окно под крепление в дне", 2, -1, (0.0, 0.0, 0.0)),
    ],
)
def test_830_button_windows_stay_open(
    built: dict[str, Triangles], name: str, axis: int, sign: int, point: tuple[float, float, float]
) -> None:
    """Штатные окна чехла 830 остаются сквозными."""
    reach = depth_to_material(built["garmin-830"], axis, sign, point)
    assert reach > THROUGH, f"{name}: луч уткнулся в материал на {reach:.1f} мм"


def test_830_lips_narrow_the_opening(
    sources: dict[str, Triangles], built: dict[str, Triangles]
) -> None:
    """Губа сужает вход ровно на заявленный захват — и только у кромки."""
    for z in (10.0, 14.0, 17.5):
        assert cavity_width(built["garmin-830"], z) == pytest.approx(
            cavity_width(sources["garmin-830"], z), abs=0.02
        ), f"на z={z} губа уже мешает — прибор не войдёт"

    cavity = cavity_width(sources["garmin-830"], CAVITY_Z)
    narrowed = cavity - cavity_width(built["garmin-830"], recipes.E830_LIP_PEAK)
    assert narrowed == pytest.approx(2 * recipes.E830_GRAB, abs=0.05)


def test_830_front_lip_grabs_the_bezel(
    sources: dict[str, Triangles], built: dict[str, Triangles]
) -> None:
    cavity = inner_face(sources["garmin-830"], 1, 1, (0.0, 0.0, CAVITY_Z))
    after = inner_face(built["garmin-830"], 1, 1, (0.0, 0.0, recipes.E830_LIP_PEAK))
    assert cavity - after == pytest.approx(recipes.E830_GRAB, abs=0.05)


# --------------------------------------------------------------------------
# Garmin Edge 840
# --------------------------------------------------------------------------

SHIFT = np.asarray(recipes.E840_CENTRE)
"""Сдвиг детали из раскладки в начало координат — координаты рецепта плюс он."""

BAND_Z = float(SHIFT[2] + 1.2)
"""Высота, на которой мерятся боковые окна: середина пояса кнопок."""


def moved(point: tuple[float, float, float]) -> tuple[float, float, float]:
    """Точка из координат исходника — в координаты готовой детали."""
    x, y, z = np.asarray(point) + SHIFT
    return float(x), float(y), float(z)


def test_840_release_holds_only_the_case(
    sources: dict[str, Triangles], built: dict[str, Triangles]
) -> None:
    """Вынос, хомуты и прокладки в релиз не попадают.

    Вся раскладка занимает 315×169 мм; если в релиз просочится хоть
    один хомут, габарит это сразу покажет.
    """
    case_body = recipes.pick_case(sources["garmin-840"], recipes.E840_PART)
    source_span = case_body.reshape(-1, 3).max(axis=0) - case_body.reshape(-1, 3).min(axis=0)
    finished = built["garmin-840"].reshape(-1, 3)
    span = finished.max(axis=0) - finished.min(axis=0)

    assert span[0] == pytest.approx(source_span[0], abs=0.05)
    assert span[2] == pytest.approx(source_span[2], abs=0.05)
    # По длине спереди прибавилась надпись, а сзади окно срезало самую
    # заднюю точку торца — отсюда только верхняя оценка.
    assert source_span[1] < span[1] < source_span[1] + recipes.RELIEF + 0.01


def test_840_stands_on_the_bed(built: dict[str, Triangles]) -> None:
    finished = built["garmin-840"].reshape(-1, 3)
    low, high = finished.min(axis=0), finished.max(axis=0)
    assert low[2] == pytest.approx(0.0, abs=1e-6)
    assert (low[0] + high[0]) / 2 == pytest.approx(0.0, abs=1e-6)


@pytest.mark.parametrize("sign", [-1, 1])
def test_840_every_button_pocket_became_a_window(built: dict[str, Triangles], sign: int) -> None:
    """Под каждой разметкой автора теперь сквозное окно."""
    for start, finish in recipes.E840_BUTTON_POCKETS[sign]:
        middle = moved((0.0, (start + finish) / 2, 0.0))
        reach = depth_to_material(built["garmin-840"], 0, sign, (0.0, middle[1], BAND_Z))
        assert reach > THROUGH, f"бок {sign:+d}, карман {start}…{finish}: окна нет"


def test_840_keeps_the_window_the_author_cut(built: dict[str, Triangles]) -> None:
    middle = moved((0.0, 36.5, 0.0))
    for sign in (-1, 1):
        reach = depth_to_material(built["garmin-840"], 0, sign, (0.0, middle[1], BAND_Z))
        assert reach > THROUGH


@pytest.mark.parametrize(
    ("sign", "y"),
    [(-1, 47.0), (-1, 81.0), (-1, 104.0), (1, 47.0), (1, 74.0), (1, 104.0)],
)
def test_840_ribs_between_the_windows_survive(
    built: dict[str, Triangles], sign: int, y: float
) -> None:
    """Между окнами остаются рёбра исходника в полную толщину.

    Ради них окна и режутся по карманам, а не одной длинной прорезью:
    сплошное окно во всю длину оставило бы бок на двух полосках.
    """
    middle = moved((0.0, y, 0.0))
    reach = depth_to_material(built["garmin-840"], 0, sign, (0.0, middle[1], BAND_Z))
    assert reach < THROUGH, f"бок {sign:+d}: ребро на y={y} пропало"


@pytest.mark.parametrize("sign", [-1, 1])
@pytest.mark.parametrize("dz", [-5.0, 7.0])
def test_840_side_rails_survive(built: dict[str, Triangles], sign: int, dz: float) -> None:
    """Над окнами и под ними бок остаётся сплошным по всей длине."""
    reach = depth_to_material(built["garmin-840"], 0, sign, (0.0, 0.0, BAND_Z + dz))
    assert reach < THROUGH


@pytest.mark.parametrize("pocket", recipes.E840_REAR_BUTTONS)
def test_840_rear_windows_open_the_bottom_buttons(
    built: dict[str, Triangles], pocket: tuple[float, float]
) -> None:
    middle = moved(((pocket[0] + pocket[1]) / 2, 0.0, 0.0))[0]
    reach = depth_to_material(built["garmin-840"], 1, -1, (middle, 0.0, BAND_Z))
    assert reach > THROUGH, "окно не попало в кнопку нижнего торца"


def test_840_rear_wall_keeps_its_middle(built: dict[str, Triangles]) -> None:
    """Между окнами задний торец остаётся сплошным.

    Кнопки нижнего торца стоят на завалах углов, в 18.7 мм от середины
    в обе стороны. Одно широкое окно сняло бы торец в поясе кнопок
    целиком — два окна по кнопкам оставляют между собой 30 мм стенки.
    """
    reach = depth_to_material(built["garmin-840"], 1, -1, (0.0, 0.0, BAND_Z))
    assert reach < THROUGH


@pytest.mark.parametrize("dz", [-5.0, 7.0])
def test_840_rear_wall_survives_above_and_below_the_window(
    built: dict[str, Triangles], dz: float
) -> None:
    """Окно вырезано только в поясе кнопок: рамка остаётся замкнутой."""
    reach = depth_to_material(built["garmin-840"], 1, -1, (0.0, 0.0, BAND_Z + dz))
    assert reach < THROUGH, "задний торец пропал целиком — рамка разойдётся"


def test_840_keeps_the_retaining_lip(built: dict[str, Triangles]) -> None:
    """Верхняя губа исходника должна пережить прорезку окон.

    Меряем на ребре у носа: там стенка сплошная на всю высоту, и
    сужение полости к кромке — это именно губа, а не край окна.
    """
    rib = moved((0.0, 104.0, 0.0))[1]

    def cavity(z: float) -> float:
        left = inner_face(built["garmin-840"], 0, -1, (0.0, rib, z))
        right = inner_face(built["garmin-840"], 0, 1, (0.0, rib, z))
        return right - left

    assert cavity(SHIFT[2] + 8.5) < cavity(BAND_Z) - 1.0


# --------------------------------------------------------------------------
# отказы вместо тихих пустышек
# --------------------------------------------------------------------------


def test_build_refuses_an_unknown_slug(tmp_path: Path) -> None:
    # Опечатка в имени не должна оборачиваться пустой сборкой с кодом 0.
    with pytest.raises(KeyError):
        build(tmp_path, only="garmin-84")


def test_preview_refuses_an_unknown_slug() -> None:
    with pytest.raises(KeyError):
        main(["preview", "--slug", "garmin-84", "--out", "preview"])


def test_saving_an_empty_body_is_refused(tmp_path: Path) -> None:
    """Сорванная булева операция возвращает пустое тело, а не исключение.

    Без проверки в релиз уехал бы STL на 84 байта, и заметили бы это
    уже на столе принтера.
    """
    cube = Manifold.cube([10.0, 10.0, 10.0])
    with pytest.raises(ValueError, match="пустое"):
        solid.save(cube - cube, tmp_path / "empty.stl")


def test_emboss_refuses_a_panel_that_missed_the_wall(sources: dict[str, Triangles]) -> None:
    """Надпись, не попавшая на стенку, — это ошибка, а не пустое тело.

    Промах означает бесконечную высоту в узле замера: плитка с буквами
    уехала бы в бесконечность, а manifold вернул бы пустое тело молча.
    """
    beyond = panel.measure(
        sources["garmin-830"], 1, 1, (0.0, 0.0, recipes.E830_TEXT_Z), (-40.0, 40.0), (-2.6, 2.6)
    )
    assert beyond.misses > 0
    with pytest.raises(ValueError, match="вышла за стенку"):
        beyond.emboss(recipes._lettering(recipes.E830_CAP), recipes.RELIEF)
