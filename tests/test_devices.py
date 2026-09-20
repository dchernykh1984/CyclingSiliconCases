"""Приёмка по модели самого прибора.

Пока модели прибора нет, про чехол можно сказать только то, что видно
в нём самом. С моделью проверяется главное: лезет ли прибор в полость
и попадают ли окна в кнопки.
"""

from __future__ import annotations

import numpy as np
import pytest

from cycling_cases import cases, devices, recipes, solid
from cycling_cases.devices import Button, Device
from cycling_cases.mesh import Triangles, ray_distances

EDGE_840 = (57.8, 85.1, 19.6)
"""Габарит Edge 840 по спецификации Garmin: ширина, длина, толщина, мм."""

EDGE_830 = (50.0, 82.0, 20.0)
"""Габарит Edge 830 — для сравнения: модель приехала под его именем."""


@pytest.fixture(scope="session")
def edge840() -> Device:
    return devices.device("garmin-840-device")


@pytest.fixture(scope="session")
def found(edge840: Device) -> tuple[Button, ...]:
    return devices.buttons(edge840)


@pytest.fixture(scope="session")
def built_840() -> Triangles:
    return solid.to_triangles(cases.case("garmin-840").build())


def test_model_is_the_840_and_not_the_830(edge840: Device) -> None:
    """Файл приехал под именем 830, но это 840 — и это надо знать.

    Разница не в допуске: по ширине между приборами 7.8 мм. Модель 830
    в чехол 830 (полость 50.6 мм) просто не вошла бы.
    """
    flat = devices.body(edge840).reshape(-1, 3)
    width, length, thickness = (flat.max(axis=0) - flat.min(axis=0))[[2, 0, 1]]

    assert width == pytest.approx(EDGE_840[0], abs=0.6)
    assert length == pytest.approx(EDGE_840[1], abs=0.6)
    assert thickness == pytest.approx(EDGE_840[2], abs=0.6)
    assert abs(width - EDGE_830[0]) > 5.0, "по ширине это никак не 830"


def test_the_model_is_one_closed_body(edge840: Device) -> None:
    body = solid.from_triangles(devices.body(edge840))
    assert body.status().name == "NoError"
    assert body.volume() > 60_000.0


def test_seven_buttons_are_found(found: tuple[Button, ...]) -> None:
    """Три кнопки на левом боку, две на правом, две на заднем торце.

    Столько их у Edge 840: питание, вверх и вниз слева, выбор и назад
    справа, lap и start/stop снизу.
    """
    by_face: dict[str, int] = {}
    for button in found:
        by_face[button.face] = by_face.get(button.face, 0) + 1
    assert by_face == {"левый бок": 3, "правый бок": 2, "задний торец": 2}


def test_buttons_stand_proud_enough_to_be_felt(found: tuple[Button, ...]) -> None:
    for button in found:
        assert button.rise > 0.9, f"{button.face}: выступ всего {button.rise:.2f} мм"


def test_every_button_sits_in_the_band_the_windows_cover(found: tuple[Button, ...]) -> None:
    """Все кнопки по высоте попадают в пояс окон с запасом."""
    low, high = recipes.E840_WINDOW_Z
    for button in found:
        assert button.height[0] > low + 1.0, f"{button.face}: кнопка ниже пояса окон"
        assert button.height[1] < high + 0.1, f"{button.face}: кнопка выше пояса окон"


def test_every_button_has_a_window_over_it(
    found: tuple[Button, ...], built_840: Triangles, edge840: Device
) -> None:
    """Главная проверка: сквозь чехол видно каждую кнопку.

    Луч пускается из середины кнопки наружу по нормали её грани. Если
    он уходит в бесконечность, над кнопкой дыра; если упирается в
    материал — кнопку закрыли.
    """
    shift = np.asarray(recipes.E840_CENTRE)
    faces = {"левый бок": (0, -1), "правый бок": (0, 1), "задний торец": (1, -1)}
    for button in found:
        axis, sign = faces[button.face]
        along = 1 if axis == 0 else 0
        start = _probe(shift, axis, sign, along, sum(button.along) / 2, sum(button.height) / 2)
        normal = np.zeros(3)
        normal[axis] = sign
        reach = ray_distances(built_840, start[None, :], -normal)[0]
        assert reach > 280.0, f"{button.face} {button.along}: кнопка закрыта чехлом"


def test_windows_leave_a_margin_around_every_button(
    found: tuple[Button, ...], built_840: Triangles
) -> None:
    """Запас от кромки кнопки до кромки окна — по всей её длине.

    Мерим не по центру, а по краям кнопки: центр мог бы попасть в окно
    и при том, что половина кнопки закрыта.
    """
    shift = np.asarray(recipes.E840_CENTRE)
    faces = {"левый бок": (0, -1), "правый бок": (0, 1), "задний торец": (1, -1)}
    margin = 0.35
    for button in found:
        axis, sign = faces[button.face]
        along = 1 if axis == 0 else 0
        normal = np.zeros(3)
        normal[axis] = sign
        for edge in (button.along[0] - margin, button.along[1] + margin):
            start = _probe(shift, axis, sign, along, edge, sum(button.height) / 2)
            reach = ray_distances(built_840, start[None, :], -normal)[0]
            assert reach > 280.0, (
                f"{button.face}: у края кнопки на {edge:.2f} нет запаса {margin} мм"
            )


def test_the_case_takes_the_device(edge840: Device) -> None:
    """Прибор входит в полость: пересечения с чехлом нет.

    Чехол печатается из TPU и что-то простил бы, но это надо знать, а
    не предполагать: пересечение больше кубического миллиметра значит,
    что чехол придётся натягивать силой.
    """
    case_body = solid.from_triangles(
        recipes.pick_case(cases.case("garmin-840").triangles(), recipes.E840_PART)
    )
    device_body = solid.from_triangles(devices.seated(edge840))
    assert (case_body ^ device_body).volume() < 1.0


def test_the_device_sits_on_the_bottom_of_the_frame(edge840: Device) -> None:
    """Задняя грань прибора ложится на дно рамки — крепление до него дойдёт."""
    flat = devices.seated(edge840).reshape(-1, 3)
    case_low = cases.case("garmin-840").triangles()
    case_low = recipes.pick_case(case_low, recipes.E840_PART).reshape(-1, 3).min(axis=0)
    # Ниже дна рамки уходит только бобышка крепления.
    assert flat[:, 2].min() == pytest.approx(case_low[2] - 3.79, abs=0.1)


def test_the_frame_holds_the_device_in(edge840: Device) -> None:
    """Вверх прибор не вынуть: рамка сверху уже, чем он сам.

    Прибор заводят снизу, со стороны крепления; дальше его держит
    сужение рамки над ним.
    """
    case = recipes.pick_case(cases.case("garmin-840").triangles(), recipes.E840_PART)
    mesh = devices.seated(edge840)
    widest = max(
        (_width(mesh, y), y) for y in np.arange(40.0, 105.0, 2.5) if np.isfinite(_width(mesh, y))
    )
    device_width, y = widest
    top = _width(case, y, z=9.0, inner=True)
    assert device_width - top > 3.0, "рамка над прибором шире его — держать нечем"


def _probe(
    shift: np.ndarray, axis: int, sign: int, along: int, at: float, height: float
) -> np.ndarray:
    """Начало луча: точка кнопки в осях готовой детали, вынесенная наружу."""
    point = np.zeros(3)
    point[along] = at
    point[2] = height
    point = point + shift
    point[axis] = sign * 300.0
    return point


def _width(mesh: Triangles, y: float, z: float = 2.0, inner: bool = False) -> float:
    edges = []
    for sign in (-1, 1):
        normal = np.array([float(sign), 0.0, 0.0])
        start = np.array([sign * 300.0, y, z])
        reach = ray_distances(mesh, start[None, :], -normal)[0]
        if not np.isfinite(reach):
            return float("nan")
        if inner:
            step = start - normal * (reach + 1e-3)
            second = ray_distances(mesh, step[None, :], -normal)[0]
            if not np.isfinite(second):
                return float("nan")
            reach += 1e-3 + second
        edges.append(start[0] - sign * reach)
    return float(edges[1] - edges[0])
