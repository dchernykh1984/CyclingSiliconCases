"""Модель самого велокомпьютера: как он садится в чехол и где у него кнопки.

Чехол мы правим вслепую до тех пор, пока не с чем его сверить. Модель
прибора это меняет: по ней видно, попадают ли окна в кнопки и лезет ли
прибор в полость вообще.

Кнопки здесь **ищутся**, а не записываются числами. Кнопка — это
местный выступ грани над её же профилем: замеряем поверхность лучами и
смотрим, где она поднимается над медианой. Замер лучами не зависит от
того, как сетка триангулирована, поэтому прореженная модель даёт те же
числа, что и полная.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from .mesh import Triangles, parts, ray_distances, read_stl

Range = tuple[float, float]


@dataclass(frozen=True, slots=True)
class Device:
    """Прибор: чья модель, в каком она масштабе и как садится в чехол."""

    slug: str
    title: str
    computer: str
    source_name: str
    case_slug: str
    units: float
    seat: tuple[tuple[float, ...], ...]
    release_shift: tuple[float, float, float] = (0.0, 0.0, 0.0)
    comment: str = ""

    @property
    def source(self) -> Path:
        from .cases import sources_dir

        return sources_dir() / self.source_name

    @property
    def matrix(self) -> NDArray[np.float64]:
        return np.asarray(self.seat, dtype=np.float64)


E840_SEAT = (
    (0.0, 0.0, 1.0, -90.0),
    (-1.0, 0.0, 0.0, 70.985),
    (0.0, -1.0, 0.0, -1.76),
)
"""Как Edge 840 садится в свой чехол: `в чехле = seat @ (модель, 1)`.

Оси: ширина прибора ложится на X чехла, длина — на −Y, толщина — на −Z.
Числа сдвига взяты из посадки, а не подобраны:

* **−90.0** — середина полости по ширине (полость 58.5 мм от −119.25
  до −60.75), прибор шириной 58.29 центрируют рёбра чехла;
* **70.985** — прибор упирается носом в переднюю стенку полости, она
  на 113.75, а до неё от середины прибора 42.765;
* **−1.76** — задняя грань прибора (без бобышки крепления она на
  +6.04 в осях модели) ложится на дно рамки, на −7.8.

Правильность этой посадки видно по кнопкам: все семь садятся в свои
карманы, и верхняя кромка каждой кнопки совпадает с кромкой кармана
с точностью 0.01 мм. Проверяется тестом.
"""

DEVICES: tuple[Device, ...] = (
    Device(
        slug="garmin-840-device",
        title="Велокомпьютер Garmin Edge 840",
        computer="Garmin Edge 840",
        source_name="Garmin840Device.stl",
        case_slug="garmin-840",
        units=0.1,
        seat=E840_SEAT,
        # Рецепт ставит готовую деталь на плоскость и центрирует её;
        # посадка прибора описана в осях исходника, поэтому для сверки
        # с готовой деталью её надо сдвинуть так же.
        release_shift=(90.0, -75.5, 7.8),
        comment="Приехал под именем «garmin edge 830 Mesh.stl», но по размерам это 840",
    ),
)


def device(slug: str) -> Device:
    for item in DEVICES:
        if item.slug == slug:
            return item
    known = ", ".join(item.slug for item in DEVICES)
    raise KeyError(f"нет прибора {slug!r}; есть {known}")


def device_for(case_slug: str) -> Device | None:
    """Модель прибора для чехла, если она у нас есть."""
    for item in DEVICES:
        if item.case_slug == case_slug:
            return item
    return None


def body(item: Device) -> Triangles:
    """Корпус прибора в миллиметрах, центрированный по габариту.

    В файле может лежать не только корпус: у модели Edge 840 рядом
    с ним десяток плоских наклеек — экранная графика. Берём самое
    крупное тело.
    """
    triangles = read_stl(item.source) * item.units
    largest = parts(triangles)[0].triangles
    flat = largest.reshape(-1, 3)
    return largest - (flat.min(axis=0) + flat.max(axis=0)) / 2


def seated(item: Device) -> Triangles:
    """Прибор в осях его чехла — там, где он в нём сидит."""
    matrix = item.matrix
    return body(item) @ matrix[:, :3].T + matrix[:, 3]


def place(item: Device, box: tuple[Range, Range, Range]) -> tuple[Range, Range, Range]:
    """Габарит из осей модели — в оси чехла."""
    corners = np.array([[x, y, z] for x in box[0] for y in box[1] for z in box[2]])
    matrix = item.matrix
    moved = corners @ matrix[:, :3].T + matrix[:, 3]
    low, high = moved.min(axis=0), moved.max(axis=0)
    return ((low[0], high[0]), (low[1], high[1]), (low[2], high[2]))


def face_surface(
    triangles: Triangles,
    axis: int,
    sign: int,
    along: int,
    across: int,
    along_range: Range,
    across_range: Range,
    step: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Поверхность грани в узлах сетки, замеренная лучами снаружи."""
    flat = triangles.reshape(-1, 3)
    start = (flat[:, axis].max() + 5.0) if sign > 0 else (flat[:, axis].min() - 5.0)
    ga = np.arange(along_range[0], along_range[1] + step / 2, step)
    gb = np.arange(across_range[0], across_range[1] + step / 2, step)
    grid_a, grid_b = np.meshgrid(ga, gb)
    origins = np.zeros((grid_a.size, 3))
    origins[:, axis] = start
    origins[:, along] = grid_a.ravel()
    origins[:, across] = grid_b.ravel()
    normal = np.zeros(3)
    normal[axis] = sign
    reach = ray_distances(triangles, origins, -normal)
    surface = sign * (start - sign * reach)
    return ga, gb, surface.reshape(grid_b.shape)


def protrusions(
    triangles: Triangles,
    axis: int,
    sign: int,
    along: int,
    across: int,
    along_range: Range,
    across_range: Range,
    step: float = 0.5,
    rise: float = 0.3,
    window: float | None = None,
    min_area: float = 6.0,
) -> list[dict[str, object]]:
    """Местные выступы грани над её собственным профилем.

    Профиль считается медианой: по всей длине сразу (`window=None`) для
    прямых боков, скользящим окном — для торца, который сам по себе
    выгнут. Без вычитания профиля кнопку не отличить от скругления
    кромки: у Edge 840 кнопки нижнего торца стоят на завале угла и по
    абсолютной высоте ниже, чем середина того же торца.
    """
    ga, gb, grid = face_surface(
        triangles, axis, sign, along, across, along_range, across_range, step
    )
    base = np.empty_like(grid)
    for row in range(grid.shape[0]):
        line = np.where(np.isfinite(grid[row]), grid[row], np.nan)
        if window is None:
            base[row] = np.nanmedian(line)
        else:
            span = max(3, int(window / step) | 1)
            base[row] = [
                np.nanmedian(line[max(0, i - span // 2) : i + span // 2 + 1])
                for i in range(len(line))
            ]
    proud = np.nan_to_num(grid - base, nan=-1e9) > rise

    labels = np.zeros_like(proud, dtype=int)
    tag = 0
    for row in range(proud.shape[0]):
        for column in range(proud.shape[1]):
            if not proud[row, column] or labels[row, column]:
                continue
            tag += 1
            stack = [(row, column)]
            labels[row, column] = tag
            while stack:
                cr, cc = stack.pop()
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        nr, nc = cr + dr, cc + dc
                        if (
                            0 <= nr < proud.shape[0]
                            and 0 <= nc < proud.shape[1]
                            and proud[nr, nc]
                            and not labels[nr, nc]
                        ):
                            labels[nr, nc] = tag
                            stack.append((nr, nc))

    found: list[dict[str, object]] = []
    for mark in range(1, tag + 1):
        mask = labels == mark
        area = float(mask.sum()) * step * step
        if area < min_area:
            continue
        rows, columns = np.nonzero(mask)
        found.append(
            {
                "along": (float(ga[columns.min()]), float(ga[columns.max()])),
                "across": (float(gb[rows.min()]), float(gb[rows.max()])),
                "rise": float((grid - base)[mask].max()),
                "area": area,
            }
        )
    found.sort(key=lambda item: item["along"][0])  # type: ignore[index,return-value]
    return found


BUTTON_BAND = (-1.5, 6.5)
"""Пояс по высоте чехла, в котором ищем кнопки, мм.

У Edge 840 все семь кнопок лежат между −0.01 и 4.82 по высоте чехла;
пояс взят с запасом в полтора миллиметра в обе стороны. Шире нельзя:
у самой лицевой грани начинается рамка экрана, и её кромка читается
как выступ.
"""

FACES: tuple[tuple[str, int, int, int, int, float | None], ...] = (
    ("левый бок", 0, -1, 1, 2, None),
    ("правый бок", 0, 1, 1, 2, None),
    ("задний торец", 1, -1, 0, 2, 16.0),
)
"""Грани чехла, на которых ищем кнопки: имя, ось нормали, сторона,
ось длины, ось высоты и окно медианы. У боков профиль постоянный по
длине, поэтому медиана берётся сразу по всей грани; торец сам по себе
выгнут, и там нужно скользящее окно."""


FACE_AXES: dict[str, tuple[int, int]] = {
    "левый бок": (0, -1),
    "правый бок": (0, 1),
    "задний торец": (1, -1),
}
"""Ось нормали и сторона для каждой грани — чтобы стрелять лучом к кнопке."""


@dataclass(frozen=True, slots=True)
class Button:
    """Кнопка прибора в осях чехла."""

    face: str
    along: Range
    height: Range
    rise: float

    def describe(self) -> str:
        return (
            f"{self.face}: {self.along[0]:8.2f}…{self.along[1]:8.2f} "
            f"× высота {self.height[0]:5.2f}…{self.height[1]:5.2f}, выступ {self.rise:.2f} мм"
        )


def probe(
    item: Device, button: Button, at: float | None = None
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Луч к кнопке готовой детали: откуда стрелять и куда.

    `at` — точка вдоль кнопки; по умолчанию её середина. Кромки кнопки
    проверяют, подставляя её края: центр может попасть в окно и тогда,
    когда половина кнопки закрыта.
    """
    axis, sign = FACE_AXES[button.face]
    along = 1 if axis == 0 else 0
    point = np.zeros(3)
    point[along] = sum(button.along) / 2 if at is None else at
    point[2] = sum(button.height) / 2
    point = point + np.asarray(item.release_shift)
    point[axis] = sign * 300.0
    normal = np.zeros(3)
    normal[axis] = float(sign)
    return point, -normal


def buttons(item: Device, step: float = 0.5) -> tuple[Button, ...]:
    """Все кнопки прибора, найденные по его модели, в осях чехла."""
    mesh = seated(item)
    flat = mesh.reshape(-1, 3)
    low, high = flat.min(axis=0), flat.max(axis=0)
    found: list[Button] = []
    for name, axis, sign, along, across, window in FACES:
        edge = 1.5  # к самым углам грань заворачивает, там ищем только шум
        spread = protrusions(
            mesh,
            axis,
            sign,
            along,
            across,
            (low[along] + edge, high[along] - edge),
            BUTTON_BAND,
            step=step,
            window=window,
            min_area=4.0 if window else 6.0,
        )
        for bump in spread:
            found.append(
                Button(
                    face=name,
                    along=bump["along"],  # type: ignore[arg-type]
                    height=bump["across"],  # type: ignore[arg-type]
                    rise=float(bump["rise"]),  # type: ignore[arg-type]
                )
            )
    return tuple(found)
