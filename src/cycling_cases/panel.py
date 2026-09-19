"""Площадка на стенке: измеренная поверхность и выпуклая надпись по ней.

Стенка чехла нигде не плоская. У Garmin 830 передняя стенка уходит
назад на 1.5 мм к краям и на 2 мм вниз, у носа 840 — на 3 мм. Плоская
плитка с буквами, приложенная к такой стенке, в середине утонула бы, а
по краям висела в воздухе.

Поэтому надпись кладётся *по поверхности*: стенка сначала измеряется
лучами, потом плитка с буквами гнётся по замеру. Высота рельефа тогда
одинакова во всех точках — это и есть то, что видно на печати.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from manifold3d import CrossSection, FillRule, Manifold
from numpy.typing import NDArray

from .lettering import Lettering
from .mesh import Triangles, ray_distances

REFINE = 0.7
"""До какой длины дробятся рёбра плитки перед изгибом, мм.

Плитку гнут по вершинам, поэтому между ними она остаётся плоской.
0.7 мм при радиусе стенки 2 мм даёт отклонение 0.015 мм — меньше
слоя печати, то есть невидимо.
"""


def _frame(axis: int, sign: int) -> tuple[NDArray[np.float64], ...]:
    """Правая тройка векторов площадки: вправо, вверх, наружу.

    «Вверх» — это Z, кроме площадок на самом верху или снизу детали,
    где вверх берётся Y. Правая тройка нужна, чтобы плитка с буквами
    не вывернулась наизнанку при повороте в мировые оси.
    """
    normal = np.zeros(3)
    normal[axis] = sign
    up = np.array([0.0, 1.0, 0.0]) if axis == 2 else np.array([0.0, 0.0, 1.0])
    right = np.cross(-normal, up)
    return right, up, normal


@dataclass(frozen=True, slots=True)
class Panel:
    """Кусок стенки, измеренный сеткой лучей.

    `origin` — мировая точка отсчёта, `heights` — расстояние от неё до
    поверхности вдоль внешней нормали в узлах сетки (u вдоль `right`,
    v вдоль `up`).
    """

    origin: NDArray[np.float64]
    right: NDArray[np.float64]
    up: NDArray[np.float64]
    normal: NDArray[np.float64]
    us: NDArray[np.float64]
    vs: NDArray[np.float64]
    heights: NDArray[np.float64]

    @property
    def misses(self) -> int:
        """Сколько узлов сетки не попало в материал."""
        return int(np.count_nonzero(~np.isfinite(self.heights)))

    @property
    def relief_span(self) -> float:
        """Насколько поверхность уходит вглубь по площадке, мм."""
        good = self.heights[np.isfinite(self.heights)]
        return float(good.max() - good.min()) if good.size else float("nan")

    @property
    def tilt(self) -> float:
        """Наибольший наклон поверхности к площадке, градусы.

        На сильно наклонённом месте выпуклость получается скошенной и
        её видимая высота падает как косинус угла; больше 45° — уже
        не надпись, а буквы, завёрнутые за кромку.
        """
        good = np.where(np.isfinite(self.heights), self.heights, np.nan)
        du = np.gradient(good, self.us, axis=1)
        dv = np.gradient(good, self.vs, axis=0)
        slope = np.sqrt(np.nan_to_num(du) ** 2 + np.nan_to_num(dv) ** 2)
        return float(np.degrees(np.arctan(np.nanmax(slope))))

    def height(self, u: NDArray[np.float64], v: NDArray[np.float64]) -> NDArray[np.float64]:
        """Билинейная выборка замера в произвольных точках площадки."""
        ui = np.clip(np.searchsorted(self.us, u) - 1, 0, len(self.us) - 2)
        vi = np.clip(np.searchsorted(self.vs, v) - 1, 0, len(self.vs) - 2)
        u0, u1 = self.us[ui], self.us[ui + 1]
        v0, v1 = self.vs[vi], self.vs[vi + 1]
        tu = np.clip((u - u0) / (u1 - u0), 0.0, 1.0)
        tv = np.clip((v - v0) / (v1 - v0), 0.0, 1.0)
        h = self.heights
        return (
            h[vi, ui] * (1 - tu) * (1 - tv)
            + h[vi, ui + 1] * tu * (1 - tv)
            + h[vi + 1, ui] * (1 - tu) * tv
            + h[vi + 1, ui + 1] * tu * tv
        )

    def to_world(self, u: float, v: float, w: float = 0.0) -> NDArray[np.float64]:
        return self.origin + u * self.right + v * self.up + w * self.normal

    def emboss(self, lettering: Lettering, relief: float, sink: float = 0.6) -> Manifold:
        """Выпуклая надпись, лежащая по измеренной поверхности.

        Плитка с буквами строится плоской, дробится на мелкие грани и
        гнётся: каждая вершина уезжает вдоль нормали на замеренную
        высоту поверхности. Внутрь стенки плитка уходит на `sink` —
        иначе объединение оставит буквы отдельными телами.
        """
        if self.misses:
            # Промах означает, что луч не встретил материала: высота в
            # этом узле бесконечна, плитка уедет в бесконечность, и
            # manifold вернёт пустое тело вместо ошибки. Лучше сказать
            # сразу и цифрами.
            raise ValueError(
                f"площадка под надпись вышла за стенку: {self.misses} узлов "
                f"из {self.heights.size} не попали в материал"
            )
        section = CrossSection([np.asarray(c) for c in lettering.contours], FillRule.NonZero)
        plate = Manifold.extrude(section, sink + relief).translate([0.0, 0.0, -sink])
        plate = plate.refine_to_length(REFINE)

        basis = np.stack([self.right, self.up, self.normal], axis=1)
        matrix = np.hstack([basis, self.origin.reshape(3, 1)])
        plate = plate.transform(matrix.tolist())

        def bend(points: NDArray[np.float64]) -> NDArray[np.float64]:
            local = (np.asarray(points) - self.origin) @ basis
            offset = self.height(local[:, 0], local[:, 1])
            return np.asarray(points) + offset[:, None] * self.normal[None, :]

        return plate.warp_batch(bend)


def measure(
    triangles: Triangles,
    axis: int,
    sign: int,
    origin: Sequence[float],
    u_range: tuple[float, float],
    v_range: tuple[float, float],
    step: float = 0.4,
) -> Panel:
    """Измерить стенку лучами снаружи внутрь.

    `origin` — мировая точка отсчёта площадки; замер ведётся от неё
    вдоль внешней нормали, так что «высота» может быть и отрицательной.
    """
    right, up, normal = _frame(axis, sign)
    start = np.asarray(origin, dtype=np.float64)

    # Луч должен стартовать заведомо снаружи детали, иначе он «увидит»
    # её изнанку и замер молча окажется с другой стороны стенки.
    along = triangles.reshape(-1, 3)[:, axis]
    reach = float(sign * (along.max() if sign > 0 else along.min()) - sign * start[axis]) + 5.0

    us = np.arange(u_range[0], u_range[1] + step / 2, step)
    vs = np.arange(v_range[0], v_range[1] + step / 2, step)
    grid_u, grid_v = np.meshgrid(us, vs)
    launch = start + grid_u.reshape(-1, 1) * right + grid_v.reshape(-1, 1) * up + reach * normal
    distances = ray_distances(triangles, launch, -normal)
    heights = (reach - distances).reshape(grid_v.shape)
    return Panel(start, right, up, normal, us, vs, heights)
