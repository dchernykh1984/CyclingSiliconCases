"""Замеры стенок для тестов: толщина и где она падает до мембраны.

Живёт рядом с тестами, а не в пакете: это инструмент приёмки, а не
часть сборки чехлов.
"""

from __future__ import annotations

import numpy as np

from cycling_cases.mesh import Triangles, ray_distances

MEMBRANE = 1.2
"""Граница между мембраной и полноценной стенкой, мм."""


def wall_thickness(triangles: Triangles, sign: int, y: float, z: float = 2.0) -> float:
    """Толщина боковой стенки в точке: расстояние между двумя гранями."""
    normal = np.array([float(sign), 0.0, 0.0])
    start = np.array([sign * 300.0, y, z])
    outer = ray_distances(triangles, start[None, :], -normal)[0]
    if not np.isfinite(outer):
        return float("nan")
    step = start - normal * (outer + 1e-4)
    inner = ray_distances(triangles, step[None, :], -normal)[0]
    return float(inner) + 1e-4 if np.isfinite(inner) else float("nan")


def thin_wall_runs(
    triangles: Triangles,
    sign: int,
    span: tuple[float, float] = (44.0, 108.0),
    step: float = 0.25,
    z: float = 2.0,
) -> list[tuple[float, float]]:
    """Отрезки по длине бока, где стенка тоньше мембраны."""
    runs: list[tuple[float, float]] = []
    start: float | None = None
    previous = span[0]
    for y in np.arange(span[0], span[1], step):
        thin = wall_thickness(triangles, sign, float(y), z) < MEMBRANE
        if thin and start is None:
            start = float(y)
        if not thin and start is not None:
            runs.append((start, previous))
            start = None
        previous = float(y)
    if start is not None:
        runs.append((start, previous))
    return runs
