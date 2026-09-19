"""Общее для тестов: собранные чехлы считаются один раз на всю сессию."""

from __future__ import annotations

import numpy as np
import pytest

from cycling_cases import cases, solid
from cycling_cases.mesh import Triangles, ray_distances


@pytest.fixture(scope="session")
def built() -> dict[str, Triangles]:
    return {item.slug: solid.to_triangles(item.build()) for item in cases.CASES}


@pytest.fixture(scope="session")
def sources() -> dict[str, Triangles]:
    return {item.slug: item.triangles() for item in cases.CASES}


FAR = 300.0
"""Откуда пускается измерительный луч, мм. Заведомо снаружи любой детали."""


def depth_to_material(
    triangles: Triangles, axis: int, sign: int, point: tuple[float, float, float]
) -> float:
    """Сколько луч летит снаружи до первой грани.

    Через сквозное окно луч пролетает деталь насквозь и утыкается в
    противоположную стенку — этим окно и отличается от стенки.
    """
    normal = np.zeros(3)
    normal[axis] = sign
    start = np.asarray(point, dtype=np.float64).copy()
    start[axis] = sign * FAR
    return float(ray_distances(triangles, start[None, :], -normal)[0])


def inner_face(
    triangles: Triangles, axis: int, sign: int, point: tuple[float, float, float]
) -> float:
    """Координата внутренней грани стенки — второй по счёту вдоль луча.

    Первая грань — наружная поверхность чехла, она от наших правок не
    меняется; полость меряется по второй.
    """
    normal = np.zeros(3)
    normal[axis] = sign
    start = np.asarray(point, dtype=np.float64).copy()
    start[axis] = sign * FAR
    outer = ray_distances(triangles, start[None, :], -normal)[0]
    if not np.isfinite(outer):
        return float("nan")
    step = start - normal * (outer + 1e-3)
    inner = ray_distances(triangles, step[None, :], -normal)[0]
    if not np.isfinite(inner):
        return float("nan")
    return float(start[axis] - sign * (outer + 1e-3 + inner))
