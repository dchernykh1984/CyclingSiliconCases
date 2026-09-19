"""Твёрдые тела: мост к manifold3d и примитивы, из которых собраны правки.

Булевы операции с чужой сеткой считает manifold3d — тот же движок, что
стоит внутри современного OpenSCAD, только без внешнего бинаря: чехлы
собираются одной командой `uv run`, и на машине сборки ничего ставить
не нужно.

Всё, что мы добавляем к чужой сетке или вычитаем из неё, живёт здесь в
виде примитивов. Сами размеры — в `recipes.py`, рядом с объяснением,
откуда они взялись.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
from manifold3d import CrossSection, FillRule, Manifold, Mesh
from numpy.typing import NDArray

from .mesh import Triangles, read_stl, write_stl

WELD = 4
"""Знаков после запятой при сшивании вершин — как в `mesh.WELD`."""


def from_triangles(triangles: Triangles) -> Manifold:
    """Сетку из STL — в тело, с которым можно делать булевы операции.

    Вершины сшиваются округлением: в STL каждый треугольник записан
    сам по себе, и без сшивания сетка для movement-движка рассыпается
    на 3938 отдельных лоскутов.
    """
    flat = np.round(triangles.reshape(-1, 3), WELD)
    vertices, index = np.unique(flat, axis=0, return_inverse=True)
    mesh = Mesh(
        vert_properties=vertices.astype(np.float32),
        tri_verts=index.reshape(-1, 3).astype(np.uint32),
    )
    body = Manifold(mesh)
    if body.status().name != "NoError":
        raise ValueError(f"сетка не годится для булевых операций: {body.status().name}")
    return body


def to_triangles(body: Manifold) -> Triangles:
    """Тело обратно в массив треугольников."""
    mesh = body.to_mesh()
    vertices = np.asarray(mesh.vert_properties, dtype=np.float64)[:, :3]
    faces = np.asarray(mesh.tri_verts, dtype=np.int64)
    return vertices[faces]


def load(path: Path) -> Manifold:
    return from_triangles(read_stl(path))


def save(body: Manifold, path: Path) -> Path:
    """Записать тело в STL, убедившись, что записывать есть что.

    Сорванная булева операция возвращает пустое тело, а не исключение.
    Без этой проверки в релиз уехал бы STL на 84 байта, и заметили бы
    это уже на столе принтера.
    """
    if body.status().name != "NoError":
        raise ValueError(f"{path.name}: тело собралось с ошибкой {body.status().name}")
    if body.is_empty() or body.num_tri() == 0:
        raise ValueError(f"{path.name}: тело пустое — булева операция срезала всё")
    return write_stl(path, to_triangles(body))


def polygon(points: Sequence[Sequence[float]]) -> CrossSection:
    """Плоский контур как сечение."""
    return CrossSection([np.asarray(points, dtype=np.float64)], FillRule.NonZero)


def rounded_rectangle(
    low: tuple[float, float], high: tuple[float, float], radius: float, steps: int = 8
) -> CrossSection:
    """Прямоугольник со скруглёнными углами.

    Окна под кнопки скругляются не для красоты: в TPU острый внутренний
    угол — начало разрыва, а чехол при надевании растягивают именно за
    кромки окон.
    """
    x0, y0 = low
    x1, y1 = high
    radius = min(radius, (x1 - x0) / 2, (y1 - y0) / 2)
    corners = [
        ((x1 - radius, y1 - radius), 0.0),
        ((x0 + radius, y1 - radius), 90.0),
        ((x0 + radius, y0 + radius), 180.0),
        ((x1 - radius, y0 + radius), 270.0),
    ]
    points: list[tuple[float, float]] = []
    for (cx, cy), start in corners:
        for step in range(steps + 1):
            angle = np.radians(start + 90.0 * step / steps)
            points.append((cx + radius * np.cos(angle), cy + radius * np.sin(angle)))
    return polygon(points)


def prism(section: CrossSection, low: float, high: float, axis: int) -> Manifold:
    """Сечение, вытянутое вдоль оси `axis` от `low` до `high`.

    manifold3d вытягивает только вдоль Z, поэтому остальные оси —
    перестановка координат готовой призмы. Сечение задаётся в тех двух
    осях, что остаются, в их обычном порядке: для axis=0 это (Y, Z),
    для axis=1 — (X, Z), для axis=2 — (X, Y).
    """
    body = Manifold.extrude(section, high - low)
    if axis == 2:
        return body.translate([0.0, 0.0, low])
    if axis == 0:
        # (u, v, w) сечения и вытяжки -> (Y, Z, X)
        matrix = [[0.0, 0.0, 1.0, low], [1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]
    else:
        # (u, v, w) -> (X, Z, Y)
        matrix = [[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, low], [0.0, 1.0, 0.0, 0.0]]
    return body.transform(matrix)


def bounds(body: Manifold) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    box = np.asarray(body.bounding_box(), dtype=np.float64)
    return box[:3], box[3:]
