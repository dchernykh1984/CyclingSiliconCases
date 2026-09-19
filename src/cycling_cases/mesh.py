"""Измерительный инструмент для сеток STL.

Главное правило проекта: геометрия проверяется расчётом, а не на глаз.
Рендер показывает далеко не всё — надпись, уехавшая за кромку на полмиллиметра,
на картинке неотличима от надписи, лежащей впритык. Поэтому всё, что должно
поместиться, посчитано здесь и закреплено тестом.

Модуль намеренно обходится одним numpy: он читает бинарный STL, разбирает
сетку на отдельные тела, режет её плоскостью и меряет расстояния. Этого
хватает и для приёмки исходников, и для проверки собственных правок.
"""

from __future__ import annotations

import struct
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

Triangles = NDArray[np.float64]
"""Сетка как массив (N, 3, 3): треугольник → три вершины → XYZ."""

Points = NDArray[np.float64]
"""Точки как массив (N, 2) или (N, 3)."""

Direction = Sequence[float] | Points
"""Направление луча: хоть кортеж, хоть массив — нормируем сами."""

WELD = 4
"""До скольких знаков округляются координаты при сшивании вершин.

Исходники нарисованы в миллиметрах, и соседние треугольники приходят из
разных программ с разницей в последнем знаке float32. Округление до 0.1 мкм
склеивает их, не трогая реальную геометрию.
"""


def read_stl(path: Path) -> Triangles:
    """Прочитать бинарный STL.

    ASCII-файлы отвергаем сразу: исходники у нас бинарные, а молчаливое
    чтение мусора дало бы «сетку» из нулей и сломало бы все замеры.
    """
    raw = path.read_bytes()
    if len(raw) < 84:
        raise ValueError(f"{path.name}: файл короче заголовка STL")
    if raw[:5] == b"solid" and b"facet normal" in raw[:2048]:
        raise ValueError(f"{path.name}: это текстовый STL, нужен бинарный")

    count = struct.unpack("<I", raw[80:84])[0]
    expected = 84 + count * 50
    if len(raw) < expected:
        raise ValueError(f"{path.name}: обещано {count} треугольников, а файла на них не хватает")

    dtype = np.dtype([("normal", "<3f4"), ("vertices", "<3,3f4"), ("attribute", "<u2")])
    records = np.frombuffer(raw, dtype=dtype, count=count, offset=84)
    return np.asarray(records["vertices"], dtype=np.float64).reshape(count, 3, 3)


def write_stl(path: Path, triangles: Triangles) -> Path:
    """Записать бинарный STL. Нормали считаем по обходу вершин."""
    first = triangles[:, 1] - triangles[:, 0]
    second = triangles[:, 2] - triangles[:, 0]
    normals = np.cross(first, second)
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = np.divide(normals, lengths, out=np.zeros_like(normals), where=lengths > 0)

    dtype = np.dtype([("normal", "<3f4"), ("vertices", "<3,3f4"), ("attribute", "<u2")])
    records = np.zeros(len(triangles), dtype=dtype)
    records["normal"] = normals
    records["vertices"] = triangles

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        handle.write(b"\0" * 80)
        handle.write(struct.pack("<I", len(triangles)))
        handle.write(records.tobytes())
    return path


def ray_distances(
    triangles: Triangles, origins: Points, direction: Direction
) -> NDArray[np.float64]:
    """До ближайшей грани вдоль луча из каждой точки; `inf`, если промах.

    Мёллер—Трумбор без ускоряющих структур: лучей мы пускаем тысячи,
    а треугольников в сетке тысячи, и произведение считается numpy за
    доли секунды. Дерево здесь только усложнило бы код.
    """
    starts = triangles[:, 0]
    first = triangles[:, 1] - starts
    second = triangles[:, 2] - starts
    ray = np.asarray(direction, dtype=np.float64)
    ray = ray / np.linalg.norm(ray)

    sideways = np.cross(ray, second)
    determinant = np.einsum("ij,ij->i", first, sideways)
    usable = np.abs(determinant) > 1e-12
    starts, first, second = starts[usable], first[usable], second[usable]
    sideways, determinant = sideways[usable], determinant[usable]

    probes = np.atleast_2d(np.asarray(origins, dtype=np.float64))
    offsets = probes[:, None, :] - starts[None, :, :]
    u = np.einsum("ijk,jk->ij", offsets, sideways) / determinant
    across = np.cross(offsets, first[None, :, :])
    v = (across @ ray) / determinant
    along = np.einsum("jk,ijk->ij", second, across) / determinant

    hit = (u >= -1e-9) & (v >= -1e-9) & (u + v <= 1 + 1e-9) & (along > 1e-9)
    return np.where(hit.any(axis=1), np.where(hit, along, np.inf).min(axis=1), np.inf)


def bounds(triangles: Triangles) -> tuple[Points, Points]:
    """Габаритная коробка: минимальный и максимальный угол."""
    flat = triangles.reshape(-1, 3)
    return flat.min(axis=0), flat.max(axis=0)


def size(triangles: Triangles) -> Points:
    """Размеры габаритной коробки по осям."""
    low, high = bounds(triangles)
    return high - low


def vertex_index(triangles: Triangles) -> NDArray[np.int64]:
    """Номера сшитых вершин для каждого треугольника, массив (N, 3)."""
    rounded = np.round(triangles.reshape(-1, 3), WELD)
    _, index = np.unique(rounded, axis=0, return_inverse=True)
    return index.reshape(len(triangles), 3).astype(np.int64)


def edge_counts(triangles: Triangles) -> dict[int, int]:
    """Сколько рёбер принадлежит одному, двум, трём треугольникам.

    У замкнутой сетки все рёбра ровно по два. Одиночные рёбра — это дырка,
    и булевы операции на такой сетке дают мусор вместо детали.
    """
    faces = vertex_index(triangles)
    pairs = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    pairs = np.sort(pairs, axis=1)
    _, counts = np.unique(pairs, axis=0, return_counts=True)
    values, totals = np.unique(counts, return_counts=True)
    return {int(value): int(total) for value, total in zip(values, totals, strict=True)}


def is_watertight(triangles: Triangles) -> bool:
    """Замкнута ли сетка — то есть можно ли её резать булевыми операциями."""
    return set(edge_counts(triangles)) == {2}


def health(triangles: Triangles) -> str:
    """Словами о том, что не так с сеткой.

    Дырка и склейка — разные беды. Ребро, принадлежащее одному треугольнику,
    это настоящая дыра: булевы операции на такой сетке дают мусор. Ребро,
    принадлежащее четырём, — это место, где деталь сама себя касается;
    manifold такое обычно переваривает, но помнить о нём стоит.
    """
    counts = edge_counts(triangles)
    if set(counts) == {2}:
        return "замкнуто"
    troubles = []
    holes = counts.get(1, 0)
    if holes:
        troubles.append(f"дырок (рёбер без пары): {holes}")
    glued = sum(total for count, total in counts.items() if count > 2)
    if glued:
        troubles.append(f"самокасаний (рёбер больше чем у двух граней): {glued}")
    return "; ".join(troubles) if troubles else "замкнуто"


def components(triangles: Triangles) -> list[Triangles]:
    """Разобрать сетку на отдельные тела.

    Скачанные модели часто оказываются целой раскладкой на стол: чехол,
    крепление, хомуты. Дальше с ними работают поштучно.
    """
    faces = vertex_index(triangles)
    parent = np.arange(int(faces.max()) + 1)

    def root(item: int) -> int:
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = int(parent[item])
        return item

    for first, second, third in faces:
        base = root(int(first))
        for other in (int(second), int(third)):
            mate = root(other)
            if mate != base:
                parent[mate] = base

    labels = np.array([root(int(face[0])) for face in faces])
    order = [labels == label for label in dict.fromkeys(labels.tolist())]
    return [triangles[mask] for mask in order]


@dataclass(frozen=True, slots=True)
class Part:
    """Одно тело внутри файла — с номером, размерами и положением."""

    number: int
    triangles: Triangles
    low: Points
    high: Points

    @property
    def size(self) -> Points:
        return self.high - self.low

    @property
    def watertight(self) -> bool:
        return is_watertight(self.triangles)

    def describe(self) -> str:
        width, depth, height = self.size
        return (
            f"#{self.number}: {len(self.triangles):6d} тр.  "
            f"{width:7.2f} × {depth:7.2f} × {height:6.2f} мм  "
            f"в точке ({self.low[0]:7.2f}, {self.low[1]:7.2f}, {self.low[2]:7.2f})"
        )


def parts(triangles: Triangles) -> list[Part]:
    """Тела файла по убыванию числа треугольников — самое крупное первым."""
    found = components(triangles)
    found.sort(key=len, reverse=True)
    result = []
    for number, mesh in enumerate(found):
        low, high = bounds(mesh)
        result.append(Part(number=number, triangles=mesh, low=low, high=high))
    return result


Segment = tuple[tuple[float, float], tuple[float, float]]


def section(triangles: Triangles, z: float) -> list[Segment]:
    """Контур детали на высоте `z` — набор отрезков в плоскости XY.

    Плоскость сечения — единственный способ померить то, что не видно
    снаружи: толщину стенки, ширину полости под велокомпьютер, зазор
    между гравировкой и кромкой.
    """
    heights = triangles[:, :, 2] - z
    crosses = ~(np.all(heights > 0, axis=1) | np.all(heights < 0, axis=1))
    segments: list[Segment] = []
    for triangle, height in zip(triangles[crosses], heights[crosses], strict=True):
        points: list[tuple[float, float]] = []
        for first, second in ((0, 1), (1, 2), (2, 0)):
            start, end = height[first], height[second]
            if start == end:
                continue
            if (start <= 0 <= end) or (end <= 0 <= start):
                ratio = start / (start - end)
                edge = triangle[first] + ratio * (triangle[second] - triangle[first])
                points.append((float(edge[0]), float(edge[1])))
        if len(points) >= 2:
            segments.append((points[0], points[1]))
    return segments


def outline_points(segments: list[Segment]) -> Points:
    """Концы отрезков контура как массив точек."""
    if not segments:
        return np.zeros((0, 2))
    return np.array([point for segment in segments for point in segment], dtype=np.float64)


def is_inside(points: Points, segments: list[Segment]) -> NDArray[np.bool_]:
    """Лежат ли точки внутри контура — луч вправо и чётность пересечений.

    Без этой проверки «отступ до кромки» врёт: у детали с вырезом точка
    посреди выреза стоит далеко от всех кромок и выглядит как надёжно
    помещающаяся, хотя материала под ней нет вовсе.
    """
    probes = np.atleast_2d(np.asarray(points, dtype=np.float64))
    if not segments:
        return np.zeros(len(probes), dtype=bool)

    starts = np.array([segment[0] for segment in segments])
    ends = np.array([segment[1] for segment in segments])
    px = probes[:, 0:1]
    py = probes[:, 1:2]

    straddles = (starts[:, 1] > py) != (ends[:, 1] > py)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = (py - starts[:, 1]) / (ends[:, 1] - starts[:, 1])
        crossing_x = starts[:, 0] + ratio * (ends[:, 0] - starts[:, 0])
    hits = straddles & (crossing_x > px)
    return (np.count_nonzero(hits, axis=1) % 2) == 1


def distance_to_outline(points: Points, segments: list[Segment]) -> NDArray[np.float64]:
    """Расстояние от каждой точки до ближайшей кромки контура."""
    probes = np.atleast_2d(np.asarray(points, dtype=np.float64))
    if not segments:
        return np.full(len(probes), np.inf)

    starts = np.array([segment[0] for segment in segments])
    ends = np.array([segment[1] for segment in segments])
    direction = ends - starts
    lengths = (direction * direction).sum(axis=1)

    offsets = probes[:, None, :] - starts[None, :, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = (offsets * direction).sum(axis=2) / np.where(lengths > 0, lengths, 1.0)
    ratio = np.clip(np.where(lengths > 0, ratio, 0.0), 0.0, 1.0)
    feet = starts[None, :, :] + ratio[:, :, None] * direction[None, :, :]
    return np.linalg.norm(probes[:, None, :] - feet, axis=2).min(axis=1)


def clearance(points: Points, segments: list[Segment]) -> float:
    """Запас до кромки; отрицательный, если хоть одна точка вне материала."""
    gaps = distance_to_outline(points, segments)
    inside = is_inside(points, segments)
    smallest = float(gaps.min())
    return smallest if bool(inside.all()) else -smallest


def box_probes(
    centre: tuple[float, float],
    width: float,
    height: float,
    angle: float = 0.0,
) -> Points:
    """Девять точек габарита надписи: углы, середины сторон и центр.

    Одних углов мало: кромка бывает волнистой и заходит в середину стороны,
    а повёрнутый вдоль детали блок углами промахивается мимо неё.
    """
    radians = np.radians(angle)
    cos, sin = float(np.cos(radians)), float(np.sin(radians))
    grid = [
        (along, across)
        for along in (-width / 2, 0.0, width / 2)
        for across in (-height / 2, 0.0, height / 2)
    ]
    return np.array(
        [
            (centre[0] + along * cos - across * sin, centre[1] + along * sin + across * cos)
            for along, across in grid
        ]
    )
