"""Превью детали картинкой: свой растеризатор на numpy.

Смотреть на деталь всё равно надо — расчёт ловит размеры, но не
ловит, скажем, надпись, уехавшую на скругление. Ставить ради этого
OpenSCAD или полноценный движок ни к чему: треугольников тут десятки
тысяч, а z-буфер с плоской заливкой пишется в полсотни строк и
работает без единой внешней зависимости.
"""

from __future__ import annotations

import struct
import zlib
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from .mesh import Triangles

BACKGROUND = (22, 22, 24)
MATERIAL = (240, 208, 140)


def write_png(path: Path, image: NDArray[np.uint8]) -> Path:
    """Записать RGB-картинку в PNG — минимальный корректный файл."""
    height, width, _ = image.shape
    raw = b"".join(b"\0" + image[row].tobytes() for row in range(height))

    def chunk(tag: bytes, payload: bytes) -> bytes:
        body = tag + payload
        return struct.pack(">I", len(payload)) + body + struct.pack(">I", zlib.crc32(body))

    data = b"\x89PNG\r\n\x1a\n"
    data += chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
    data += chunk(b"IDAT", zlib.compress(raw, 6))
    data += chunk(b"IEND", b"")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def _basis(eye: NDArray[np.float64], target: NDArray[np.float64]) -> NDArray[np.float64]:
    forward = target - eye
    forward = forward / np.linalg.norm(forward)
    up = np.array([0.0, 0.0, 1.0])
    if abs(float(forward @ up)) > 0.999:
        up = np.array([0.0, 1.0, 0.0])
    right = np.cross(forward, up)
    right = right / np.linalg.norm(right)
    return np.stack([right, np.cross(right, forward), forward])


def render(
    triangles: Triangles,
    path: Path,
    eye: tuple[float, float, float],
    target: tuple[float, float, float] | None = None,
    size: tuple[int, int] = (900, 700),
    scale: float | None = None,
) -> Path:
    """Отрисовать сетку в PNG: ортографическая камера, плоская заливка."""
    flat = triangles.reshape(-1, 3)
    # Центр берём по габариту, а не по среднему вершин: у надписи и
    # скруглений вершин густо, и среднее уводит камеру с детали.
    middle = (flat.min(axis=0) + flat.max(axis=0)) / 2
    centre = np.asarray(target if target is not None else middle, dtype=np.float64)
    station = np.asarray(eye, dtype=np.float64)
    basis = _basis(station, centre)
    camera = (triangles - station) @ basis.T

    width, height = size
    if scale is None:
        scale = 0.8 * min(size) / float((flat.max(axis=0) - flat.min(axis=0)).max())
    screen_x = camera[:, :, 0] * scale + width / 2
    screen_y = height / 2 - camera[:, :, 1] * scale
    depth = camera[:, :, 2]

    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = np.divide(normals, lengths, out=np.zeros_like(normals), where=lengths > 0)
    light = np.array([0.35, -0.55, 0.75])
    light = light / np.linalg.norm(light)
    shade = np.clip(np.abs(normals @ light), 0.0, 1.0) * 0.75 + 0.22

    image = np.full((height, width, 3), BACKGROUND, np.uint8)
    zbuffer = np.full((height, width), np.inf)
    for index in np.argsort(-depth.mean(axis=1)):
        xs, ys, zs = screen_x[index], screen_y[index], depth[index]
        left = int(max(0, np.floor(xs.min())))
        right = int(min(width - 1, np.ceil(xs.max())))
        bottom = int(max(0, np.floor(ys.min())))
        top = int(min(height - 1, np.ceil(ys.max())))
        if left > right or bottom > top:
            continue
        area = (xs[1] - xs[0]) * (ys[2] - ys[0]) - (xs[2] - xs[0]) * (ys[1] - ys[0])
        if abs(area) < 1e-9:
            continue
        grid_x, grid_y = np.meshgrid(np.arange(left, right + 1), np.arange(bottom, top + 1))
        first = ((xs[1] - grid_x) * (ys[2] - grid_y) - (xs[2] - grid_x) * (ys[1] - grid_y)) / area
        second = ((xs[2] - grid_x) * (ys[0] - grid_y) - (xs[0] - grid_x) * (ys[2] - grid_y)) / area
        third = 1.0 - first - second
        inside = (first >= 0) & (second >= 0) & (third >= 0)
        if not inside.any():
            continue
        z = first * zs[0] + second * zs[1] + third * zs[2]
        window = zbuffer[bottom : top + 1, left : right + 1]
        mask = inside & (z < window)
        if not mask.any():
            continue
        colour = (np.array(MATERIAL) * shade[index]).astype(np.uint8)
        image[bottom : top + 1, left : right + 1][mask] = colour
        window[mask] = z[mask]
    return write_png(Path(path), image)


CAMERAS: dict[str, tuple[float, float, float]] = {
    "front": (0.0, 400.0, 30.0),
    "iso": (170.0, 240.0, 160.0),
    "back": (-140.0, -240.0, 150.0),
    "top": (0.0, 0.1, 400.0),
}
"""Ракурсы превью, как смещение камеры от центра детали.

Четыре вида — минимум, на котором видно и надпись, и окна под кнопки,
и раскладку в целом. Один вид врёт: буква, уехавшая за кромку, в
лоб выглядит как буква на месте.
"""
