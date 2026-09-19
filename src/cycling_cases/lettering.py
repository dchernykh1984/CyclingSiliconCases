"""Надпись в контуры: буквы из TTF, разложенные в плоские полигоны.

Шрифт лежит в репозитории (`assets/fonts`) копией, а не берётся из
системы: надпись на чехле должна получаться одинаковой и на ноутбуке,
и на сборочной машине.

Взят Roboto Black — самое жирное начертание из семейства. Толщина
штриха важнее рисунка: у Roboto Black стойка буквы примерно 0.23 от
высоты прописной, то есть при высоте 4 мм это 0.94 мм — два периметра
сопла 0.4 мм. Начертание полегче дало бы штрих в один периметр, и
выпуклая надпись на TPU поплыла бы.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from fontTools.pens.recordingPen import RecordingPen
from fontTools.ttLib import TTFont
from numpy.typing import NDArray

FONT = Path(__file__).parent / "assets" / "fonts" / "Roboto-Black.ttf"

STEM_RATIO = 0.23
"""Толщина стойки буквы в долях высоты прописной — для проверки печатности."""

CURVE_STEPS = 12
"""На сколько отрезков дробится дуга буквы.

Двенадцать — это примерно 0.1 мм хорды при высоте буквы 4 мм: мельче
слайсер всё равно не различит, крупнее — видно грани на круглых буквах.
"""

Contour = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class Lettering:
    """Готовая строка: контуры в миллиметрах и её габарит."""

    contours: tuple[Contour, ...]
    width: float
    cap_height: float

    @property
    def stem(self) -> float:
        """Ожидаемая толщина стойки буквы."""
        return STEM_RATIO * self.cap_height

    def centred(self) -> Lettering:
        """Сдвинуть строку так, чтобы её центр был в начале координат."""
        shift = np.array([-self.width / 2.0, -self.cap_height / 2.0])
        return Lettering(
            tuple(contour + shift for contour in self.contours), self.width, self.cap_height
        )

    def extent(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        points = np.vstack(self.contours)
        return points.min(axis=0), points.max(axis=0)


def _glyph_contours(font: TTFont, char: str, scale: float) -> list[Contour]:
    """Контуры одного знака в миллиметрах, начало — на базовой линии слева."""
    pen = RecordingPen()
    font.getGlyphSet()[font.getBestCmap()[ord(char)]].draw(pen)

    contours: list[list[NDArray[np.float64]]] = []
    current: list[NDArray[np.float64]] = []
    cursor = np.zeros(2)
    for operation, arguments in pen.value:
        if operation == "moveTo":
            if current:
                contours.append(current)
            cursor = np.asarray(arguments[0], dtype=np.float64)
            current = [cursor]
        elif operation == "lineTo":
            cursor = np.asarray(arguments[0], dtype=np.float64)
            current.append(cursor)
        elif operation == "qCurveTo":
            points = [np.asarray(point, dtype=np.float64) for point in arguments]
            chain = [cursor, *points]
            # В TrueType подряд идущие контрольные точки делят дугу пополам:
            # между соседними лежит неявная точка на кривой.
            for index in range(1, len(chain) - 1):
                control = chain[index]
                start = cursor if index == 1 else (chain[index - 1] + control) / 2
                end = chain[-1] if index == len(chain) - 2 else (control + chain[index + 1]) / 2
                for step in range(1, CURVE_STEPS + 1):
                    t = step / CURVE_STEPS
                    current.append((1 - t) ** 2 * start + 2 * (1 - t) * t * control + t * t * end)
            cursor = chain[-1]
        elif operation == "curveTo":
            first, second, end = (np.asarray(point, dtype=np.float64) for point in arguments[-3:])
            for step in range(1, CURVE_STEPS + 1):
                t = step / CURVE_STEPS
                current.append(
                    (1 - t) ** 3 * cursor
                    + 3 * (1 - t) ** 2 * t * first
                    + 3 * (1 - t) * t**2 * second
                    + t**3 * end
                )
            cursor = end
        elif operation == "closePath" and current:
            contours.append(current)
            current = []
    if current:
        contours.append(current)
    return [np.asarray(contour) * scale for contour in contours]


def line(text: str, cap_height: float, tracking: float = 0.0, font_path: Path = FONT) -> Lettering:
    """Строка заданной высоты прописной, начало — базовая линия слева.

    `tracking` добавляется к каждому межбуквенному промежутку: им
    надпись подгоняют под ширину стенки, не трогая высоту букв.
    """
    font = TTFont(font_path)
    scale = cap_height / font["OS/2"].sCapHeight
    metrics = font["hmtx"]
    cmap = font.getBestCmap()

    contours: list[Contour] = []
    pen_x = 0.0
    for char in text:
        if char not in cmap and ord(char) not in cmap:
            raise KeyError(f"в шрифте нет знака {char!r}")
        if not char.isspace():
            for contour in _glyph_contours(font, char, scale):
                contours.append(contour + np.array([pen_x, 0.0]))
        pen_x += metrics[cmap[ord(char)]][0] * scale + tracking
    return Lettering(tuple(contours), pen_x - tracking, cap_height)
