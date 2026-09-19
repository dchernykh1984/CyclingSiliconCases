"""Надпись: размеры строки считаются, а не подбираются на глаз."""

from __future__ import annotations

import numpy as np
import pytest

from cycling_cases import lettering, recipes


def test_cap_height_is_exact_on_a_flat_letter() -> None:
    """У «T» верх и низ плоские — её высота и есть заказанная."""
    low, high = lettering.line("T", 4.0).extent()
    assert high[1] - low[1] == pytest.approx(4.0, abs=0.02)
    assert low[1] == pytest.approx(0.0, abs=0.02)


def test_round_letters_overshoot_only_slightly() -> None:
    """Круглые буквы по типографской традиции чуть выше прописной.

    Площадка под надпись мерится с запасом именно на этот выступ; если
    он вдруг вырастет, запас перестанет быть запасом.
    """
    low, high = lettering.line(recipes.SLOGAN, 4.0).extent()
    assert 4.0 <= high[1] - low[1] <= 4.15


def test_width_scales_with_height() -> None:
    narrow = lettering.line(recipes.SLOGAN, 2.0).width
    wide = lettering.line(recipes.SLOGAN, 4.0).width
    assert wide == pytest.approx(2 * narrow, rel=1e-6)


def test_stem_is_thick_enough_for_a_nozzle() -> None:
    """Стойка буквы — не меньше двух периметров сопла 0.4 мм."""
    for cap in (recipes.E830_CAP, recipes.E840_CAP):
        assert lettering.line(recipes.SLOGAN, cap).stem >= 0.8


def test_centred_line_is_centred() -> None:
    low, high = lettering.line(recipes.SLOGAN, 4.0).centred().extent()
    assert np.allclose((low + high) / 2, [0.0, 0.0], atol=0.1)


def test_tracking_only_changes_the_width() -> None:
    plain = lettering.line(recipes.SLOGAN, 4.0)
    spaced = lettering.line(recipes.SLOGAN, 4.0, tracking=0.5)
    assert spaced.width > plain.width
    assert spaced.cap_height == plain.cap_height


def test_unknown_character_is_refused() -> None:
    with pytest.raises(KeyError):
        lettering.line("UBT ☃", 4.0)
