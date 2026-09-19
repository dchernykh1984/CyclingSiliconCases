"""Проверки измерительного инструмента — на фигурах, ответ для которых известен."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from cycling_cases.mesh import (
    bounds,
    box_probes,
    clearance,
    components,
    distance_to_outline,
    edge_counts,
    health,
    is_inside,
    is_watertight,
    ray_distances,
    read_stl,
    section,
    size,
    write_stl,
)


def box(width: float, depth: float, height: float, shift: tuple[float, float, float]) -> np.ndarray:
    """Замкнутый параллелепипед из двенадцати треугольников."""
    low = np.array(shift, dtype=np.float64)
    high = low + np.array([width, depth, height], dtype=np.float64)
    corners = np.array(
        [[x, y, z] for x in (low[0], high[0]) for y in (low[1], high[1]) for z in (low[2], high[2])]
    )
    faces = [
        (0, 1, 3),
        (0, 3, 2),
        (4, 6, 7),
        (4, 7, 5),
        (0, 4, 5),
        (0, 5, 1),
        (2, 3, 7),
        (2, 7, 6),
        (0, 2, 6),
        (0, 6, 4),
        (1, 5, 7),
        (1, 7, 3),
    ]
    return np.array([[corners[a], corners[b], corners[c]] for a, b, c in faces])


def test_box_is_watertight() -> None:
    assert edge_counts(box(10, 20, 30, (0, 0, 0))) == {2: 18}
    assert is_watertight(box(10, 20, 30, (0, 0, 0)))
    assert health(box(10, 20, 30, (0, 0, 0))) == "замкнуто"


def test_hole_is_reported_as_a_hole() -> None:
    # Выкидываем одну грань: получается коробка с дыркой.
    broken = box(10, 20, 30, (0, 0, 0))[:-1]
    assert not is_watertight(broken)
    assert "дырок" in health(broken)


def test_bounds_and_size() -> None:
    mesh = box(10, 20, 30, (-5, 1, 2))
    low, high = bounds(mesh)
    assert low == pytest.approx([-5, 1, 2])
    assert high == pytest.approx([5, 21, 32])
    assert size(mesh) == pytest.approx([10, 20, 30])


def test_two_boxes_are_two_bodies() -> None:
    mesh = np.concatenate([box(10, 10, 10, (0, 0, 0)), box(10, 10, 10, (50, 0, 0))])
    assert len(components(mesh)) == 2


def test_touching_boxes_are_one_body() -> None:
    # Общая грань сшивает тела: именно так раскладка на стол иногда
    # оказывается одним телом, хотя на вид это разные детали.
    mesh = np.concatenate([box(10, 10, 10, (0, 0, 0)), box(10, 10, 10, (10, 0, 0))])
    assert len(components(mesh)) == 1


def test_section_gives_the_outline() -> None:
    mesh = box(10, 20, 30, (0, 0, 0))
    segments = section(mesh, 15.0)
    assert segments, "на середине высоты сечение не может быть пустым"
    points = np.array([point for segment in segments for point in segment])
    assert points[:, 0].min() == pytest.approx(0.0)
    assert points[:, 0].max() == pytest.approx(10.0)
    assert points[:, 1].min() == pytest.approx(0.0)
    assert points[:, 1].max() == pytest.approx(20.0)


def test_inside_and_distance() -> None:
    segments = section(box(10, 20, 30, (0, 0, 0)), 15.0)
    probes = np.array([[5.0, 10.0], [-1.0, 10.0], [1.0, 1.0]])
    assert list(is_inside(probes, segments)) == [True, False, True]
    gaps = distance_to_outline(probes, segments)
    assert gaps[0] == pytest.approx(5.0)
    assert gaps[1] == pytest.approx(1.0)
    assert gaps[2] == pytest.approx(1.0)


def test_clearance_is_negative_outside() -> None:
    # Ради этого всё и затевалось: точка в вырезе стоит далеко от кромок,
    # и без проверки «внутри ли» она выглядела бы прекрасно помещающейся.
    segments = section(box(10, 20, 30, (0, 0, 0)), 15.0)
    assert clearance(np.array([[5.0, 10.0]]), segments) == pytest.approx(5.0)
    assert clearance(np.array([[5.0, 10.0], [20.0, 10.0]]), segments) < 0


def test_box_probes_cover_the_edges() -> None:
    probes = box_probes((0.0, 0.0), 10.0, 4.0)
    assert len(probes) == 9, "углы, середины сторон и центр"
    assert probes[:, 0].min() == pytest.approx(-5.0)
    assert probes[:, 1].max() == pytest.approx(2.0)


def test_rotated_probes_follow_the_angle() -> None:
    probes = box_probes((0.0, 0.0), 10.0, 0.0, angle=90.0)
    assert probes[:, 1].max() == pytest.approx(5.0)
    assert probes[:, 0].max() == pytest.approx(0.0, abs=1e-9)


def test_write_and_read_round_trip(tmp_path: Path) -> None:
    mesh = box(10, 20, 30, (1, 2, 3))
    path = write_stl(tmp_path / "box.stl", mesh)
    assert read_stl(path) == pytest.approx(mesh)


def test_text_stl_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "ascii.stl"
    path.write_text("solid box\n facet normal 0 0 1\n" + "x" * 100, encoding="utf-8")
    with pytest.raises(ValueError, match="бинарный"):
        read_stl(path)


def test_ray_hits_the_near_face() -> None:
    mesh = box(10, 20, 30, (0, 0, 0))
    hit = ray_distances(mesh, np.array([[5.0, 10.0, 100.0]]), (0, 0, -1))
    assert hit[0] == pytest.approx(70.0)


def test_ray_that_misses_returns_infinity() -> None:
    mesh = box(10, 20, 30, (0, 0, 0))
    hit = ray_distances(mesh, np.array([[50.0, 10.0, 100.0]]), (0, 0, -1))
    assert not np.isfinite(hit[0])


def test_ray_ignores_what_is_behind_it() -> None:
    # Замер стенки идёт лучом снаружи внутрь, и грань за спиной луча —
    # это противоположная стенка: приняв её за свою, замер молча
    # показал бы толщину всей детали.
    mesh = box(10, 20, 30, (0, 0, 0))
    hit = ray_distances(mesh, np.array([[5.0, 10.0, 100.0]]), (0, 0, 1))
    assert not np.isfinite(hit[0])


def test_ray_distances_measure_a_wall() -> None:
    # Две грани подряд вдоль луча — это и есть толщина стенки.
    mesh = np.concatenate([box(2, 20, 30, (0, 0, 0)), box(2, 20, 30, (10, 0, 0))])
    first = ray_distances(mesh, np.array([[-5.0, 10.0, 15.0]]), (1, 0, 0))[0]
    second = ray_distances(mesh, np.array([[-5.0 + first + 1e-3, 10.0, 15.0]]), (1, 0, 0))[0]
    assert first == pytest.approx(5.0)
    assert second == pytest.approx(2.0, abs=1e-2)


def test_many_rays_at_once() -> None:
    mesh = box(10, 20, 30, (0, 0, 0))
    origins = np.array([[x, 10.0, 100.0] for x in (2.0, 5.0, 8.0, 50.0)])
    hits = ray_distances(mesh, origins, (0, 0, -1))
    assert hits[:3] == pytest.approx([70.0, 70.0, 70.0])
    assert not np.isfinite(hits[3])


def test_rays_survive_a_mesh_they_cannot_hit() -> None:
    # Все треугольники параллельны лучу: пересекать нечего. Раньше
    # `min` по пустой оси ронял замер вместо честного «промах».
    flat = np.array([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]])
    hits = ray_distances(flat, np.array([[0.2, 0.2, 5.0], [9.0, 9.0, 5.0]]), (1, 0, 0))
    assert not np.isfinite(hits).any()


def test_chunking_does_not_change_the_answer(monkeypatch: pytest.MonkeyPatch) -> None:
    # Лучи считаются пачками, чтобы память не росла с их числом.
    # Размер пачки не должен влиять на результат.
    mesh = box(10, 20, 30, (0, 0, 0))
    origins = np.array([[x, 10.0, 100.0] for x in np.linspace(0.5, 9.5, 40)])
    whole = ray_distances(mesh, origins, (0, 0, -1))
    monkeypatch.setattr("cycling_cases.mesh.CHUNK", 1)
    assert ray_distances(mesh, origins, (0, 0, -1)) == pytest.approx(whole)
