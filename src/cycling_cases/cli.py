"""Командная строка: собрать чехлы, посмотреть исходник, сделать превью."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from . import __version__, devices, preview, solid
from .cases import CASES, COMPANIONS, build, case, sources_dir
from .mesh import edge_counts, health, parts, ray_distances, read_stl, write_stl


def _find(name: str) -> Path:
    path = Path(name)
    if path.exists():
        return path
    candidate = sources_dir() / name
    return candidate if candidate.exists() else path


def command_list(args: argparse.Namespace) -> int:
    for item in CASES:
        print(f"{item.slug:12s}  {item.title:32s}  из {item.source_name}")
        if item.comment:
            print(f"{'':12s}  {item.comment}")
    for vessel in COMPANIONS:
        print(f"{vessel.slug:12s}  {vessel.title:32s}  из {vessel.source_name}")
        if vessel.comment:
            print(f"{'':12s}  {vessel.comment}")
    return 0


def command_build(args: argparse.Namespace) -> int:
    for path in build(Path(args.out), only=args.slug):
        triangles = read_stl(path)
        low, high = triangles.reshape(-1, 3).min(axis=0), triangles.reshape(-1, 3).max(axis=0)
        print(f"{path}  {len(triangles)} тр.  {np.round(high - low, 2)} мм  {health(triangles)}")
    return 0


def command_inspect(args: argparse.Namespace) -> int:
    """Показать, что лежит в файле: тела, размеры, замкнутость.

    Скачанные модели часто оказываются целой раскладкой на стол, и
    первое, что нужно знать перед правкой, — из скольких тел она
    состоит.
    """
    path = _find(args.stl)
    triangles = read_stl(path)
    found = parts(triangles)
    print(f"{path.name}: {len(triangles)} треугольников, тел {len(found)}")
    for part in found:
        print(f"  {part.describe()}  {health(part.triangles)}")
    edges = edge_counts(triangles)
    print(f"  рёбра по числу треугольников: {dict(sorted(edges.items()))}")
    if 1 in edges:
        print("  внимание: в сетке дырки — булевы операции дадут мусор, её надо чинить")
    return 0


def command_split(args: argparse.Namespace) -> int:
    """Разложить файл на отдельные тела — по STL на тело."""
    path = _find(args.stl)
    out = Path(args.out)
    stem = path.stem.lower()
    for part in parts(read_stl(path)):
        target = write_stl(out / f"{stem}-part-{part.number}.stl", part.triangles)
        print(f"{target}  {part.describe()}")
    return 0


def command_preview(args: argparse.Namespace) -> int:
    """Отрендерить PNG по каждому чехлу — чтобы посмотреть глазами."""
    if args.slug:
        case(args.slug)
    out = Path(args.out)
    for item in CASES:
        if args.slug and item.slug != args.slug:
            continue
        triangles = solid.to_triangles(item.build())
        flat = triangles.reshape(-1, 3)
        centre = (flat.min(axis=0) + flat.max(axis=0)) / 2
        for name, offset in preview.CAMERAS.items():
            target = out / f"{item.slug}-{name}.png"
            preview.render(
                triangles,
                target,
                eye=tuple(centre + np.asarray(offset)),
                target=tuple(centre),
                size=(args.width, args.height),
            )
            print(target)
    return 0


def command_fit(args: argparse.Namespace) -> int:
    """Показать, как прибор садится в чехол и видно ли его кнопки."""
    item = case(args.slug)
    device = devices.device_for(item.slug)
    if device is None:
        print(f"{item.slug}: модели прибора нет, сверять не с чем")
        return 1

    finished = solid.to_triangles(item.build())
    inside = solid.from_triangles(devices.seated(device))
    print(f"{item.title} + {device.title}")
    print(f"  прибор {inside.volume() / 1000:.1f} см³, кнопок {len(devices.buttons(device))}")
    closed = 0
    for button in devices.buttons(device):
        start, direction = devices.probe(device, button)
        edges = [
            devices.probe(device, button, at=button.along[0]),
            devices.probe(device, button, at=button.along[1]),
        ]
        reach = [
            ray_distances(finished, point[None, :], way)[0]
            for point, way in [(start, direction), *edges]
        ]
        verdict = "окно есть" if min(reach) > 280.0 else "ЗАКРЫТА"
        closed += min(reach) <= 280.0
        print(f"  {button.describe()}  → {verdict}")
    print("  все кнопки открыты" if not closed else f"  закрытых кнопок: {closed}")
    return 0 if not closed else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cycling-cases",
        description="Чехлы для велокомпьютеров Garmin: сборка и разбор моделей",
    )
    parser.add_argument("--version", action="version", version=__version__)
    commands = parser.add_subparsers(dest="command", required=True)

    listing = commands.add_parser("list", help="перечислить чехлы")
    listing.set_defaults(handler=command_list)

    builder = commands.add_parser("build", help="собрать чехлы в каталог")
    builder.add_argument("--out", default="dist", help="куда складывать STL")
    builder.add_argument("--slug", default=None, help="собрать только один чехол")
    builder.set_defaults(handler=command_build)

    inspector = commands.add_parser("inspect", help="что лежит в STL: тела, размеры, замкнутость")
    inspector.add_argument("stl", help="путь к файлу или имя файла в input_data")
    inspector.set_defaults(handler=command_inspect)

    splitter = commands.add_parser("split", help="разложить STL на отдельные тела")
    splitter.add_argument("stl", help="путь к файлу или имя файла в input_data")
    splitter.add_argument("--out", default="dist/parts", help="куда складывать тела")
    splitter.set_defaults(handler=command_split)

    fitting = commands.add_parser("fit", help="как прибор садится в чехол и видно ли кнопки")
    fitting.add_argument("slug", help="имя чехла, например garmin-840")
    fitting.add_argument("--source", action="store_true", help=argparse.SUPPRESS)
    fitting.set_defaults(handler=command_fit)

    viewer = commands.add_parser("preview", help="отрендерить превью каждого чехла")
    viewer.add_argument("--out", default="preview", help="куда складывать PNG")
    viewer.add_argument("--slug", default=None, help="только один чехол")
    viewer.add_argument("--width", type=int, default=900)
    viewer.add_argument("--height", type=int, default=700)
    viewer.set_defaults(handler=command_preview)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.handler(args))
