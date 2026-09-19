"""Командная строка: собрать чехлы, посмотреть исходник, сделать превью."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from . import __version__
from .cases import CASES, build, case, sources_dir
from .mesh import edge_counts, health, parts, read_stl
from .openscad import RenderTask, run

PREVIEW_CAMERAS: dict[str, str] = {
    "top": "0,0,0,0,0,0,0",
    "iso": "0,0,0,60,0,25,0",
    "front": "0,0,0,90,0,0,0",
}
"""Камеры для превью: сверху, в три четверти и сбоку.

Три ракурса — минимум, на котором видно и раскладку, и стенки. Один вид
врёт: деталь, съехавшая по высоте, сверху выглядит идеально.
"""


def command_build(args: argparse.Namespace) -> int:
    for path in build(Path(args.out), binary=args.openscad):
        print(path)
    return 0


def command_list(args: argparse.Namespace) -> int:
    for item in CASES:
        state = "своя модель" if item.modified else "копия исходника"
        print(f"{item.slug:12s}  {item.title:28s}  {state}")
        if item.comment:
            print(f"{'':12s}  {item.comment}")
    return 0


def command_inspect(args: argparse.Namespace) -> int:
    """Показать, что лежит в файле: тела, размеры, замкнутость.

    Скачанные модели часто оказываются целой раскладкой на стол, и первое,
    что нужно знать перед правкой, — из скольких тел она состоит.
    """
    path = Path(args.stl)
    if not path.exists():
        candidate = sources_dir() / args.stl
        path = candidate if candidate.exists() else path
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


def command_preview(args: argparse.Namespace) -> int:
    """Отрендерить PNG по каждому чехлу — чтобы посмотреть глазами."""
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    viewer = Path(args.viewer) if args.viewer else None
    for item in CASES:
        model = item.model
        for name, camera in PREVIEW_CAMERAS.items():
            target = out / f"{item.slug}-{name}.png"
            if model is None:
                source = item.source.resolve()
                scratch = out / f"{item.slug}.scad"
                scratch.write_text(f'import("{source}");\n', encoding="utf-8")
                task = RenderTask(filename=target.name, model=scratch)
            else:
                task = RenderTask(filename=target.name, model=model, definitions=item.definitions)
            run(
                target,
                task,
                binary=args.openscad,
                render=True,
                extra=[
                    f"--imgsize={args.width},{args.height}",
                    f"--camera={camera}",
                    "--viewall",
                    "--autocenter",
                    "--colorscheme=Tomorrow",
                ],
            )
            print(target)
    if viewer is not None:  # pragma: no cover - удобство, не логика
        print(f"смотреть: {viewer}")
    return 0


def command_split(args: argparse.Namespace) -> int:
    """Разложить файл на отдельные тела — по STL на тело."""
    from .mesh import write_stl

    path = Path(args.stl)
    if not path.exists():
        candidate = sources_dir() / args.stl
        path = candidate if candidate.exists() else path
    out = Path(args.out)
    stem = case(args.slug).slug if args.slug else path.stem.lower()
    for part in parts(read_stl(path)):
        target = write_stl(out / f"{stem}-part-{part.number}.stl", part.triangles)
        print(f"{target}  {part.describe()}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cycling-cases",
        description="Чехлы для велокомпьютеров Garmin: сборка и разбор моделей",
    )
    parser.add_argument("--version", action="version", version=__version__)
    parser.add_argument(
        "--openscad",
        default=None,
        help="путь к бинарю openscad (по умолчанию из PATH или $OPENSCAD)",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    listing = commands.add_parser("list", help="перечислить чехлы")
    listing.set_defaults(handler=command_list)

    builder = commands.add_parser("build", help="собрать все чехлы в каталог")
    builder.add_argument("--out", default="dist", help="куда складывать STL")
    builder.set_defaults(handler=command_build)

    inspector = commands.add_parser("inspect", help="что лежит в STL: тела, размеры, замкнутость")
    inspector.add_argument("stl", help="путь к файлу или имя файла в input_data")
    inspector.set_defaults(handler=command_inspect)

    splitter = commands.add_parser("split", help="разложить STL на отдельные тела")
    splitter.add_argument("stl", help="путь к файлу или имя файла в input_data")
    splitter.add_argument("--out", default="dist/parts", help="куда складывать тела")
    splitter.add_argument("--slug", default=None, help="имя чехла для префикса файлов")
    splitter.set_defaults(handler=command_split)

    preview = commands.add_parser("preview", help="отрендерить превью каждого чехла")
    preview.add_argument("--out", default="preview", help="куда складывать PNG")
    preview.add_argument("--width", type=int, default=900)
    preview.add_argument("--height", type=int, default=800)
    preview.add_argument("--viewer", default=None, help=argparse.SUPPRESS)
    preview.set_defaults(handler=command_preview)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    handler = args.handler
    return int(handler(args))
