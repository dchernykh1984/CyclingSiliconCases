"""Каталог чехлов и сборка."""

from __future__ import annotations

import filecmp
from pathlib import Path

import pytest

from cycling_cases import cases
from cycling_cases.cases import CASES, MODELS, build, case, render_plan, repository_root
from cycling_cases.cli import build_parser, main
from cycling_cases.openscad import RenderTask, command, scad_literal


def test_repository_root_is_found_by_its_sources() -> None:
    # Корень ищется по наличию `input_data`, а не отсчётом шагов от пакета:
    # при установке колесом пакет лежит в site-packages, и шаги промахиваются.
    assert (repository_root() / "input_data").is_dir()


def test_repository_root_falls_back_to_the_working_directory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Пакет, установленный колесом, лежит вдали от исходников: тогда корень
    # ищется от рабочего каталога. Эту ветку легко сломать и не заметить,
    # потому что в режиме разработки она никогда не выполняется.
    real = repository_root()
    monkeypatch.setattr(cases, "PACKAGE_ROOT", tmp_path / "site-packages" / "cycling_cases")
    monkeypatch.chdir(real)
    assert repository_root() == real


def test_slugs_are_unique_and_tidy() -> None:
    slugs = [item.slug for item in CASES]
    assert len(slugs) == len(set(slugs))
    for slug in slugs:
        assert slug == slug.lower()
        assert " " not in slug, "имя уходит в имя файла, пробелам там не место"


def test_case_lookup() -> None:
    assert case("garmin-830").computer == "Garmin Edge 830"
    with pytest.raises(KeyError):
        case("garmin-1030")


def test_filenames_are_named_after_the_case() -> None:
    assert {item.filename for item in CASES} == {"garmin-830.stl", "garmin-840.stl"}


def test_models_of_modified_cases_are_shipped() -> None:
    # Пока чехол уходит копией, модели у него нет; как только появится —
    # она должна лежать в пакете, иначе сборка упадёт уже в CI.
    for item in CASES:
        model = item.model
        if item.modified:
            assert model is not None and model.is_file(), f"нет модели для {item.slug}"
            assert model.parent == MODELS


def test_plan_covers_exactly_the_modified_cases() -> None:
    planned = {task.filename for task in render_plan()}
    assert planned == {item.filename for item in CASES if item.modified}


def test_build_copies_sources_while_there_are_no_edits(tmp_path: Path) -> None:
    # Первый релиз отдаёт ровно то, что лежит в `input_data`: файл копируется
    # байт в байт, без прогона через OpenSCAD, — иначе «то же самое»
    # превратилось бы в пересохранённую сетку.
    built = build(tmp_path)
    assert {path.name for path in built} == {item.filename for item in CASES}
    for item in CASES:
        if not item.modified:
            assert filecmp.cmp(item.source, tmp_path / item.filename, shallow=False)


def test_scad_literals_are_escaped() -> None:
    assert scad_literal("UBT") == '"UBT"'
    assert scad_literal('он сказал "да"') == '"он сказал \\"да\\""'
    assert scad_literal(True) == "true"
    assert scad_literal(2.5) == "2.5"


def test_openscad_command_asks_for_the_fast_engine() -> None:
    task = RenderTask(filename="x.stl", model=Path("model.scad"), definitions={"text_size": 5})
    line = command(Path("out/x.stl"), task, known=frozenset({"manifold", "binstl"}))
    assert "--backend=manifold" in line
    assert "--export-format=binstl" in line
    assert "-D" in line and "text_size=5" in line


def test_old_openscad_gets_no_unknown_flags() -> None:
    # Сборка 2021 года о таких ключах не знает и просто не запустится.
    task = RenderTask(filename="x.stl", model=Path("model.scad"))
    line = command(Path("out/x.stl"), task)
    assert [flag for flag in line if flag.startswith("--")] == []


def test_cli_knows_its_commands() -> None:
    parser = build_parser()
    for name in ("list", "build", "inspect", "split", "preview"):
        assert parser.parse_args([name, *(["x.stl"] if name in {"inspect", "split"} else [])])


def test_cli_lists_cases(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["list"]) == 0
    printed = capsys.readouterr().out
    for item in CASES:
        assert item.slug in printed
