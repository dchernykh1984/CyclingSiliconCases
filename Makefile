UV      ?= uv
DIST    ?= dist
PREVIEW ?= preview

.PHONY: help install lint format typecheck test build preview inspect clean

help:
	@echo "Разработка:"
	@echo "  make install     поставить зависимости (uv) и хуки pre-commit"
	@echo "  make lint        ruff check + ruff format --check"
	@echo "  make typecheck   mypy"
	@echo "  make test        pytest"
	@echo ""
	@echo "Чехлы:"
	@echo "  make build       собрать STL в ./$(DIST)"
	@echo "  make preview     отрендерить PNG в ./$(PREVIEW) (нужен openscad)"
	@echo "  make inspect     показать, из каких тел состоят исходники"
	@echo ""
	@echo "  make clean       удалить ./$(DIST), ./$(PREVIEW) и кэши"

install:
	$(UV) sync
	$(UV) run pre-commit install

lint:
	$(UV) run ruff check .
	$(UV) run ruff format --check .

format:
	$(UV) run ruff format .
	$(UV) run ruff check --fix .

typecheck:
	$(UV) run mypy

test:
	$(UV) run pytest

build:
	$(UV) run cycling-cases build --out $(DIST)

preview:
	$(UV) run cycling-cases preview --out $(PREVIEW)

inspect:
	@for stl in input_data/*.stl; do $(UV) run cycling-cases inspect "$$stl"; done

clean:
	rm -rf $(DIST) $(PREVIEW) .pytest_cache .ruff_cache .mypy_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
