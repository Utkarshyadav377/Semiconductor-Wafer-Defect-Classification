.PHONY: install train evaluate gradcam test lint

install:
	pip install -e . -r requirements-dev.txt

train:
	python scripts/train.py --config configs/default.yaml

evaluate:
	python scripts/evaluate.py --config configs/default.yaml

gradcam:
	python scripts/gradcam.py --config configs/default.yaml

test:
	pytest -q

lint:
	ruff check .
