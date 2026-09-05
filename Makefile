.PHONY: setup pretorneo backtest notebook test clean

setup:
	pip install -r requirements.txt
	python scripts/build_squad_values.py

pretorneo:
	python scripts/run_pretournament.py --sims 30000

backtest:
	python src/backtest.py

notebook:
	python scripts/build_notebook.py

test:
	pytest -q

clean:
	rm -rf outputs/*.csv outputs/*.json src/__pycache__ .pytest_cache
