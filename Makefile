.PHONY: up down reset etl econometrics simulate dashboard logs shell

## ── Docker lifecycle ──────────────────────────────────────────────────────────
up:
	docker compose up -d --build

down:
	docker compose down

reset:
	docker compose down -v --remove-orphans
	docker compose up -d --build

## ── Pipeline phases (run inside runner container) ────────────────────────────
etl:
	docker compose run --rm dashboard python run_pipeline.py --phase etl

econometrics:
	docker compose run --rm dashboard python run_pipeline.py --phase econometrics

copula:
	docker compose run --rm dashboard python run_pipeline.py --phase copula

simulate:
	docker compose run --rm dashboard python run_pipeline.py --phase simulate

full:
	docker compose run --rm dashboard python run_pipeline.py --phase all

## ── Local (no Docker) ────────────────────────────────────────────────────────
install:
	pip install -r requirements.txt

local-etl:
	python run_pipeline.py --phase etl

local-full:
	python run_pipeline.py --phase all

local-dashboard:
	streamlit run dashboard/app.py

## ── Utilities ─────────────────────────────────────────────────────────────────
logs:
	docker compose logs -f

shell:
	docker compose exec dashboard bash
