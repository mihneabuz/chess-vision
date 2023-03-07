dev: model-base
	docker compose -f dev-compose.yml up --build

model-base:
	docker build -t model-base ./models
