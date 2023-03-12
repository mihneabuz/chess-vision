dev: model-base
	FILE_SERVER_TOKEN=$$(head -c 100 /dev/random | md5) docker compose -f dev-compose.yml up --build

model-base:
	docker build -t model-base ./models
