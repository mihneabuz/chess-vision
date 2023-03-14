dev: model-base
	FILE_SERVER_TOKEN=$$(head -c 1000 /dev/random | md5sum | cut -d' ' -f1) docker compose -f dev-compose.yml up --build

model-base:
	docker buildx build -t model-base ./models
