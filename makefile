dev: model-base
	FILE_SERVER_TOKEN=$$(head -c 1000 /dev/random | md5sum | cut -d' ' -f1) docker compose -f dev-compose.yml up --build

model-base:
	docker buildx build -t model-base ./models

zip-weights:
	zip ./models/weights.zip ./models/board_segmentation_weights ./models/piece_classification_weights

unzip-weights:
	unzip ./models/weights.zip


export FILE_SERVER_TOKEN = d0ddcb5ddaf48ca987672d8a33c26ae8

prod: model-base
	docker swarm init
	docker service create --name registry --publish published=5000,target=5000 registry:2
	docker compose -f prod-compose.yml build
	docker compose -f prod-compose.yml push
	docker stack deploy --compose-file prod-compose.yml chess-vision

prod-cleanup:
	docker stack rm chess-vision
	docker service rm registry
	docker swarm leave --force
