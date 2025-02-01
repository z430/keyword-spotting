CONTAINER_NAME := kws-app:dev

DEV_CMD := docker run -it --rm --gpus=all --privileged=true --net=host -v /media/ssd:/media/ssd


build-container: Dockerfile
	docker build -f $< -t ${CONTAINER_NAME} .

run-dev: build-container
	${DEV_CMD} ${CONTAINER_NAME}
