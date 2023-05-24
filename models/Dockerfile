from python:3.9-slim-buster as BASE

WORKDIR /model

COPY requirements.docker.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

ENV MODULES_PATH "/model"

COPY . .
