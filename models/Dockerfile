from python:3.10-buster as BASE

RUN apt update
RUN apt install -y libgl1

WORKDIR /model

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
