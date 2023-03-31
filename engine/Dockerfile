FROM rust:latest AS base

WORKDIR app

COPY ./Cargo.toml .
COPY ./Cargo.lock .

RUN mkdir src
RUN echo 'fn main() { panic!("THIS SHOULD NOT RUN") }' > src/main.rs

RUN cargo build --release

RUN rm -r ./src/*

from base as runner

COPY ./src ./src

RUN touch -a -m ./src/main.rs

RUN cargo build --release

CMD [ "./target/release/engine" ]
