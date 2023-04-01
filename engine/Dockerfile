FROM rust:latest AS rust-stockfish

RUN git clone https://github.com/official-stockfish/Stockfish.git
WORKDIR Stockfish
RUN make -C src -j build ARCH=x86-64-modern
RUN mv src/stockfish /
WORKDIR /

FROM rust-stockfish as builder

WORKDIR app

COPY ./Cargo.toml .
COPY ./Cargo.lock .

RUN mkdir src
RUN echo 'fn main() { panic!("THIS SHOULD NOT RUN") }' > src/main.rs

RUN cargo build --release

RUN rm -r ./src/*

from builder as runner

COPY ./src ./src

RUN touch -a -m ./src/main.rs

RUN cargo build --release

ENV STOCKFISH=/stockfish

CMD [ "./target/release/engine" ]
