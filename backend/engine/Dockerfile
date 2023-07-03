FROM rust:1.70 AS stockfish-builder

RUN git clone https://github.com/official-stockfish/Stockfish.git
WORKDIR Stockfish
RUN make -C src -j build ARCH=x86-64-modern


FROM rust:latest as rust-deps

WORKDIR app

COPY ./Cargo.toml .
COPY ./Cargo.lock .

RUN mkdir src
RUN echo 'fn main() { panic!("THIS SHOULD NOT RUN") }' > src/main.rs

RUN cargo build --release

RUN rm -r ./src/*


from rust-deps as rust-builder

COPY ./src ./src

RUN touch -a -m ./src/main.rs

RUN cargo build --release


from debian:bullseye-slim as runner

WORKDIR /app

COPY --from=stockfish-builder /Stockfish/src/stockfish /stockfish
COPY --from=rust-builder /app/target/release/engine /engine

ENV STOCKFISH=/stockfish

CMD [ "/engine" ]
