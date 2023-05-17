FROM rust:latest AS deps

COPY ./shared ./shared

WORKDIR app

COPY ./upload/Cargo.toml .
COPY ./upload/Cargo.lock .

RUN mkdir src
RUN echo 'fn main() { panic!("THIS SHOULD NOT RUN") }' > src/main.rs

RUN cargo build --release

RUN rm -r ./src/*


from deps as builder

COPY ./upload/src ./src

RUN touch -a -m ./src/main.rs

RUN cargo build --release


from debian:bullseye-slim as runner

WORKDIR /app

COPY --from=builder /app/target/release/upload ./upload

CMD [ "./upload" ]
