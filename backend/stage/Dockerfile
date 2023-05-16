FROM rustlang/rust:nightly AS builder

RUN apt update
RUN apt install -y libpython3.9-dev

COPY ./shared ./shared

WORKDIR app

COPY ./stage .

RUN cargo build --release


from model-base as runner

COPY --from=builder /app/target/release/app /stage

CMD [ "/stage" ]
