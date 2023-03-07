from model-base as model-with-rust

RUN apt update
RUN apt install -y

WORKDIR rust

RUN curl https://sh.rustup.rs -sSf -o rustup
RUN chmod +x rustup
RUN ./rustup -y


from model-with-rust as model-with-rust-deps

WORKDIR app

COPY ./Cargo.toml .
COPY ./Cargo.lock .

RUN mkdir src
RUN echo 'fn main() { panic!("THIS SHOULD NOT RUN") }' > src/main.rs

RUN PATH="$HOME/.cargo/bin:$PATH" cargo build --release

RUN rm -r ./src/*


from model-with-rust-deps as runner

COPY ./src ./src

RUN touch -a -m ./src/main.rs

RUN PATH="$HOME/.cargo/bin:$PATH" cargo build --release

CMD [ "./target/release/app" ]
