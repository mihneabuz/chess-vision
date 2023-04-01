mod board;
mod engine;

use std::io::Result;
use std::net::SocketAddr;
use std::sync::{Arc, Mutex};

use axum::extract::State;
use axum::routing::{get, post};
use axum::{Json, Router, Server};

use serde::{Deserialize, Serialize};

#[derive(Deserialize, Debug)]
pub struct Request {
    #[serde(with = "serde_bytes")]
    pub pieces: Vec<u8>,
    pub black: bool,
}

#[derive(Serialize, Debug)]
pub struct Response {
    pub success: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<String>,
}

impl Response {
    fn error() -> Self {
        Self {
            success: false,
            data: None,
        }
    }

    fn from_data(data: Option<String>) -> Self {
        Self {
            success: data.is_some(),
            data,
        }
    }
}

type Engine = Arc<Mutex<engine::Engine>>;

#[tokio::main]
async fn main() -> Result<()> {
    let engine = Arc::new(Mutex::new(engine::Engine::new()));

    let app = Router::new()
        .route("/running", get(|| async { "yes" }))
        .route("/generate", post(generate))
        .with_state(engine);

    Server::bind(&SocketAddr::from(([0, 0, 0, 0], 80)))
        .serve(app.into_make_service())
        .await
        .unwrap();

    Ok(())
}

async fn generate(State(engine): State<Engine>, Json(payload): Json<Request>) -> Json<Response> {
    println!("{:?}", payload);

    let fen = match board::decode(payload) {
        Some(fen) => fen,
        None => return Json(Response::error()),
    };

    println!("{:?}", fen);

    let data = engine.lock().unwrap().generate(fen).ok();

    Json(Response::from_data(data))
}
