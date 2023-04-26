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
    #[serde(flatten)]
    pub data: Option<Payload>,
}

#[derive(Serialize, Debug)]
pub struct Payload {
    #[serde(rename = "move")]
    pub mov: String,
    pub from: u8,
    pub to: u8
}

impl Response {
    fn error() -> Self {
        Self {
            success: false,
            data: None,
        }
    }

    fn from_payload(payload: Payload) -> Self {
        Self {
            success: true,
            data: Some(payload)
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
    let Some(fen) = board::decode_board(payload.pieces, payload.black) else {
        return Json(Response::error());
    };

    println!("{:?}", fen);

    let Some(mov) = engine.lock().unwrap().generate(fen).ok() else {
        return Json(Response::error());
    };

    let Some((from, to)) = board::decode_move(&mov, payload.black) else {
        return Json(Response::error());
    };

    Json(Response::from_payload(Payload { mov, from, to }))
}
