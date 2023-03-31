use std::io::Result;
use std::net::SocketAddr;

use axum::routing::{get, post};
use axum::{Json, Router, Server};

use serde::{Deserialize, Serialize};

#[derive(Deserialize, Debug)]
struct MoveRequest {
    #[serde(with = "serde_bytes")]
    pieces: Vec<u8>,
    black: bool,
}

#[derive(Serialize, Debug)]
struct MoveResponse {
    success: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    data: Option<String>,
}

#[tokio::main]
async fn main() -> Result<()> {
    let app = Router::new()
        .route("/running", get(|| async { "yes" }))
        .route("/generate", post(generate))
        .with_state(());

    Server::bind(&SocketAddr::from(([0, 0, 0, 0], 80)))
        .serve(app.into_make_service())
        .await
        .unwrap();

    Ok(())
}

async fn generate(Json(payload): Json<MoveRequest>) -> Json<MoveResponse> {
    println!("{:?}", payload);

    if payload.pieces.len() != 64 {
        return Json(MoveResponse {
            success: false,
            data: None,
        });
    }

    Json(MoveResponse {
        success: false,
        data: None,
    })
}



