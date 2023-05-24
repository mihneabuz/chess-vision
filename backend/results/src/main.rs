mod utils;
use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::Arc;
use tokio::sync::Mutex;

use axum::{
    extract::{Path, State},
    routing::get,
    Json, Router, Server,
};
use serde::Serialize;

use shared::message::{connect, consumer, ErrorMessage, Message};

#[derive(Serialize, Debug)]
pub struct Response {
    pub done: bool,
    #[serde(flatten)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub payload: Option<Payload>,
}

#[derive(Serialize, Debug)]
pub struct Payload {
    pub success: bool,
    #[serde(flatten)]
    pub result: PayloadResult,
}

#[derive(Serialize, Debug)]
#[serde(rename_all = "snake_case")]
pub enum PayloadResult {
    #[serde(with = "serde_bytes")]
    Pieces(Vec<u8>),
    Error(String),
}

type Results = Arc<Mutex<HashMap<String, Result<[u8; 64], String>>>>;

#[tokio::main]
async fn main() {
    let results: Results = Arc::new(Mutex::new(HashMap::new()));
    let conn = &*Box::leak(Box::new(connect(&utils::amqp_addr()).await.unwrap()));

    let results_ref = Arc::clone(&results);
    tokio::spawn(async move {
        consumer(conn, &utils::queue(), |message: Message| async {
            let result = message.data.try_into().unwrap();
            results_ref.lock().await.insert(message.id, Ok(result));
        })
        .await
        .expect("queue failed");
    });

    let results_ref = Arc::clone(&results);
    tokio::spawn(async move {
        consumer(conn, &utils::fail_queue(), |error: ErrorMessage| async {
            results_ref.lock().await.insert(error.id, Err(error.message));
        })
        .await
        .expect("error queue failed");
    });

    let app = Router::new()
        .route("/running", get(|| async { "yes" }))
        .route("/:id", get(root))
        .with_state(results);

    Server::bind(&SocketAddr::from(([0, 0, 0, 0], 80)))
        .serve(app.into_make_service())
        .await
        .unwrap();
}

async fn root(State(results): State<Results>, Path(id): Path<String>) -> Json<Response> {
    let item = results.lock().await.remove(&id);
    match item {
        Some(Ok(pieces)) => Json(Response {
            done: true,
            payload: Some(Payload {
                success: true,
                result: PayloadResult::Pieces(pieces.into()),
            }),
        }),
        Some(Err(message)) => Json(Response {
            done: true,
            payload: Some(Payload {
                success: false,
                result: PayloadResult::Error(message),
            }),
        }),
        None => Json(Response {
            done: false,
            payload: None,
        }),
    }
}
