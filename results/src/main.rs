mod utils;

use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::{Arc, Mutex};

use axum::extract::{Path, State};
use axum::routing::get;
use axum::{Json, Router, Server};

use futures::StreamExt;
use lapin::options::BasicConsumeOptions;
use lapin::types::FieldTable;
use lapin::{self, Connection, ConnectionProperties, Consumer};
use serde::{Deserialize, Serialize};
use tokio_retry::Retry;

#[derive(Deserialize, Debug)]
pub struct Message {
    pub id: String,
    pub hash: String,
    #[serde(with = "serde_bytes")]
    pub data: Vec<u8>,
}

#[derive(Deserialize, Debug)]
pub struct ErrorMessage {
    pub id: String,
    pub message: String,
}

#[derive(Serialize, Debug)]
#[serde(rename_all = "snake_case")]
pub enum PayloadResult {
    #[serde(with = "serde_bytes")]
    Pieces(Vec<u8>),
    Error(String)
}

#[derive(Serialize, Debug)]
pub struct Payload {
    pub success: bool,
    #[serde(flatten)]
    pub result: PayloadResult
}

#[derive(Serialize, Debug)]
pub struct Response {
    pub done: bool,
    #[serde(flatten)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub payload: Option<Payload>
}

type Results = Arc<Mutex<HashMap<String, Result<[u8; 64], String>>>>;

#[tokio::main]
async fn main() -> lapin::Result<()> {
    let results: Results = Arc::new(Mutex::new(HashMap::new()));

    let results_ref = Arc::clone(&results);
    tokio::spawn(async move {
        let mut consumer = get_consumer(&utils::amqp_addr(), &utils::queue())
            .await
            .expect("Could not connect to final stage");

        while let Some(delivery) = consumer.next().await {
            if let Ok(Ok(message)) = delivery.map(|d| serde_json::from_slice::<Message>(&d.data)) {
                results_ref
                    .lock()
                    .unwrap()
                    .insert(message.id, Ok(message.data.try_into().unwrap()));
            }
        }
    });

    let results_ref = Arc::clone(&results);
    tokio::spawn(async move {
        let mut consumer = get_consumer(&utils::amqp_addr(), &utils::fail_queue())
            .await
            .expect("Could not connect to fail queue");

        while let Some(delivery) = consumer.next().await {
            if let Ok(Ok(message)) = delivery.map(|d| serde_json::from_slice::<ErrorMessage>(&d.data)) {
                results_ref
                    .lock()
                    .unwrap()
                    .insert(message.id, Err(message.message));
            }
        }
    });

    let app = Router::new()
        .route("/running", get(|| async { "yes" }))
        .route("/:id", get(root))
        .with_state(results);

    Server::bind(&SocketAddr::from(([0, 0, 0, 0], 80)))
        .serve(app.into_make_service())
        .await
        .unwrap();

    Ok(())
}

async fn root(State(results): State<Results>, Path(id): Path<String>) -> Json<Response> {
    let item = results.lock().unwrap().remove(&id);
    match item {
        Some(Ok(pieces)) => Json(Response {
            done: true,
            payload: Some(Payload {
                success: true,
                result: PayloadResult::Pieces(pieces.into())
            })
        }),
        Some(Err(message)) => Json(Response {
            done: true,
            payload: Some(Payload {
                success: false,
                result: PayloadResult::Error(message)
            })
        }),
        None => Json(Response {
            done: false,
            payload: None
        }),
    }
}

async fn get_consumer(addr: &str, queue: &str) -> lapin::Result<Consumer> {
    let retry_strategy = utils::retry_strategy();

    let conn = Retry::spawn(retry_strategy.clone(), || async {
        Connection::connect(addr, ConnectionProperties::default()).await
    })
    .await?;

    println!("connected to message queue");

    let (opts, table) = (BasicConsumeOptions::default(), FieldTable::default());
    let consumer = Retry::spawn(retry_strategy, || async {
        let channel = conn.create_channel().await?;
        channel.basic_consume(queue, "service", opts, table.clone()).await
    })
    .await?;

    Ok(consumer)
}
