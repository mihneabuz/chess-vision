mod utils;

use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::{Arc, RwLock};

use axum::extract::{Path, State};
use axum::routing::get;
use axum::{Json, Router, Server};

use futures::StreamExt;
use lapin::options::BasicConsumeOptions;
use lapin::types::FieldTable;
use lapin::{Connection, ConnectionProperties, Consumer, Result};
use serde::{Deserialize, Serialize};
use tokio_retry::Retry;

#[derive(Deserialize, Debug)]
pub struct Message {
    pub id: String,
    pub hash: String,
    #[serde(with = "serde_bytes")]
    pub data: Vec<u8>,
}

#[derive(Serialize, Debug)]
pub struct Response {
    pub success: bool,
    #[serde(with = "serde_bytes")]
    pub pieces: Option<Vec<u8>>,
}

type Results = Arc<RwLock<HashMap<String, [u8; 64]>>>;

#[tokio::main]
async fn main() -> Result<()> {
    let results: Results = Arc::new(RwLock::new(HashMap::new()));

    let results_ref = Arc::clone(&results);
    tokio::spawn(async move {
        let mut consumer = get_consumer().await.expect("Could not connect to final stage");

        while let Some(delivery) = consumer.next().await {
            if let Ok(Ok(message)) = delivery.map(|d| serde_json::from_slice::<Message>(&d.data)) {
                results_ref
                    .write()
                    .unwrap()
                    .insert(message.id, message.data.try_into().unwrap());
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
    match results.read().unwrap().get(&id) {
        Some(result) => Json(Response {
            success: true,
            pieces: Some(result.clone().try_into().unwrap()),
        }),
        None => Json(Response {
            success: false,
            pieces: None,
        }),
    }
}

async fn get_consumer() -> Result<Consumer> {
    let retry_strategy = utils::retry_strategy();

    let conn = Retry::spawn(retry_strategy.clone(), || async {
        Connection::connect(&utils::amqp_addr(), ConnectionProperties::default()).await
    })
    .await?;

    println!("connected to message queue");

    let message_queue = utils::queue();
    let (opts, table) = (BasicConsumeOptions::default(), FieldTable::default());
    let consumer = Retry::spawn(retry_strategy, || async {
        let channel = conn.create_channel().await?;
        channel.basic_consume(&message_queue, "service", opts, table.clone()).await
    })
    .await?;

    Ok(consumer)
}
