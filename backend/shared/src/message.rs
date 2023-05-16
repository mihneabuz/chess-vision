use std::iter;
use std::time::Duration;

use futures::{stream::StreamExt, Future};
use lapin::{
    options::{BasicAckOptions, BasicConsumeOptions, BasicPublishOptions, QueueDeclareOptions},
    types::FieldTable,
    BasicProperties, Connection, ConnectionProperties, Result,
};
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use tokio::sync::mpsc::Receiver;
use tokio_retry::{
    strategy::{jitter, FixedInterval},
    Retry,
};

#[derive(Deserialize, Serialize, Debug)]
pub struct Message {
    pub id: String,
    pub hash: String,
    #[serde(with = "serde_bytes")]
    pub data: Vec<u8>,
}

impl Message {
    pub fn new(id: String, hash: String, data: Vec<u8>) -> Self {
        Self { id, hash, data }
    }
}

#[derive(Deserialize, Serialize, Debug)]
pub struct ErrorMessage {
    pub id: String,
    pub message: String,
}

impl ErrorMessage {
    pub fn new(id: String, message: String) -> Self {
        Self { id, message }
    }
}

fn retry_strategy() -> iter::Map<FixedInterval, fn(Duration) -> Duration> {
    FixedInterval::from_millis(500).map(jitter)
}

pub async fn connect(addr: &str) -> Result<Connection> {
    Retry::spawn(retry_strategy(), || {
        Connection::connect(addr, ConnectionProperties::default())
    })
    .await
}

pub async fn consumer<T, F, K>(conn: &Connection, queue: &str, f: F) -> Result<()>
where
    F: Fn(T) -> K,
    T: DeserializeOwned,
    K: Future,
{
    let mut consumer = Retry::spawn(retry_strategy(), || async {
        let channel = conn.create_channel().await?;
        let table = FieldTable::default();

        let opts = QueueDeclareOptions::default();
        channel.queue_declare(queue, opts, table.clone()).await?;

        let opts = BasicConsumeOptions::default();
        channel.basic_consume(queue, "service", opts, table).await
    })
    .await?;

    let opts = BasicAckOptions::default();

    while let Some(delivery) = consumer.next().await {
        if let Ok(delivery) = delivery {
            if let Ok(message) = serde_json::from_slice::<T>(&delivery.data) {
                f(message).await;

                Retry::spawn(retry_strategy(), || delivery.acker.ack(opts))
                    .await
                    .expect("could not send ack");
            }
        }
    }

    Ok(())
}

pub async fn simple_publisher<T>(
    conn: &Connection,
    queue: &str,
    mut receiver: Receiver<T>,
) -> Result<()>
where
    T: Serialize,
{
    let channel = Retry::spawn(retry_strategy(), || async {
        let channel = conn.create_channel().await?;

        let table = FieldTable::default();
        let opts = QueueDeclareOptions::default();
        channel.queue_declare(queue, opts, table).await?;

        Result::Ok(channel)
    })
    .await?;

    let opts = BasicPublishOptions::default();
    let props = BasicProperties::default();

    loop {
        let message = receiver.recv().await.unwrap();
        let bytes = &serde_json::to_vec(&message).unwrap();
        channel
            .basic_publish("", queue, opts, bytes, props.clone())
            .await?
            .await?;
    }
}

pub async fn publisher<T, E>(
    conn: &Connection,
    queue: &str,
    fail_queue: &str,
    mut receiver: Receiver<std::result::Result<T, E>>,
) -> Result<()>
where
    T: Serialize,
    E: Serialize,
{
    let channel = Retry::spawn(retry_strategy(), || async {
        let channel = conn.create_channel().await?;
        let table = FieldTable::default();

        let opts = QueueDeclareOptions::default();
        channel.queue_declare(queue, opts, table.clone()).await?;
        channel.queue_declare(fail_queue, opts, table).await?;

        Result::Ok(channel)
    })
    .await?;

    let opts = BasicPublishOptions::default();
    let props = BasicProperties::default();

    loop {
        let message = receiver.recv().await.unwrap();

        let (route, bytes) = match message {
            Ok(message) => (queue, serde_json::to_vec(&message).unwrap()),
            Err(error) => (fail_queue, serde_json::to_vec(&error).unwrap()),
        };

        channel
            .basic_publish("", route, opts, &bytes, props.clone())
            .await?
            .await?;
    }
}
