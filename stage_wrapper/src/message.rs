use std::sync::mpsc::Sender;

use bytes::Bytes;
use futures::stream::StreamExt;
use lapin::options::{BasicPublishOptions, QueueDeclareOptions};
use lapin::{options::BasicConsumeOptions, types::FieldTable, Connection};
use lapin::{BasicProperties, ConnectionProperties, Result};
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc::UnboundedReceiver;
use tokio_retry::Retry;

use crate::utils;

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
pub struct FailMessage {
    pub id: String,
    pub message: String,
}

impl FailMessage {
    pub fn new(id: String, message: String) -> Self {
        Self { id, message }
    }
}

pub type StageResult = std::result::Result<Message, FailMessage>;

pub async fn connect() -> Result<Connection> {
    Connection::connect(&utils::amqp_addr(), ConnectionProperties::default()).await
}

pub async fn consumer(conn: &Connection, sender: Sender<(Message, Bytes)>) -> Result<()> {
    let message_queue = utils::current_queue();
    let mut consumer = Retry::spawn(utils::retry_strategy(), || async {
        conn.create_channel()
            .await?
            .basic_consume(
                &message_queue,
                "service",
                BasicConsumeOptions::default(),
                FieldTable::default(),
            )
            .await
    })
    .await?;

    while let Some(delivery) = consumer.next().await {
        if let Ok(delivery) = delivery {
            if let Ok(message) = serde_json::from_slice::<Message>(&delivery.data) {
                let image_bytes = crate::file::fetch_image(&message.id).await.unwrap();

                if format!("{:?}", md5::compute(&image_bytes)) != message.hash {
                    println!("IMAGE HASH WRONG!");
                }

                sender.send((message, image_bytes)).unwrap();
            }
        }
    }

    Ok(())
}

pub async fn publisher(conn: &Connection, mut receiver: UnboundedReceiver<StageResult>) -> Result<()> {
    let channel = conn.create_channel().await?;

    let message_queue = utils::next_queue();
    channel
        .queue_declare(&message_queue, QueueDeclareOptions::default(), FieldTable::default())
        .await?;

    let fail_queue = utils::fail_queue();
    channel
        .queue_declare(&fail_queue, QueueDeclareOptions::default(), FieldTable::default())
        .await?;

    loop {
        let message = receiver.recv().await.unwrap();
        match message {
            Ok(res) => {
                channel
                    .basic_publish(
                        "",
                        &message_queue,
                        BasicPublishOptions::default(),
                        &serde_json::to_vec(&res).unwrap(),
                        BasicProperties::default(),
                    )
                    .await?
                    .await?;
            }

            Err(err) => {
                channel
                    .basic_publish(
                        "",
                        &message_queue,
                        BasicPublishOptions::default(),
                        &serde_json::to_vec(&err).unwrap(),
                        BasicProperties::default(),
                    )
                    .await?
                    .await?;
            }
        }
    }
}
