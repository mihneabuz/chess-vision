use std::sync::mpsc::Sender;

use bytes::Bytes;
use futures::stream::StreamExt;
use lapin::options::{BasicPublishOptions, QueueDeclareOptions};
use lapin::{options::BasicConsumeOptions, types::FieldTable, Connection};
use lapin::{BasicProperties, Result};
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc::UnboundedReceiver;

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

pub async fn consumer(conn: &Connection, sender: Sender<(Message, Bytes)>) -> Result<()> {
    let channel = conn.create_channel().await?;

    let message_queue = utils::current_queue();
    let mut consumer = loop {
        if let Ok(consumer) = channel
            .basic_consume(
                &message_queue,
                "service",
                BasicConsumeOptions::default(),
                FieldTable::default(),
            )
            .await
        {
            break consumer;
        }
    };

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

pub async fn publisher(conn: &Connection, mut receiver: UnboundedReceiver<Message>) -> Result<()> {
    let channel = conn.create_channel().await?;

    let message_queue = utils::next_queue();
    channel
        .queue_declare(&message_queue, QueueDeclareOptions::default(), FieldTable::default())
        .await?;

    loop {
        let message = receiver.recv().await.unwrap();
        let payload = serde_json::to_vec(&message).unwrap();
        println!("{:?}", message);
        channel
            .basic_publish(
                "",
                &message_queue,
                BasicPublishOptions::default(),
                &payload,
                BasicProperties::default(),
            )
            .await?
            .await?;
    }
}
