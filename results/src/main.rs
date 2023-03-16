mod utils;

use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use futures::StreamExt;
use lapin::options::{BasicConsumeOptions, QueueDeclareOptions};
use lapin::types::FieldTable;
use lapin::{Connection, ConnectionProperties, Consumer, Result};
use serde::{Deserialize, Serialize};
use tokio_retry::Retry;

#[derive(Deserialize, Serialize, Debug)]
pub struct Message {
    pub id: String,
    pub hash: String,
    #[serde(with = "serde_bytes")]
    pub data: Vec<u8>,
}

#[tokio::main]
async fn main() -> Result<()> {
    let map: HashMap<String, [u8; 64]> = HashMap::new();
    let results = Arc::new(RwLock::new(map));

    let mut consumer = get_consumer().await?;

    while let Some(delivery) = consumer.next().await {
        if let Ok(parsed) = delivery.map(|d| serde_json::from_slice::<Message>(&d.data)) {
            if let Ok(message) = parsed {
                results
                    .write()
                    .unwrap()
                    .insert(message.id, message.data.try_into().unwrap())
                    .unwrap();
            }
        }
    }

    Ok(())
}

async fn get_consumer() -> Result<Consumer> {
    let retry_strategy = utils::retry_strategy();

    let conn = Retry::spawn(retry_strategy.clone(), || async {
        Connection::connect(&utils::amqp_addr(), ConnectionProperties::default()).await
    })
    .await?;

    let channel = conn.create_channel().await?;
    println!("connected to message queue");

    let message_queue = utils::queue();
    let (opts, table) = (QueueDeclareOptions::default(), FieldTable::default());
    channel.queue_declare(&message_queue, opts, table.clone()).await?;

    let opts = BasicConsumeOptions::default();
    let consumer = Retry::spawn(retry_strategy, || async {
        channel.basic_consume(&message_queue, "service", opts, table.clone()).await
    })
    .await?;

    Ok(consumer)
}
