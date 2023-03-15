mod utils;

use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use futures::StreamExt;
use lapin::options::{BasicConsumeOptions, QueueDeclareOptions};
use lapin::types::FieldTable;
use lapin::{Connection, ConnectionProperties, Result, Consumer};

#[tokio::main]
async fn main() -> Result<()> {
    let map: HashMap<String, [u8; 64]> = HashMap::new();
    let results = RwLock::new(Arc::new(map));

    let mut consumer = get_consumer().await?;

    while let Some(delivery) = consumer.next().await {
        println!("{:?}", delivery);
    }

    Ok(())
}

async fn get_consumer() -> Result<Consumer> {
    let conn = loop {
        let conn = Connection::connect(&utils::amqp_addr(), ConnectionProperties::default()).await;
        if let Ok(conn) = conn {
            break Box::leak(Box::new(conn));
        }

        tokio::time::sleep(std::time::Duration::from_millis(200)).await;
    };

    let channel = conn.create_channel().await?;
    println!("connected to message queue");

    let message_queue = utils::queue();
    let (opts, table) = (QueueDeclareOptions::default(), FieldTable::default());
    channel
        .queue_declare(&message_queue, opts, table.clone())
        .await?;

    let opts = BasicConsumeOptions::default();
    let consumer = loop {
        if let Ok(consumer) = channel
            .basic_consume(&message_queue, "service", opts, table.clone())
            .await
        {
            break consumer;
        }

        tokio::time::sleep(std::time::Duration::from_millis(200)).await;
    };

    Ok(consumer)
}
