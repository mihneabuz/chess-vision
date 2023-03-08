mod file;
mod message;
mod utils;
mod wrapper;

use bytes::Bytes;
use futures::join;
use lapin::{Connection, ConnectionProperties, Result};
use message::{consumer, publisher, Message};

use crate::file::fetch_file_sync;

#[tokio::main]
async fn main() -> Result<()> {
    println!("starting stage {:?}", utils::model_name());

    let conn = loop {
        let conn = Connection::connect(&utils::amqp_addr(), ConnectionProperties::default()).await;
        if let Ok(conn) = conn {
            break Box::leak(Box::new(conn));
        }

        tokio::time::sleep(std::time::Duration::from_millis(200)).await;
    };

    println!("connected to message queue");

    let (consume_sender, consume_receiver) = std::sync::mpsc::channel::<(Message, Bytes)>();
    let (publish_sender, publish_receiver) = tokio::sync::mpsc::unbounded_channel::<Message>();

    let worker = tokio::task::spawn_blocking(move || {
        let service = wrapper::create(utils::service_type()).expect("could not load service");

        let resource = service.resource().expect("could not get service resource");
        let data = fetch_file_sync(&resource).expect("could not fetch resource for service");
        service.configure(&data).expect("could not configure service");

        let mut messages = Vec::new();
        loop {
            while let Ok((message, image)) = consume_receiver.try_recv() {
                messages.push((message, image));
            }

            if messages.len() > 0 {
                println!("processing {:?} images", messages.len());

                let data = messages
                    .iter()
                    .map(|(message, image)| -> (&[u8], &[u8]) { (&image, &message.data) })
                    .collect::<Vec<(&[u8], &[u8])>>();

                let result = service.process(&data).unwrap();

                messages.drain(..).map(|m| m.0).zip(result.into_iter()).for_each(
                    |(message, data)| {
                        publish_sender.send(Message::new(message.id, message.hash, data)).unwrap();
                    },
                );
            }

            std::thread::sleep(std::time::Duration::from_millis(100));
        }
    });

    let h1 = tokio::spawn(consumer(conn, consume_sender));
    let h2 = tokio::spawn(publisher(conn, publish_receiver));

    join! { h1, h2, worker };

    Ok(())
}
