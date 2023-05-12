mod file;
mod message;
mod utils;
mod wrapper;

use bytes::Bytes;
use futures::join;
use lapin::Result;
use message::{connect, consumer, publisher, FailMessage, Message, StageResult};

use crate::file::fetch_file_sync;

use tokio_retry::Retry;

#[tokio::main]
async fn main() -> Result<()> {
    let stage = utils::model_name();
    println!("starting stage {:?}", stage);

    let conn = Box::leak(Box::new(Retry::spawn(utils::retry_strategy(), connect).await?));

    println!("connected to message queue");

    let (consume_sender, consume_receiver) = std::sync::mpsc::channel::<(Message, Bytes)>();
    let (publish_sender, publish_receiver) = tokio::sync::mpsc::unbounded_channel::<StageResult>();

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
                        let result = if data.len() > 0 {
                            Ok(Message::new(message.id, message.hash, data))
                        } else {
                            let err = format!("processing failed on {}", stage);
                            Err(FailMessage::new(message.id, err))
                        };

                        publish_sender.send(result).unwrap();
                    },
                );
            }

            std::thread::sleep(std::time::Duration::from_millis(100));
        }
    });

    let h2 = tokio::spawn(publisher(conn, publish_receiver));
    let h1 = tokio::spawn(consumer(conn, consume_sender));

    join! { h1, h2, worker };

    Ok(())
}
