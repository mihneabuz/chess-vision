mod file;
mod utils;
mod wrapper;

use bytes::Bytes;

use file::{fetch_file_sync, fetch_image};
use shared::message::{connect, consumer, publisher, ErrorMessage, Message};

type StageResult = std::result::Result<Message, ErrorMessage>;

#[tokio::main]
async fn main() {
    let (consume_sender, mut consume_receiver) = tokio::sync::mpsc::channel::<(Message, Bytes)>(20);
    let (publish_sender, publish_receiver) = tokio::sync::mpsc::channel::<StageResult>(20);

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
                            let err = format!("processing failed on {}", utils::model_name());
                            Err(ErrorMessage::new(message.id, err))
                        };

                        publish_sender.blocking_send(result).unwrap();
                    },
                );
            }

            std::thread::sleep(std::time::Duration::from_millis(100));
        }
    });

    let conn = &*Box::leak(Box::new(connect(&utils::amqp_addr()).await.unwrap()));

    let consumer = async move {
        consumer(conn, &utils::current_queue(), |message: Message| async {
            let image_bytes = fetch_image(&message.id).await.unwrap();

            if format!("{:?}", md5::compute(&image_bytes)) != message.hash {
                println!("IMAGE HASH WRONG!");
            }

            let input = (message, image_bytes);

            consume_sender.send(input).await.unwrap();
        })
        .await
        .expect("consumer failed");
    };

    let publisher = async move {
        publisher(conn, &utils::next_queue(), &utils::fail_queue(), publish_receiver)
            .await
            .expect("publisher failed");
    };

    println!("stage {:?} started", utils::model_name());

    tokio::join! { tokio::spawn(consumer), tokio::spawn(publisher), worker };
}
