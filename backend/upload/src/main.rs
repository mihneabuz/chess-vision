mod file;
mod utils;

use std::net::SocketAddr;
use std::sync::Arc;

use axum::{
    extract::{Multipart, State},
    routing::{get, post},
    Json, Router, Server,
};
use serde::Serialize;
use tokio::sync::{
    mpsc::{self, Sender},
    Mutex,
};
use ulid::Ulid;

use shared::message::{connect, simple_publisher, Message};

#[derive(Serialize, Debug)]
struct Response {
    success: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    id: Option<String>,
}

type MessageSender = Arc<Mutex<Sender<Message>>>;

async fn upload(State(sender): State<MessageSender>, mut multipart: Multipart) -> Json<Response> {
    let mut id = None;
    while let Ok(Some(field)) = multipart.next_field().await {
        if let Some("image") = field.name() {
            if let Ok(image) = field.bytes().await {
                let ulid = Ulid::new().to_string();

                if let Some(hash) = file::handle_image(image, &ulid).await {
                    sender
                        .lock()
                        .await
                        .send(Message::new(ulid.clone(), hash, vec![]))
                        .await
                        .unwrap();

                    id = Some(ulid);
                }
            }
        }
    }

    Json(Response {
        success: id.is_some(),
        id,
    })
}

#[tokio::main]
async fn main() {
    let (sender, receiver) = mpsc::channel::<Message>(64);

    tokio::spawn(async move {
        let conn = connect(&utils::amqp_addr()).await.unwrap();
        simple_publisher(&conn, &utils::queue(), receiver)
            .await
            .expect("publisher failed");
    });

    let app = Router::new()
        .route("/running", get(|| async { "yes" }))
        .route("/", post(upload))
        .with_state(Arc::new(Mutex::new(sender)));

    Server::bind(&SocketAddr::from(([0, 0, 0, 0], 80)))
        .serve(app.into_make_service())
        .await
        .unwrap();
}
