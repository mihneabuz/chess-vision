use anyhow::{anyhow, Result};
use bytes::Bytes;
use lazy_static::lazy_static;
use reqwest::Client;
use tokio_retry::{
    strategy::{jitter, FixedInterval},
    Retry,
};

lazy_static! {
    static ref CLIENT: Client = Client::new();
    static ref FILE_SERVER: String = crate::utils::file_server();
    static ref FILE_TOKEN: String = crate::utils::file_server_token();
}

pub async fn fetch_image(id: &str) -> Result<Bytes> {
    fetch_file(&format!("image_{}", id)).await
}

pub async fn fetch_file(filename: &str) -> Result<Bytes> {
    let file_server = FILE_SERVER.as_str();
    let token = FILE_TOKEN.as_str();

    let url = format!("http://{}/files/{}?token={}", file_server, filename, token);

    let content = Retry::spawn(FixedInterval::from_millis(1000).map(jitter), || async {
        let response = CLIENT.get(&url).send().await?;

        if !response.status().is_success() {
            return Err(anyhow!("bad"));
        }

        Ok(response.bytes().await?)
    })
    .await?;

    Ok(content)
}

pub fn fetch_file_sync(filename: &str) -> Result<Bytes> {
    let rt = tokio::runtime::Runtime::new().unwrap();
    rt.block_on(fetch_file(filename))
}
