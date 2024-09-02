use std::time::Duration;

use async_channel::{Receiver, Sender};
use indicatif::{ProgressBar, ProgressStyle};
use tokio::{
    fs::File,
    io::{AsyncWriteExt, BufWriter},
};
use vector_store_relevancy_benchmark::{
    DB_PEDIA_OPENAI_TEXT_EMBEDDING_3_LARGE_DIMENSIONS, DB_PEDIA_OPENAI_TEXT_EMBEDDING_3_LARGE_PATH,
    DB_PEDIA_OPENAI_TEXT_EMBEDDING_ADA_002_DIMENSIONS, DB_PEDIA_OPENAI_TEXT_EMBEDDING_ADA_002_PATH,
};

#[tokio::main]
async fn main() {
    // Number of documents feched each http call, a too high value will get us http 500
    let batch_size = 100;

    let progress = indicatif::MultiProgress::new();
    let style = ProgressStyle::default_bar()
        .template(
            "{msg:11} [{elapsed_precise}] {bar:40.cyan/blue} ETA: {eta} - {percent}% ({pos}/{len})",
        )
        .unwrap();
    let dispatch_progress = indicatif::ProgressBar::new(1_000_000 / batch_size)
        .with_message("Dispatching")
        .with_style(style.clone());
    progress.add(dispatch_progress.clone());
    let downloading_progress = indicatif::ProgressBar::new(1_000_000 / batch_size)
        .with_message("Downloaded")
        .with_style(style.clone());
    progress.add(downloading_progress.clone());
    let writing_progress = indicatif::ProgressBar::new(1_000_000)
        .with_message("Written")
        .with_style(style.clone());
    progress.add(writing_progress.clone());

    dispatch_progress.tick();
    downloading_progress.tick();
    writing_progress.tick();

    let pool_size = 200;
    let (writing_sender, writing_receiver) = async_channel::bounded(1);
    let writer = tokio::spawn(writer(writing_receiver, writing_progress));

    let (url_sender, url_receiver) = async_channel::bounded(1);

    for _ in 0..pool_size {
        tokio::spawn(downloader(
            url_receiver.clone(),
            writing_sender.clone(),
            downloading_progress.clone(),
        ));
    }

    for offset in dispatch_progress.wrap_iter(0..(1_000_000 / batch_size)) {
        let base_url = "https://datasets-server.huggingface.co/rows?dataset=Qdrant%2Fdbpedia-entities-openai3-text-embedding-3-large-3072-1M&config=default&split=train";
        let url = format!("{base_url}&offset={offset}&length={batch_size}");
        url_sender.send(url).await.unwrap();
    }

    writer.await.unwrap();
}

async fn downloader(url: Receiver<String>, sender: Sender<Vec<f32>>, progress: ProgressBar) {
    let client = reqwest::Client::new();
    while let Ok(url) = url.recv().await {
        let mut retry = 0;
        let request = loop {
            match client.get(&url).send().await {
                Ok(v) => break v,
                Err(e) => {
                    eprintln!("{e}");
                    if retry == 10 {
                        continue;
                    } else {
                        retry += 1;
                        tokio::time::sleep(Duration::from_secs(2 * retry)).await;
                    }
                }
            }
        };
        progress.inc(1);
        let reader = request.bytes().await.unwrap();

        for json in serde_json::Deserializer::from_slice(&reader).into_iter::<serde_json::Value>() {
            let Ok(json) = json else {
                eprintln!("{}", json.unwrap_err());
                continue;
            };
            let serde_json::Value::Array(ref array) = json["rows"] else {
                continue;
            };
            for row in array {
                let embeddings = &row["row"];
                let small = &embeddings["text-embedding-ada-002-1536-embedding"];
                let large = &embeddings["text-embedding-3-large-3072-embedding"];
                if let Ok(small) = serde_json::from_value::<Vec<f32>>(small.clone()) {
                    assert_eq!(
                        small.len(),
                        DB_PEDIA_OPENAI_TEXT_EMBEDDING_ADA_002_DIMENSIONS
                    );
                    sender.send(small).await.unwrap();
                }
                if let Ok(large) = serde_json::from_value::<Vec<f32>>(large.clone()) {
                    assert_eq!(
                        large.len(),
                        DB_PEDIA_OPENAI_TEXT_EMBEDDING_3_LARGE_DIMENSIONS
                    );
                    sender.send(large).await.unwrap();
                }
            }
        }
    }
}

async fn writer(receiver: Receiver<Vec<f32>>, progress: ProgressBar) {
    let large_writer = File::create(DB_PEDIA_OPENAI_TEXT_EMBEDDING_3_LARGE_PATH)
        .await
        .unwrap();
    let mut large_writer = BufWriter::new(large_writer);
    let ada_writer = File::create(DB_PEDIA_OPENAI_TEXT_EMBEDDING_ADA_002_PATH)
        .await
        .unwrap();
    let mut ada_writer = BufWriter::new(ada_writer);

    let mut small_embedder_counter = 0;
    let mut large_embedder_counter = 0;

    while let Ok(received) = receiver.recv().await {
        progress.inc(1);
        if received.len() == DB_PEDIA_OPENAI_TEXT_EMBEDDING_ADA_002_DIMENSIONS {
            let small: &[u8] = bytemuck::cast_slice(&received);
            ada_writer.write_all(small).await.unwrap();
            small_embedder_counter += 1;
        }
        if received.len() == DB_PEDIA_OPENAI_TEXT_EMBEDDING_3_LARGE_DIMENSIONS {
            let large: &[u8] = bytemuck::cast_slice(&received);
            large_writer.write_all(large).await.unwrap();
            large_embedder_counter += 1;
        }
    }

    progress.finish();
    println!("small_embedder_counter: {small_embedder_counter}");
    println!("large_embedder_counter: {large_embedder_counter}");
}
