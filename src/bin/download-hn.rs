use tokio::io::AsyncWriteExt;
use vector_store_relevancy_benchmark::*;

struct ToDownload {
    pub url: &'static str,
    pub path: &'static str,
}

#[tokio::main]
async fn main() {
    let client = reqwest::Client::new();
    let to_download = [
        ToDownload {
            url: "https://static.wilsonl.in/hackerverse/dataset/post-embs-data.mat",
            path: HN_POSTS_PATH,
        },
        ToDownload {
            url: "https://static.wilsonl.in/hackerverse/dataset/toppost-embs-data.mat",
            path: HN_TOP_POSTS_PATH,
        },
    ];

    let mut tasks = Vec::new();
    for to_download in to_download {
        let client = client.clone();
        let task = tokio::spawn(async move {
            let mut file = tokio::fs::File::create_new(to_download.path).await.unwrap();
            let mut payload = client.get(to_download.url).send().await.unwrap();
            while let Some(item) = payload.chunk().await.unwrap() {
                file.write_all(&item).await.unwrap();
            }
        });
        tasks.push(task);
    }

    for task in tasks {
        task.await.unwrap();
    }
}
