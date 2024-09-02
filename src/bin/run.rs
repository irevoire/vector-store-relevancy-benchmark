use vector_store_relevancy_benchmark::{bench_over_all_distances, MatLEView};

fn hn_top_post() -> MatLEView<f32> {
    MatLEView::new("Hackernews top posts", "assets/hn-top-posts.mat", 1024)
}

fn hn_posts() -> MatLEView<f32> {
    MatLEView::new("Hackernews posts", "assets/hn-posts.mat", 512)
}

fn db_pedia_3_large() -> MatLEView<f32> {
    MatLEView::new(
        "db pedia OpenAI text-embedding 3 large",
        "assets/db-pedia-OpenAI-text-embedding-3-large.mat",
        3072,
    )
}

fn db_pedia_ada_002_large() -> MatLEView<f32> {
    MatLEView::new(
        "db pedia OpenAI text-embedding ada  002",
        "assets/db-pedia-OpenAI-text-embedding-ada-002.mat",
        1536,
    )
}

fn main() {
    let take = 10_000;
    for dataset in [
        &hn_top_post(),
        &hn_posts(),
        &db_pedia_3_large(),
        &db_pedia_ada_002_large(),
    ] {
        let vectors: Vec<(u32, &[f32])> = dataset
            .iter()
            .enumerate()
            .map(|(i, v)| (i as u32, v))
            .take(take)
            .collect();

        dataset.header();
        bench_over_all_distances(dataset.dimensions(), vectors.as_slice());
        println!();
    }
}
