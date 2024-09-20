use benchmarks::{bench_over_all_distances, MatLEView};

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

fn db_pedia_3_large_reduced_to_1536() -> MatLEView<f32> {
    let mut mat = MatLEView::new(
        "db pedia OpenAI text-embedding 3 large 1536",
        "assets/db-pedia-OpenAI-text-embedding-3-large.mat",
        3072,
    );
    mat.reduce_dimensions_to(1536);
    mat
}

fn db_pedia_ada_002_large() -> MatLEView<f32> {
    MatLEView::new(
        "db pedia OpenAI text-embedding ada  002",
        "assets/db-pedia-OpenAI-text-embedding-ada-002.mat",
        1536,
    )
}

fn wikipedia_768() -> MatLEView<f32> {
    MatLEView::new(
        "wikipedia 22 12 simple embeddings",
        "assets/wikipedia-22-12-simple-embeddings.mat",
        768,
    )
}

fn main() {
    let take = 100_000;
    for dataset in [
        // &hn_posts(),
        // &hn_top_post(),
        // &db_pedia_3_large_reduced_to_1536(),
        // &db_pedia_3_large(),
        // &db_pedia_ada_002_large(),
        &wikipedia_768(),
    ] {
        let vectors: Vec<(u32, &[f32])> =
            dataset.iter().enumerate().map(|(i, v)| (i as u32, v)).take(take).collect();

        dataset.header();
        bench_over_all_distances(
            dataset.dimensions(),
            dataset.reduced_dimensions(),
            vectors.as_slice(),
        );
        println!();
    }
}
