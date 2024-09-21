use benchmarks::{bench_over_all_distances, MatLEView};
use clap::{Parser, ValueEnum};
use enum_iterator::Sequence;

#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, ValueEnum, Sequence)]
enum Dataset {
    /// Hackernews posts (512)
    HnPosts,
    /// Wikipedia (768)
    Wikipedia,
    /// Hackernews top posts (1024)
    HnTopPost,
    /// db pedia OpenAI text-embedding ada 002 (1536)
    DbPediaAda002,
    /// db pedia OpenAI text-embedding 3 large (3072)
    DbPedia3Large,
}

impl From<Dataset> for MatLEView<f32> {
    fn from(dataset: Dataset) -> Self {
        match dataset {
            Dataset::HnPosts => MatLEView::new("Hackernews posts", "assets/hn-posts.mat", 512),
            Dataset::Wikipedia => MatLEView::new(
                "wikipedia 22 12 simple embeddings",
                "assets/wikipedia-22-12-simple-embeddings.mat",
                768,
            ),
            Dataset::HnTopPost => {
                MatLEView::new("Hackernews top posts", "assets/hn-top-posts.mat", 1024)
            }
            Dataset::DbPediaAda002 => MatLEView::new(
                "db pedia OpenAI text-embedding ada  002",
                "assets/db-pedia-OpenAI-text-embedding-ada-002.mat",
                1536,
            ),
            Dataset::DbPedia3Large => MatLEView::new(
                "db pedia OpenAI text-embedding 3 large",
                "assets/db-pedia-OpenAI-text-embedding-3-large.mat",
                3072,
            ),
        }
    }
}

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    /// The datasets to run and all of them are ran if empty.
    #[arg(value_enum)]
    datasets: Vec<Dataset>,

    /// Number of vectors to evaluate from the datasets.
    #[arg(long, default_value_t = 100_000)]
    count: usize,
}

fn main() {
    let Args { datasets, count } = Args::parse();

    let datasets: Vec<MatLEView<_>> = if datasets.is_empty() {
        enum_iterator::all::<Dataset>().map(Into::into).collect()
    } else {
        datasets.into_iter().map(Into::into).collect()
    };

    for dataset in datasets {
        let vectors: Vec<_> =
            dataset.iter().enumerate().map(|(i, v)| (i as u32, v)).take(count).collect();
        dataset.header();
        bench_over_all_distances(dataset.dimensions(), vectors.as_slice());
        println!();
    }
}
