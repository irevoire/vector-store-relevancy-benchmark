use std::fs::File;

use memmap2::Mmap;
use vector_store_relevancy_benchmark::{bench_over_all_distances, MatLEView};

fn main() -> anyhow::Result<()> {
    let embs_data_file = File::open("assets/hn-top-posts.mat")?;
    let embs_data = unsafe { Mmap::map(&embs_data_file)? };
    const HN_TOP_POST_DIMENSIONS: usize = 512;
    let embs_data = MatLEView::<HN_TOP_POST_DIMENSIONS, f32>::new(&embs_data);

    let vectors: Vec<(u32, &[f32])> = embs_data
        .iter()
        .enumerate()
        .map(|(i, v)| (i as u32, v.as_slice()))
        .collect();

    bench_over_all_distances(HN_TOP_POST_DIMENSIONS, vectors.as_slice());
    Ok(())
}
