mod arroy_bench;
mod qdrant;

use std::fs::File;
use std::mem;
use std::{fmt, marker::PhantomData};

use arroy::distances::*;
use arroy::{
    internals::{Leaf, UnalignedVector},
    ItemId,
};
use arroy_bench::measure_arroy_distance;
use bytemuck::{AnyBitPattern, PodCastError};
use memmap2::Mmap;
use qdrant_client::qdrant::{quantization_config, ScalarQuantizationBuilder};

use crate::qdrant::measure_qdrant_distance;

pub const RECALL_TESTED: [usize; 6] = [1, 10, 20, 50, 100, 500];

pub fn bench_over_all_distances(dimensions: usize, vectors: &[(u32, &[f32])]) {
    println!("{} vectors are used for this measure", vectors.len());
    println!("Recall tested is {RECALL_TESTED:?}");

    for func in &[
        bench_qdrant_distance::<Angular>(),
        bench_qdrant_distance::<BinaryQuantizedAngular>(),
        bench_qdrant_distance::<Euclidean>(),
        bench_qdrant_distance::<BinaryQuantizedEuclidean>(),
        bench_qdrant_distance::<Manhattan>(),
        bench_qdrant_distance::<BinaryQuantizedManhattan>(),
        bench_qdrant_distance::<DotProduct>(),
        // angular
        // bench_arroy_distance::<BinaryQuantizedAngular, 1>(),
        bench_arroy_distance::<BinaryQuantizedAngular, 3>(),
        // bench_arroy_distance::<BinaryQuantizedAngular, 6>(),
        // bench_arroy_distance::<BinaryQuantizedAngular, 10>(),
        // bench_arroy_distance::<BinaryQuantizedAngular, 50>(),
        // bench_arroy_distance::<BinaryQuantizedAngular, 100>(),
        bench_arroy_distance::<Angular, 1>(),
        // manhattan
        // bench_arroy_distance::<BinaryQuantizedManhattan, 1>(),
        bench_arroy_distance::<BinaryQuantizedManhattan, 3>(),
        // bench_arroy_distance::<BinaryQuantizedManhattan, 6>(),
        // bench_arroy_distance::<BinaryQuantizedManhattan, 10>(),
        // bench_arroy_distance::<BinaryQuantizedManhattan, 50>(),
        // bench_arroy_distance::<BinaryQuantizedManhattan, 100>(),
        bench_arroy_distance::<Manhattan, 1>(),
        // euclidean
        // bench_arroy_distance::<BinaryQuantizedEuclidean, 1>(),
        bench_arroy_distance::<BinaryQuantizedEuclidean, 3>(),
        // bench_arroy_distance::<BinaryQuantizedEuclidean, 6>(),
        // bench_arroy_distance::<BinaryQuantizedEuclidean, 10>(),
        // bench_arroy_distance::<BinaryQuantizedEuclidean, 50>(),
        // bench_arroy_distance::<BinaryQuantizedEuclidean, 100>(),
        bench_arroy_distance::<Euclidean, 1>(),
        // dot-product
        bench_arroy_distance::<DotProduct, 1>(),
    ] {
        (func)(dimensions, vectors);
    }
}

/// A generalist distance trait that contains the informations required to configure every engine
trait Distance: arroy::Distance {
    const BINARY_QUANTIZED: bool;
    type RealDistance: arroy::Distance;
    const QDRANT_DISTANCE: qdrant_client::qdrant::Distance;
    fn qdrant_quantization_config() -> quantization_config::Quantization;
}

macro_rules! arroy_distance {
    ($distance:ty => qdrant: $qdrant:ident) => {
        impl Distance for $distance {
            const BINARY_QUANTIZED: bool = false;
            type RealDistance = $distance;
            const QDRANT_DISTANCE: qdrant_client::qdrant::Distance =
                qdrant_client::qdrant::Distance::$qdrant;
            fn qdrant_quantization_config() -> quantization_config::Quantization {
                ScalarQuantizationBuilder::default().into()
            }
        }
    };
    ($distance:ty => real: $real:ty, qdrant: $qdrant:ident) => {
        impl Distance for $distance {
            const BINARY_QUANTIZED: bool = true;
            type RealDistance = $real;
            const QDRANT_DISTANCE: qdrant_client::qdrant::Distance =
                qdrant_client::qdrant::Distance::$qdrant;
            fn qdrant_quantization_config() -> quantization_config::Quantization {
                qdrant_client::qdrant::BinaryQuantization::default().into()
            }
        }
    };
}

arroy_distance!(BinaryQuantizedAngular => real: Angular, qdrant: Cosine);
arroy_distance!(Angular => qdrant: Cosine);
arroy_distance!(BinaryQuantizedEuclidean => real: Euclidean, qdrant: Euclid);
arroy_distance!(Euclidean => qdrant: Euclid);
arroy_distance!(BinaryQuantizedManhattan => real: Manhattan, qdrant: Manhattan);
arroy_distance!(Manhattan => qdrant: Manhattan);
arroy_distance!(DotProduct => qdrant: Dot);

fn bench_arroy_distance<D: Distance, const OVERSAMPLING: usize>(
) -> &'static (dyn Fn(usize, &[(u32, &[f32])]) + 'static) {
    &measure_arroy_distance::<D, D::RealDistance, OVERSAMPLING> as &dyn Fn(usize, &[(u32, &[f32])])
}

fn bench_qdrant_distance<D: Distance>() -> &'static (dyn Fn(usize, &[(u32, &[f32])]) + 'static) {
    &measure_qdrant_distance::<D> as &dyn Fn(usize, &[(u32, &[f32])])
}

pub struct MatLEView<T> {
    name: &'static str,
    mmap: Mmap,
    dimensions: usize,
    _marker: PhantomData<T>,
}

impl<T: AnyBitPattern> MatLEView<T> {
    pub fn new(name: &'static str, path: &str, dimensions: usize) -> MatLEView<T> {
        let file = File::open(path).unwrap();
        let mmap = unsafe { Mmap::map(&file).unwrap() };

        assert!((mmap.len() / mem::size_of::<T>()) % dimensions == 0);
        MatLEView {
            name,
            mmap,
            dimensions,
            _marker: PhantomData,
        }
    }

    pub fn header(&self) {
        println!(
            "{} - {} vectors of {} dimensions",
            self.name,
            self.len(),
            self.dimensions
        );
    }

    pub fn dimensions(&self) -> usize {
        self.dimensions
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn len(&self) -> usize {
        (self.mmap.len() / mem::size_of::<T>()) / self.dimensions
    }

    pub fn get(&self, index: usize) -> Option<Result<&[T], PodCastError>> {
        let tsize = mem::size_of::<T>();
        if (index * self.dimensions + self.dimensions) * tsize < self.mmap.len() {
            let start = index * self.dimensions;
            let bytes = &self.mmap[start * tsize..(start + self.dimensions) * tsize];
            match bytemuck::try_cast_slice::<u8, T>(bytes) {
                Ok(slice) => Some(Ok(slice)),
                Err(e) => Some(Err(e)),
            }
        } else {
            None
        }
    }

    pub fn iter(&self) -> impl Iterator<Item = &[T]> {
        (0..self.len() - 1).map(|i| self.get(i).unwrap().unwrap())
    }

    pub fn get_all(&self) -> Vec<&[T]> {
        self.iter().collect()
    }
}

fn partial_sort_by<'a, D: arroy::Distance>(
    mut vectors: impl Iterator<Item = (ItemId, &'a [f32])>,
    sort_by: &[f32],
    elements: usize,
) -> Vec<(ItemId, &'a [f32], f32)> {
    let mut ret = Vec::with_capacity(elements);
    ret.extend(
        vectors
            .by_ref()
            .take(elements)
            .map(|(i, v)| (i, v, distance::<D>(sort_by, v))),
    );
    ret.sort_by(|(_, _, left), (_, _, right)| left.total_cmp(right));

    if ret.is_empty() {
        return ret;
    }

    for (item_id, vector) in vectors {
        let distance = distance::<D>(sort_by, vector);
        if distance < ret.last().unwrap().2 {
            match ret.binary_search_by(|(_, _, d)| d.total_cmp(&distance)) {
                Ok(i) | Err(i) => {
                    ret.pop();
                    ret.insert(i, (item_id, vector, distance))
                }
            }
        }
    }

    ret
}

fn distance<D: arroy::Distance>(left: &[f32], right: &[f32]) -> f32 {
    let left = UnalignedVector::from_slice(left);
    let left = Leaf {
        header: D::new_header(&left),
        vector: left,
    };
    let right = UnalignedVector::from_slice(right);
    let right = Leaf {
        header: D::new_header(&right),
        vector: right,
    };

    D::built_distance(&left, &right)
}

pub struct Recall(pub f32);

impl fmt::Debug for Recall {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.0 {
            // red
            f32::NEG_INFINITY..=0.25 => write!(f, "\x1b[1;31m")?,
            // yellow
            0.25..=0.5 => write!(f, "\x1b[1;33m")?,
            // green
            0.5..=0.75 => write!(f, "\x1b[1;32m")?,
            // blue
            0.75..=0.90 => write!(f, "\x1b[1;34m")?,
            // cyan
            0.90..=0.999 => write!(f, "\x1b[1;36m")?,
            // underlined cyan
            0.999..=f32::INFINITY => write!(f, "\x1b[1;4;36m")?,
            _ => (),
        }
        write!(f, "{:.2}\x1b[0m", self.0)
    }
}

pub const HN_TOP_POSTS_PATH: &str = "assets/hn-top-posts.mat";
pub const HN_TOP_POSTS_DIMENSIONS: usize = 1024;
pub fn hn_top_posts() -> MatLEView<f32> {
    MatLEView::new(
        "Hackernews top posts",
        HN_TOP_POSTS_PATH,
        HN_TOP_POSTS_DIMENSIONS,
    )
}

pub const HN_POSTS_PATH: &str = "assets/hn-posts.mat";
pub const HN_POSTS_DIMENSIONS: usize = 512;
pub fn hn_posts() -> MatLEView<f32> {
    MatLEView::new("Hackernews posts", HN_POSTS_PATH, HN_POSTS_DIMENSIONS)
}

pub const DB_PEDIA_OPENAI_TEXT_EMBEDDING_3_LARGE_PATH: &str =
    "assets/db-pedia-OpenAI-text-embedding-3-large.mat";
pub const DB_PEDIA_OPENAI_TEXT_EMBEDDING_3_LARGE_DIMENSIONS: usize = 3072;
// pub fn hn_posts() -> MatLEView<f32> {
//     MatLEView::new("Hackernews posts", "assets/hn-posts.mat", 512)
// }

pub const DB_PEDIA_OPENAI_TEXT_EMBEDDING_ADA_002_PATH: &str =
    "assets/db-pedia-OpenAI-text-embedding-3-large.mat";
pub const DB_PEDIA_OPENAI_TEXT_EMBEDDING_ADA_002_DIMENSIONS: usize = 1536;
// pub fn hn_posts() -> MatLEView<f32> {
//     MatLEView::new("Hackernews posts", "assets/hn-posts.mat", 512)
// }
