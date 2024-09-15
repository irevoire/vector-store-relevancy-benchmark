mod arroy_bench;
mod dataset;
mod qdrant;

pub use dataset::*;

use std::fmt;
use std::fmt::Write;

use arroy::distances::*;
use arroy::{
    internals::{Leaf, UnalignedVector},
    ItemId,
};
use arroy_bench::measure_arroy_distance;
use qdrant_client::qdrant::{quantization_config, ScalarQuantizationBuilder};

use crate::qdrant::measure_qdrant_distance;

pub const RECALL_TESTED: [usize; 6] = [1, 10, 20, 50, 100, 500];

pub fn bench_over_all_distances(dimensions: usize, vectors: &[(u32, &[f32])]) {
    println!(
        "\x1b[1m{}\x1b[0m vectors are used for this measure",
        vectors.len()
    );
    let mut recall_tested = String::new();
    RECALL_TESTED
        .iter()
        .for_each(|recall| write!(&mut recall_tested, "{recall:4}, ").unwrap());
    let recall_tested = recall_tested.trim_end_matches(", ");
    println!("Recall tested is:         [{recall_tested}]");

    for func in &[
        // angular
        // bench_qdrant_distance::<Angular>(),
        // bench_arroy_distance::<Angular, 1>(),
        // bench_qdrant_distance::<BinaryQuantizedAngular>(),
        // bench_arroy_distance::<BinaryQuantizedAngular, 1>(),
        // bench_arroy_distance::<BinaryQuantizedAngular, 1>(),
        // bench_arroy_distance::<BinaryQuantizedAngular, 1>(),
        // bench_arroy_distance::<BinaryQuantizedAngular, 3>(),
        // manhattan
        // bench_qdrant_distance::<Manhattan>(),
        // bench_arroy_distance::<Manhattan, 1>(),
        // bench_qdrant_distance::<BinaryQuantizedManhattan>(),
        // bench_arroy_distance::<BinaryQuantizedManhattan, 1>(),
        // bench_arroy_distance::<BinaryQuantizedManhattan, 3>(),
        // euclidean
        // bench_qdrant_distance::<Euclidean>(),
        // bench_arroy_distance::<Euclidean, 1>(),
        // bench_qdrant_distance::<BinaryQuantizedEuclidean>(),
        // bench_arroy_distance::<BinaryQuantizedEuclidean, 1>(),
        bench_arroy_distance::<BinaryQuantizedEuclidean, 1>(),
        bench_arroy_distance::<BinaryQuantizedEuclidean, 1>(),
        bench_arroy_distance::<BinaryQuantizedEuclidean, 1>(),
        // bench_arroy_distance::<BinaryQuantizedEuclidean, 1>(),
        // bench_arroy_distance::<BinaryQuantizedEuclidean, 3>(),
        // dot-product
        // bench_qdrant_distance::<DotProduct>(),
        // bench_arroy_distance::<DotProduct, 1>(),
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

fn bench_arroy_distance<D: Distance, const OVERSAMPLING: usize>() -> fn(usize, &[(u32, &[f32])]) {
    measure_arroy_distance::<D, D::RealDistance, OVERSAMPLING>
}

fn bench_qdrant_distance<D: Distance>() -> fn(usize, &[(u32, &[f32])]) {
    measure_qdrant_distance::<D>
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
