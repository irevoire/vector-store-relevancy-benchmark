use std::fs::File;
use std::mem;
use std::num::NonZeroUsize;
use std::{fmt, marker::PhantomData};

use arroy::distances::*;
use arroy::{
    internals::{self, Leaf, NodeCodec, UnalignedVector},
    Database, Distance, ItemId, Writer,
};
use byte_unit::{Byte, UnitType};
use bytemuck::{AnyBitPattern, PodCastError};
use heed::{EnvOpenOptions, RwTxn};
use memmap2::Mmap;
use rand::{rngs::StdRng, seq::SliceRandom, SeedableRng};

const TWENTY_HUNDRED_MIB: usize = 200 * 1024 * 1024 * 1024;

pub const RECALL_TESTED: [usize; 6] = [1, 10, 20, 50, 100, 500];

pub fn bench_over_all_distances(dimensions: usize, vectors: &[(u32, &[f32])]) {
    println!("{} vectors are used for this measure", vectors.len());
    println!("Recall tested is {RECALL_TESTED:?}");

    for func in &[
        // angular
        bench_arroy_distance::<BinaryQuantizedAngular, 1>(),
        bench_arroy_distance::<BinaryQuantizedAngular, 3>(),
        bench_arroy_distance::<BinaryQuantizedAngular, 6>(),
        bench_arroy_distance::<Angular, 1>(),
        // manhattan
        bench_arroy_distance::<BinaryQuantizedManhattan, 1>(),
        bench_arroy_distance::<BinaryQuantizedManhattan, 3>(),
        bench_arroy_distance::<BinaryQuantizedManhattan, 6>(),
        bench_arroy_distance::<Manhattan, 1>(),
        // euclidean
        bench_arroy_distance::<BinaryQuantizedEuclidean, 1>(),
        bench_arroy_distance::<BinaryQuantizedEuclidean, 3>(),
        bench_arroy_distance::<BinaryQuantizedEuclidean, 6>(),
        bench_arroy_distance::<Euclidean, 1>(),
        // dot-product
        bench_arroy_distance::<DotProduct, 1>(),
    ] {
        (func)(dimensions, vectors);
    }
}

trait ArroyDistance: Distance {
    type RealDistance: Distance;
}

macro_rules! arroy_distance {
    ($distance:ty) => {
        impl ArroyDistance for $distance {
            type RealDistance = $distance;
        }
    };
    ($distance:ty => $real:ty) => {
        impl ArroyDistance for $distance {
            type RealDistance = $real;
        }
    };
}

arroy_distance!(BinaryQuantizedAngular => Angular);
arroy_distance!(Angular);
arroy_distance!(BinaryQuantizedEuclidean => Euclidean);
arroy_distance!(Euclidean);
arroy_distance!(BinaryQuantizedManhattan => Manhattan);
arroy_distance!(Manhattan);
arroy_distance!(DotProduct);

fn bench_arroy_distance<D: ArroyDistance, const OVERSAMPLING: usize>(
) -> &'static (dyn Fn(usize, &[(u32, &[f32])]) + 'static) {
    &measure_distance::<D, D::RealDistance, OVERSAMPLING> as &dyn Fn(usize, &[(u32, &[f32])])
}

// fn bench_qdrant_distance() -> &'static (dyn Fn(usize, &[(u32, &[f32])]) + 'static) {
//     &measure_qdrant_distance as &dyn Fn(usize, &[(u32, &[f32])])
// }

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

pub fn measure_distance<
    ArroyDistance: Distance,
    PerfectDistance: Distance,
    const OVERSAMPLING: usize,
>(
    dimensions: usize,
    points: &[(u32, &[f32])],
) {
    let dir = tempfile::tempdir().unwrap();
    let env = unsafe {
        EnvOpenOptions::new()
            .map_size(TWENTY_HUNDRED_MIB)
            .open(dir.path())
    }
    .unwrap();

    let now = std::time::Instant::now();
    let mut rng = StdRng::seed_from_u64(13);
    let mut wtxn = env.write_txn().unwrap();

    let database = env
        .create_database::<internals::KeyCodec, NodeCodec<ArroyDistance>>(&mut wtxn, None)
        .unwrap();
    load_into_arroy(&mut rng, &mut wtxn, database, dimensions, points);
    wtxn.commit().unwrap();
    let database_size =
        Byte::from_u64(env.non_free_pages_size().unwrap()).get_appropriate_unit(UnitType::Binary);
    let rtxn = env.read_txn().unwrap();

    let time_to_index = now.elapsed();

    let now = std::time::Instant::now();
    let reader = arroy::Reader::open(&rtxn, 0, database).unwrap();

    let mut recalls = Vec::new();
    for number_fetched in RECALL_TESTED {
        if number_fetched > points.len() {
            break;
        }
        let mut correctly_retrieved = 0;
        for _ in 0..100 {
            let querying = points.choose(&mut rng).unwrap();

            let relevant = partial_sort_by::<PerfectDistance>(
                points.iter().map(|(i, v)| (*i, *v)),
                querying.1,
                number_fetched,
            );

            let arroy = reader
                .nns_by_item(
                    &rtxn,
                    querying.0,
                    number_fetched,
                    None,
                    Some(NonZeroUsize::new(OVERSAMPLING).unwrap()),
                    None,
                )
                .unwrap()
                .unwrap();

            for ret in arroy {
                if relevant.iter().any(|(id, _, _)| *id == ret.0) {
                    correctly_retrieved += 1;
                }
            }
        }

        let recall = correctly_retrieved as f32 / (number_fetched as f32 * 100.0);
        recalls.push(Recall(recall));
    }
    let time_to_search = now.elapsed();

    let distance_name = ArroyDistance::name();
    println!(
        "{distance_name:30} x{OVERSAMPLING}: {recalls:?}, indexed for: {time_to_index:02.2?}, searched for: {time_to_search:02.2?}, size on disk: {database_size:#}"
    );
}

fn partial_sort_by<'a, D: Distance>(
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

fn distance<D: Distance>(left: &[f32], right: &[f32]) -> f32 {
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

fn load_into_arroy<D: Distance>(
    rng: &mut StdRng,
    wtxn: &mut RwTxn,
    database: Database<D>,
    dimensions: usize,
    points: &[(ItemId, &[f32])],
) {
    let writer = Writer::<D>::new(database, 0, dimensions);
    for (i, vector) in points.iter() {
        assert_eq!(vector.len(), dimensions);
        writer.add_item(wtxn, *i, vector).unwrap();
    }
    writer.build(wtxn, rng, None).unwrap();
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
