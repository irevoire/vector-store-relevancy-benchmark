use std::mem;
use std::{fmt, marker::PhantomData};

use arroy::distances::*;
use arroy::{
    internals::{self, Leaf, NodeCodec, UnalignedVector},
    Database, Distance, ItemId, Writer,
};
use bytemuck::{AnyBitPattern, PodCastError};
use heed::{EnvOpenOptions, RwTxn};
use rand::{rngs::StdRng, seq::SliceRandom, SeedableRng};

const TWENTY_HUNDRED_MIB: usize = 2 * 1024 * 1024 * 1024;

pub fn bench_over_all_distances(dimensions: usize, vectors: &[(u32, &[f32])]) {
    let recall_tested = [1, 10, 20, 50, 100, 500];

    for (distance_name, func) in &[
        (
            BinaryQuantizedAngular::name(),
            &measure_distance::<BinaryQuantizedAngular, Angular>
                as &dyn Fn(usize, &[(u32, &[f32])], usize) -> f32,
        ),
        (
            Angular::name(),
            &measure_distance::<Angular, Angular> as &dyn Fn(usize, &[(u32, &[f32])], usize) -> f32,
        ),
        (
            BinaryQuantizedManhattan::name(),
            &measure_distance::<BinaryQuantizedManhattan, Manhattan>
                as &dyn Fn(usize, &[(u32, &[f32])], usize) -> f32,
        ),
        (
            Manhattan::name(),
            &measure_distance::<Manhattan, Manhattan>
                as &dyn Fn(usize, &[(u32, &[f32])], usize) -> f32,
        ),
        (
            BinaryQuantizedEuclidean::name(),
            &measure_distance::<BinaryQuantizedEuclidean, Euclidean>
                as &dyn Fn(usize, &[(u32, &[f32])], usize) -> f32,
        ),
        (
            Euclidean::name(),
            &measure_distance::<Euclidean, Euclidean>
                as &dyn Fn(usize, &[(u32, &[f32])], usize) -> f32,
        ),
        (
            DotProduct::name(),
            &measure_distance::<DotProduct, DotProduct>
                as &dyn Fn(usize, &[(u32, &[f32])], usize) -> f32,
        ),
    ] {
        let now = std::time::Instant::now();
        let mut recall = Vec::new();
        for number_fetched in recall_tested {
            let rec = (func)(dimensions, vectors, number_fetched);
            recall.push(Recall(rec));
        }
        println!("{distance_name:30}: {recall:?}, took {:?}", now.elapsed());
    }
}

pub struct MatLEView<'m, const DIM: usize, T> {
    bytes: &'m [u8],
    _marker: PhantomData<T>,
}

impl<const DIM: usize, T: AnyBitPattern> MatLEView<'_, DIM, T> {
    pub fn new(bytes: &[u8]) -> MatLEView<DIM, T> {
        assert!((bytes.len() / mem::size_of::<T>()) % DIM == 0);
        MatLEView {
            bytes,
            _marker: PhantomData,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn len(&self) -> usize {
        (self.bytes.len() / mem::size_of::<T>()) / DIM
    }

    pub fn get(&self, index: usize) -> Option<Result<&[T; DIM], PodCastError>> {
        let tsize = mem::size_of::<T>();
        if (index * DIM + DIM) * tsize < self.bytes.len() {
            let start = index * DIM;
            let bytes = &self.bytes[start * tsize..(start + DIM) * tsize];
            match bytemuck::try_cast_slice::<u8, T>(bytes) {
                Ok(slice) => Some(Ok(slice.try_into().unwrap())),
                Err(e) => Some(Err(e)),
            }
        } else {
            None
        }
    }

    pub fn iter(&self) -> impl Iterator<Item = &[T; DIM]> {
        (0..self.len() - 1).map(|i| self.get(i).unwrap().unwrap())
    }

    pub fn get_all(&self) -> Vec<&[T; DIM]> {
        self.iter().collect()
    }
}

pub fn measure_distance<ArroyDistance: Distance, PerfectDistance: Distance>(
    dimensions: usize,
    points: &[(u32, &[f32])],
    number_fetched: usize,
) -> f32 {
    let dir = tempfile::tempdir().unwrap();
    let env = unsafe {
        EnvOpenOptions::new()
            .map_size(TWENTY_HUNDRED_MIB)
            .open(dir.path())
    }
    .unwrap();

    let mut rng = StdRng::seed_from_u64(13);
    let mut wtxn = env.write_txn().unwrap();

    let database = env
        .create_database::<internals::KeyCodec, NodeCodec<ArroyDistance>>(&mut wtxn, None)
        .unwrap();
    load_into_arroy(&mut rng, &mut wtxn, database, dimensions, points);

    let reader = arroy::Reader::open(&wtxn, 0, database).unwrap();

    let mut correctly_retrieved = 0;
    for _ in 0..100 {
        let querying = points.choose(&mut rng).unwrap();

        let relevant = partial_sort_by::<PerfectDistance>(
            points.iter().map(|(i, v)| (*i, *v)),
            querying.1,
            number_fetched,
        );

        let arroy = reader
            .nns_by_item(&wtxn, querying.0, number_fetched, None, None)
            .unwrap()
            .unwrap();

        for ret in arroy {
            if relevant.iter().any(|(id, _, _)| *id == ret.0) {
                correctly_retrieved += 1;
            }
        }
    }

    correctly_retrieved as f32 / (number_fetched as f32 * 100.0)
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
