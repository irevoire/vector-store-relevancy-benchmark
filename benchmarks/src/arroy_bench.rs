use std::num::NonZeroUsize;
use std::time::Duration;

use arroy::internals::{self, NodeCodec};
use arroy::{Database, ItemId, Writer};
use byte_unit::{Byte, UnitType};
use heed::{EnvOpenOptions, RwTxn};
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use roaring::RoaringBitmap;

use crate::{partial_sort_by, Recall, RECALL_TESTED, RNG_SEED};
const TWENTY_HUNDRED_MIB: usize = 2000 * 1024 * 1024 * 1024;

pub fn measure_arroy_distance<
    ArroyDistance: arroy::Distance,
    PerfectDistance: arroy::Distance,
    const OVERSAMPLING: usize,
    const FILTER_SUBSET_PERCENT: usize,
>(
    dimensions: usize,
    points: &[(u32, &[f32])],
) {
    let dir = tempfile::tempdir().unwrap();
    let env =
        unsafe { EnvOpenOptions::new().map_size(TWENTY_HUNDRED_MIB).open(dir.path()) }.unwrap();

    let now = std::time::Instant::now();
    let mut arroy_seed = StdRng::seed_from_u64(13);
    let mut rng = StdRng::seed_from_u64(RNG_SEED);
    let mut wtxn = env.write_txn().unwrap();

    let database = env
        .create_database::<internals::KeyCodec, NodeCodec<ArroyDistance>>(&mut wtxn, None)
        .unwrap();
    let inserted = load_into_arroy(&mut arroy_seed, &mut wtxn, database, dimensions, points);
    wtxn.commit().unwrap();

    let filtered_percentage = FILTER_SUBSET_PERCENT as f32;
    let candidates = if FILTER_SUBSET_PERCENT >= 100 {
        None
    } else {
        let count = (inserted.len() as f32 * (filtered_percentage / 100.0)) as usize;
        Some(inserted.iter().take(count).collect::<RoaringBitmap>())
    };

    let database_size =
        Byte::from_u64(env.non_free_pages_size().unwrap()).get_appropriate_unit(UnitType::Binary);

    let time_to_index = now.elapsed();
    let mut total_duration = Duration::default();

    let mut recalls = Vec::new();
    for number_fetched in RECALL_TESTED {
        if number_fetched > points.len() {
            break;
        }
        let queries: Vec<_> = (0..100).map(|_| points.choose(&mut rng).unwrap()).collect();

        let (correctly_retrieved, duration) = queries
            .into_par_iter()
            .map(|querying| {
                let rtxn = env.read_txn().unwrap();
                let reader = arroy::Reader::open(&rtxn, 0, database).unwrap();

                let relevant = partial_sort_by::<PerfectDistance>(
                    points
                        .iter()
                        // Only evaluate the candidate points
                        .filter(|(id, _)| {
                            candidates.as_ref().map_or(true, |cand| cand.contains(*id))
                        })
                        .map(|(i, v)| (*i, *v)),
                    querying.1,
                    number_fetched,
                );

                let now = std::time::Instant::now();
                let mut arroy = reader.nns(number_fetched);
                arroy.oversampling(NonZeroUsize::new(OVERSAMPLING).unwrap());

                if let Some(ref candidates) = candidates {
                    arroy.candidates(candidates);
                }
                let arroy = arroy.by_item(&rtxn, querying.0).unwrap().unwrap();

                let elapsed = now.elapsed();

                let mut correctly_retrieved = Some(0);
                for ret in arroy {
                    if relevant.iter().any(|(id, _, _)| *id == ret.0) {
                        if let Some(correctly_retrieved) = &mut correctly_retrieved {
                            *correctly_retrieved += 1;
                        }
                    } else if let Some(cand) = candidates.as_ref() {
                        // We set the counter to -1 if we return a filtered out candidated
                        if !cand.contains(ret.0) {
                            correctly_retrieved = None;
                        }
                    }
                }

                (correctly_retrieved, elapsed)
            })
            .reduce(
                || (Some(0), Duration::default()),
                |(aanswer, aduration), (banswer, bduration)| {
                    (aanswer.zip(banswer).map(|(a, b)| a + b), aduration + bduration)
                },
            );

        total_duration += duration;
        let recall = correctly_retrieved.unwrap_or(-1) as f32 / (number_fetched as f32 * 100.0);
        recalls.push(Recall(recall));
    }
    let time_to_search = total_duration;

    // make the distance name smaller
    let distance_name = ArroyDistance::name().replace("binary quantized", "bq");
    println!(
        "[arroy]  {distance_name:16} x{OVERSAMPLING}: {recalls:?}, \
        indexed for: {time_to_index:02.2?}, \
        searched for: {time_to_search:02.2?}, \
        size on disk: {database_size:#.2}, \
        searched in {filtered_percentage:#.2}%"
    );
}

fn load_into_arroy<D: arroy::Distance>(
    rng: &mut StdRng,
    wtxn: &mut RwTxn,
    database: Database<D>,
    dimensions: usize,
    points: &[(ItemId, &[f32])],
) -> RoaringBitmap {
    let writer = Writer::<D>::new(database, 0, dimensions);
    let mut candidates = RoaringBitmap::new();
    for (i, vector) in points.iter() {
        assert_eq!(vector.len(), dimensions);
        writer.add_item(wtxn, *i, vector).unwrap();
        assert!(candidates.push(*i));
    }
    writer.builder(rng).build(wtxn).unwrap();
    candidates
}
