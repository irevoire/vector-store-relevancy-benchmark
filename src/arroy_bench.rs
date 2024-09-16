use std::num::NonZeroUsize;

use arroy::{
    internals::{self, NodeCodec},
    Database, ItemId, Writer,
};
use byte_unit::{Byte, UnitType};
use heed::{EnvOpenOptions, RwTxn};
use rand::{rngs::StdRng, seq::SliceRandom, SeedableRng};

use crate::{partial_sort_by, Recall, RECALL_TESTED, RNG_SEED};
const TWENTY_HUNDRED_MIB: usize = 2000 * 1024 * 1024 * 1024;

pub fn measure_arroy_distance<
    ArroyDistance: arroy::Distance,
    PerfectDistance: arroy::Distance,
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
    let mut arroy_seed = StdRng::seed_from_u64(13);
    let mut rng = StdRng::seed_from_u64(RNG_SEED);
    let mut wtxn = env.write_txn().unwrap();

    let database = env
        .create_database::<internals::KeyCodec, NodeCodec<ArroyDistance>>(&mut wtxn, None)
        .unwrap();
    load_into_arroy(&mut arroy_seed, &mut wtxn, database, dimensions, points);
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

    // make the distance name smaller
    let distance_name = ArroyDistance::name().replace("binary quantized", "bq");
    println!(
        "[arroy]  {distance_name:16} x{OVERSAMPLING}: {recalls:?}, indexed for: {time_to_index:02.2?}, searched for: {time_to_search:02.2?}, size on disk: {database_size:#.2}"
    );
}

fn load_into_arroy<D: arroy::Distance>(
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
