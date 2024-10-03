use std::collections::HashMap;
use std::net::Ipv4Addr;
use std::str::FromStr;
use std::time::{Duration, Instant};

use byte_unit::{Byte, UnitType};
use qdrant_client::qdrant::point_id::PointIdOptions;
use qdrant_client::qdrant::{
    Condition, CreateCollectionBuilder, Filter, PointId, PointStruct, Range, SearchParamsBuilder,
    SearchPointsBuilder, UpsertPointsBuilder, VectorParamsBuilder,
};
use qdrant_client::{Payload, Qdrant};
use rand::prelude::SliceRandom;
use rand::rngs::StdRng;
use rand::SeedableRng;
use roaring::RoaringBitmap;

use crate::scenarios::*;
use crate::{partial_sort_by, Distance, Recall, RECALL_TESTED, RNG_SEED};

pub fn prepare_and_run<D, F>(points: &[(u32, &[f32])], execute: F)
where
    D: Distance,
    F: FnOnce(Duration, &Qdrant),
{
    let before_build = Instant::now();
    let dimensions = points[0].1.len();

    let points: Vec<_> = points
        .iter()
        .map(|(id, vector)| {
            PointStruct::new(
                *id as u64,
                vector.to_vec(),
                Payload::try_from(serde_json::json!({ "id": *id })).unwrap(),
            )
        })
        .collect();

    let client = tokio::runtime::Runtime::new().unwrap().block_on(async {
        let ip = Ipv4Addr::from_str("127.0.0.1").unwrap();
        let port = 6334;
        let url = format!("http://{}:{}/", ip, port);
        let client = Qdrant::from_url(&url).timeout(Duration::from_secs(1800)).build().unwrap();
        let collection_name = "hello";

        let _ = client.delete_collection(collection_name).await;

        client
            .create_collection(
                CreateCollectionBuilder::new(collection_name)
                    .vectors_config(VectorParamsBuilder::new(dimensions as u64, D::QDRANT_DISTANCE))
                    .quantization_config(D::qdrant_quantization_config()),
            )
            .await
            .unwrap();
        tokio::time::sleep(Duration::from_secs(1)).await;

        client
            .upsert_points_chunked(
                UpsertPointsBuilder::new(collection_name, points.clone()).wait(true),
                1000,
            )
            .await
            .unwrap();

        client
    });

    (execute)(before_build.elapsed(), &client);
}

pub fn run_scenarios<D: Distance>(
    env: &heed::Env,
    time_to_index: Duration,
    distance: &ScenarioDistance,
    search: Vec<&ScenarioSearch>,
    queries: Vec<(&u32, &&[f32], HashMap<ScenarioFiltering, (Option<RoaringBitmap>, Vec<u32>)>)>,
    database: arroy::Database<D>,
) {
    // let database_size =
    //     Byte::from_u64(env.non_free_pages_size().unwrap()).get_appropriate_unit(UnitType::Binary);

    // let mut database_size = 0u64;
    // let collection_path = format!("storage/collections/{collection_name}");
    // for result in walkdir::WalkDir::new(collection_path) {
    //     let entry = match result {
    //         Ok(entry) => entry,
    //         _ => continue,
    //     };
    //     database_size += entry.metadata().map_or(0, |metadata| metadata.len());
    // }
    // let database_size = Byte::from_u64(database_size).get_appropriate_unit(UnitType::Binary);

    // println!("indexing: {time_to_index:02.2?}, size: {database_size:#.2}");

    for ScenarioSearch { oversampling, filtering } in &search {
        //     let mut time_to_search = Duration::default();
        //     let mut recalls = Vec::new();
        //     for number_fetched in RECALL_TESTED {
        //         let (correctly_retrieved, duration) = queries
        //             .par_iter()
        //             .map(|(&id, _target, relevants)| {
        //                 let rtxn = env.read_txn().unwrap();
        //                 let reader = arroy::Reader::open(&rtxn, 0, database).unwrap();

        //                 let (candidates, relevants) = &relevants[filtering];
        //                 // Only keep the top number fetched documents.
        //                 let relevants = relevants.get(..number_fetched).unwrap_or(relevants);

        //                 let now = std::time::Instant::now();
        //                 let arroy_answer = reader
        //                     .nns_by_item(
        //                         &rtxn,
        //                         id,
        //                         number_fetched,
        //                         None,
        //                         oversampling.to_non_zero_usize(),
        //                         candidates.as_ref(),
        //                     )
        //                     .unwrap()
        //                     .unwrap();
        //                 let elapsed = now.elapsed();

        //                 let mut correctly_retrieved = Some(0);
        //                 for (id, _dist) in arroy_answer {
        //                     if relevants.contains(&id) {
        //                         if let Some(cr) = &mut correctly_retrieved {
        //                             *cr += 1;
        //                         }
        //                     } else if let Some(cand) = candidates.as_ref() {
        //                         // We set the counter to -1 if we return a filtered out candidated
        //                         if !cand.contains(id) {
        //                             correctly_retrieved = None;
        //                         }
        //                     }
        //                 }

        //                 (correctly_retrieved, elapsed)
        //             })
        //             .reduce(
        //                 || (Some(0), Duration::default()),
        //                 |(aanswer, aduration), (banswer, bduration)| {
        //                     (aanswer.zip(banswer).map(|(a, b)| a + b), aduration + bduration)
        //                 },
        //             );

        //         time_to_search += duration;
        //         // If non-candidate documents are returned we show a recall of -1
        //         let recall =
        //             correctly_retrieved.map_or(-1.0, |cr| cr as f32 / (number_fetched as f32 * 100.0));
        //         recalls.push(Recall(recall));
        //     }

        //     let filtered_percentage = filtering.to_ratio_f32() * 100.0;
        //     println!(
        //         "[arroy]  {distance:16?} {oversampling}: {recalls:?}, \
        //                                 searched for: {time_to_search:02.2?}, \
        //                                 searched in {filtered_percentage:#.2}%"
        //     );
    }
}

pub fn measure_qdrant_distance<
    D: Distance,
    const EXACT: bool,
    const FILTER_SUBSET_PERCENT: usize,
>(
    dimensions: usize,
    points: &[(u32, &[f32])],
) {
    let filtered_percentage = FILTER_SUBSET_PERCENT as f32;
    let candidates_range = if FILTER_SUBSET_PERCENT >= 100 {
        None
    } else {
        let count = (points.len() as f32 * (filtered_percentage / 100.0)) as usize;
        Some([points[0].0, points[count - 1].0])
    };

    let points: Vec<_> = points
        .iter()
        .map(|(id, vector)| {
            PointStruct::new(
                *id as u64,
                vector.to_vec(),
                Payload::try_from(serde_json::json!({ "id": *id })).unwrap(),
            )
        })
        .collect();

    tokio::runtime::Runtime::new().unwrap().block_on(async {
        let ip = Ipv4Addr::from_str("127.0.0.1").unwrap();
        let port = 6334;
        let url = format!("http://{}:{}/", ip, port);
        let client = Qdrant::from_url(&url).timeout(Duration::from_secs(1800)).build().unwrap();
        let collection_name = "hello";

        let _ = client.delete_collection(collection_name).await;

        client
            .create_collection(
                CreateCollectionBuilder::new(collection_name)
                    .vectors_config(VectorParamsBuilder::new(dimensions as u64, D::QDRANT_DISTANCE))
                    .quantization_config(D::qdrant_quantization_config()),
            )
            .await
            .unwrap();
        tokio::time::sleep(Duration::from_secs(1)).await;

        let before_build = Instant::now();
        let mut rng = StdRng::seed_from_u64(RNG_SEED);
        client
            .upsert_points_chunked(
                UpsertPointsBuilder::new(collection_name, points.clone()).wait(true),
                1000,
            )
            .await
            .unwrap();

        let mut database_size = 0u64;
        let collection_path = format!("storage/collections/{collection_name}");
        for result in walkdir::WalkDir::new(collection_path) {
            let entry = match result {
                Ok(entry) => entry,
                _ => continue,
            };
            database_size += entry.metadata().map_or(0, |metadata| metadata.len());
        }
        let database_size = Byte::from_u64(database_size).get_appropriate_unit(UnitType::Binary);
        let time_to_index = before_build.elapsed();

        let mut duration_secs = 0.0;
        let mut recalls = Vec::new();
        for number_fetched in RECALL_TESTED {
            if number_fetched > points.len() {
                break;
            }
            let mut correctly_retrieved = Some(0);
            for _ in 0..100 {
                let querying = points.choose(&mut rng).unwrap();

                let relevant = partial_sort_by::<D::RealDistance>(
                    points
                        .iter()
                        .filter(|point| {
                            // Only evaluate the candidate points
                            let get_id = |ps: &PointStruct| -> u64 {
                                match ps.id.as_ref().unwrap().point_id_options.as_ref().unwrap() {
                                    PointIdOptions::Num(id) => *id,
                                    _ => panic!("what!"),
                                }
                            };
                            match candidates_range {
                                Some([l, h]) => (l..=h).contains(&(get_id(point) as u32)),
                                None => true,
                            }
                        })
                        .map(|point| (get_id_from_point(point), get_vector_from_point(point))),
                    get_vector_from_point(querying),
                    number_fetched,
                );

                let search_builder = SearchPointsBuilder::new(
                    collection_name,
                    get_vector_from_point(querying),
                    number_fetched as u64,
                )
                .with_vectors(false)
                .params(SearchParamsBuilder::default().exact(EXACT));

                let response = client
                    .search_points(match candidates_range {
                        Some([l, h]) => {
                            let range = Range {
                                gte: Some(l as f64),
                                lte: Some(h as f64),
                                ..Default::default()
                            };
                            let condition = Condition::range("id", range);
                            search_builder.filter(Filter::must(Some(condition)))
                        }
                        None => search_builder,
                    })
                    .await
                    .unwrap();

                for point in response.result {
                    if relevant
                        .iter()
                        .any(|(id, _, _)| *id == get_id_from_id(point.id.as_ref().unwrap()))
                    {
                        if let Some(correctly_retrieved) = &mut correctly_retrieved {
                            *correctly_retrieved += 1;
                        }
                    } else if let Some([l, h]) = candidates_range {
                        if !(l..=h).contains(&get_id_from_id(point.id.as_ref().unwrap())) {}
                    }
                }

                duration_secs += response.time;
            }

            let recall = correctly_retrieved.unwrap_or(-1) as f32 / (number_fetched as f32 * 100.0);
            recalls.push(Recall(recall));
        }
        let time_to_search = Duration::try_from_secs_f64(duration_secs).unwrap();

        let mut distance_name = D::QDRANT_DISTANCE.as_str_name().to_string();
        if D::BINARY_QUANTIZED {
            distance_name.insert_str(0, "bq ");
        }
        if EXACT {
            distance_name.push_str(" exact");
        }
        println!(
            "[qdrant] {distance_name:16} x1: {recalls:?}, \
            indexed for: {time_to_index:02.2?}, \
            searched for: {time_to_search:02.2?}, \
            size on disk: {database_size:#.2}, \
            searched in {filtered_percentage:#.2}%"
        );
    });
}

fn get_id_from_id(id: &PointId) -> u32 {
    match id.point_id_options.as_ref().unwrap() {
        qdrant_client::qdrant::point_id::PointIdOptions::Num(n) => *n as u32,
        qdrant_client::qdrant::point_id::PointIdOptions::Uuid(_) => todo!("uuid not supported"),
    }
}

fn get_id_from_point(point: &PointStruct) -> u32 {
    get_id_from_id(point.id.as_ref().unwrap())
}

fn get_vector_from_point(point: &PointStruct) -> &[f32] {
    match point.vectors.as_ref().unwrap().vectors_options.as_ref().unwrap() {
        qdrant_client::qdrant::vectors::VectorsOptions::Vector(vec) => vec.data.as_slice(),
        qdrant_client::qdrant::vectors::VectorsOptions::Vectors(_) => todo!(),
    }
}
