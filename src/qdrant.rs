use std::{net::Ipv4Addr, str::FromStr, time::Duration};

use byte_unit::{Byte, UnitType};
use qdrant_client::{
    qdrant::{
        CreateCollectionBuilder, PointId, PointStruct, SearchParamsBuilder, SearchPointsBuilder,
        UpsertPointsBuilder, VectorParamsBuilder,
    },
    Payload, Qdrant,
};
use rand::prelude::SliceRandom;
use rand::{rngs::StdRng, SeedableRng};

use crate::{partial_sort_by, Distance, Recall, RECALL_TESTED};

pub fn measure_qdrant_distance<D: Distance, const EXACT: bool>(
    dimensions: usize,
    points: &[(u32, &[f32])],
) {
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
        let client = Qdrant::from_url(&url).build().unwrap();
        let collection_name = "hello";

        let _ = client.delete_collection(collection_name).await;

        client
            .create_collection(
                CreateCollectionBuilder::new(collection_name)
                    .vectors_config(VectorParamsBuilder::new(
                        dimensions as u64,
                        D::QDRANT_DISTANCE,
                    ))
                    .quantization_config(D::qdrant_quantization_config()
                    ),
            )
            .await
            .unwrap();
        tokio::time::sleep(Duration::from_secs(1)).await;

        let now = std::time::Instant::now();
        let mut rng = StdRng::seed_from_u64(13);
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
        let time_to_index = now.elapsed();

        let mut recalls = Vec::new();
        for number_fetched in RECALL_TESTED {
            if number_fetched > points.len() {
                break;
            }
            let mut correctly_retrieved = 0;
            for _ in 0..100 {
                let querying = points.choose(&mut rng).unwrap();

                let relevant = partial_sort_by::<D::RealDistance>(
                    points
                        .iter()
                        .map(|point| (get_id_from_point(point), get_vector_from_point(point))),
                    get_vector_from_point(querying),
                    number_fetched,
                );

                let qdrant = client
                    .search_points(
                        SearchPointsBuilder::new(
                            collection_name,
                            get_vector_from_point(querying),
                            number_fetched as u64,
                        )
                        .params(SearchParamsBuilder::default().exact(EXACT)),
                    )
                    .await
                    .unwrap();

                for point in qdrant.result {
                    if relevant
                        .iter()
                        .any(|(id, _, _)| *id == get_id_from_id(point.id.as_ref().unwrap()))
                    {
                        correctly_retrieved += 1;
                    }
                }
            }

            let recall = correctly_retrieved as f32 / (number_fetched as f32 * 100.0);
            recalls.push(Recall(recall));
        }
        let time_to_search = now.elapsed();

        let mut distance_name = D::QDRANT_DISTANCE.as_str_name().to_string();
        if D::BINARY_QUANTIZED {
            distance_name.insert_str(0, "bq ");
        }
        if EXACT {
            distance_name.push_str(" exact");
        }
        println!(
            "[qdrant] {distance_name:16} x1: {recalls:?}, indexed for: {time_to_index:02.2?}, searched for: {time_to_search:02.2?}, size on disk: {database_size:#.2}"
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
    match point
        .vectors
        .as_ref()
        .unwrap()
        .vectors_options
        .as_ref()
        .unwrap()
    {
        qdrant_client::qdrant::vectors::VectorsOptions::Vector(vec) => vec.data.as_slice(),
        qdrant_client::qdrant::vectors::VectorsOptions::Vectors(_) => todo!(),
    }
}
