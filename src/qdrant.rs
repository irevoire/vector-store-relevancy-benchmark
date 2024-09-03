use dockertest::{waitfor::MessageSource, DockerTest, Image, Source, TestBodySpecification};
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

pub fn measure_qdrant_distance<D: Distance>(dimensions: usize, points: &[(u32, &[f32])]) {
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

    let mut docker = DockerTest::new();
    let image = Image::with_repository("qdrant/qdrant").source(Source::DockerHub);
    let mut spec = TestBodySpecification::with_image(image);
    spec.modify_port_map(6333, 6333);
    spec.modify_port_map(6334, 6334);
    docker.provide_container(spec);
    docker.run(|ops| async move {
        // A handle to operate on the Container.
        let container = ops.handle("qdrant/qdrant");
        container
            .assert_message("Qdrant gRPC listening on 6334", MessageSource::Stdout, 10)
            .await;

        let (ip, port) = container.host_port(6334).unwrap();
        let url = format!("http://{}:{}/", ip, port);
        let client = Qdrant::from_url(&url).build().unwrap();

        let collection_name = "hello";
        client
            .create_collection(
                CreateCollectionBuilder::new(collection_name)
                    .vectors_config(VectorParamsBuilder::new(
                        dimensions as u64,
                        D::QDRANT_DISTANCE,
                    ))
                    .quantization_config(D::qdrant_quantization_config()),
            )
            .await
            .unwrap();

        let now = std::time::Instant::now();
        let mut rng = StdRng::seed_from_u64(13);
        client
            .upsert_points_chunked(UpsertPointsBuilder::new(collection_name, points.clone()).wait(true), 1000)
            .await
            .unwrap();
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
                    points.iter().map(|point| (get_id_from_point(point), get_vector_from_point(point))),
                    get_vector_from_point(querying),
                    number_fetched,
                );

            let qdrant = client
                .search_points(
                    SearchPointsBuilder::new(collection_name, get_vector_from_point(querying), number_fetched as u64)
                        .params(SearchParamsBuilder::default().exact(true)),
                )
                .await.unwrap();

                for point in qdrant.result {
                    if relevant.iter().any(|(id, _, _)| *id == get_id_from_id(point.id.as_ref().unwrap())) {
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
            distance_name.push_str(" bq");
        }
        println!(
            "[qdrant] {distance_name:12} x1: {recalls:?}, indexed for: {time_to_index:02.2?}, searched for: {time_to_search:02.2?}"
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
