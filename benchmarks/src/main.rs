use std::collections::HashMap;
use std::fmt;
use std::num::NonZeroUsize;
use std::time::{Duration, Instant};

use arroy::distances::Angular;
use benchmarks::{prepare_and_run, MatLEView, Recall, RECALL_TESTED, RNG_SEED};
use byte_unit::{Byte, UnitType};
use clap::{Parser, ValueEnum};
use enum_iterator::Sequence;
use itertools::{iproduct, Itertools};
use ordered_float::OrderedFloat;
use rand::rngs::StdRng;
use rand::seq::SliceRandom as _;
use rand::SeedableRng;
use rayon::iter::{IntoParallelRefIterator as _, ParallelIterator};
use rayon::slice::ParallelSliceMut;
use roaring::RoaringBitmap;
use slice_group_by::GroupBy;

#[derive(Debug, Copy, Clone, ValueEnum, Sequence)]
enum Dataset {
    /// Hackernews posts (512)
    HnPosts,
    /// Wikipedia (768)
    Wikipedia,
    /// Hackernews top posts (1024)
    HnTopPost,
    /// db pedia OpenAI text-embedding ada 002 (1536)
    DbPediaAda002,
    /// db pedia OpenAI text-embedding 3 large (3072)
    DbPedia3Large,
}

impl From<Dataset> for MatLEView<f32> {
    fn from(dataset: Dataset) -> Self {
        match dataset {
            Dataset::HnPosts => MatLEView::new("Hackernews posts", "assets/hn-posts.mat", 512),
            Dataset::Wikipedia => MatLEView::new(
                "wikipedia 22 12 simple embeddings",
                "assets/wikipedia-22-12-simple-embeddings.mat",
                768,
            ),
            Dataset::HnTopPost => {
                MatLEView::new("Hackernews top posts", "assets/hn-top-posts.mat", 1024)
            }
            Dataset::DbPediaAda002 => MatLEView::new(
                "db pedia OpenAI text-embedding ada  002",
                "assets/db-pedia-OpenAI-text-embedding-ada-002.mat",
                1536,
            ),
            Dataset::DbPedia3Large => MatLEView::new(
                "db pedia OpenAI text-embedding 3 large",
                "assets/db-pedia-OpenAI-text-embedding-3-large.mat",
                3072,
            ),
        }
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, ValueEnum, Sequence)]
enum ScenarioContender {
    Qdrant,
    Arroy,
    // Typesense,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, ValueEnum, Sequence)]
enum ScenarioDistance {
    Cosine,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, ValueEnum, Sequence)]
enum ScenarioOversampling {
    X1,
    X3,
}

impl ScenarioOversampling {
    pub fn to_non_zero_usize(self) -> Option<NonZeroUsize> {
        match self {
            ScenarioOversampling::X1 => None,
            ScenarioOversampling::X3 => NonZeroUsize::new(3),
        }
    }
}

impl fmt::Display for ScenarioOversampling {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ScenarioOversampling::X1 => f.write_str("x1"),
            ScenarioOversampling::X3 => f.write_str("x3"),
        }
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, ValueEnum, Sequence)]
enum ScenarioFiltering {
    NoFilter,
    Filter50,
    Filter25,
    Filter15,
    Filter10,
    Filter8,
    Filter6,
    Filter2,
    Filter1,
}

impl ScenarioFiltering {
    pub fn to_f32(self) -> f32 {
        match self {
            ScenarioFiltering::NoFilter => 1.0,
            ScenarioFiltering::Filter50 => 0.50,
            ScenarioFiltering::Filter25 => 0.25,
            ScenarioFiltering::Filter15 => 0.15,
            ScenarioFiltering::Filter10 => 0.1,
            ScenarioFiltering::Filter8 => 0.08,
            ScenarioFiltering::Filter6 => 0.06,
            ScenarioFiltering::Filter2 => 0.02,
            ScenarioFiltering::Filter1 => 0.01,
        }
    }
}

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    /// The datasets to run and all of them are ran if empty.
    #[arg(long, value_enum)]
    datasets: Vec<Dataset>,

    #[arg(long, value_enum)]
    contenders: Vec<ScenarioContender>,

    #[arg(long, value_enum)]
    distances: Vec<ScenarioDistance>,

    #[arg(long, value_enum)]
    over_samplings: Vec<ScenarioOversampling>,

    #[arg(long, value_enum)]
    filterings: Vec<ScenarioFiltering>,

    /// Number of vectors to evaluate from the datasets.
    #[arg(long, default_value_t = 10_000)]
    count: usize,
}

#[derive(Debug, PartialEq, Eq, PartialOrd, Ord)]
struct ScenarioSearch {
    oversampling: ScenarioOversampling,
    filtering: ScenarioFiltering,
}

fn main() {
    let Args { datasets, count, contenders, distances, over_samplings, filterings } = Args::parse();

    let datasets = set_or_all::<_, MatLEView<f32>>(datasets);
    let contenders = set_or_all::<_, ScenarioContender>(contenders);
    let distances = set_or_all::<_, ScenarioDistance>(distances);
    let over_samplings = set_or_all::<_, ScenarioOversampling>(over_samplings);
    let filterings = set_or_all::<_, ScenarioFiltering>(filterings);

    let scenaris: Vec<_> = iproduct!(datasets, distances, contenders, over_samplings, filterings)
        .map(|(dataset, distance, contender, oversampling, filtering)| {
            (dataset, distance, contender, ScenarioSearch { oversampling, filtering })
        })
        .sorted()
        .collect();

    let mut previous_dataset = None;
    for grp in scenaris
        .linear_group_by(|(da, dia, ca, _), (db, dib, cb, _)| da == db && dia == dib && ca == cb)
    {
        let (dataset, distance, contender, _) = &grp[0];
        let search: Vec<&ScenarioSearch> = grp.iter().map(|(_, _, _, s)| s).collect();

        if previous_dataset != Some(dataset.name()) {
            previous_dataset = Some(dataset.name());
            dataset.header();
            if dataset.len() != count {
                println!("\x1b[1m{count}\x1b[0m vectors are used for this measure");
            }
        }

        let points: Vec<_> =
            dataset.iter().take(count).enumerate().map(|(i, v)| (i as u32, v)).collect();

        // let mut recall_tested = String::new();
        // RECALL_TESTED.iter().for_each(|recall| write!(&mut recall_tested, "{recall:4}, ").unwrap());
        // let recall_tested = recall_tested.trim_end_matches(", ");
        // println!("Recall tested is:             [{recall_tested}]");

        let max = RECALL_TESTED.iter().max().copied().unwrap();
        let mut rng = StdRng::seed_from_u64(RNG_SEED);
        let queries: Vec<_> = (0..100)
            .map(|_| points.choose(&mut rng).unwrap())
            .map(|(id, target)| {
                let mut points = points.clone();
                points.par_sort_unstable_by_key(|(_, v)| {
                    OrderedFloat(benchmarks::distance::<Angular>(target, v))
                });

                // We collect the different filtered versions here.
                let filtered: HashMap<_, _> = search
                    .iter()
                    .map(|ScenarioSearch { filtering, .. }| {
                        let candidates = match filtering {
                            ScenarioFiltering::NoFilter => None,
                            filtering => {
                                let total = points.len() as f32;
                                let filtering = filtering.to_f32();
                                Some(
                                    points
                                        .iter()
                                        .map(|(id, _)| id)
                                        .take((total * filtering) as usize)
                                        .collect::<RoaringBitmap>(),
                                )
                            }
                        };

                        // This is the real expected answer without the filtered out candidates.
                        let answer = points
                            .iter()
                            .map(|(id, _)| *id)
                            .filter(|&id| candidates.as_ref().map_or(true, |c| c.contains(id)))
                            .take(max)
                            .collect::<Vec<_>>();

                        (*filtering, (candidates, answer))
                    })
                    .collect();

                (id, target, filtered)
            })
            .collect();

        match contender {
            ScenarioContender::Qdrant => todo!(),
            ScenarioContender::Arroy => {
                match distance {
                    ScenarioDistance::Cosine => {
                        let before_build = Instant::now();
                        prepare_and_run::<Angular, _>(&points, |env, database| {
                            let time_to_index = before_build.elapsed();
                            let database_size = Byte::from_u64(env.non_free_pages_size().unwrap())
                                .get_appropriate_unit(UnitType::Binary);

                            for ScenarioSearch { oversampling, filtering } in &search {
                                let mut time_to_search = Duration::default();
                                let mut recalls = Vec::new();
                                for number_fetched in RECALL_TESTED {
                                    let (correctly_retrieved, duration) = queries
                                        .par_iter()
                                        .map(|(&id, _target, relevants)| {
                                            let rtxn = env.read_txn().unwrap();
                                            let reader =
                                                arroy::Reader::open(&rtxn, 0, database).unwrap();

                                            let (candidates, relevants) = &relevants[filtering];
                                            let now = std::time::Instant::now();
                                            let arroy_answer = reader
                                                .nns_by_item(
                                                    &rtxn,
                                                    id,
                                                    number_fetched,
                                                    None,
                                                    oversampling.to_non_zero_usize(),
                                                    candidates.as_ref(),
                                                )
                                                .unwrap()
                                                .unwrap();
                                            let elapsed = now.elapsed();

                                            let mut correctly_retrieved = Some(0);
                                            for (id, _dist) in arroy_answer {
                                                if relevants.iter().any(|&rid| rid == id) {
                                                    if let Some(cr) = &mut correctly_retrieved {
                                                        *cr += 1;
                                                    }
                                                } else if let Some(cand) = candidates.as_ref() {
                                                    // We set the counter to -1 if we return a filtered out candidated
                                                    if !cand.contains(id) {
                                                        correctly_retrieved = None;
                                                    }
                                                }
                                            }

                                            (correctly_retrieved, elapsed)
                                        })
                                        .reduce(
                                            || (Some(0), Duration::default()),
                                            |(aanswer, aduration), (banswer, bduration)| {
                                                (
                                                    aanswer.zip(banswer).map(|(a, b)| a + b),
                                                    aduration + bduration,
                                                )
                                            },
                                        );

                                    time_to_search += duration;
                                    // If non-cnadidate documents are returned we show a recall of -1
                                    let recall = correctly_retrieved.map_or(-1.0, |cr| {
                                        cr as f32 / (number_fetched as f32 * 100.0)
                                    });
                                    recalls.push(Recall(recall));
                                }

                                let filtered_percentage = filtering.to_f32() * 100.0;
                                println!(
                                    "[arroy]  {distance:16?} {oversampling}: {recalls:?}, \
                                    indexed for: {time_to_index:02.2?}, \
                                    searched for: {time_to_search:02.2?}, \
                                    size on disk: {database_size:#.2}, \
                                    searched in {filtered_percentage:#.2}%"
                                );
                            }
                        })
                    }
                }
            }
        }

        println!();
    }

    // for dataset in datasets {
    //     let vectors: Vec<_> =
    //         dataset.iter().enumerate().map(|(i, v)| (i as u32, v)).take(count).collect();
    //     dataset.header();
    //     bench_over_all_distances(dataset.dimensions(), vectors.as_slice());
    //     println!();
    // }
}

fn set_or_all<S, T>(datasets: Vec<S>) -> Vec<T>
where
    S: Sequence,
    S: Into<T>,
{
    if datasets.is_empty() {
        enum_iterator::all::<S>().map(Into::into).collect()
    } else {
        datasets.into_iter().map(Into::into).collect()
    }
}
