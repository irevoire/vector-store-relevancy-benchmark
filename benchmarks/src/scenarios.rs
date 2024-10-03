use std::fmt;
use std::num::NonZeroUsize;

use clap::ValueEnum;
use enum_iterator::Sequence;

use crate::MatLEView;

#[derive(Debug, Copy, Clone, ValueEnum, Sequence)]
pub enum Dataset {
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
pub enum ScenarioContender {
    Qdrant,
    Arroy,
    // Typesense,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, ValueEnum, Sequence)]
pub enum ScenarioDistance {
    Cosine,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, ValueEnum, Sequence)]
pub enum ScenarioOversampling {
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
pub enum ScenarioFiltering {
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
    pub fn to_ratio_f32(self) -> f32 {
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

#[derive(Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct ScenarioSearch {
    pub oversampling: ScenarioOversampling,
    pub filtering: ScenarioFiltering,
}
