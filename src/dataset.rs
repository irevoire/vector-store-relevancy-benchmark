use std::{fs::File, marker::PhantomData, mem};

use bytemuck::{AnyBitPattern, PodCastError};
use memmap2::Mmap;

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
