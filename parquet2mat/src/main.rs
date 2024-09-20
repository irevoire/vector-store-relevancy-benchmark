use std::fs::File;
use std::io::{self, Write};
use std::path::PathBuf;

use anyhow::{bail, Context};
use clap::Parser;
use parquet::file::reader::{FileReader, SerializedFileReader};
use parquet::record::{Field, Row};

/// Small tool that takes mltiple parquet files and extracts the embeddings to output them flat in a matrix file.
#[derive(Parser)]
struct Args {
    /// The set of parquet files to ouput in the output file.
    files: Vec<PathBuf>,

    /// The name of the embedding field to extract.
    #[arg(long)]
    embedding_name: String,

    /// The output file name.
    #[arg(long, default_value = "output.mat")]
    output: PathBuf,
}

fn main() -> anyhow::Result<()> {
    let Args {
        files,
        embedding_name,
        output: output_path,
    } = Args::parse();

    let output = File::create(&output_path)
        .with_context(|| format!("while opening {}", output_path.display()))?;
    let mut output = io::BufWriter::new(output);
    let mut total_embeddings_count: usize = 0;

    for file_path in files {
        let file = File::open(&file_path)?;
        let reader = SerializedFileReader::new(file)?;

        let mut embeddings_count = 0;
        for result in reader.get_row_iter(None)? {
            let row: Row = result?;
            for (name, field) in row.get_column_iter() {
                if name == &embedding_name {
                    embeddings_count += 1;
                    let list = match field {
                        Field::ListInternal(list) => list,
                        _ => bail!(
                            "this is not a list while processing {}",
                            file_path.display()
                        ),
                    };

                    let mut floats = Vec::with_capacity(list.len());
                    for element in list.elements() {
                        match element {
                            // Field::Float16(f16) => todo!(),
                            // Field::Float(_) => todo!(),
                            Field::Double(double) => floats.push(*double as f32),
                            _ => bail!(
                                "this is not a list of doubles while processing {}",
                                file_path.display()
                            ),
                        }
                    }

                    output.write_all(bytemuck::cast_slice(&floats))?;
                }
            }
        }

        println!("{} embeddings appended to the output.", embeddings_count);
        total_embeddings_count += embeddings_count;
    }

    output.flush()?;
    println!(
        "done appending {} embeddings into {}.",
        total_embeddings_count,
        output_path.display()
    );

    Ok(())
}
