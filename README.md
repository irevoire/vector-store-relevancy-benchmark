## Vector store benchmark

**Currently the arroy dependencies point to a local directory and the benchmarks cannot be run.**

## Usage

Download the datasets you want to use:
```bash
cargo run --release --bin download-db-pedia-OpenAI-text-embedding-3-large 
cargo run --release --bin download-hn 
```

It's going to take a long time.

Then run the benchmark with:
```bash
cargo run --release --bin run
```
