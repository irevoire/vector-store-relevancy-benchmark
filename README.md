## Vector store benchmark

**Currently the arroy dependencies point to a local directory and the benchmarks cannot be run.**

## Usage

Download the datasets you want to use:
```bash
./download.sh
```

It's going to take a long time.

Make sure you have a qdrant server running with the default parameter.
You can [find the binaries](https://github.com/qdrant/qdrant/releases/latest) on their GitHub releases page.
```bash
./qdrant
```

Then run the benchmark with:
```bash
cargo run --release
```
