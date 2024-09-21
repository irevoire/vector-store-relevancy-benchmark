#!/usr/bin/env bash

mkdir -p assets
mkdir -p tmp-assets

# Function to download a file, retrying a specified number of times
download_file() {
    # Max number of retries
    url=$1
    max_retries=3
    retries=0

    cd tmp-assets

    while [ "$retries" -lt "$max_retries" ]; do
        echo "Downloading $url"
        curl -sSLO "$url" && break
        ((retries++))
        echo "Retry number $retries"
    done
    echo "Downloaded $url"
}


# Define the download_file function as a shell function
export -f download_file

# Build the parquet2mat tool in advance
cargo build --release -p parquet2mat

#  /$$$$$$$                                    /$$                           /$$
#  | $$__  $$                                  | $$                          | $$
#  | $$  \ $$  /$$$$$$  /$$  /$$  /$$ /$$$$$$$ | $$  /$$$$$$   /$$$$$$   /$$$$$$$  /$$$$$$$
#  | $$  | $$ /$$__  $$| $$ | $$ | $$| $$__  $$| $$ /$$__  $$ |____  $$ /$$__  $$ /$$_____/
#  | $$  | $$| $$  \ $$| $$ | $$ | $$| $$  \ $$| $$| $$  \ $$  /$$$$$$$| $$  | $$|  $$$$$$
#  | $$  | $$| $$  | $$| $$ | $$ | $$| $$  | $$| $$| $$  | $$ /$$__  $$| $$  | $$ \____  $$
#  | $$$$$$$/|  $$$$$$/|  $$$$$/$$$$/| $$  | $$| $$|  $$$$$$/|  $$$$$$$|  $$$$$$$ /$$$$$$$/
#  |_______/  \______/  \_____/\___/ |__/  |__/|__/ \______/  \_______/ \_______/|_______/

# Max number of parallel downloads
max_parallel=10

# Call the function to download parquet files in parallel
output=assets/db-pedia-OpenAI-text-embedding-ada-002.mat
if [ ! -f $output ]; then
    cat dbpedia-1536.urls | xargs -n 1 -P $max_parallel -I {} bash -c 'download_file {}'
    cargo run --release -p parquet2mat -- \
        tmp-assets/* \
        --embedding-name 'text-embedding-3-large-1536-embedding' \
        --output $output
    rm -rf tmp-assets/*
fi

# Call the function to download parquet files in parallel
output=assets/db-pedia-OpenAI-text-embedding-3-large.mat
if [ ! -f $output ]; then
    cat dbpedia-3072.urls | xargs -n 1 -P $max_parallel -I {} bash -c 'download_file {}'
    cargo run --release -p parquet2mat -- \
        tmp-assets/* \
        --embedding-name 'text-embedding-3-large-3072-embedding' \
        --output $output
    rm -rf tmp-assets/*
fi

# Call the function to download parquet files in parallel
output=assets/wikipedia-22-12-simple-embeddings.mat
if [ ! -f $output ]; then
    cat wikipedia-768.urls | xargs -n 1 -P $max_parallel -I {} bash -c 'download_file {}'
    cargo run --release -p parquet2mat -- \
        tmp-assets/* \
        --embedding-name 'emb' \
        --output $output
    rm -rf tmp-assets/*
fi

output=assets/hn-posts.mat
if [ ! -f $output ]; then
    curl -o $output 'https://static.wilsonl.in/hackerverse/dataset/post-embs-data.mat'
fi

output=assets/hn-top-posts.mat
if [ ! -f $output ]; then
    curl -o $output 'https://static.wilsonl.in/hackerverse/dataset/toppost-embs-data.mat'
fi

rm -rf tmp-assets
