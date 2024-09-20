#!/usr/bin/env bash

mkdir -p assets

mkdir -p tmp-assets
rm -rf tmp-assets/*

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
cat dbpedia-1536.urls | xargs -n 1 -P $max_parallel -I {} bash -c 'download_file {}'
cargo run --release -p parquet2mat -- \
    tmp-assets/* \
    --embedding-name 'text-embedding-3-large-1536-embedding' \
    --output assets/db-pedia-OpenAI-text-embedding-ada-002.mat
rm -rf tmp-assets/*

# Call the function to download parquet files in parallel
cat dbpedia-3072.urls | xargs -n 1 -P $max_parallel -I {} bash -c 'download_file {}'
cargo run --release -p parquet2mat -- \
    tmp-assets/* \
    --embedding-name 'text-embedding-3-large-3072-embedding' \
    --output assets/db-pedia-OpenAI-text-embedding-3-large.mat
rm -rf tmp-assets/*

curl -o assets/hn-posts.mat 'https://static.wilsonl.in/hackerverse/dataset/post-embs-data.mat'
curl -o assets/hn-top-posts.mat 'https://static.wilsonl.in/hackerverse/dataset/toppost-embs-data.mat'
