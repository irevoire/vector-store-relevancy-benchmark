#!/usr/bin/env bash

mkdir -p assets

cargo run --release --bin download-db-pedia-OpenAI-text-embedding-3-large

curl -o assets/hn-posts.mat 'https://static.wilsonl.in/hackerverse/dataset/post-embs-data.mat'
curl -o assets/hn-top-posts.mat 'https://static.wilsonl.in/hackerverse/dataset/toppost-embs-data.mat'
