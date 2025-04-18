#!/bin/bash

# Run script to test ColBERT implementation for Subtask 4b
# This script will activate the virtual environment and run the retrieval script

# Make sure we're in the correct directory
cd "$(dirname "$0")/.."

# Activate virtual environment
source ragatouille_env/bin/activate

# Install any missing dependencies
pip install tqdm rank_bm25

# Run the optimized version with specific parameters
# --batch_size: Number of queries to process at once
# --max_queries: Maximum number of queries to process (None for all)
# --num_workers: Number of parallel workers (set to 1 for stability)
# --test_size: Number of queries to use for time estimation
# --skip_indexing: Skip index creation if it already exists

# Small test run first with 50 queries
echo "Running small test with 50 queries..."
python colbertv2/colbert_retriever_optimized.py --batch_size 10 --max_queries 50 --num_workers 1 --test_size 10

# If you want to run on the full dataset after testing:
# echo "Running on full dataset..."
# python colbertv2/colbert_retriever_optimized.py --batch_size 20 --num_workers 1 --skip_indexing 
# python colbertv2/colbert_retriever_optimized.py --batch_size 20 --num_workers 1 --skip_indexing --use_wandb
