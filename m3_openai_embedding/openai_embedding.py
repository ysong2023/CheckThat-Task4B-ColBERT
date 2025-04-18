#!/usr/bin/env python
# coding: utf-8

# # OpenAI Embedding Approach
# 
# ### CLEF 2025 - CheckThat! Lab  - Task 4 Scientific Web Discourse - Subtask 4b (Scientific Claim Source Retrieval)
# 
# This script implements an approach using OpenAI embeddings for the source retrieval task. It includes:
# - Code to load data (collection set and query set)
# - Code to generate OpenAI embeddings for papers and tweets
# - Code to calculate cosine similarity and retrieve top matches
# - Code to evaluate the model using MRR@k

import numpy as np
import pandas as pd
import os
import time
from tqdm import tqdm
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Set your OpenAI API key from environment variable
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Configuration
config = {
    "embedding_model": "text-embedding-3-small",
    "batch_size": 100,
    "top_k": 5,
}

# Paths to data files
PATH_COLLECTION_DATA = 'subtask4b_collection_data.pkl'
PATH_QUERY_TRAIN_DATA = 'subtask4b_query_tweets_train.tsv'
PATH_QUERY_DEV_DATA = 'subtask4b_query_tweets_dev.tsv'

print("Loading collection and query data...")
# Load the collection set (CORD-19 papers)
df_collection = pd.read_pickle(PATH_COLLECTION_DATA)
# Load the query sets (tweets)
df_query_train = pd.read_csv(PATH_QUERY_TRAIN_DATA, sep='\t')
df_query_dev = pd.read_csv(PATH_QUERY_DEV_DATA, sep='\t')

print("Collection set shape:", df_collection.shape)
print("Train query set shape:", df_query_train.shape)
print("Dev query set shape:", df_query_dev.shape)

# Function to get OpenAI embeddings with rate limiting and retry logic
def get_embedding(text, model="text-embedding-3-small", max_retries=5):
    """
    Get embeddings from OpenAI API with retry logic for rate limiting
    """
    delay = 1
    for attempt in range(max_retries):
        try:
            text = text.replace("\n", " ")
            response = client.embeddings.create(
                input=[text],
                model=model
            )
            return response.data[0].embedding
        except Exception as e:
            if "rate limit" in str(e).lower() and attempt < max_retries - 1:
                print(f"Rate limit hit, retrying in {delay} seconds...")
                time.sleep(delay)
                delay *= 2  # Exponential backoff
            else:
                print(f"Error generating embedding: {e}")
                # Return a zero vector as fallback
                return [0.0] * (1536 if "3-small" in model else 3072)  # Adjust size based on model

# Function to get embeddings in batches to optimize API calls
def get_embeddings_batch(texts, model="text-embedding-3-small", batch_size=100):
    """
    Get embeddings for a list of texts in batches
    """
    all_embeddings = []
    
    for i in tqdm(range(0, len(texts), batch_size), desc="Generating embeddings"):
        batch_texts = texts[i:i+batch_size]
        # Clean texts
        batch_texts = [text.replace("\n", " ") for text in batch_texts]
        
        try:
            response = client.embeddings.create(
                input=batch_texts,
                model=model
            )
            batch_embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(batch_embeddings)
            
            # Avoid rate limits
            if i + batch_size < len(texts):
                time.sleep(0.5)
        except Exception as e:
            print(f"Error in batch {i}-{i+batch_size}: {e}")
            # Fallback to individual processing with retries
            batch_embeddings = []
            for text in batch_texts:
                emb = get_embedding(text, model)
                batch_embeddings.append(emb)
            all_embeddings.extend(batch_embeddings)
    
    return all_embeddings

# Create a cache directory for storing embeddings to avoid regenerating them
os.makedirs("embedding_cache", exist_ok=True)

# Generate or load paper embeddings
paper_embeddings_file = "embedding_cache/paper_embeddings.npy"
paper_cord_uids_file = "embedding_cache/paper_cord_uids.npy"

start_time = time.time()
if os.path.exists(paper_embeddings_file) and os.path.exists(paper_cord_uids_file):
    print("Loading cached paper embeddings...")
    paper_embeddings = np.load(paper_embeddings_file)
    cord_uids = np.load(paper_cord_uids_file, allow_pickle=True)
    print(f"Loaded embeddings in {time.time() - start_time:.2f} seconds")
else:
    print("Generating paper embeddings (this may take some time)...")
    # Create text representations of papers (title + abstract)
    paper_texts = df_collection[['title', 'abstract']].apply(
        lambda x: f"{x['title']} {x['abstract']}", axis=1).tolist()
    cord_uids = df_collection['cord_uid'].tolist()
    
    # Get embeddings for papers
    paper_embeddings = get_embeddings_batch(paper_texts, model=config["embedding_model"], batch_size=config["batch_size"])
    paper_embeddings = np.array(paper_embeddings)
    
    # Save embeddings to cache
    np.save(paper_embeddings_file, paper_embeddings)
    np.save(paper_cord_uids_file, np.array(cord_uids, dtype=object))
    print(f"Generated embeddings in {time.time() - start_time:.2f} seconds")

# Function to get top cord_uids for a given query
def get_top_cord_uids_embedding(query_text, top_k=5):
    """
    Retrieve top k papers for a query using embedding similarity
    """
    # Get the embedding for the query
    query_embedding = get_embedding(query_text, model=config["embedding_model"])
    
    # Calculate cosine similarity between query and all papers
    similarities = cosine_similarity([query_embedding], paper_embeddings)[0]
    
    # Get indices of top k similar papers
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    top_similarities = [similarities[i] for i in top_indices]
    
    # Return the cord_uids of top k papers and their similarities
    return [cord_uids[i] for i in top_indices], top_similarities

# Process the dev set first (it's smaller) to verify the approach
print("Processing dev set...")
# To avoid reprocessing, we'll use a cache file
dev_predictions_file = "embedding_cache/dev_predictions.pkl"
dev_similarities_file = "embedding_cache/dev_similarities.pkl"

start_time = time.time()
if os.path.exists(dev_predictions_file) and os.path.exists(dev_similarities_file):
    print("Loading cached dev predictions...")
    df_query_dev = pd.read_pickle(dev_predictions_file)
    similarities_dev = pd.read_pickle(dev_similarities_file)
    print(f"Loaded predictions in {time.time() - start_time:.2f} seconds")
else:
    # Apply embedding-based retrieval to dev set
    results = [get_top_cord_uids_embedding(x, top_k=config["top_k"]) for x in tqdm(df_query_dev['tweet_text'], desc="Processing dev queries")]
    df_query_dev['embedding_topk'] = [r[0] for r in results]
    similarities_dev = [r[1] for r in results]
    
    # Save to cache
    df_query_dev.to_pickle(dev_predictions_file)
    pd.to_pickle(similarities_dev, dev_similarities_file)
    print(f"Processed dev set in {time.time() - start_time:.2f} seconds")

# Process the train set if needed (you might skip this for development as it's larger)
process_train = False  # Set to True if you want to process the training set
train_predictions_file = "embedding_cache/train_predictions.pkl"
train_similarities_file = "embedding_cache/train_similarities.pkl"

if process_train:
    start_time = time.time()
    if os.path.exists(train_predictions_file) and os.path.exists(train_similarities_file):
        print("Loading cached train predictions...")
        df_query_train = pd.read_pickle(train_predictions_file)
        similarities_train = pd.read_pickle(train_similarities_file)
        print(f"Loaded train predictions in {time.time() - start_time:.2f} seconds")
    else:
        print("Processing train set...")
        results = [get_top_cord_uids_embedding(x, top_k=config["top_k"]) for x in tqdm(df_query_train['tweet_text'], desc="Processing train queries")]
        df_query_train['embedding_topk'] = [r[0] for r in results]
        similarities_train = [r[1] for r in results]
        
        # Save to cache
        df_query_train.to_pickle(train_predictions_file)
        pd.to_pickle(similarities_train, train_similarities_file)
        print(f"Processed train set in {time.time() - start_time:.2f} seconds")

# Evaluate using Mean Reciprocal Rank (MRR@k)
def get_performance_mrr(data, col_gold, col_pred, list_k=[1, 5, 10]):
    """
    Calculate Mean Reciprocal Rank for different k values
    """
    d_performance = {}
    for k in list_k:
        data["in_topx"] = data.apply(
            lambda x: (1/([i for i in x[col_pred][:k]].index(x[col_gold]) + 1) 
                      if x[col_gold] in [i for i in x[col_pred][:k]] else 0), 
            axis=1
        )
        d_performance[k] = data["in_topx"].mean()
    return d_performance

# Evaluate the embedding-based approach
print("Evaluating the embedding-based approach...")
results_dev = get_performance_mrr(df_query_dev, 'cord_uid', 'embedding_topk', list_k=[1, 5, 10])
print(f"Results on the dev set: {results_dev}")

if process_train:
    results_train = get_performance_mrr(df_query_train, 'cord_uid', 'embedding_topk', list_k=[1, 5, 10])
    print(f"Results on the train set: {results_train}")

# Export results for submission
df_query_dev['preds'] = df_query_dev['embedding_topk'].apply(lambda x: x[:5])
df_query_dev[['post_id', 'preds']].to_csv('predictions.tsv', index=None, sep='\t')

print("Done! Predictions saved to 'predictions.tsv'")

# Compare with BM25 baseline (optional)
run_bm25_comparison = False

if run_bm25_comparison:
    print("Running BM25 baseline for comparison...")
    from rank_bm25 import BM25Okapi
    
    # Create the BM25 corpus
    corpus = df_collection[['title', 'abstract']].apply(
        lambda x: f"{x['title']} {x['abstract']}", axis=1).tolist()
    tokenized_corpus = [doc.split(' ') for doc in corpus]
    bm25 = BM25Okapi(tokenized_corpus)
    
    def get_top_cord_uids_bm25(query):
        tokenized_query = query.split(' ')
        doc_scores = bm25.get_scores(tokenized_query)
        indices = np.argsort(-doc_scores)[:5]
        bm25_topk = [cord_uids[i] for i in indices]
        return bm25_topk
    
    # Apply BM25 to dev set
    df_query_dev['bm25_topk'] = df_query_dev['tweet_text'].apply(get_top_cord_uids_bm25)
    
    # Evaluate BM25
    results_bm25_dev = get_performance_mrr(df_query_dev, 'cord_uid', 'bm25_topk')
    print(f"BM25 results on the dev set: {results_bm25_dev}")
    
    # Print improvement
    for k in results_dev.keys():
        improvement = results_dev[k] - results_bm25_dev[k]
        improvement_pct = improvement/results_bm25_dev[k]*100
        print(f"Improvement for MRR@{k}: {improvement:.4f} ({improvement_pct:.2f}%)") 