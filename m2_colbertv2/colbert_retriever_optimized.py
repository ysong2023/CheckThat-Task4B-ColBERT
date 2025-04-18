"""
Optimized ColBERT implementation for the CheckThat 2025 Subtask 4b (Scientific Claim Source Retrieval)

This script uses RAGatouille with ColBERT to retrieve the mentioned papers from tweets.
Optimized with batch processing and parallel execution for better performance.
"""

import os
import time
import pandas as pd
import numpy as np
from tqdm import tqdm
import pickle
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

# Try to import wandb, but continue if not available
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    print("Warning: wandb not installed. Experiment tracking will be disabled.")
    print("To enable, install with: pip install wandb")
    WANDB_AVAILABLE = False

# Force CPU-only mode for stability in WSL
os.environ['CUDA_VISIBLE_DEVICES'] = ''

def process_batch(batch_queries, model, k=5):
    """Process a batch of queries using the ColBERT model"""
    results = []
    for query in batch_queries:
        search_results = model.search(query, k=k)
        results.append([result['document_id'] for result in search_results])
    return results

def get_performance_mrr(data, col_gold, col_pred, list_k=[1, 5, 10]):
    """Calculate Mean Reciprocal Rank for different k values"""
    d_performance = {}
    for k in list_k:
        data["in_topx"] = data.apply(
            lambda x: (1/([i for i in x[col_pred][:k]].index(x[col_gold]) + 1) 
                       if x[col_gold] in [i for i in x[col_pred][:k]] else 0), 
            axis=1
        )
        d_performance[k] = data["in_topx"].mean()
    return d_performance

def run_bm25_baseline(corpus, cord_uids, queries, k=5):
    """Run BM25 baseline on the queries"""
    from rank_bm25 import BM25Okapi
    
    # Tokenize corpus for BM25
    tokenized_corpus = [doc.split(' ') for doc in corpus]
    bm25 = BM25Okapi(tokenized_corpus)
    
    results = []
    for query in tqdm(queries, desc="BM25 search"):
        tokenized_query = query.split(' ')
        doc_scores = bm25.get_scores(tokenized_query)
        indices = np.argsort(-doc_scores)[:k]
        results.append([cord_uids[x] for x in indices])
    
    return results

def log_to_wandb(data, key=None):
    """Log data to wandb if available"""
    if WANDB_AVAILABLE and WANDB_INITIALIZED:
        if key:
            wandb.log({key: data})
        else:
            wandb.log(data)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run ColBERT retrieval for Scientific Claim Source Retrieval')
    parser.add_argument('--batch_size', type=int, default=10, help='Batch size for processing queries')
    parser.add_argument('--max_queries', type=int, default=None, help='Maximum number of queries to process (None for all)')
    parser.add_argument('--num_workers', type=int, default=None, help='Number of parallel workers (None for auto)')
    parser.add_argument('--test_size', type=int, default=20, help='Number of test queries for time estimation')
    parser.add_argument('--skip_indexing', action='store_true', help='Skip index creation if it already exists')
    parser.add_argument('--top_k', type=int, default=5, help='Number of results to retrieve')
    parser.add_argument('--wandb_key', type=str, default="97df37db0da7e2ca101b94391db47d874ab98d24", help='Weights & Biases API key')
    parser.add_argument('--wandb_project', type=str, default="scientific-web-discourse", help='Weights & Biases project name')
    parser.add_argument('--wandb_entity', type=str, default="raquelzacano37-university-of-british-columbia", help='Weights & Biases entity name')
    parser.add_argument('--use_wandb', action='store_true', help='Enable Weights & Biases logging')
    args = parser.parse_args()

    # Set number of workers
    if args.num_workers is None:
        args.num_workers = max(1, multiprocessing.cpu_count() - 1)  # Leave one CPU for system
    
    # Initialize flag to track if wandb is initialized
    WANDB_INITIALIZED = False
    
    # Initialize Weights & Biases if available and enabled
    if WANDB_AVAILABLE and args.use_wandb:
        try:
            wandb.login(key=args.wandb_key)
        except Exception as e:
            print(f"Failed to login to wandb: {e}")
            WANDB_AVAILABLE = False
    
    print(f"Starting optimized ColBERT implementation with {args.num_workers} workers...")
    
    # Load datasets
    print("Loading datasets...")
    df_collection = pd.read_pickle("../subtask_4b/subtask4b_collection_data.pkl")
    df_query_train = pd.read_csv("../subtask_4b/subtask4b_query_tweets_train.tsv", sep='\t')
    df_query_dev = pd.read_csv("../subtask_4b/subtask4b_query_tweets_dev.tsv", sep='\t')
    
    print(f"Collection set: {df_collection.shape[0]} papers")
    print(f"Train query set: {df_query_train.shape[0]} tweets")
    print(f"Dev query set: {df_query_dev.shape[0]} tweets")
    
    # Initialize wandb run after loading data to get accurate collection size
    if WANDB_AVAILABLE and args.use_wandb:
        try:
            wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                name="4b_colbertv2_retrieval",
                config={
                    "model": "colbert-ir/colbertv2.0",
                    "retrieval_method": "ColBERT late interaction",
                    "batch_size": args.batch_size,
                    "top_k": args.top_k,
                    "collection_size": df_collection.shape[0],
                    "query_size": df_query_dev.shape[0] if args.max_queries is None else min(args.max_queries, df_query_dev.shape[0]),
                    "num_workers": args.num_workers,
                    "use_faiss": False,
                    "cpu_only": True,
                }
            )
            
            # Log hardware specs
            wandb.config.update({
                "cpu": "i9-15700",
                "ram": "64GB",
                "gpu": "GeForce RTX 4080 (not used)",
            })
            
            # Mark wandb as initialized
            WANDB_INITIALIZED = True
            print("Weights & Biases initialized successfully")
        except Exception as e:
            print(f"Failed to initialize wandb: {e}")
            WANDB_AVAILABLE = False
    
    # Prepare the collection - combine title and abstract for better retrieval
    print("Preparing document collection...")
    corpus = df_collection[['title', 'abstract']].apply(lambda x: f"{x['title']} {x['abstract']}", axis=1).tolist()
    cord_uids = df_collection['cord_uid'].tolist()
    
    # Create document list with IDs
    documents = corpus
    document_ids = cord_uids
    
    # Import RAGatouille
    from ragatouille import RAGPretrainedModel
    
    # Check if index exists and create if needed
    index_path = os.path.join('.ragatouille', 'colbert', 'indexes', 'papers_index')
    
    indexing_time = 0
    if not os.path.exists(index_path) or not args.skip_indexing:
        # Create an index with our papers collection
        print("Loading ColBERT model...")
        RAG = RAGPretrainedModel.from_pretrained("colbert-ir/colbertv2.0")
        
        print("Creating index from documents (this may take a while)...")
        start_time = time.time()
        
        index_path = RAG.index(
            collection=documents,
            document_ids=document_ids,
            index_name="papers_index",
            use_faiss=False  # Using CPU-only mode for stability in WSL
        )
        
        indexing_time = time.time() - start_time
        print(f"Index creation completed in {indexing_time:.2f} seconds")
        log_to_wandb(indexing_time, "indexing_time_seconds")
    else:
        # Use from_index() when skipping indexing to properly load the existing index
        print(f"Using existing index at {index_path}")
        RAG = RAGPretrainedModel.from_index(index_path)
    
    # Test on a small subset to estimate time
    print(f"Running test retrieval on {args.test_size} queries...")
    test_queries = df_query_dev['tweet_text'].iloc[:args.test_size].tolist()
    
    start_time = time.time()
    test_batches = [test_queries[i:i + args.batch_size] for i in range(0, len(test_queries), args.batch_size)]
    test_results = []
    
    for batch in tqdm(test_batches, desc="Test batches"):
        test_results.extend(process_batch(batch, RAG, k=args.top_k))
    
    test_time = time.time() - start_time
    print(f"Test retrieval completed in {test_time:.2f} seconds for {args.test_size} queries")
    log_to_wandb({
        "test_query_time_seconds": test_time,
        "test_query_time_per_query": test_time / args.test_size
    })
    
    # Calculate estimated time for full dataset
    total_queries = df_query_dev.shape[0] if args.max_queries is None else min(args.max_queries, df_query_dev.shape[0])
    estimated_time = (test_time / args.test_size) * total_queries
    print(f"Estimated time for {total_queries} queries: {estimated_time:.2f} seconds ({estimated_time/60:.2f} minutes)")
    
    # Get queries to process
    max_queries = args.max_queries
    queries = df_query_dev['tweet_text'].iloc[:max_queries].tolist() if max_queries else df_query_dev['tweet_text'].tolist()
    query_ids = df_query_dev['post_id'].iloc[:max_queries].tolist() if max_queries else df_query_dev['post_id'].tolist()
    
    # Process in batches
    print(f"Processing {len(queries)} queries in batches of {args.batch_size}...")
    batches = [queries[i:i + args.batch_size] for i in range(0, len(queries), args.batch_size)]
    
    # Process batches in parallel
    start_time = time.time()
    colbert_results = []
    
    if args.num_workers > 1:
        print(f"Using {args.num_workers} parallel workers...")
        with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
            futures = {executor.submit(process_batch, batch, RAG, args.top_k): batch for batch in batches}
            for future in tqdm(as_completed(futures), total=len(batches), desc="Processing batches"):
                batch_results = future.result()
                colbert_results.extend(batch_results)
    else:
        print("Using single worker processing...")
        for batch in tqdm(batches, desc="Processing batches"):
            batch_results = process_batch(batch, RAG, k=args.top_k)
            colbert_results.extend(batch_results)
    
    retrieval_time = time.time() - start_time
    print(f"Retrieval completed in {retrieval_time:.2f} seconds ({retrieval_time/60:.2f} minutes)")
    log_to_wandb({
        "full_retrieval_time_seconds": retrieval_time,
        "retrieval_time_per_query": retrieval_time / len(queries),
        "queries_per_second": len(queries) / retrieval_time
    })
    
    # Create results dataframe
    results_df = pd.DataFrame({
        'post_id': query_ids,
        'preds': colbert_results
    })
    
    # Save results
    results_df.to_csv('colbert_predictions.tsv', sep='\t', index=False)
    print("Results saved to colbert_predictions.tsv")
    
    # Log sample predictions to wandb
    if WANDB_AVAILABLE and args.use_wandb:
        try:
            wandb.log({"sample_predictions": wandb.Table(dataframe=results_df.head(10))})
        except Exception as e:
            print(f"Failed to log sample predictions to wandb: {e}")
    
    # Merge results with ground truth for evaluation
    eval_df = df_query_dev.iloc[:max_queries].copy() if max_queries else df_query_dev.copy()
    eval_df['colbert_topk'] = colbert_results
    
    # Calculate MRR scores
    mrr_scores = get_performance_mrr(eval_df, 'cord_uid', 'colbert_topk')
    print(f"ColBERT MRR@k scores: {mrr_scores}")
    
    # Log MRR scores to wandb
    for k, score in mrr_scores.items():
        log_to_wandb({f"colbert_mrr@{k}": score})
    
    # Compare with BM25 baseline if requested
    try:
        print("Running BM25 baseline for comparison...")
        bm25_start_time = time.time()
        bm25_results = run_bm25_baseline(corpus, cord_uids, queries, k=args.top_k)
        bm25_time = time.time() - bm25_start_time
        
        # Add BM25 results to evaluation dataframe
        eval_df['bm25_topk'] = bm25_results
        
        # Calculate BM25 MRR scores
        bm25_mrr_scores = get_performance_mrr(eval_df, 'cord_uid', 'bm25_topk')
        print(f"BM25 MRR@k scores: {bm25_mrr_scores}")
        
        # Log BM25 results to wandb
        log_to_wandb({
            "bm25_retrieval_time_seconds": bm25_time,
            "bm25_retrieval_time_per_query": bm25_time / len(queries)
        })
        
        bm25_improvements = {}
        for k, score in bm25_mrr_scores.items():
            improvement = (mrr_scores[k] - bm25_mrr_scores[k]) / bm25_mrr_scores[k] * 100
            bm25_improvements[f"bm25_mrr@{k}"] = score
            bm25_improvements[f"improvement_mrr@{k}"] = improvement
        log_to_wandb(bm25_improvements)
        
        # Compare results
        print("\nComparison:")
        for k in sorted(mrr_scores.keys()):
            improvement = (mrr_scores[k] - bm25_mrr_scores[k]) / bm25_mrr_scores[k] * 100
            print(f"MRR@{k}: ColBERT = {mrr_scores[k]:.4f}, BM25 = {bm25_mrr_scores[k]:.4f}, Improvement: {improvement:.2f}%")
    
    except Exception as e:
        print(f"Couldn't run BM25 baseline: {e}")
    
    # Log overall metrics
    log_to_wandb({
        "total_runtime_seconds": retrieval_time + (0 if args.skip_indexing else indexing_time),
        "total_documents": len(documents),
        "total_queries": len(queries)
    })
    
    # Finish wandb run
    if WANDB_AVAILABLE and args.use_wandb:
        try:
            wandb.finish()
        except Exception as e:
            print(f"Failed to finish wandb run: {e}")
    
    print("\nTask completed!") 