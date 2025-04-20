"""
Optimized ColBERT implementation for the CheckThat 2025 Subtask 4b (Scientific Claim Source Retrieval)

This script uses RAGatouille with ColBERT to retrieve the mentioned papers from tweets.
Optimized with batch processing and parallel execution for better performance.
GPU-accelerated version.
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
import torch
import importlib
import sys

# Force GPU usage
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Use the first GPU
torch.cuda.set_device(0)

# Print GPU information for debugging
print("\n===== GPU INFORMATION =====")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU device: {torch.cuda.get_device_name(0)}")
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    # Force some tensor operations on GPU to verify it's working
    print("Testing GPU with a sample operation:")
    test_tensor = torch.rand(1000, 1000).cuda()
    start = time.time()
    result = torch.matmul(test_tensor, test_tensor)
    print(f"GPU tensor operation took {time.time() - start:.4f} seconds")
    print(f"Tensor device: {result.device}")
    del test_tensor, result
    torch.cuda.empty_cache()
print("===========================\n")

# Import and modify FAISS to force GPU usage
try:
    import faiss
    print("\n===== FAISS INFORMATION =====")
    print(f"FAISS version: {faiss.__version__}")
    print(f"FAISS GPU support: {faiss.get_num_gpus()} GPU(s) detected")
    
    # Force FAISS to use GPU by monkey patching
    if faiss.get_num_gpus() > 0:
        print("Setting up FAISS with GPU resources")
        # Create global GPU resources
        global_res = faiss.StandardGpuResources()
        
        # Monkey patch the index_factory function to use GPU
        original_index_factory = faiss.index_factory
        def gpu_index_factory(*args, **kwargs):
            index = original_index_factory(*args, **kwargs)
            return faiss.index_cpu_to_gpu(global_res, 0, index)
        
        # Replace the index_factory function
        try:
            faiss.index_factory = gpu_index_factory
            print("Successfully patched FAISS to use GPU")
        except Exception as e:
            print(f"Failed to patch FAISS: {e}")
    print("=============================\n")
except ImportError:
    print("FAISS not found. GPU acceleration for vector search may not work properly.")

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

# Patch RAGatouille to force GPU usage
def patch_ragatouille():
    """Attempt to patch RAGatouille to use GPU explicitly"""
    try:
        # Import RAGatouille
        from ragatouille import RAGPretrainedModel
        
        # We'll use a post-initialization approach instead of patching the constructor
        original_init = RAGPretrainedModel.__init__
        
        def patched_init(self, *args, **kwargs):
            # Call the original init first
            original_init(self, *args, **kwargs)
            
            # Then move model to GPU if available
            if torch.cuda.is_available():
                print("Moving RAGatouille model to GPU after initialization...")
                if hasattr(self, 'colbert_model') and self.colbert_model is not None:
                    self.colbert_model = self.colbert_model.to('cuda')
                    print("✓ ColBERT model moved to GPU")
        
        # Apply the patch to __init__ instead
        RAGPretrainedModel.__init__ = patched_init
        print("Successfully patched RAGatouille to explicitly use GPU")
    except Exception as e:
        print(f"Failed to patch RAGatouille: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run ColBERT retrieval for Scientific Claim Source Retrieval')
    parser.add_argument('--batch_size', type=int, default=10, help='Batch size for processing queries')
    parser.add_argument('--max_queries', type=int, default=None, help='Maximum number of queries to process (None for all)')
    parser.add_argument('--num_workers', type=int, default=None, help='Number of parallel workers (None for auto)')
    parser.add_argument('--test_size', type=int, default=20, help='Number of test queries for time estimation')
    parser.add_argument('--skip_indexing', action='store_true', help='Skip index creation if it already exists')
    parser.add_argument('--top_k', type=int, default=5, help='Number of results to retrieve')
    args = parser.parse_args()

    # Set number of workers
    if args.num_workers is None:
        args.num_workers = max(1, multiprocessing.cpu_count() - 1)  # Leave one CPU for system
    
    print(f"Starting optimized ColBERT implementation with {args.num_workers} workers...")
    
    # Patch RAGatouille to explicitly use GPU
    patch_ragatouille()
    
    # Load datasets
    print("Loading datasets...")
    df_collection = pd.read_pickle("subtask_4b/subtask4b_collection_data.pkl")
    df_query_train = pd.read_csv("subtask_4b/subtask4b_query_tweets_train.tsv", sep='\t')
    df_query_dev = pd.read_csv("subtask_4b/subtask4b_query_tweets_dev.tsv", sep='\t')
    
    print(f"Collection set: {df_collection.shape[0]} papers")
    print(f"Train query set: {df_query_train.shape[0]} tweets")
    print(f"Dev query set: {df_query_dev.shape[0]} tweets")
    
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
        
        # Check if ColBERT model is on GPU
        if hasattr(RAG, 'colbert_model') and RAG.colbert_model is not None:
            device = next(RAG.colbert_model.parameters()).device
            print(f"ColBERT model is on device: {device}")
        
        print("Creating index from documents (this may take a while)...")
        print("GPU will be used for indexing if available")
        start_time = time.time()
        
        # Explicit configuration for GPU
        gpu_config = {
            'collection': documents,
            'document_ids': document_ids,
            'index_name': "papers_index",
            'use_faiss': True  # Enable FAISS with GPU
        }
        
        # Create index
        print("Creating index with FAISS, GPU will be used through our monkey-patched FAISS")
        index_path = RAG.index(**gpu_config)
        
        indexing_time = time.time() - start_time
        print(f"Index creation completed in {indexing_time:.2f} seconds")
    else:
        # Use from_index() when skipping indexing to properly load the existing index
        print(f"Using existing index at {index_path}")
        RAG = RAGPretrainedModel.from_index(index_path)
        
        # Move model to GPU if it's not already there
        if hasattr(RAG, 'colbert_model') and RAG.colbert_model is not None:
            RAG.colbert_model = RAG.colbert_model.to('cuda')
            device = next(RAG.colbert_model.parameters()).device
            print(f"ColBERT model moved to device: {device}")
        
        # Verify FAISS is using GPU
        if 'faiss' in sys.modules:
            print(f"\nFAISS GPU information:")
            print(f"GPU devices available: {faiss.get_num_gpus()}")
            
            # Test with a small index to verify GPU operation
            try:
                print("Testing FAISS GPU operation:")
                d = 64  # dimension
                test_data = np.random.random((1000, d)).astype('float32')
                
                # Create a CPU index
                cpu_start = time.time()
                cpu_index = faiss.IndexFlatL2(d)
                cpu_index.add(test_data)
                cpu_time = time.time() - cpu_start
                print(f"CPU index operation took {cpu_time:.4f} seconds")
                
                # Create a GPU index
                if faiss.get_num_gpus() > 0:
                    gpu_start = time.time()
                    res = faiss.StandardGpuResources()
                    gpu_index = faiss.index_cpu_to_gpu(res, 0, faiss.IndexFlatL2(d))
                    gpu_index.add(test_data)
                    gpu_time = time.time() - gpu_start
                    print(f"GPU index operation took {gpu_time:.4f} seconds")
                    print(f"Speedup: {cpu_time/gpu_time:.2f}x")
            except Exception as e:
                print(f"Error testing FAISS: {e}")
    
    # Test on a small subset to estimate time
    print(f"Running test retrieval on {args.test_size} queries...")
    test_queries = df_query_dev['tweet_text'].iloc[:args.test_size].tolist()
    
    start_time = time.time()
    test_batches = [test_queries[i:i + args.batch_size] for i in range(0, len(test_queries), args.batch_size)]
    test_results = []
    
    # Force GPU sync before timing
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    for batch in tqdm(test_batches, desc="Test batches"):
        test_results.extend(process_batch(batch, RAG, k=args.top_k))
    
    # Force GPU sync after timing
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    test_time = time.time() - start_time
    print(f"Test retrieval completed in {test_time:.2f} seconds for {args.test_size} queries")
    
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
    
    # Create results dataframe
    results_df = pd.DataFrame({
        'post_id': query_ids,
        'preds': colbert_results
    })
    
    # Save results
    results_df.to_csv('colbert_predictions.tsv', sep='\t', index=False)
    print("Results saved to colbert_predictions.tsv")
    
    # Merge results with ground truth for evaluation
    eval_df = df_query_dev.iloc[:max_queries].copy() if max_queries else df_query_dev.copy()
    eval_df['colbert_topk'] = colbert_results
    
    # Calculate MRR scores
    mrr_scores = get_performance_mrr(eval_df, 'cord_uid', 'colbert_topk')
    print(f"ColBERT MRR@k scores: {mrr_scores}")
    
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
        
        # Compare results
        print("\nComparison:")
        for k in sorted(mrr_scores.keys()):
            improvement = (mrr_scores[k] - bm25_mrr_scores[k]) / bm25_mrr_scores[k] * 100
            print(f"MRR@{k}: ColBERT = {mrr_scores[k]:.4f}, BM25 = {bm25_mrr_scores[k]:.4f}, Improvement: {improvement:.2f}%")
    
    except Exception as e:
        print(f"Couldn't run BM25 baseline: {e}")
    
    # Print final GPU memory stats
    if torch.cuda.is_available():
        print("\nFinal GPU memory stats:")
        print(f"Allocated: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
        print(f"Cached: {torch.cuda.memory_reserved(0) / 1e9:.2f} GB")
        print(f"Max allocated: {torch.cuda.max_memory_allocated(0) / 1e9:.2f} GB")
    
    print(f"\nTask completed! Total time: {retrieval_time + (0 if args.skip_indexing else indexing_time):.2f} seconds") 