#!/usr/bin/env python
"""
Fine-tune a ColBERT model using RAGatouille's official API.
This script loads triplets from ragatouille_training_data.json and fine-tunes the model.
"""

import os
import argparse
import json
import torch
import time
from tqdm import tqdm
import pandas as pd
import numpy as np
from ragatouille import RAGTrainer

def check_gpu():
    """Check if GPU is available and print GPU info"""
    if torch.cuda.is_available():
        print("\n===== GPU INFORMATION =====")
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
        return True
    else:
        print("No GPU available. Fine-tuning will be slow on CPU.")
        return False

def finetune_colbert(training_data_path, output_dir, base_model="colbert-ir/colbertv2.0",
                   batch_size=32, epochs=3, learning_rate=1e-5):
    """
    Fine-tune ColBERT model using RAGatouille's RAGTrainer
    """
    # Load training data
    with open(training_data_path, 'r') as f:
        training_data = json.load(f)
    
    print(f"Loaded {len(training_data)} training examples")
    
    # Make sure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Check GPU availability
    use_gpu = check_gpu()
    
    # Initialize RAGTrainer
    print(f"Initializing RAGTrainer with base model: {base_model}")
    trainer = RAGTrainer(
        model_name=os.path.basename(output_dir),  # Name for your fine-tuned model
        pretrained_model_name=base_model  # Base model to fine-tune
    )
    
    # Prepare the training data
    print("Preparing training data...")
    # RAGatouille expects triplets in the format (query, positive, negative)
    # Our data is already in the right format, just need to extract it properly
    
    # First, extract all unique documents to build a corpus
    all_documents = []
    for triplet in training_data:
        all_documents.append(triplet["positive"])
        all_documents.append(triplet["negative"])
    all_documents = list(set(all_documents))  # Remove duplicates
    
    # Create triplets in the format RAGatouille expects
    formatted_triplets = []
    for triplet in training_data:
        formatted_triplets.append((
            triplet["query"],          # Query
            triplet["positive"],       # Positive document
            triplet["negative"]        # Negative document
        ))
    
    # We'll save the processed data in a directory next to the output dir
    data_dir = os.path.join(os.path.dirname(output_dir), "training_data")
    os.makedirs(data_dir, exist_ok=True)
    
    # Prepare the training data with RAGatouille
    trainer.prepare_training_data(
        raw_data=formatted_triplets,
        data_out_path=data_dir,
        all_documents=all_documents  # Add the corpus for mining additional negatives
    )
    
    # Start the training
    try:
        start_time = time.time()
        
        print(f"Starting fine-tuning with batch_size={batch_size}")
        trainer.train(
            batch_size=batch_size
        )
        
        training_time = time.time() - start_time
        print(f"Fine-tuning completed in {training_time:.2f} seconds ({training_time/60:.2f} minutes)")
        
        # Print final GPU memory stats
        if use_gpu:
            print("\nFinal GPU memory stats:")
            print(f"Allocated: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
            print(f"Cached: {torch.cuda.memory_reserved(0) / 1e9:.2f} GB")
            print(f"Max allocated: {torch.cuda.max_memory_allocated(0) / 1e9:.2f} GB")
        
        return True
        
    except Exception as e:
        print(f"Error during fine-tuning: {e}")
        return False

def evaluate_model(model_dir, test_queries_path, collection_path, k=5):
    """
    Evaluate the fine-tuned model on test queries
    """
    from ragatouille import RAGPretrainedModel
    import time
    
    # Load test queries and collection
    test_queries = pd.read_csv(test_queries_path, sep='\t')
    collection = pd.read_pickle(collection_path)
    
    # Prepare documents
    documents = collection[['title', 'abstract']].apply(
        lambda x: f"{x['title']} {x['abstract']}", axis=1).tolist()
    document_ids = collection['cord_uid'].tolist()
    
    # Load fine-tuned model
    print(f"Loading fine-tuned model from {model_dir}")
    model = RAGPretrainedModel.from_pretrained(model_dir)
    
    # Create index
    print("Creating search index...")
    index_path = model.index(
        collection=documents,
        document_ids=document_ids,
        index_name="eval_index"
    )
    
    # Run search on test queries
    print(f"Evaluating on {len(test_queries)} queries...")
    all_results = []
    
    start_time = time.time()
    for idx, row in tqdm(test_queries.iterrows(), total=len(test_queries)):
        query = row['tweet_text']
        results = model.search(query, k=k)
        retrieved_docs = [r['document_id'] for r in results]
        all_results.append(retrieved_docs)
    
    eval_time = time.time() - start_time
    print(f"Evaluation completed in {eval_time:.2f} seconds ({eval_time/len(test_queries):.2f} seconds per query)")
    
    # Calculate MRR
    mrr_values = calculate_mrr(test_queries, all_results)
    print(f"MRR@1: {mrr_values[1]}")
    print(f"MRR@5: {mrr_values[5]}")
    print(f"MRR@10: {mrr_values[10]}")
    
    return mrr_values

def calculate_mrr(test_queries, results, k_values=[1, 5, 10]):
    """
    Calculate Mean Reciprocal Rank
    """
    mrr_dict = {}
    for k in k_values:
        reciprocal_ranks = []
        for idx, row in test_queries.iterrows():
            gt_doc = row['cord_uid']
            retrieved = results[idx][:k]
            
            if gt_doc in retrieved:
                rank = retrieved.index(gt_doc) + 1
                reciprocal_ranks.append(1.0 / rank)
            else:
                reciprocal_ranks.append(0.0)
        
        mrr_dict[k] = np.mean(reciprocal_ranks)
    
    return mrr_dict

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Fine-tune a ColBERT model using RAGatouille')
    parser.add_argument('--training_data', default='ragatouille_training_data.json',
                       help='Path to the training data file')
    parser.add_argument('--output_dir', default='./finetuned_colbert',
                       help='Directory to save the fine-tuned model')
    parser.add_argument('--base_model', default='colbert-ir/colbertv2.0',
                       help='Base ColBERT model to fine-tune')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=3,
                       help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-5,
                       help='Learning rate')
    parser.add_argument('--evaluate', action='store_true',
                       help='Evaluate the model after training')
    parser.add_argument('--test_queries', default='subtask_4b/subtask4b_query_tweets_dev.tsv',
                       help='Path to test queries for evaluation')
    parser.add_argument('--collection', default='subtask_4b/subtask4b_collection_data.pkl',
                       help='Path to document collection')
    
    args = parser.parse_args()
    
    # Check GPU availability
    has_gpu = check_gpu()
    
    # Fine-tune the model
    print(f"Fine-tuning ColBERT model from {args.base_model}")
    success = finetune_colbert(
        args.training_data,
        args.output_dir,
        args.base_model,
        args.batch_size,
        args.epochs,
        args.learning_rate
    )
    
    # Evaluate if requested and training succeeded
    if success and args.evaluate:
        print("\nEvaluating fine-tuned model...")
        results = evaluate_model(
            args.output_dir,
            args.test_queries,
            args.collection
        )
        
    print("All done!") 