#!/usr/bin/env python
"""
Generate fine-tuning triplets for ColBERT using the dev set and OpenAI API.
This script will create training data in the format required by RAGatouille for ColBERT fine-tuning.

Each triplet consists of:
- Query: Tweet text from the dev set
- Positive document: The cited paper (from ground truth)
- Negative document: Similar but incorrect paper (hard negative) generated using OpenAI API
"""

import os
import time
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
import pickle
import json
import random
from openai import OpenAI
import dotenv

# Load environment variables
dotenv.load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

def get_hard_negative(query, positive_doc, all_docs, top_k=5):
    """
    Use OpenAI API to generate a hard negative document similar to the positive document
    but not identical, based on the query.
    """
    # Sample some random documents as candidates for hard negatives
    positive_cord_uid = positive_doc['cord_uid']
    candidate_docs = random.sample([doc for doc in all_docs if doc['cord_uid'] != positive_cord_uid], min(50, len(all_docs)-1))
    
    try:
        # Call OpenAI API to find a hard negative
        system_prompt = """You are an AI assistant that helps select hard negative examples for training retrieval models.
        A hard negative is a document that seems relevant to the query but is not the correct answer.
        It should be semantically similar to the positive document but different enough that it's not the correct answer."""
        
        user_prompt = f"""Query: {query}
        
        Positive document (correct answer):
        {positive_doc['title']}
        {positive_doc['abstract']}
        
        Please select ONE document from the candidate documents below that would be a good "hard negative" - 
        a document that seems relevant to the query but is actually incorrect:
        
        Candidate documents:
        {[f"Doc {i}: {doc['title']}" for i, doc in enumerate(candidate_docs)]}
        
        Respond with only the document number (e.g., "Doc 5") that would be the best hard negative."""
        
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=50,
            temperature=0.2
        )
        
        # Extract the chosen document number
        resp_text = response.choices[0].message.content.strip()
        for i in range(len(candidate_docs)):
            if f"Doc {i}" in resp_text:
                return candidate_docs[i]
        
        # If unable to parse properly, return a random document
        return random.choice(candidate_docs)
    
    except Exception as e:
        print(f"Error using OpenAI API: {e}")
        # Fallback: return a random document that's not the positive
        return random.choice(candidate_docs)

def generate_triplets(queries_df, collection_df, output_file, num_samples=None):
    """
    Generate training triplets for ColBERT fine-tuning:
    (query, positive document, negative document)
    """
    # Create a dictionary mapping cord_uid to document
    cord_uid_to_doc = {row['cord_uid']: row for _, row in collection_df.iterrows()}
    
    # Sample a subset if specified
    if num_samples and num_samples < len(queries_df):
        queries_df = queries_df.sample(num_samples, random_state=42)
    
    all_docs = [row for _, row in collection_df.iterrows()]
    
    triplets = []
    for idx, row in tqdm(queries_df.iterrows(), total=len(queries_df), desc="Generating triplets"):
        query = row['tweet_text']
        positive_cord_uid = row['cord_uid']
        
        if positive_cord_uid not in cord_uid_to_doc:
            print(f"Warning: positive document {positive_cord_uid} not found in collection")
            continue
        
        positive_doc = cord_uid_to_doc[positive_cord_uid]
        
        # Get a hard negative using OpenAI
        negative_doc = get_hard_negative(query, positive_doc, all_docs)
        
        # Create the triplet
        triplet = {
            "query": query,
            "positive": f"{positive_doc['title']} {positive_doc['abstract']}",
            "negative": f"{negative_doc['title']} {negative_doc['abstract']}"
        }
        
        triplets.append(triplet)
        
        # Sleep to avoid rate limiting
        time.sleep(0.2)
    
    # Save the triplets to a file
    with open(output_file, 'w') as f:
        json.dump(triplets, f, indent=2)
    
    print(f"Generated {len(triplets)} triplets and saved to {output_file}")
    return triplets

def create_ragatouille_training_data(triplets, output_file):
    """
    Convert triplets to the RAGatouille training format
    """
    ragatouille_data = []
    for t in triplets:
        ragatouille_data.append({
            "query": t["query"],
            "positive": t["positive"],
            "negative": t["negative"]
        })
    
    with open(output_file, 'w') as f:
        json.dump(ragatouille_data, f, indent=2)
    
    print(f"Created RAGatouille training data with {len(ragatouille_data)} examples and saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate fine-tuning triplets for ColBERT')
    parser.add_argument('--queries', default='subtask_4b/subtask4b_query_tweets_dev.tsv', 
                       help='Path to the query tweets file')
    parser.add_argument('--collection', default='subtask_4b/subtask4b_collection_data.pkl',
                       help='Path to the collection file')
    parser.add_argument('--output', default='triplets.json',
                       help='Path to save the triplets')
    parser.add_argument('--ragatouille_output', default='ragatouille_training_data.json',
                       help='Path to save the RAGatouille formatted training data')
    parser.add_argument('--num_samples', type=int, default=100,
                       help='Number of samples to use (None for all)')
    
    args = parser.parse_args()
    
    # Check for OpenAI API key
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        # Create a .env file with the API key
        with open('.env', 'w') as f:
            f.write(f"OPENAI_API_KEY={os.environ.get('OPENAI_API_KEY', '')}")
        print("Created .env file. Please set your OpenAI API key in this file.")
        exit(1)
    
    # Load data
    print("Loading data...")
    queries_df = pd.read_csv(args.queries, sep='\t')
    collection_df = pd.read_pickle(args.collection)
    
    print(f"Loaded {len(queries_df)} queries and {len(collection_df)} documents")
    
    # Generate triplets
    triplets = generate_triplets(queries_df, collection_df, args.output, args.num_samples)
    
    # Create RAGatouille training data
    create_ragatouille_training_data(triplets, args.ragatouille_output) 