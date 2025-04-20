#!/usr/bin/env python
# coding: utf-8

# # DocT5Query Implementation for Scientific Claim Source Retrieval
# CLEF 2025 - CheckThat! Lab - Task 4 Scientific Web Discourse - Subtask 4b

import numpy as np
import pandas as pd
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import T5ForConditionalGeneration, T5Tokenizer, AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import json
from typing import List, Dict, Tuple
from rank_bm25 import BM25Okapi

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Configuration
config = {
    "model_name": "t5-base",
    "max_length": 512,
    "batch_size": 8,
    "learning_rate": 2e-5,
    "num_epochs": 10,
    "num_queries": 3
}

# Function to load data same as in baseline
def load_data():
    # Paths to data files
    PATH_COLLECTION_DATA = 'subtask_4b/subtask4b_collection_data.pkl'  # Updated path
    PATH_QUERY_DATA = 'subtask_4b/subtask4b_query_tweets.tsv'  # Updated path
    
    # Load data
    df_collection = pd.read_pickle(PATH_COLLECTION_DATA)
    df_query = pd.read_csv(PATH_QUERY_DATA, sep='\t')
    
    return df_collection, df_query

# Create dataset class for fine-tuning
class DocQueryDataset(Dataset):
    def __init__(self, papers, tweets, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.papers = papers
        self.tweets = tweets
        
    def __len__(self):
        return len(self.papers)
    
    def __getitem__(self, idx):
        paper = self.papers[idx]
        tweet = self.tweets[idx]
        
        # For doc2query, we want model to generate a tweet given a paper
        inputs = self.tokenizer(
            f"Generate tweet: {paper}", 
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        
        outputs = self.tokenizer(
            tweet,
            truncation=True,
            max_length=128,
            padding="max_length",
            return_tensors="pt"
        )
        
        return {
            "input_ids": inputs.input_ids.squeeze(),
            "attention_mask": inputs.attention_mask.squeeze(),
            "labels": outputs.input_ids.squeeze()
        }

# Function to prepare training data
def prepare_training_data(df_collection, df_query):
    # Create mapping from cord_uid to tweet
    paper_tweet_map = {}
    for _, row in df_query.iterrows():
        cord_uid = row['cord_uid']
        tweet = row['tweet_text']
        if cord_uid in paper_tweet_map:
            paper_tweet_map[cord_uid].append(tweet)
        else:
            paper_tweet_map[cord_uid] = [tweet]
    
    # Prepare training data
    papers = []
    tweets = []
    
    for _, row in df_collection.iterrows():
        cord_uid = row['cord_uid']
        if cord_uid in paper_tweet_map:
            paper_text = f"{row['title']} {row['abstract']}"
            for tweet in paper_tweet_map[cord_uid]:
                papers.append(paper_text)
                tweets.append(tweet)
    
    return papers, tweets

# Function to fine-tune the T5 model
def finetune_t5(papers, tweets, output_dir="./model", epochs=3, batch_size=8):
    # Initialize tokenizer and model
    tokenizer = T5Tokenizer.from_pretrained("t5-base")
    model = T5ForConditionalGeneration.from_pretrained("t5-base")
    model = model.to(device)
    
    # Split data
    train_papers, val_papers, train_tweets, val_tweets = train_test_split(
        papers, tweets, test_size=0.1, random_state=42
    )
    
    # Create datasets
    train_dataset = DocQueryDataset(train_papers, train_tweets, tokenizer)
    val_dataset = DocQueryDataset(val_papers, val_tweets, tokenizer)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Setup optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=5e-5)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=total_steps
    )
    
    # Training loop
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        
        # Training
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} - Training"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            train_loss += loss.item()
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
        
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} - Validation"):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs.loss
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        
        print(f"Epoch {epoch+1} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
    
    # Save the model
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    return model, tokenizer

# Function to generate query expansions for documents
def expand_documents(df_collection, model, tokenizer, num_queries=5):
    model.eval()
    expanded_corpus = []
    cord_uids = []
    
    for i, row in tqdm(df_collection.iterrows(), total=len(df_collection), desc="Expanding documents"):
        paper_text = f"{row['title']} {row['abstract']}"
        cord_uid = row['cord_uid']
        
        # Generate queries
        input_ids = tokenizer(
            f"Generate tweet: {paper_text}", 
            return_tensors="pt", 
            max_length=512, 
            truncation=True
        ).input_ids.to(device)
        
        expanded_text = paper_text
        
        with torch.no_grad():
            for _ in range(num_queries):
                outputs = model.generate(
                    input_ids,
                    max_length=64,
                    num_beams=5,
                    no_repeat_ngram_size=2,
                    top_k=50,
                    top_p=0.95,
                    do_sample=True
                )
                generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                expanded_text += " " + generated_text
        
        expanded_corpus.append(expanded_text)
        cord_uids.append(cord_uid)
    
    return expanded_corpus, cord_uids

# Retrieval function using BM25 on expanded corpus
def retrieve_with_bm25(expanded_corpus, cord_uids, df_query):
    # Tokenize the corpus
    tokenized_corpus = [doc.split(' ') for doc in expanded_corpus]
    bm25 = BM25Okapi(tokenized_corpus)
    
    # Retrieve documents for each query
    text2bm25top = {}
    
    def get_top_cord_uids(query):
        if query in text2bm25top.keys():
            return text2bm25top[query]
        else:
            tokenized_query = query.split(' ')
            doc_scores = bm25.get_scores(tokenized_query)
            indices = np.argsort(-doc_scores)[:1000]
            bm25_topk = [cord_uids[x] for x in indices]

            text2bm25top[query] = bm25_topk
            return bm25_topk
    
    df_query['doct5query_topk'] = df_query['tweet_text'].apply(lambda x: get_top_cord_uids(x))
    
    return df_query

# Evaluation function - same as baseline
def get_performance_mrr(data, col_gold, col_pred, list_k=[1, 5, 10]):
    d_performance = {}
    for k in list_k:
        data["in_topx"] = data.apply(lambda x: (1/([i for i in x[col_pred][:k]].index(x[col_gold]) + 1) if x[col_gold] in [i for i in x[col_pred][:k]] else 0), axis=1)
        d_performance[k] = data["in_topx"].mean()
    return d_performance

class PaperTweetDataset(Dataset):
    def __init__(self, papers: List[str], tweets: List[str], labels: List[int]):
        self.papers = papers
        self.tweets = tweets
        self.labels = labels
        
    def __len__(self):
        return len(self.papers)
    
    def __getitem__(self, idx):
        return self.papers[idx], self.tweets[idx], self.labels[idx]

class DocT5Query:
    def __init__(self, model_name: str = "t5-base"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name).to(self.device)
        
    def train(self, train_data: List[Dict], val_data: List[Dict], 
              num_epochs: int = 10, batch_size: int = 8, 
              learning_rate: float = 2e-5, save_dir: str = "./model"):
        """Train the DocT5Query model."""
        # Create datasets
        train_dataset = PaperTweetDataset(
            papers=[item["paper"] for item in train_data],
            tweets=[item["tweet"] for item in train_data],
            labels=[item["label"] for item in train_data]
        )
        
        val_dataset = PaperTweetDataset(
            papers=[item["paper"] for item in val_data],
            tweets=[item["tweet"] for item in val_data],
            labels=[item["label"] for item in val_data]
        )
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        # Setup training
        optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()
        
        best_val_loss = float('inf')
        os.makedirs(save_dir, exist_ok=True)
        
        for epoch in range(num_epochs):
            # Training
            self.model.train()
            train_loss = 0
            train_steps = 0
            
            for papers, tweets, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
                # Prepare inputs
                inputs = self.tokenizer(
                    papers,
                    padding=True,
                    truncation=True,
                    max_length=config["max_length"],
                    return_tensors="pt"
                ).to(self.device)
                
                targets = self.tokenizer(
                    tweets,
                    padding=True,
                    truncation=True,
                    max_length=config["max_length"],
                    return_tensors="pt"
                ).to(self.device)
                
                # Forward pass
                outputs = self.model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    labels=targets["input_ids"]
                )
                
                loss = outputs.loss
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                train_steps += 1
            
            avg_train_loss = train_loss / train_steps
            
            # Validation
            self.model.eval()
            val_loss = 0
            val_steps = 0
            
            with torch.no_grad():
                for papers, tweets, labels in val_loader:
                    inputs = self.tokenizer(
                        papers,
                        padding=True,
                        truncation=True,
                        max_length=config["max_length"],
                        return_tensors="pt"
                    ).to(self.device)
                    
                    targets = self.tokenizer(
                        tweets,
                        padding=True,
                        truncation=True,
                        max_length=config["max_length"],
                        return_tensors="pt"
                    ).to(self.device)
                    
                    outputs = self.model(
                        input_ids=inputs["input_ids"],
                        attention_mask=inputs["attention_mask"],
                        labels=targets["input_ids"]
                    )
                    
                    loss = outputs.loss
                    val_loss += loss.item()
                    val_steps += 1
            
            avg_val_loss = val_loss / val_steps
            
            print(f"Epoch {epoch+1}/{num_epochs}")
            print(f"Average training loss: {avg_train_loss:.4f}")
            print(f"Average validation loss: {avg_val_loss:.4f}")
            
            # Save best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                model_path = os.path.join(save_dir, "best_model.pt")
                torch.save(self.model.state_dict(), model_path)
                print("Saved best model")
    
    def expand_document(self, document: str, num_queries: int = 3) -> List[str]:
        """Generate queries for a document."""
        self.model.eval()
        
        # Prepare input
        inputs = self.tokenizer(
            document,
            padding=True,
            truncation=True,
            max_length=config["max_length"],
            return_tensors="pt"
        ).to(self.device)
        
        # Generate queries
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=config["max_length"],
                num_return_sequences=num_queries,
                num_beams=num_queries,
                early_stopping=True
            )
        
        # Decode queries
        queries = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return queries
    
    def save_model(self, path: str):
        """Save the model to disk."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.model.state_dict(), path)
    
    def load_model(self, path: str):
        """Load the model from disk."""
        self.model.load_state_dict(torch.load(path))

def main():
    print("Loading data...")
    df_collection, df_query = load_data()
    
    # Check if model already exists
    model_path = "./model"
    if os.path.exists(model_path) and os.path.isdir(model_path):
        print("Loading existing model...")
        tokenizer = T5Tokenizer.from_pretrained(model_path)
        model = T5ForConditionalGeneration.from_pretrained(model_path)
        model = model.to(device)
    else:
        print("Preparing training data...")
        papers, tweets = prepare_training_data(df_collection, df_query)
        print(f"Training with {len(papers)} paper-tweet pairs...")
        model, tokenizer = finetune_t5(papers, tweets, output_dir=model_path)
    
    print("Expanding documents with DocT5Query...")
    expanded_corpus, cord_uids = expand_documents(df_collection, model, tokenizer)
    
    print("Retrieving documents...")
    df_query = retrieve_with_bm25(expanded_corpus, cord_uids, df_query)
    
    print("Evaluating results...")
    results = get_performance_mrr(df_query, 'cord_uid', 'doct5query_topk')
    print(results)
    
    # Compare with baseline
    baseline = {1: 0.5078930751420754, 5: 0.5511050305198906, 10: 0.5561281167206236}
    print("\nBaseline (BM25):")
    print(baseline)
    print("\nDocT5Query:")
    print(results)
    
    improvements = {k: results[k] - baseline[k] for k in baseline}
    print("\nImprovements:")
    print(improvements)

    # Load data
    try:
        with open("subtask_4b/data/train.json", "r") as f:
            train_data = json.load(f)
        
        with open("subtask_4b/data/val.json", "r") as f:
            val_data = json.load(f)
    except FileNotFoundError:
        print("JSON files not found. Using direct data from DataFrame...")
        # Create synthetic data from the loaded DataFrames
        train_data = []
        val_data = []
        
        # Create mapping from cord_uid to tweet
        paper_tweet_map = {}
        for _, row in df_query.iterrows():
            cord_uid = row['cord_uid']
            tweet = row['tweet_text']
            if cord_uid in paper_tweet_map:
                paper_tweet_map[cord_uid].append(tweet)
            else:
                paper_tweet_map[cord_uid] = [tweet]
        
        # Create train and validation data
        for _, row in df_collection.iterrows():
            cord_uid = row['cord_uid']
            if cord_uid in paper_tweet_map:
                paper_text = f"{row['title']} {row['abstract']}"
                for tweet in paper_tweet_map[cord_uid]:
                    data_item = {
                        "paper": paper_text,
                        "tweet": tweet,
                        "label": 1  # Positive example
                    }
                    if np.random.random() < 0.9:  # 90% to train, 10% to val
                        train_data.append(data_item)
                    else:
                        val_data.append(data_item)
        
        print(f"Created {len(train_data)} training samples and {len(val_data)} validation samples")
    
    # Initialize model
    model = DocT5Query(model_name=config["model_name"])
    
    # Train model
    model.train(
        train_data=train_data,
        val_data=val_data,
        num_epochs=config["num_epochs"],
        batch_size=config["batch_size"],
        learning_rate=config["learning_rate"],
        save_dir="./model"
    )
    
    print("Training completed!")

if __name__ == "__main__":
    main() 