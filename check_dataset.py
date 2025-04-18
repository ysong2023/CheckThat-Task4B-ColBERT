import pandas as pd
import os

# Print current directory and files
print("Current directory:", os.getcwd())
print("Files in subtask_4b directory:", os.listdir("subtask_4b"))

# Check file sizes
for file in os.listdir("subtask_4b"):
    print(f"File: {file}, Size: {os.path.getsize(os.path.join('subtask_4b', file))/(1024*1024):.2f} MB")

# Try to load and print dataset sizes
try:
    train = pd.read_csv("subtask_4b/subtask4b_query_tweets_train.tsv", sep='\t')
    print(f"Train dataset: {train.shape[0]} rows, {train.shape[1]} columns")
except Exception as e:
    print(f"Error loading train data: {e}")

try:
    dev = pd.read_csv("subtask_4b/subtask4b_query_tweets_dev.tsv", sep='\t')
    print(f"Dev dataset: {dev.shape[0]} rows, {dev.shape[1]} columns")
except Exception as e:
    print(f"Error loading dev data: {e}")

try:
    collection = pd.read_pickle("subtask_4b/subtask4b_collection_data.pkl")
    print(f"Collection dataset: {collection.shape[0]} rows, {collection.shape[1]} columns")
except Exception as e:
    print(f"Error loading collection data: {e}") 