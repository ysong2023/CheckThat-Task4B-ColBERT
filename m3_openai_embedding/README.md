# OpenAI Embedding Method for Scientific Claim Source Retrieval

## Overview

This document describes an approach using OpenAI embeddings for the CLEF 2025 CheckThat! Lab Task 4b (Scientific Claim Source Retrieval). The task involves retrieving the correct scientific paper from the CORD-19 collection that is implicitly referenced in a tweet.

The approach leverages OpenAI's pre-trained text embedding models to generate dense vector representations (embeddings) of both papers and tweets. These embeddings capture semantic meaning beyond simple keyword matching, allowing for more effective matching between tweets and their referenced papers.

### Key advantages over baseline:

- **Semantic understanding**: Captures the meaning of text rather than just matching keywords
- **Contextual awareness**: Understands relationships between concepts even with different terminology
- **Zero-shot capability**: No fine-tuning required, uses pre-trained models directly
- **Handling of implicit references**: Better at detecting references when exact paper wording isn't used

## Implementation Details

### Data Processing

The implementation processes two main data sources:
1. **Collection set**: CORD-19 academic papers' metadata (title, abstract, etc.)
2. **Query set**: Tweets with implicit references to papers

### Key Components

1. **Embedding Generation**: Using OpenAI's text-embedding models to create vector representations
2. **Similarity Calculation**: Computing cosine similarity between tweet and paper embeddings
3. **Retrieval**: Selecting top-k papers with highest similarity scores
4. **Evaluation**: Using Mean Reciprocal Rank (MRR@k) metrics

## Critical Code Snippets

### Generating Embeddings

```python
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
                return [0.0] * (1536 if "3-small" in model else 3072)
```

### Paper Representation

```python
# Create text representations of papers (title + abstract)
paper_texts = df_collection[['title', 'abstract']].apply(
    lambda x: f"{x['title']} {x['abstract']}", axis=1).tolist()
cord_uids = df_collection['cord_uid'].tolist()

# Get embeddings for papers
paper_embeddings = get_embeddings_batch(paper_texts)
paper_embeddings = np.array(paper_embeddings)
```

### Retrieval Function

```python
def get_top_cord_uids_embedding(query_text, top_k=5):
    """
    Retrieve top k papers for a query using embedding similarity
    """
    # Get the embedding for the query
    query_embedding = get_embedding(query_text)
    
    # Calculate cosine similarity between query and all papers
    similarities = cosine_similarity([query_embedding], paper_embeddings)[0]
    
    # Get indices of top k similar papers
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    
    # Return the cord_uids of top k papers
    return [cord_uids[i] for i in top_indices]
```

### Evaluation Metric

```python
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
```

## Performance Comparison

Our OpenAI embedding approach demonstrates significant improvements over the BM25 baseline:

| Method | MRR@1 | MRR@5 | MRR@10 |
|--------|-------|-------|--------|
| BM25 Baseline | 0.5076 | 0.5509 | 0.5559 |
| OpenAI Embedding | 0.6136 | 0.6767 | 0.6767 |
| Improvement | +0.1060 (20.88%) | +0.1258 (22.84%) | +0.1208 (21.73%) |

The results show substantial improvements across all evaluation metrics, with gains of over 20% for each metric, indicating a major advance in retrieval performance.

Compared to other models in the literature:

| Method | MRR@1 | MRR@5 | MRR@10 |
|--------|-------|-------|--------|
| BM25 | 0.5079 | 0.5511 | 0.5561 |
| DocT5Query | 0.5224 | 0.5629 | 0.5679 |
| OpenAI Embedding | 0.6136 | 0.6767 | 0.6767 |
| Improvement over BM25 | +0.1057 (20.81%) | +0.1256 (22.79%) | +0.1206 (21.69%) |
| Improvement over DocT5Query | +0.0912 (17.46%) | +0.1138 (20.22%) | +0.1088 (19.16%) |

Our OpenAI embedding approach significantly outperforms both the BM25 baseline and the DocT5Query method across all metrics, with improvements ranging from 17% to nearly 23%.

## Usage Guidelines

To use this approach:

1. **Setup**: Install required dependencies and set your OpenAI API key
```bash
pip install openai numpy pandas scikit-learn tqdm
export OPENAI_API_KEY="your-api-key-here"
```

2. **Run**: Execute the script to generate predictions
```bash
python openai_embedding.py
```

3. **Results**: Find predictions in `predictions_embedding.tsv`

## Limitations and Future Work

- **API costs**: Using OpenAI's API incurs costs based on the number of tokens processed
- **Rate limits**: API rate limits can slow down processing for large datasets
- **Model variations**: Different embedding models (ada, text-embedding-3-small/large) offer different trade-offs between performance and cost
- **Hybrid approaches**: Combining embedding similarity with traditional retrieval methods could further improve performance

Future work could explore:
- Fine-tuning embedding models specifically for scientific paper retrieval
- Incorporating additional metadata (authors, journal, etc.) into the similarity calculation
- Exploring cross-encoder approaches for re-ranking the initial candidate set

## Conclusion

The OpenAI embedding approach offers a significant improvement over traditional lexical retrieval methods for scientific claim source retrieval. By leveraging pre-trained language models, it can better understand the semantic relationship between tweets and scientific papers, resulting in more accurate retrieval performance. 