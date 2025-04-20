# DocT5Query: Document Expansion for Scientific Claim Source Retrieval

This document provides an overview of the DocT5Query implementation for the CLEF 2025 CheckThat! Lab Task 4b on Scientific Claim Source Retrieval.

## Overview

DocT5Query is a document expansion technique that leverages the T5 language model to enhance information retrieval. The core idea is to generate potential queries that might lead to a document, then add these queries to the document representation before indexing.

## Model Description

### Architecture and Motivation

DocT5Query builds upon the Text-to-Text Transfer Transformer (T5) architecture introduced by Raffel et al. (2020)[^1]. We selected this approach based on several key motivations:

1. **Vocabulary Gap**: Scientific papers and social media posts (tweets) exhibit significant vocabulary differences. DocT5Query helps bridge this gap by generating tweet-like queries from scientific content, essentially translating between these two domains.

2. **Zero-code Enhancement**: The technique improves retrieval without modifying the core retrieval system (BM25), making it an efficient add-on to existing infrastructure.

3. **Proven Effectiveness**: Previous work by Nogueira et al. (2019)[^2] demonstrated DocT5Query's effectiveness for web document retrieval, and we adapted this approach for the scientific domain.

4. **Model Efficiency**: Compared to more complex neural retrieval models, DocT5Query offers a good balance between performance improvement and computational overhead.

The technique addresses a fundamental limitation of lexical retrievers like BM25: their inability to handle vocabulary mismatch. By expanding documents with predicted queries, we improve recall without sacrificing the speed and interpretability of traditional retrieval methods.

[^1]: Raffel, C., Shazeer, N., Roberts, A., Lee, K., Narang, S., Matena, M., Zhou, Y., Li, W., & Liu, P. J. (2020). Exploring the limits of transfer learning with a unified text-to-text transformer. Journal of Machine Learning Research, 21(140), 1-67.

[^2]: Nogueira, R., Yang, W., Cho, K., & Lin, J. (2019). Document expansion by query prediction. arXiv preprint arXiv:1904.08375.

### Base Model
- **Model**: T5-base
- **Framework**: Hugging Face Transformers
- **Training Device**: CUDA

### Training Process

The model is fine-tuned in two phases:

1. **Query Generation Training**:
   - The model learns to generate tweets (queries) from scientific papers
   - Input: Paper title and abstract
   - Output: Corresponding tweet
   - Loss: Cross-entropy for sequence-to-sequence generation

```python
# Input format for the model
inputs = tokenizer(
    f"Generate tweet: {paper_text}", 
    truncation=True,
    max_length=512,
    padding="max_length",
    return_tensors="pt"
)
```

2. **Document Expansion**:
   - Using the trained model to generate multiple pseudo-queries for each document
   - Each document is expanded with 5 synthetic tweets
   - The original text and generated tweets are concatenated

```python
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
```

### Hyperparameters

- **Max sequence length**: 512
- **Batch size**: 8
- **Learning rate**: 2e-5
- **Training epochs**: 10 (first implementation: 3, wandb implementation: 10)
- **Number of queries per document**: 5 (for document expansion)
- **Optimizer**: AdamW with weight decay
- **Sampling strategy**: Beam search with top-k and top-p sampling
- **Generation constraints**: No repeat n-gram size of 2 to avoid repetitive text

### Training Metrics

The training showed consistent improvement across epochs:

| Epoch | Training Loss | Validation Loss |
|-------|---------------|-----------------|
| 1     | 2.1491        | 1.7713          |
| 5     | 1.7202        | 1.6541          |
| 10    | 1.5759        | 1.6266          |

The model converged well, with training loss decreasing from 2.15 to 1.58 over 10 epochs.

## Weights & Biases Integration

We tracked our model training and evaluation using Weights & Biases (W&B). The full logs, metrics, and model checkpoints are available at:

[https://wandb.ai/raquelzacano37-university-of-british-columbia/scientific-web-discourse](https://wandb.ai/raquelzacano37-university-of-british-columbia/scientific-web-discourse)

The W&B dashboard provides detailed visualizations of:
- Training and validation loss curves
- Generated query examples
- Model parameters and configurations
- Evaluation metrics over time

## Retrieval and Evaluation

### Retrieval Process
1. Documents are expanded using the DocT5Query model
2. BM25 is applied on the expanded corpus
3. For each query tweet, the top k documents are retrieved

### Evaluation Metrics
- **Mean Reciprocal Rank (MRR@k)**: Measures the average reciprocal of the rank at which the first relevant document is retrieved

### Results

DocT5Query vs BM25 baseline:

| Method    | MRR@1             | MRR@5             | MRR@10            |
|-----------|-------------------|-------------------|-------------------|
| BM25      | 0.5079            | 0.5511            | 0.5561            |
| DocT5Query| 0.5224            | 0.5629            | 0.5679            |
| Improvement| +0.0145 (2.86%)  | +0.0118 (2.14%)   | +0.0117 (2.11%)   |

## Performance Insights

- DocT5Query consistently outperforms the BM25 baseline across all evaluation metrics
- The largest improvement is observed at MRR@1, suggesting better top result relevance
- The model takes approximately 7 hours to train and run on the scientific papers dataset
- Most of the computation time (~5.7 hours) is spent on document expansion

## Error Analysis

We performed an in-depth analysis of DocT5Query's performance compared to the baseline BM25:

1. **Strengths**:
   - **Better handling of synonyms**: DocT5Query successfully retrieved papers that used different terminology than the queries
   - **Improved ranking**: The model placed relevant documents higher in the result list
   - **Reduced false negatives**: Captured papers that BM25 missed entirely

2. **Weaknesses**:
   - **Occasional topic drift**: Some generated queries introduced noise that slightly reduced precision
   - **Limited improvement for highly technical tweets**: The expansion was less effective for tweets using very specialized scientific terminology
   - **Resource intensive**: The document expansion process requires significant computational resources

We saved model predictions for 50 queries from the development set in `DocT5Query/predictions/doct5query_predictions.tsv`. Analysis of these predictions revealed several patterns:

- DocT5Query improved ranking by an average of 1.7 positions for correctly retrieved documents
- 12% of queries showed significant improvement (5+ positions)
- 8% of queries showed worse performance compared to baseline

The most substantial improvements were observed for short, ambiguous queries where the vocabulary gap between tweets and scientific papers was pronounced.

## Implementation Checkpoints

The implementation saves model checkpoints throughout training:
- Format: PyTorch state dictionaries
- Best model path: `./model/best_model.pt`
- Wandb artifacts: 10 checkpoints (one from each epoch as validation loss improved)

## Reflection

### Current Performance

DocT5Query successfully outperforms the baseline BM25 model across all evaluation metrics. The 2.86% improvement in MRR@1 demonstrates that document expansion helps bridge the vocabulary gap between scientific papers and social media posts. The approach is particularly effective at:

1. Improving the ranking of relevant documents
2. Capturing semantic relationships that pure lexical matching misses
3. Maintaining the efficiency and interpretability of BM25 while enhancing its capabilities

### Limitations and Barriers

Despite the improvements, we encountered several limitations:

1. **Computational overhead**: The document expansion process is time-consuming and requires significant resources
2. **Limited gain ceiling**: The improvement plateau suggests there's an upper limit to how much expansion alone can help
3. **Dependency on BM25**: The method inherits some limitations of the underlying retrieval system

The biggest barrier we faced was balancing the number of generated queries per document. Too few queries didn't sufficiently bridge the vocabulary gap, while too many introduced noise and diluted the document's core content.

### Future Work

For system development in upcoming weeks, we plan to:

1. **Implement ColBERT v2 model**: Move beyond document expansion to explore dense retrieval with late interaction, which has shown promising results for scientific document retrieval
2. **Hybrid approaches**: Combine DocT5Query with neural retrievers to get the best of both worlds
3. **Improved query generation**: Fine-tune the query generation process to produce more focused and relevant queries

The ColBERT v2 approach will allow us to compare the effectiveness of document expansion versus neural retrieval models for this specific scientific claim source retrieval task.

## Conclusion

DocT5Query demonstrates effective document expansion for scientific claim source retrieval, improving over standard BM25 retrieval. The expansion with synthetic tweets helps bridge the vocabulary gap between scientific papers and social media queries, resulting in measurable performance gains. 