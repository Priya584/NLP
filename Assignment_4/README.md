# NLP Assignment 4: N-gram Language Models and Smoothing

## Overview
This assignment implements n-gram language models (unigram through quadrigram) with various smoothing techniques for Gujarati text, and computes sentence probabilities.

## Components

### 1. N-gram Model Construction
Builds probabilistic language models:
- **Unigram**: P(w) = count(w) / total_words
- **Bigram**: P(w₂|w₁) = count(w₁,w₂) / count(w₁)
- **Trigram**: P(w₃|w₁,w₂) = count(w₁,w₂,w₃) / count(w₁,w₂)
- **Quadrigram**: P(w₄|w₁,w₂,w₃) = count(w₁,w₂,w₃,w₄) / count(w₁,w₂,w₃)

**Features**:
- Automatic context handling with `<s>` and `</s>` markers
- Efficient counting using `Counter` from collections
- Probability normalization

### 2. Smoothing Techniques

#### a) Add-One (Laplace) Smoothing
```
P(w₂|w₁) = (count(w₁,w₂) + 1) / (count(w₁) + V)
```
where V = vocabulary size

#### b) Add-K Smoothing
```
P(w₂|w₁) = (count(w₁,w₂) + k) / (count(w₁) + k*V)
```
Tested with k=0.5 for better probability distribution

#### c) Add-Type (Token-Type) Smoothing
```
P(w₂|w₁) = (count(w₁,w₂) + 1) / (count(w₁) + num_bigram_types)
```
Alternative smoothing based on unique n-gram types

### 3. Sentence Probability Calculation
Computes probability of complete sentences:
- Uses log-space arithmetic to prevent underflow
- Handles very long sentences
- Samples 1000 random sentences for evaluation

**Output Format**:
```
Sentence: [Gujarati text]
Add-One: log-prob = -104.85, prob = 2.91e-46
Add-K: log-prob = -100.30, prob = 2.76e-44
Add-Type: log-prob = -119.32, prob = 1.51e-52
```

### 4. Model Evaluation
- Computes average log-probability over sample
- Compares different smoothing methods
- Analyzes impact of smoothing on rare events

## Dependencies
```python
pandas
collections.Counter
collections.defaultdict
csv
math
random
```

## Dataset
- **Input**: `tokenized_gujarati_sentences.parquet`
- **Corpus Size**: 14,811 sentences
- **Vocabulary**: 6,278 unique words (from 1000 samples)

## Usage

### Build N-gram Model
```python
unigram_model = build_ngram_model(sentences, n=1)
bigram_model = build_ngram_model(sentences, n=2)
trigram_model = build_ngram_model(sentences, n=3)
quadrigram_model = build_ngram_model(sentences, n=4)
```

### Save Model to CSV
```python
save_ngram_to_csv(model, "Bigram", "bigram_probs.csv")
```

### Compute Sentence Probability
```python
log_prob, prob = sentence_prob(tokens, method="add_one")
log_prob, prob = sentence_prob(tokens, method="add_k", k=0.5)
log_prob, prob = sentence_prob(tokens, method="add_type")
```

### Evaluate Model
```python
avg_logp = avg_logprob_over_samples(samples, "add_one")
```

## Output Files
- `unigram_probs.csv` - Unigram probabilities
- `bigram_probs.csv` - Bigram probabilities  
- `trigram_probs.csv` - Trigram probabilities
- `quadrigram_probs.csv` - Quadrigram probabilities
- `sentence_probabilities.csv` - Sentence-level probabilities

## Key Concepts

### Smoothing Necessity
- Handles zero-probability problem for unseen n-grams
- Prevents multiplication by zero in probability calculations
- Redistributes probability mass from seen to unseen events

### Log-Space Computation
- Prevents numerical underflow with very small probabilities
- Converts products to sums: log(a×b) = log(a) + log(b)
- Essential for long sentences

### Trade-offs
- **Add-1**: Simple but over-smooths (gives too much mass to unseen events)
- **Add-k** (k<1): More flexible, better probability distribution
- **Add-Type**: Alternative formulation, may not normalize to 1

## Sample Results
```
Vocabulary size: 6,278 words
Average log-prob (Add-One): -72.45
Average log-prob (Add-K k=0.5): -68.32
Average log-prob (Add-Type): -84.17
```

## Applications
- Language modeling for Gujarati
- Text generation
- Spelling correction
- Machine translation
- Speech recognition


