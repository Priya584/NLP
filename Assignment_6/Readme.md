# NLP Assignment 6: PMI and TF-IDF Analysis

## Overview
This assignment implements Pointwise Mutual Information (PMI) computation and TF-IDF vectorization for Gujarati text, along with nearest neighbor search for sentence similarity.

## Components

### 1. Data Splitting
Systematic division of corpus into:
- **Training Set**: 80,000 sentences (80%)
- **Validation Set**: 10,000 sentences (10%)
- **Test Set**: 10,000 sentences (10%)

**Features**:
- Reproducible splits (random_state=42)
- Proper stratification
- Load from Parquet format

### 2. Pointwise Mutual Information (PMI)

Measures association strength between words in bigrams:

**Formula**:
```
PMI(w₁, w₂) = log₂(P(w₂|w₁) / P(w₂))
```

**Interpretation**:
- PMI > 0: Words appear together more than by chance (positive association)
- PMI = 0: Words are independent
- PMI < 0: Words avoid each other (negative association)
- PMI = -∞: Bigram never observed

**Features**:
- Uses pre-computed unigram and bigram probabilities
- Computes PMI for all unique bigrams in validation and test sets
- Handles zero-probability cases
- Base-2 logarithm for information-theoretic interpretation

**Results**:
- Validation: 106,764 unique bigrams
- Test: 106,090 unique bigrams

### 3. TF-IDF Vectorization

Transforms text into numerical feature vectors:

**TF (Term Frequency)**:
```
TF(t,d) = count(t in d)
```

**IDF (Inverse Document Frequency)**:
```
IDF(t) = log(N / df(t))
```

**TF-IDF Score**:
```
TF-IDF(t,d) = TF(t,d) × IDF(t)
```

where:
- N = total number of documents
- df(t) = number of documents containing term t

**Features**:
- Fit on training data only (prevents data leakage)
- Transform validation and test sets using training vocabulary
- Preserves Gujarati Unicode characters
- Whitespace tokenization
- No lowercasing (important for Gujarati script)

**Matrix Shapes**:
```
Train: (80,000 sentences, 120,665 features)
Val:   (10,000 sentences, 120,665 features)
Test:  (10,000 sentences, 120,665 features)
```

### 4. Nearest Neighbor Search

Finds most similar sentences using cosine distance:

**Distance Metric**:
```
cosine_distance(u, v) = 1 - (u·v) / (||u|| × ||v||)
```

**Features**:
- k=2 nearest neighbors (self + closest match)
- Brute-force algorithm for accuracy
- Cosine similarity metric
- Searches within each dataset (val with val, test with test)

**Sample Results**:
```
Original: આ આધુનિક હોસ્પિટલ આજે ખાલી બેડ્ સ...
Neighbor: સ .
Cosine Distance: 0.7046

Original: દહેજના ૪ અગ્રણીઓ ડિટેઇન...
Neighbor: એક સોશિયલ વર્કર હોવા છતાં...
Cosine Distance: 0.8233
```

## Dependencies
```python
pandas
numpy
sklearn.feature_extraction.text.TfidfVectorizer
sklearn.neighbors.NearestNeighbors
sklearn.model_selection.train_test_split
math
```

## Dataset
- **Input**: `tokenized_gujarati_sentences.parquet`
- **Probability Files**: 
  - `unigram_probs.csv` (138,132 words)
  - `bigram_probs.csv` (822,206 bigrams)
- **Total Sentences**: 100,000

## Usage

### Load and Split Data
```python
train, val, test = load_and_split_data(
    'tokenized_gujarati_sentences.parquet',
    col_idx=0,
    val_sz=0.1,
    tst_sz=0.1
)
```

### Calculate PMI
```python
pmi_score = calc_pmi(word1, word2, unigram_probs, bigram_probs)
val_pmi = get_pmi_for_set(val_sentences, unigram_probs, bigram_probs)
```

### TF-IDF Vectorization
```python
vectorizer = TfidfVectorizer(
    tokenizer=lambda x: x.split(),
    lowercase=False,
    token_pattern=None
)
X_train = vectorizer.fit_transform(train_sentences)
X_val = vectorizer.transform(val_sentences)
X_test = vectorizer.transform(test_sentences)
```

### Find Nearest Neighbors
```python
nn_model = NearestNeighbors(n_neighbors=2, metric='cosine', algorithm='brute')
nn_model.fit(X_val)
distances, indices = nn_model.kneighbors(X_val)
```

## Key Concepts

### PMI Applications
- **Collocation Detection**: Find meaningful word pairs
- **Word Association**: Measure semantic relatedness
- **Feature Selection**: Identify informative word pairs
- **Semantic Networks**: Build word association graphs

### TF-IDF Properties
- **Downweights Common Words**: Reduces importance of frequent terms
- **Emphasizes Distinctive Terms**: Highlights document-specific vocabulary
- **Sparse Representation**: Most values are zero
- **Scalable**: Works with large vocabularies

### Cosine Similarity
- **Range**: [0, 1] for distance; [0, 1] for similarity
- **Angular**: Measures angle between vectors, not magnitude
- **Normalized**: Length-independent comparison
- **Efficient**: Fast computation with sparse matrices

## Analysis Insights

### PMI Patterns
- High PMI indicates strong collocations (idioms, technical terms)
- Negative PMI suggests words that don't co-occur
- Useful for identifying multi-word expressions

### TF-IDF Distribution
- Vocabulary size: 120,665 unique tokens
- Most features have low values (sparse matrix)
- Captures document uniqueness effectively

### Nearest Neighbors
- Cosine distance ranges from 0.70 to 0.82 in samples
- Shorter sentences often match with similar short sentences
- Semantic similarity visible in matches
- Some matches may be coincidental (common words)

## Output
The script prints:
1. Data split statistics
2. PMI computation progress
3. TF-IDF matrix shapes
4. Sample nearest neighbor pairs with distances

## Applications
- **Document Similarity**: Find related texts
- **Information Retrieval**: Search engines
- **Duplicate Detection**: Identify similar content
- **Recommendation Systems**: Suggest related documents
- **Clustering**: Group similar texts
- **Topic Modeling**: Discover themes in corpus


