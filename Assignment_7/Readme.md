# NLP Assignment 7: Advanced Text Analysis

## Overview
This assignment extends the work from Assignment 6, providing deeper analysis of PMI scores, TF-IDF vectorization, and nearest neighbor search for Gujarati text corpus.

## Components

### 1. Data Processing Pipeline
Complete workflow for text analysis:
- Load Parquet dataset
- Split into train/validation/test sets
- Load pre-computed n-gram probabilities
- Compute linguistic features
- Perform similarity analysis

### 2. PMI (Pointwise Mutual Information) Analysis
Extended analysis of word associations:

**Computation**:
```
PMI(w₁, w₂) = log₂(P(w₂|w₁) / P(w₂))
```

**Features**:
- Processes entire validation and test sets
- Handles missing probabilities gracefully
- Identifies collocations and word associations
- Returns -∞ for unseen bigrams

**Results Summary**:
- **Validation Set**: 106,764 unique bigrams analyzed
- **Test Set**: 106,090 unique bigrams analyzed
- **Input Data**: 138,132 unigrams, 822,206 bigrams

### 3. TF-IDF Vectorization
Sophisticated feature extraction:

**Configuration**:
- Custom tokenizer: whitespace splitting
- Lowercase: False (preserves Gujarati script)
- Token pattern: None (custom tokenization)

**Vocabulary Learning**:
- Fit only on training data (80,000 sentences)
- Vocabulary size: 120,665 unique features
- Transform validation and test using learned vocabulary

**Output Matrices**:
```
Training:   (80,000 × 120,665) sparse matrix
Validation: (10,000 × 120,665) sparse matrix  
Test:       (10,000 × 120,665) sparse matrix
```

### 4. Nearest Neighbor Search
Cosine similarity-based sentence matching:

**Algorithm**: k-Nearest Neighbors
- k=2 (self + 1 neighbor)
- Metric: Cosine distance
- Algorithm: Brute-force (exact search)

**Search Strategy**:
- Validation sentences searched within validation set
- Test sentences searched within test set
- Prevents train-test contamination

### 5. Sample Results Analysis

**Example 1**:
```
Original (idx 0): આ આધુનિક હોસ્પિટલ આજે ખાલી બેડ્ સ...
Neighbor (idx 3839): સ .
Distance: 0.7046
```

**Example 2**:
```
Original (idx 1): દહેજના ૪ અગ્રણીઓ ડિટેઇન...
Neighbor (idx 9837): એક સોશિયલ વર્કર હોવા છતાં...
Distance: 0.8233
```

**Example 3**:
```
Original (idx 2): એવી સ્થિતિમાં ડ્રોપલેટનો...
Neighbor (idx 5162): ચા વાળા વડાપ્રધાન બની શકે...
Distance: 0.7121
```

## Dependencies
```python
pandas
numpy
scikit-learn
math
collections
```

## Dataset Specifications
- **Format**: Parquet (efficient columnar storage)
- **Total Sentences**: 100,000 Gujarati sentences
- **Training Split**: 80,000 sentences
- **Validation Split**: 10,000 sentences
- **Test Split**: 10,000 sentences
- **Vocabulary**: 120,665 unique tokens

## Input Files
1. `tokenized_gujarati_sentences.parquet` - Main corpus
2. `unigram_probs.csv` - Unigram probabilities
3. `bigram_probs.csv` - Bigram probabilities

## Code Structure

### Main Function
```python
main()
```
Orchestrates entire pipeline:
1. Load and split data
2. Load probability files
3. Compute PMI scores
4. Build TF-IDF vectors
5. Find nearest neighbors

### Helper Functions
- `load_and_split_data()` - Data loading and splitting
- `calc_pmi()` - PMI computation for word pairs
- `get_pmi_for_set()` - Batch PMI calculation
- `find_nn()` - Nearest neighbor search

## Usage

### Run Complete Pipeline
```python
if __name__ == "__main__":
    main()
```

### Configure Parameters
```python
TOKENIZED_FILE = 'tokenized_gujarati_sentences.parquet'
TEXT_COLUMN_INDEX = 0
UNIGRAM_CSV = 'unigram_probs.csv'
BIGRAM_CSV = 'bigram_probs.csv'
VAL_SIZE = 0.10
TEST_SIZE = 0.10
```

### Custom PMI Analysis
```python
pmi_score = calc_pmi("word1", "word2", unigram_probs, bigram_probs)
```

### Custom Vectorization
```python
vectorizer = TfidfVectorizer(
    tokenizer=lambda x: x.split(),
    lowercase=False,
    token_pattern=None
)
X = vectorizer.fit_transform(sentences)
```

## Key Insights

### PMI Distribution
- Most bigrams have moderate PMI values
- High PMI indicates strong collocations
- Many bigrams have -∞ PMI (never co-occur)
- Useful for identifying multi-word expressions

### TF-IDF Characteristics
- Highly sparse matrices (~99% zeros)
- Captures document-specific terms effectively
- Vocabulary learned from training prevents overfitting
- Cosine distance effective for similarity

### Nearest Neighbor Patterns
- Distance range: 0.70 - 0.82 in samples
- Shorter sentences tend to match other short sentences
- Semantic similarity visible but not perfect
- Some matches based on common function words

## Performance Considerations

### Memory Efficiency
- Sparse matrix representation saves memory
- Parquet format reduces I/O time
- Streaming computation for large datasets

### Computational Complexity
- TF-IDF: O(n × m) where n=docs, m=vocab
- Nearest Neighbors: O(n²) for brute-force
- PMI: O(unique_bigrams)

## Applications

### Information Retrieval
- Find similar documents
- Query expansion using PMI
- Document ranking

### Text Mining
- Collocation extraction
- Phrase identification
- Semantic clustering

### Content Analysis
- Duplicate detection
- Plagiarism checking
- Topic discovery

### Recommendation Systems
- Content-based filtering
- Similar article suggestions
- User preference modeling

## Evaluation Metrics

### For PMI
- Count of high-PMI bigrams (collocations)
- Distribution analysis
- Comparison with linguistic intuition

### For TF-IDF + NN
- Nearest neighbor accuracy
- Distance distribution
- Semantic coherence of matches

## Future Enhancements
1. Word embeddings (Word2Vec, FastText)
2. Contextual embeddings (BERT, RoBERTa)
3. Approximate nearest neighbors (faster search)
4. Cross-lingual similarity
5. Fine-tuned distance metrics

## Notes
- Gujarati script requires proper font support for visualization
- Unicode handling is critical for correct tokenization
- Reproducibility ensured through fixed random seeds
- Modular design allows easy experimentation


