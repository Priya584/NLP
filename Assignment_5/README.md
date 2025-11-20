# NLP Assignment 5: Advanced Smoothing and Text Generation

## Overview
This assignment implements sophisticated smoothing techniques (Good-Turing, Katz Backoff, Kneser-Ney) and text generation algorithms (Greedy, Beam Search) for Gujarati language modeling.

## Components

### 1. Good-Turing Smoothing
Redistributes probability mass based on frequency-of-frequencies:

**Formula**:
```
c* = (c + 1) × N(c+1) / N(c)
P*(w) = c* / N
```
where:
- c = original count
- N(c) = number of n-grams appearing exactly c times
- c* = adjusted count

**Features**:
- Handles unseen n-grams using N₁ (singleton count)
- Builds frequency-of-frequency tables
- Works for unigram through quadrigram models
- Computes probability for unseen events

**Output**: Frequency tables with columns:
- C (MLE): Original count
- Nc: Frequency of frequency
- C*: Adjusted count
- P*: Smoothed probability

### 2. Deleted Interpolation (Quadrigrams)
Combines multiple n-gram orders using weighted mixture:

**Formula**:
```
P(w₄|w₁,w₂,w₃) = λ₁P₁(w₄) + λ₂P₂(w₄|w₃) + λ₃P₃(w₄|w₂,w₃) + λ₄P₄(w₄|w₁,w₂,w₃)
```
where Σλᵢ = 1

**Optimization**:
- Grid search over λ values (step=0.2)
- Validates on held-out validation set
- Maximizes validation log-likelihood

**Best Results**:
```
Best λ's: (0.6, 0.4, 0.0, 0.0)
Validation log-likelihood: -30,816.31
```

### 3. Katz Backoff Model
Hierarchical backoff scheme for unseen n-grams:

**Strategy**:
1. Try highest-order n-gram (quadrigram)
2. If not found, back off to trigram with discount α
3. Continue backing off to bigram, then unigram
4. Return small constant for completely unknown words

**Formula**:
```
P(w|ctx) = {
    P_observed(w|ctx)           if seen in training
    α × P(w|shorter_ctx)        otherwise
}
```

**Implementation**:
- Single unified probability dictionary
- Recursive backoff logic
- Alpha parameter (default=0.4) controls backoff weight

### 4. Kneser-Ney Smoothing (Recursive)
State-of-the-art smoothing using continuation probability:

**Unigram Base Case** (Formula 2):
```
P_kn(w) = N(•,w) / N(•,•)
```
where N(•,w) = number of unique contexts word w appears in

**Higher-Order** (Formula 1):
```
P_kn(w|ctx) = max(count(ctx,w) - d, 0) / count(ctx) + λ(ctx) × P_kn(w|shorter_ctx)
```
where:
- d = discount parameter (0.75)
- λ(ctx) = (d × unique_words(ctx)) / count(ctx)

**Features**:
- Approximates counts from probabilities (×1,000,000)
- Recursive implementation
- Handles contexts of varying lengths
- Superior performance on rare events

### 5. Text Generation Algorithms

#### a) Greedy Generation
- Selects word with maximum probability at each step
- Fast but deterministic
- May produce repetitive text

#### b) Beam Search
- Maintains top-k hypotheses (beam_size=5)
- Scores using log-probability
- Balances quality and diversity
- More coherent output than greedy

#### c) Sampling-Based Generation
- Random sampling according to probability distribution
- Introduces diversity
- Works with both greedy and beam search
- Produces more natural-sounding text

## Dependencies
```python
pandas
numpy
collections.Counter
collections.defaultdict
math
random
heapq
```

## Dataset
- **Input**: `tokenized_gujarati_sentences.parquet`
- **Training**: 80,000 sentences (80%)
- **Validation**: 10,000 sentences (10%)
- **Test**: 10,000 sentences (10%)
- **Vocabulary**: 40,240 unique words
- **Total Tokens**: 258,501

## Usage

### Build Good-Turing Model
```python
unigram_gt = good_turing_model(unigram_counts, V, 1)
bigram_gt = good_turing_model(bigram_counts, V, 2)
sentence_logprob(tokens, bigram_gt, n=2)
```

### Deleted Interpolation
```python
best_lambdas, score = grid_search_lambdas_fast(val_set, step=0.2, max_sent=200)
logprob = sentence_logprob_interpolated(tokens, best_lambdas)
```

### Katz Backoff
```python
katz = KatzBackoff(all_probs, alpha=0.4)
prob = katz.prob("<s> <s> <s>", "આ")  # P(w|context)
```

### Kneser-Ney
```python
knr = KneserNeyRecursive(d=0.75)
prob = knr.prob("આ એક", "છે")  # P(w|context)
```

### Text Generation
```python
sentence = generate_sentence_greedy(models, n=3, max_len=20)
sentence = generate_sentence_beam(models, n=3, beam_size=5, max_len=20)
```

## Output Files
- `frequency_table_n1_top100.csv` - Unigram GT frequencies
- `frequency_table_n2_top100.csv` - Bigram GT frequencies
- `frequency_table_n3_top100.csv` - Trigram GT frequencies
- `frequency_table_n4_top100.csv` - Quadrigram GT frequencies

## Sample Generated Text

### Trigram Greedy:
```
આ ઉપરાંત , તમે તમારા ઘરની કિંમત આપો જેથી તે એક સારો વિચાર છે .
```

### Trigram Beam Search:
```
જો કે , જો કે , જો કે , જો કે , જો કે ...
```

### Quadrigram with Sampling:
```
કેબિનેટ કક્ષાના મંત્રીઓની બેઠક યોજવામાં આવી હતી .
```

## Key Insights

### Good-Turing
- Best for well-behaved frequency distributions
- Requires sufficient data for each count level
- May not work well for very large counts

### Katz Backoff
- Simple and efficient
- Clear interpretation (try specific, fall back to general)
- May waste probability mass on unseen contexts

### Kneser-Ney
- State-of-the-art performance
- Better handling of rare words
- Uses word diversity (continuation probability)
- More complex implementation

### Generation Quality
- **Greedy**: Fast, deterministic, repetitive
- **Beam**: Better quality, explores alternatives
- **Sampling**: Most diverse, more natural
- Higher n-grams produce more coherent text

## Evaluation Metrics
- Log-likelihood on validation set
- Perplexity (exp(-avg_logprob))
- Generated text coherence
- Coverage of unseen n-grams

## Applications
- Language modeling
- Text generation
- Machine translation
- Speech recognition
- Autocomplete systems


