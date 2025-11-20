# HMM Part-of-Speech Tagging - Assignment

Implementation of Hidden Markov Model (HMM) for automatic Part-of-Speech tagging using the Viterbi algorithm.

## Assignment Tasks Completed

### Task (a): Data Splitting
- ✅ Split data into training (80%) and testing (20%) sets
- ✅ Random shuffling for unbiased distribution
- 123 training sentences, 31 testing sentences from 154 total

### Task (b): Probability Calculation
- ✅ **Emission Probabilities**: P(word | tag)
- ✅ **Transition Probabilities**: P(tag_i | tag_{i-1})
- ✅ **Initial Probabilities**: P(tag at sentence start)

### Task (c): Viterbi Decoding
- ✅ Implemented Viterbi algorithm for optimal tag sequence prediction
- ✅ Dynamic programming approach with O(T² × N) complexity
- ✅ Backpointer matrix for path reconstruction

### Task (d): Performance Evaluation
- ✅ **Accuracy**: 74.91%
- ✅ **Correct Predictions**: 612 out of 817 words
- ✅ Evaluation on test set with unseen data

## Requirements

```bash
pip install numpy
```

## Dataset

- **File**: `pos_tagdata.txt`
- **Format**: `word_TAG` pairs separated by spaces
- **Tags**: 38 unique POS tags (DT, NN, VB, JJ, etc.)
- **Vocabulary**: 1,167 unique words

## Usage

Run the program:
```bash
python exam.py
```

Output:
```
Loading data...
Total sentences: 154

Splitting data (80:20)...
Training sentences: 123
Testing sentences: 31

Training HMM model...
Number of unique tags: 38
Number of unique words: 1167

Evaluating on test set...

Results:
Accuracy: 74.91%
Correct predictions: 612/817
```

## Implementation Details

### HMM Components

1. **Training Phase**
   - Parse sentences to extract words and tags
   - Count occurrences of tags, tag transitions, and word-tag pairs
   - Convert counts to probabilities

2. **Viterbi Algorithm**
   - Uses log probabilities to prevent underflow
   - Maintains probability matrix and backpointer matrix
   - Finds most likely tag sequence efficiently

3. **Evaluation**
   - Compares predicted tags with true tags
   - Calculates overall accuracy percentage

### Code Structure

```python
class HMM_POS_Tagger:
    - parse_sentence()    # Extract words and tags
    - train()             # Learn probabilities from data
    - viterbi()           # Decode optimal tag sequence
    - predict()           # Tag new sentence
    - evaluate()          # Calculate accuracy
```

## Results

| Metric | Value |
|--------|-------|
| Accuracy | 74.91% |
| Correct | 612 |
| Total | 817 |

**Sample Output:**
```
Words:      The cat sleeps
True Tags:  DT NN VBZ
Predicted:  DT NN VBZ
```

## Key Features

- No external NLP libraries used (pure implementation)
- Handles unknown words with smoothing (1e-10)
- Efficient dynamic programming approach
- Modular, object-oriented code design

## Files

- `exam.py` - Main implementation
- `pos_tagdata.txt` - Training/testing data
- `README.md` - Documentation
- `requirements.txt` - Dependencies

## How It Works

1. **Load Data**: Read tagged sentences from file
2. **Split**: 80% training, 20% testing (random shuffle)
3. **Train**: Calculate emission, transition, and initial probabilities
4. **Predict**: Use Viterbi algorithm to find best tag sequence
5. **Evaluate**: Measure accuracy on test set


