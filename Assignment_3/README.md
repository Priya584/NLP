# NLP Assignment 3: Trie-Based Stemming and Frequency Analysis

## Overview
This assignment implements trie data structures for unsupervised stemming and performs frequency-based stopword removal on Gujarati text corpus.

## Components

### 1. Trie Data Structure
Custom implementation with:
- **TrieNode**: Stores children, frequency count, and end-of-word marker
- **Trie**: Supports word insertion and stem/suffix identification
- **Branching Detection**: Identifies morphological boundaries using branch points

### 2. Prefix Trie Stemming
Builds trie from word beginnings to identify common prefixes:
- Inserts words character-by-character
- Finds stem at last branching node
- Example: "investigation" → stem="investigat", suffix="ion"

**Sample Output**:
```
investigation = investigat+ion
election = electi+on
irregularities = irregularit+ies
manner = manner
```

### 3. Suffix Trie Stemming
Builds trie from reversed words to identify common suffixes:
- Processes words in reverse order
- Identifies suffix patterns
- Example: "investigation" → stem="inve", suffix="stigation"

**Sample Output**:
```
investigation = inve+stigation
election = +election
irregularities = irregu+larities
```

### 4. Branching Probability Analysis
Analyzes trie structure to understand morphological patterns:
- Computes branching probability at each depth
- Generates visualization plots showing:
  - Depth vs. branching probability for prefix trie
  - Depth vs. branching probability for suffix trie
- Helps identify optimal stem boundaries

### 5. Gujarati Text Frequency Analysis
Processes tokenized Gujarati sentences:
- Builds word frequency distribution
- Visualizes top 100 most frequent words
- Implements stopword removal by frequency threshold

**Thresholds Tested**:
- Remove words appearing ≤ 500 times
- Remove words appearing ≤ 100 times
- Remove words appearing ≤ 200 times

## Dependencies
```python
matplotlib
pandas
pyarrow
collections
```

## Dataset
- **Input**: `brown_nouns.txt` (English nouns)
- **Input**: `tokenized_gujarati_sentences.parquet` (Gujarati corpus)
- **Font Required**: NotoSansGujarati-Regular.ttf for visualization

## Usage

### Build Trie
```python
prefix_trie = build_prefix_trie(words)
suffix_trie = build_suffix_trie(words)
```

### Find Stem and Suffix
```python
stem, suffix = prefix_trie.find_split("investigation")
print(f"{word} = {stem}+{suffix}")
```

### Analyze Branching
```python
stats = collect_branching_stats(prefix_trie)
plot_branching_probability(stats, "Prefix Trie")
```

### Frequency Analysis
```python
freq_dict = build_frequency_distribution(sentences)
plot_top_words(freq_df, "Top 100 Words")
plot_after_stopword_removal(threshold=500)
```

## Key Insights

### Prefix vs. Suffix Tries
- **Prefix Tries**: Better for identifying root words and derivational morphology
- **Suffix Tries**: Better for inflectional morphology and common endings

### Branching Points
- High branching probability indicates morpheme boundaries
- Helps in unsupervised morphological segmentation
- Different patterns for prefixes vs. suffixes

### Stopword Removal
- Frequency-based approach removes common function words
- Different thresholds affect vocabulary size and information retention
- Visualizations show distribution changes

## Output Files
- Branching probability plots (matplotlib figures)
- Filtered word frequency visualizations
- Statistical summaries

## Applications
- Unsupervised stemming for low-resource languages
- Morphological analysis without labeled data
- Stopword identification
- Vocabulary reduction


