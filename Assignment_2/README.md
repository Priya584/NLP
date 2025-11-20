# NLP Assignment 2: Finite State Automata and Morphological Analysis

## Overview
This assignment implements Deterministic Finite Automata (DFA) for pattern recognition and Finite State Transducers (FST) for English noun morphological analysis.

## Components

### 1. DFA Implementation
- **Task**: Recognize lowercase English words containing only letters
- **States**: 
  - q0 (initial state)
  - q1 (accepting state for valid words)
  - q_dead (reject state)
- **Input Symbols**: `{letter, other}`
- **Visualization**: Generates state transition diagram using `visual-automata` library

**Test Cases**:
- Accepted: "cat", "dog", "a", "zebra"
- Rejected: "dog1", "1dog", "DogHouse", "Dog_house", " cats", ""

### 2. Noun Morphological Analyzer
Implements FST-like analysis to identify:
- Singular nouns (N+SG)
- Plural nouns (N+PL)
- Irregular plural forms

**Features**:
- Handles irregular nouns (men, women, children, teeth, etc.)
- Processes regular plurals (-s, -es, -ies endings)
- Detects sibilant endings (sh, ch, x, z, s)
- Validates word format (lowercase, alphabetic)

### 3. Brown Corpus Processing
- Reads nouns from `brown_nouns.txt` (202,793 nouns)
- Performs morphological analysis on each noun
- Outputs results in multiple formats:
  - Compressed Parquet (with Snappy compression)
  - CSV format

## Dependencies
```python
visual-automata
pandas
pyarrow
graphviz
```

## Usage

### Run DFA Simulation
```python
run_simulation("cat")  # Returns True
run_simulation("1dog")  # Returns False
```

### Analyze Nouns
```python
analyze_noun("cats")     # Returns "cat+N+PL"
analyze_noun("children") # Returns "child+N+PL"
analyze_noun("book")     # Returns "book+N+SG"
```

### Process Corpus
```python
process_file_to_parquet("brown_nouns.txt", "noun_analysis.parquet")
process_file_to_csv("brown_nouns.txt", "noun_analysis.csv")
```

## Output Files
- `dfa_diagram.png` - Visual representation of DFA
- `noun_analysis.parquet` - Compressed analysis results
- `noun_analysis.csv` - CSV format analysis results

## FST Design
The morphological analyzer follows a formal FST design with:
- Defined input/output alphabets
- State transition table
- Context-sensitive rules for plural formation
- Special handling for sibilants and irregular forms

## Key Concepts
- **DFA**: Recognizes valid patterns in strings
- **FST**: Transforms input strings to output strings
- **Morphological Analysis**: Breaking down words into root + grammatical features
- **State Minimization**: Efficient automaton design


