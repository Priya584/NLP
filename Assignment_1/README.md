## Gujarati Tokenizer – Sentence & Word Level

This tokenizer processes Gujarati text into clean, meaningful units using regex-based rules. The tokenizer works on streaming data from the `IndicCorpV2` dataset.

---

### Sentence Tokenization

We use a custom sentence boundary detector for Gujarati that avoids incorrect splits on common patterns like dates, currency, and titles.

#### 🛡️ Protected Patterns

Before splitting, we **temporarily replace** sensitive patterns with placeholders:

| Pattern | Meaning |
|--------|---------|
| `તા\s*\.` | Used in dates (e.g., `તા. 5 ઓગસ્ટ`) |
| `રૂા\s*\.` | Currency notation (`રૂા. 500`) |
| `શ્રી\s*\.` / `શ્રીમતી\s*\.` | Honorifics |
| `ડૉ\s*\.` | Doctor title |

These are later **restored** after sentence splitting.

#### 📍 Sentence Split Pattern

```python
pattern = r'(?<!\d)\.(?!\d)|[\u0964!?]'
```

**Explanation:**

- `(?<!\d)\.(?!\d)` → Matches a dot (`.`) **only if it’s not between digits**  
  → Prevents splits on numbers like `3.14`

- `[\u0964!?]` → Matches:
  - `\u0964` → Danda `।` (used in Gujarati/Hindi)
  - `!` → Exclamation
  - `?` → Question mark

---

### Word Tokenization

We tokenize Gujarati sentences into words, punctuation, and symbols using:

```python
pattern = r'(?:રૂા\.?|[$])\d[\d,.]*|\d[\d,.]*|[\u0A80-\u0AFF\w]+(?:-[\u0A80-\u0AFF\w]+)*|[$]|[.,\u0964!?;:()"\"'-]'
```

### 🔍 Word Tokenizer Pattern Explanation

| Pattern | Matches |
|--------|---------|
| `(?:રૂા\.?\|[$])\d[\d,.]*` | Currency like `રૂા.500` or `$500` |
| `\d[\d,.]*` | Numbers like `12,000` or `3.14` |
| `[\u0A80-\u0AFF\w]+(?:-[\u0A80-\u0AFF\w]+)*` | Gujarati words with optional hyphenation |
| `[$]` | Standalone dollar sign |
| `[.,\u0964!?;:()"\'-]` | Punctuation marks |


---

### Output

- Sentence tokenizer stores **one sentence per row**
- Paragraph tokenizer joins tokenized sentences per paragraph
- Final data is stored in `.parquet` format using **Snappy compression**
