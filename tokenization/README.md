# Tokenization

This project demonstrates fundamental tokenization techniques for Natural Language Processing (NLP) using Python and NLTK (Natural Language Toolkit).

## What is Tokenization?

Tokenization is the process of breaking down text into smaller units called **tokens**. These tokens can be words, sentences, or other meaningful segments. Tokenization is a fundamental preprocessing step in NLP and information retrieval systems, enabling computers to analyze and process human language.

## Why Tokenization Matters

- **Text Analysis**: Breaking text into manageable pieces for analysis
- **Feature Extraction**: Converting text into numerical representations for machine learning
- **Information Retrieval**: Indexing and searching documents efficiently
- **Text Preprocessing**: Cleaning and normalizing text data for downstream tasks

## Installation

This project uses UV for dependency management. Install dependencies with:

```bash
uv sync
```

Or with pip:

```bash
pip install nltk
```

**Note**: On first run, the script automatically downloads the required NLTK `punkt_tab` tokenizer data (requires internet connection).

## Tokenization Methods

This project demonstrates three tokenization approaches:

### 1. Word Tokenization

Splits text into individual words and punctuation marks.

```python
word_tokens = nltk.word_tokenize(text)
```

**Example:**
```
Input: "Machine learning é um campo da inteligência artificial..."

Output: ['Machine', 'learning', 'é', 'um', 'campo', 'da', 'inteligência', 
         'artificial', 'que', 'permite', ...]
```

**Use Cases:**
- Counting word frequencies
- Building vocabulary
- Text classification
- Basic text analysis

### 2. Sentence Tokenization

Splits text into complete sentences, intelligently handling sentence boundaries.

```python
sentence_tokens = nltk.sent_tokenize(text)
```

**Example:**
```
Input: "Machine learning é um campo... Sem serem programados explicitamente..."

Output: [
    'Machine learning é um campo da inteligência artificial que permite...',
    'Sem serem programados explicitamente para cada tarefa.'
]
```

**Use Cases:**
- Text summarization
- Sentence-level analysis
- Document segmentation
- Natural language understanding

### 3. Preprocessed Tokenization

Custom preprocessing pipeline that combines tokenization with cleaning steps.

```python
def preprocess(input: list[str]):
    tokens = nltk.word_tokenize(input.lower())
    return [word for word in tokens if word.isalnum()]
```

**Processing Steps:**
1. Convert text to lowercase (normalization)
2. Tokenize into words
3. Filter out non-alphanumeric tokens (remove punctuation)

**Example:**
```
Input: "Machine learning é o aprendizado automático de máquinas a partir de dados."

Output: ['machine', 'learning', 'é', 'o', 'aprendizado', 'automático', 
         'de', 'máquinas', 'a', 'partir', 'de', 'dados']
```

**Use Cases:**
- Text cleaning for machine learning
- Removing noise from text
- Preparing data for vectorization
- Information retrieval preprocessing

## Usage

Run the demonstration script:

```bash
python 01-tokenization.py
```

Or with UV:

```bash
uv run 01-tokenization.py
```

### Expected Output

The script demonstrates all three tokenization methods on Portuguese text:

1. **Word tokens**: List of individual words and punctuation
2. **Sentence tokens**: List of complete sentences
3. **Preprocessed documents**: Cleaned, lowercase tokens for three sample documents

## Example Code

```python
import nltk

# Download required data
nltk.download("punkt_tab")

# Sample text
text = "Machine learning é um campo da inteligência artificial..."

# Word tokenization
word_tokens = nltk.word_tokenize(text)
print(word_tokens)

# Sentence tokenization
sentence_tokens = nltk.sent_tokenize(text)
print(sentence_tokens)

# Custom preprocessing
def preprocess(input: str):
    tokens = nltk.word_tokenize(input.lower())
    return [word for word in tokens if word.isalnum()]

# Apply to documents
documents = ["Text 1...", "Text 2...", "Text 3..."]
preprocessed = [" ".join(preprocess(doc)) for doc in documents]
print(preprocessed)
```

## Key Concepts

### Tokens
The individual units produced by tokenization (words, sentences, etc.)

### Normalization
Converting text to a standard form (e.g., lowercase) to reduce variations

### Filtering
Removing unwanted elements like punctuation or special characters

### Alphanumeric Tokens
Tokens containing only letters and numbers (no punctuation or symbols)

## Multi-Language Support

NLTK's tokenizers support multiple languages, including:
- English
- Portuguese (as demonstrated in this project)
- Spanish, French, German, and many others

The `punkt` tokenizer uses trained models for language-specific sentence boundaries.

## Dependencies

- **nltk** (v3.9.2): Natural Language Toolkit
- **punkt_tab**: NLTK tokenizer models (downloaded automatically)

## Next Steps

After mastering tokenization, you can explore:

1. **Stemming/Lemmatization**: Reducing words to root forms
2. **Stop Word Removal**: Filtering common words
3. **TF-IDF**: Term frequency-inverse document frequency
4. **Word Embeddings**: Vector representations of words
5. **Document Indexing**: Building search systems

## Project Context

This project is part of a larger curriculum on **fundamentals of retrieval systems**, where tokenization serves as the foundation for text processing and information retrieval.

## License

This is an educational project for learning NLP fundamentals.
