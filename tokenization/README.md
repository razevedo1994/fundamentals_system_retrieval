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

## Information Retrieval Systems

This project demonstrates four fundamental information retrieval approaches, building upon tokenization techniques. Each approach represents a different paradigm for searching and ranking documents.

### 1. Vector Space Model with TF-IDF (`02-tokenization.py`)

The **Vector Space Model** represents documents and queries as vectors in a high-dimensional space, where each dimension corresponds to a unique term in the corpus.

**Key Concepts:**

- **TF-IDF (Term Frequency-Inverse Document Frequency)**: A weighting scheme that balances how frequently a term appears in a document (TF) against how rare it is across all documents (IDF)
  - **TF**: Measures term importance within a document
  - **IDF**: Reduces weight of common terms, increases weight of rare terms
  - Formula: `TF-IDF(t,d) = TF(t,d) × IDF(t)`

- **Cosine Similarity**: Measures similarity between document and query vectors by computing the cosine of the angle between them
  - Range: 0 (orthogonal/unrelated) to 1 (identical direction)
  - Ignores magnitude, focuses on orientation

**Implementation:**

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Create TF-IDF matrix
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(preprocessed_docs)

# Search function
def search_tfidf(query, vectorizer, tfidf_matrix):
    query_vector = vectorizer.transform([query])
    similarities = cosine_similarity(tfidf_matrix, query_vector).flatten()
    return sorted(enumerate(similarities), key=lambda x: x[1], reverse=True)
```

**Advantages:**
- Simple and intuitive geometric interpretation
- Effective for keyword-based retrieval
- Captures term importance naturally

**Use Cases:**
- Document similarity comparison
- Basic search engines
- Content recommendation systems

See `02-tokenization.py:32-48` for complete implementation.

---

### 2. Boolean Retrieval Model (`03-tokenization.py`)

The **Boolean Retrieval Model** treats retrieval as a logical problem, using Boolean operators (AND, OR, NOT) to match exact query conditions.

**Key Concepts:**

- **Boolean Operators**:
  - `AND`: Document must contain all terms
  - `OR`: Document must contain at least one term
  - `NOT`: Document must not contain the term

- **Inverted Index**: Data structure mapping terms to documents containing them
  - Enables fast lookup of documents by term
  - Foundation of most search engines

- **Binary Matching**: Documents either match or don't match (no ranking)

**Implementation:**

```python
from whoosh.index import create_in
from whoosh.fields import Schema, ID, TEXT
from whoosh.qparser import QueryParser

# Create index schema
schema = Schema(title=ID(stored=True, unique=True), 
                content=TEXT(stored=True))

# Index documents
index = create_in("index_dir", schema)
writer = index.writer()
for i, doc in enumerate(documents):
    writer.add_document(title=str(i), content=doc)
writer.commit()

# Boolean search
def boolean_search(query, index):
    parser = QueryParser("content", schema=index.schema)
    parsed_query = parser.parse(query)  # e.g., "machine E learning"
    
    with index.searcher() as searcher:
        results = searcher.search(parsed_query)
        return [(hit["title"], hit["content"]) for hit in results]
```

**Advantages:**
- Precise control over search conditions
- Deterministic results
- Efficient for exact matching

**Limitations:**
- No ranking of results
- All-or-nothing matching
- Requires users to construct complex queries

**Use Cases:**
- Legal document search
- Database queries
- Technical documentation retrieval

See `03-tokenization.py:42-64` for complete implementation.

---

### 3. Probabilistic Retrieval Model - BM25 (`04-tokenization.py`)

The **Probabilistic Retrieval Model** ranks documents based on the probability of relevance to a query. **BM25** (Best Matching 25) is the most widely-used probabilistic ranking function.

**Key Concepts:**

- **Probability Ranking Principle**: Documents should be ranked by decreasing probability of relevance
  
- **BM25 Algorithm**: Sophisticated ranking function that considers:
  - **Term Frequency (TF)**: With saturation to prevent over-weighting repeated terms
  - **Document Length Normalization**: Adjusts for varying document lengths
  - **IDF**: Penalizes common terms, rewards rare terms
  
- **Formula Components**:
  - `k1`: Controls term frequency saturation (typically 1.2-2.0)
  - `b`: Controls document length normalization (typically 0.75)

**Implementation:**

```python
from rank_bm25 import BM25Okapi
import numpy as np

# Tokenize documents
tokenized_docs = [preprocess(doc) for doc in documents]

# Create BM25 index
bm25 = BM25Okapi(tokenized_docs)

# Search function
def search_bm25(query, bm25):
    tokenized_query = preprocess(query)
    scores = bm25.get_scores(tokenized_query)
    return scores

# Rank results
results = search_bm25("machine learning", bm25)
for i in np.argsort(results)[::-1]:
    print(f"Document {i}: {documents[i]}")
```

**Advantages:**
- State-of-the-art ranking effectiveness
- Handles document length naturally
- Robust across different document collections
- Used by major search engines (Elasticsearch, Solr)

**Improvements over TF-IDF:**
- Non-linear term frequency scaling
- Better document length normalization
- Tunable parameters for optimization

**Use Cases:**
- Production search engines
- Academic paper retrieval
- E-commerce product search
- Question-answering systems

See `04-tokenization.py:33-48` for complete implementation.

---

## Comparison of Retrieval Models

| Model | Ranking | Complexity | Precision | Recall | Best For |
|-------|---------|------------|-----------|--------|----------|
| **Boolean** | No | Low | High | Low | Exact matching, legal search |
| **Vector Space (TF-IDF)** | Yes | Medium | Medium | High | General search, similarity |
| **Probabilistic (BM25)** | Yes | High | High | High | Production systems, relevance |

## Running the Examples

Each script demonstrates a different retrieval approach:

```bash
# Basic tokenization
uv run 01-tokenization.py

# Vector Space Model with TF-IDF
uv run 02-tokenization.py

# Boolean Retrieval Model
uv run 03-tokenization.py

# Probabilistic Retrieval Model (BM25)
uv run 04-tokenization.py
```

## Dependencies

- **nltk**: Natural language processing and tokenization
- **scikit-learn**: TF-IDF vectorization and cosine similarity
- **whoosh**: Full-text indexing and Boolean search
- **rank-bm25**: BM25 probabilistic ranking

## Next Steps

After mastering these retrieval systems, you can explore:

1. **Stemming/Lemmatization**: Reducing words to root forms
2. **Stop Word Removal**: Filtering common words (see `03-tokenization.py:33`)
3. **Query Expansion**: Enriching queries with synonyms
4. **Learning to Rank**: Machine learning-based ranking
5. **Neural Retrieval**: Dense vector embeddings and semantic search

## Project Context

This project is part of a larger curriculum on **fundamentals of retrieval systems**, progressing from basic tokenization to advanced ranking algorithms that power modern search engines.

## License

This is an educational project for learning NLP and Information Retrieval fundamentals.
