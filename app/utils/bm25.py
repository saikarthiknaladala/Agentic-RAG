"""BM25 implementation for keyword-based search."""
import math
from typing import List, Dict, Tuple
from collections import Counter
import re


class BM25:
    """
    BM25 (Best Matching 25) ranking function for keyword search.

    BM25 is a probabilistic ranking function that scores documents based on
    term frequency (TF) and inverse document frequency (IDF).

    Parameters:
        k1: Controls term frequency saturation (default: 1.5)
        b: Controls document length normalization (default: 0.75)
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus: List[List[str]] = []
        self.doc_lengths: List[int] = []
        self.avg_doc_length: float = 0.0
        self.doc_count: int = 0
        self.idf_cache: Dict[str, float] = {}

    def tokenize(self, text: str) -> List[str]:
        """Tokenize text into lowercase words."""
        # Remove special characters and convert to lowercase
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        tokens = text.split()
        return [t for t in tokens if len(t) > 2]  # Filter very short tokens

    def fit(self, corpus: List[str]):
        """
        Index the corpus for BM25 scoring.

        Args:
            corpus: List of document texts
        """
        self.corpus = [self.tokenize(doc) for doc in corpus]
        self.doc_lengths = [len(doc) for doc in self.corpus]
        self.doc_count = len(corpus)
        self.avg_doc_length = sum(self.doc_lengths) / self.doc_count if self.doc_count > 0 else 0

        # Pre-compute IDF values
        self._compute_idf()

    def _compute_idf(self):
        """Compute IDF (Inverse Document Frequency) for all terms."""
        # Count documents containing each term
        df = Counter()
        for doc in self.corpus:
            unique_terms = set(doc)
            for term in unique_terms:
                df[term] += 1

        # Calculate IDF: log((N - df + 0.5) / (df + 0.5) + 1)
        for term, doc_freq in df.items():
            idf = math.log((self.doc_count - doc_freq + 0.5) / (doc_freq + 0.5) + 1.0)
            self.idf_cache[term] = idf

    def get_idf(self, term: str) -> float:
        """Get IDF score for a term."""
        return self.idf_cache.get(term, 0.0)

    def score_document(self, query_terms: List[str], doc_index: int) -> float:
        """
        Calculate BM25 score for a document given query terms.

        Args:
            query_terms: Tokenized query
            doc_index: Index of document in corpus

        Returns:
            BM25 score
        """
        if doc_index >= len(self.corpus):
            return 0.0

        doc = self.corpus[doc_index]
        doc_length = self.doc_lengths[doc_index]
        score = 0.0

        # Count term frequencies in the document
        term_freqs = Counter(doc)

        # Calculate normalized document length
        norm_doc_length = doc_length / self.avg_doc_length if self.avg_doc_length > 0 else 0

        for term in query_terms:
            if term not in term_freqs:
                continue

            tf = term_freqs[term]
            idf = self.get_idf(term)

            # BM25 formula
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * norm_doc_length)

            score += idf * (numerator / denominator)

        return score

    def search(self, query: str, top_k: int = 10) -> List[Tuple[int, float]]:
        """
        Search the corpus using BM25.

        Args:
            query: Query text
            top_k: Number of top results to return

        Returns:
            List of (doc_index, score) tuples sorted by score
        """
        query_terms = self.tokenize(query)

        # Score all documents
        scores = []
        for i in range(self.doc_count):
            score = self.score_document(query_terms, i)
            if score > 0:
                scores.append((i, score))

        # Sort by score (descending) and return top_k
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    def add_documents(self, new_docs: List[str]):
        """
        Add new documents to the index.

        Args:
            new_docs: List of new document texts
        """
        if not self.corpus:
            self.fit(new_docs)
        else:
            # Extend the corpus
            new_tokenized = [self.tokenize(doc) for doc in new_docs]
            self.corpus.extend(new_tokenized)
            self.doc_lengths.extend([len(doc) for doc in new_tokenized])
            self.doc_count = len(self.corpus)
            self.avg_doc_length = sum(self.doc_lengths) / self.doc_count

            # Recompute IDF values
            self._compute_idf()
