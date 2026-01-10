"""Custom vector store implementation without external databases."""
import numpy as np
import json
import os
from typing import List, Tuple, Dict, Any
import logging
from pathlib import Path


logger = logging.getLogger(__name__)


class VectorStore:
    """
    Custom in-memory vector store with persistence.

    Features:
    - Cosine similarity search
    - Efficient numpy-based operations
    - Persistent storage to disk
    - No external vector database dependencies
    """

    def __init__(self, persist_dir: str = "vector_store"):
        """
        Initialize vector store.

        Args:
            persist_dir: Directory for persistent storage
        """
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(exist_ok=True)

        self.vectors: np.ndarray = np.array([])  # Shape: (n_docs, embedding_dim)
        self.documents: List[str] = []  # Document texts
        self.metadata: List[Dict[str, Any]] = []  # Document metadata
        self.embedding_dim: int = 0

        # Try to load existing store
        self.load()

    def add_documents(
        self,
        texts: List[str],
        embeddings: List[List[float]],
        metadata: List[Dict[str, Any]]
    ):
        """
        Add documents with embeddings to the store.

        Args:
            texts: Document texts
            embeddings: Document embeddings
            metadata: Document metadata
        """
        if not texts:
            return

        # Convert embeddings to numpy array
        new_vectors = np.array(embeddings, dtype=np.float32)

        # Initialize or validate embedding dimension
        if self.embedding_dim == 0:
            self.embedding_dim = new_vectors.shape[1]
        elif new_vectors.shape[1] != self.embedding_dim:
            raise ValueError(
                f"Embedding dimension mismatch: expected {self.embedding_dim}, "
                f"got {new_vectors.shape[1]}"
            )

        # Normalize vectors for cosine similarity
        new_vectors = self._normalize_vectors(new_vectors)

        # Append to existing store
        if self.vectors.size == 0:
            self.vectors = new_vectors
        else:
            self.vectors = np.vstack([self.vectors, new_vectors])

        self.documents.extend(texts)
        self.metadata.extend(metadata)

        logger.info(f"Added {len(texts)} documents to vector store. Total: {len(self.documents)}")

    def _normalize_vectors(self, vectors: np.ndarray) -> np.ndarray:
        """
        Normalize vectors to unit length for cosine similarity.

        Args:
            vectors: Input vectors

        Returns:
            Normalized vectors
        """
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        # Avoid division by zero
        norms = np.where(norms == 0, 1, norms)
        return vectors / norms

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        threshold: float = 0.0
    ) -> List[Tuple[int, float, str, Dict[str, Any]]]:
        """
        Search for similar documents using cosine similarity.

        Args:
            query_embedding: Query embedding vector
            top_k: Number of top results to return
            threshold: Minimum similarity threshold

        Returns:
            List of (index, similarity, text, metadata) tuples
        """
        if self.vectors.size == 0:
            return []

        # Convert query to numpy and normalize
        query_vector = np.array(query_embedding, dtype=np.float32).reshape(1, -1)
        query_vector = self._normalize_vectors(query_vector)

        # Compute cosine similarity (dot product of normalized vectors)
        similarities = np.dot(self.vectors, query_vector.T).flatten()

        # Get top_k indices sorted by similarity
        top_indices = np.argsort(similarities)[::-1][:top_k]

        # Filter by threshold and prepare results
        results = []
        for idx in top_indices:
            similarity = float(similarities[idx])
            if similarity >= threshold:
                results.append((
                    int(idx),
                    similarity,
                    self.documents[idx],
                    self.metadata[idx]
                ))

        return results

    def get_document(self, index: int) -> Tuple[str, Dict[str, Any]]:
        """
        Get document by index.

        Args:
            index: Document index

        Returns:
            (text, metadata) tuple
        """
        if 0 <= index < len(self.documents):
            return self.documents[index], self.metadata[index]
        raise IndexError(f"Document index {index} out of range")

    def save(self):
        """Save vector store to disk."""
        try:
            # Save vectors
            np.save(self.persist_dir / "vectors.npy", self.vectors)

            # Save documents and metadata
            data = {
                "documents": self.documents,
                "metadata": self.metadata,
                "embedding_dim": self.embedding_dim
            }

            with open(self.persist_dir / "data.json", "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

            logger.info(f"Saved vector store with {len(self.documents)} documents")

        except Exception as e:
            logger.error(f"Error saving vector store: {str(e)}")

    def load(self):
        """Load vector store from disk."""
        try:
            vectors_path = self.persist_dir / "vectors.npy"
            data_path = self.persist_dir / "data.json"

            if vectors_path.exists() and data_path.exists():
                # Load vectors
                self.vectors = np.load(vectors_path)

                # Load documents and metadata
                with open(data_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                self.documents = data["documents"]
                self.metadata = data["metadata"]
                self.embedding_dim = data["embedding_dim"]

                logger.info(f"Loaded vector store with {len(self.documents)} documents")

        except Exception as e:
            logger.error(f"Error loading vector store: {str(e)}")

    def clear(self):
        """Clear all data from the store."""
        self.vectors = np.array([])
        self.documents = []
        self.metadata = []
        self.embedding_dim = 0

        # Remove persisted files
        for file in self.persist_dir.glob("*"):
            file.unlink()

        logger.info("Cleared vector store")

    def size(self) -> int:
        """Get number of documents in store."""
        return len(self.documents)

    def get_all_texts(self) -> List[str]:
        """Get all document texts."""
        return self.documents.copy()
