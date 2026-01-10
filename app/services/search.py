"""Search service combining semantic and keyword search with re-ranking."""
import logging
from typing import List, Dict, Any, Tuple
import numpy as np

from app.services.vector_store import VectorStore
from app.services.llm_service import OllamaService
from app.utils.bm25 import BM25


logger = logging.getLogger(__name__)


class SearchService:
    """
    Hybrid search service combining semantic and keyword search.

    Features:
    - Semantic search using vector embeddings (cosine similarity)
    - Keyword search using BM25
    - Hybrid search with configurable weights
    - Re-ranking for improved retrieval
    """

    def __init__(
        self,
        vector_store: VectorStore,
        llm_service: OllamaService,
        semantic_weight: float = 0.7,
        keyword_weight: float = 0.3
    ):
        """
        Initialize search service.

        Args:
            vector_store: Vector store instance
            llm_service: LLM service instance
            semantic_weight: Weight for semantic search (0-1)
            keyword_weight: Weight for keyword search (0-1)
        """
        self.vector_store = vector_store
        self.llm_service = llm_service
        self.semantic_weight = semantic_weight
        self.keyword_weight = keyword_weight
        self.bm25 = BM25()

        # Initialize BM25 with existing documents
        self._rebuild_bm25_index()

    def _rebuild_bm25_index(self):
        """Rebuild BM25 index with all documents from vector store."""
        if self.vector_store.size() > 0:
            documents = self.vector_store.get_all_texts()
            self.bm25.fit(documents)
            logger.info(f"Built BM25 index with {len(documents)} documents")

    def add_documents(
        self,
        texts: List[str],
        embeddings: List[List[float]],
        metadata: List[Dict[str, Any]]
    ):
        """
        Add documents to both vector store and BM25 index.

        Args:
            texts: Document texts
            embeddings: Document embeddings
            metadata: Document metadata
        """
        # Add to vector store
        self.vector_store.add_documents(texts, embeddings, metadata)

        # Update BM25 index
        self.bm25.add_documents(texts)

    def semantic_search(
        self,
        query: str,
        top_k: int = 10,
        threshold: float = 0.0
    ) -> List[Tuple[int, float]]:
        """
        Perform semantic search using embeddings.

        Args:
            query: Query text
            top_k: Number of results
            threshold: Minimum similarity threshold

        Returns:
            List of (doc_index, similarity_score) tuples
        """
        # Generate query embedding
        query_embedding = self.llm_service.generate_embedding(query)

        # Search vector store
        results = self.vector_store.search(query_embedding, top_k, threshold)

        # Return (index, score) tuples
        return [(idx, score) for idx, score, _, _ in results]

    def keyword_search(
        self,
        query: str,
        top_k: int = 10
    ) -> List[Tuple[int, float]]:
        """
        Perform keyword search using BM25.

        Args:
            query: Query text
            top_k: Number of results

        Returns:
            List of (doc_index, bm25_score) tuples
        """
        return self.bm25.search(query, top_k)

    def hybrid_search(
        self,
        query: str,
        top_k: int = 10,
        threshold: float = 0.0
    ) -> List[Tuple[int, float, Dict[str, float]]]:
        """
        Perform hybrid search combining semantic and keyword search.

        Args:
            query: Query text
            top_k: Number of results
            threshold: Minimum similarity threshold for semantic search

        Returns:
            List of (doc_index, combined_score, score_breakdown) tuples
        """
        # Get semantic results
        semantic_results = self.semantic_search(query, top_k * 2, threshold)

        # Get keyword results
        keyword_results = self.keyword_search(query, top_k * 2)

        # Normalize scores
        semantic_scores = self._normalize_scores(semantic_results)
        keyword_scores = self._normalize_scores(keyword_results)

        # Combine scores
        combined_scores = {}
        score_breakdowns = {}

        # Add semantic scores
        for idx, norm_score in semantic_scores.items():
            combined_scores[idx] = self.semantic_weight * norm_score
            score_breakdowns[idx] = {
                "semantic": norm_score,
                "keyword": 0.0
            }

        # Add keyword scores
        for idx, norm_score in keyword_scores.items():
            if idx in combined_scores:
                combined_scores[idx] += self.keyword_weight * norm_score
                score_breakdowns[idx]["keyword"] = norm_score
            else:
                combined_scores[idx] = self.keyword_weight * norm_score
                score_breakdowns[idx] = {
                    "semantic": 0.0,
                    "keyword": norm_score
                }

        # Sort by combined score
        sorted_results = sorted(
            combined_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_k]

        # Return with breakdowns
        return [
            (idx, score, score_breakdowns[idx])
            for idx, score in sorted_results
        ]

    def _normalize_scores(
        self,
        results: List[Tuple[int, float]]
    ) -> Dict[int, float]:
        """
        Normalize scores to 0-1 range using min-max normalization.

        Args:
            results: List of (doc_index, score) tuples

        Returns:
            Dict mapping doc_index to normalized score
        """
        if not results:
            return {}

        scores = [score for _, score in results]
        min_score = min(scores)
        max_score = max(scores)

        # Avoid division by zero
        if max_score == min_score:
            return {idx: 1.0 for idx, _ in results}

        normalized = {}
        for idx, score in results:
            norm_score = (score - min_score) / (max_score - min_score)
            normalized[idx] = norm_score

        return normalized

    def rerank_results(
        self,
        query: str,
        results: List[Tuple[int, float, Dict[str, float]]]
    ) -> List[Tuple[int, float, Dict[str, Any]]]:
        """
        Re-rank results using LLM-based relevance scoring.

        Args:
            query: Original query
            results: List of (doc_index, score, score_breakdown) tuples

        Returns:
            Re-ranked list of (doc_index, final_score, metadata) tuples
        """
        if not results:
            return []

        reranked = []

        for idx, hybrid_score, breakdown in results:
            doc_text, metadata = self.vector_store.get_document(idx)

            # Use LLM to score relevance
            relevance_score = self._score_relevance_with_llm(query, doc_text)

            # Combine hybrid score with LLM relevance
            final_score = 0.6 * hybrid_score + 0.4 * relevance_score

            metadata_with_scores = {
                **metadata,
                "hybrid_score": hybrid_score,
                "semantic_score": breakdown["semantic"],
                "keyword_score": breakdown["keyword"],
                "relevance_score": relevance_score,
                "final_score": final_score
            }

            reranked.append((idx, final_score, metadata_with_scores))

        # Sort by final score
        reranked.sort(key=lambda x: x[1], reverse=True)

        return reranked

    def _score_relevance_with_llm(
        self,
        query: str,
        document: str,
        max_length: int = 500
    ) -> float:
        """
        Score document relevance using LLM.

        Args:
            query: Query text
            document: Document text
            max_length: Max document length to send to LLM

        Returns:
            Relevance score (0-1)
        """
        # Truncate document if too long
        if len(document) > max_length:
            document = document[:max_length] + "..."

        prompt = f"""Rate the relevance of this document to the query on a scale of 0.0 to 1.0.

Query: "{query}"

Document: "{document}"

Respond with ONLY a number between 0.0 and 1.0, nothing else."""

        try:
            response = self.llm_service.generate_text(
                prompt,
                temperature=0.2,
                max_tokens=10
            )

            # Extract score
            score_str = response.strip()
            score = float(score_str)
            return max(0.0, min(1.0, score))  # Clamp to 0-1

        except Exception as e:
            logger.error(f"Error scoring relevance: {str(e)}")
            return 0.5  # Default to middle score

    def search(
        self,
        query: str,
        top_k: int = 5,
        threshold: float = 0.3,
        use_reranking: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Main search method combining all techniques.

        Args:
            query: Query text
            top_k: Number of results
            threshold: Minimum similarity threshold
            use_reranking: Whether to apply re-ranking

        Returns:
            List of search result dicts
        """
        # Hybrid search
        hybrid_results = self.hybrid_search(query, top_k * 2, threshold)

        if not hybrid_results:
            return []

        # Re-rank if enabled
        if use_reranking:
            final_results = self.rerank_results(query, hybrid_results)[:top_k]
        else:
            final_results = [
                (idx, score, {"hybrid_score": score, **breakdown})
                for idx, score, breakdown in hybrid_results
            ][:top_k]

        # Format results
        formatted_results = []
        for idx, final_score, metadata in final_results:
            doc_text, doc_metadata = self.vector_store.get_document(idx)

            result = {
                "chunk_id": str(idx),
                "text": doc_text,
                "source_file": doc_metadata.get("source_file", "unknown"),
                "page_number": doc_metadata.get("page_number"),
                "similarity_score": final_score,
                "scores": metadata
            }
            formatted_results.append(result)

        return formatted_results
