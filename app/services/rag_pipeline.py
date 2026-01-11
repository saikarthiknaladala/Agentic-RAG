"""RAG Pipeline orchestrator integrating all components."""
import logging
from typing import List, Dict, Any, Optional
from app.services.llm_service import OllamaService
from app.services.search import SearchService
from app.models.schemas import QueryResponse, Citation


logger = logging.getLogger(__name__)


class RAGPipeline:
    """
    Main RAG pipeline orchestrator.

    Pipeline stages:
    1. Query refusal check (PII, legal, medical)
    2. Intent detection
    3. Query transformation
    4. Hybrid search (semantic + keyword)
    5. Re-ranking
    6. Citation filtering (similarity threshold)
    7. Answer generation
    8. Hallucination detection
    9. Answer shaping by intent
    """

    def __init__(
        self,
        llm_service: OllamaService,
        search_service: SearchService,
        similarity_threshold: float = 0.3
    ):
        """
        Initialize RAG pipeline.

        Args:
            llm_service: LLM service instance
            search_service: Search service instance
            similarity_threshold: Minimum similarity for citations
        """
        self.llm_service = llm_service
        self.search_service = search_service
        self.similarity_threshold = similarity_threshold

    def process_query(
        self,
        query: str,
        top_k: int = 5,
        similarity_threshold: Optional[float] = None
    ) -> QueryResponse:
        """
        Process user query through the RAG pipeline.

        Args:
            query: User query
            top_k: Number of chunks to retrieve
            similarity_threshold: Override default threshold

        Returns:
            QueryResponse with answer and metadata
        """
        threshold = similarity_threshold or self.similarity_threshold

        # Stage 1: Check for refusal (PII, legal, medical)
        refusal_check = self.llm_service.check_pii_and_refusal(query)
        if refusal_check["should_refuse"]:
            return QueryResponse(
                answer=self._get_refusal_message(refusal_check),
                intent=refusal_check["category"],
                citations=[],
                metadata={"refused": True, "reason": refusal_check["reason"]}
            )

        # Stage 2: Detect intent
        intent_data = self.llm_service.detect_intent(query)
        intent_category = intent_data.get("intent_category", "knowledge_query")
        requires_search = intent_data.get("requires_search", True)

        # Handle non-search intents
        if not requires_search:
            return QueryResponse(
                answer=self._handle_non_search_intent(query, intent_category),
                intent=intent_category,
                citations=[],
                metadata={"requires_search": False}
            )

        # Stage 3: Transform query for better retrieval
        transformed_query = self.llm_service.transform_query(query)
        logger.info(f"Transformed query: '{query}' -> '{transformed_query}'")

        # Stage 4 & 5: Hybrid search with re-ranking
        search_results = self.search_service.search(
            transformed_query,
            top_k=top_k,
            threshold=threshold,
            use_reranking=True
        )

        # Stage 6: Citation filtering
        if not search_results or all(r["similarity_score"] < threshold for r in search_results):
            return QueryResponse(
                answer="I don't have sufficient evidence in the knowledge base to answer this question accurately. The available information doesn't meet the confidence threshold required for a reliable response.",
                intent=intent_category,
                citations=[],
                metadata={
                    "insufficient_evidence": True,
                    "threshold": threshold,
                    "max_score": max([r["similarity_score"] for r in search_results]) if search_results else 0
                }
            )

        # Filter chunks below threshold
        valid_chunks = [r for r in search_results if r["similarity_score"] >= threshold]

        # Stage 7: Generate answer
        answer = self._generate_answer(query, valid_chunks, intent_category)

        # Stage 8: Hallucination detection
        chunk_texts = [chunk["text"] for chunk in valid_chunks]
        hallucination_check = self.llm_service.check_hallucination(answer, chunk_texts)

        if not hallucination_check["is_supported"]:
            logger.warning(f"Potential hallucination detected. Unsupported claims: {hallucination_check['unsupported_claims']}")
            # Regenerate with stricter prompt
            answer = self._generate_answer(query, valid_chunks, intent_category, strict_mode=True)

        # Stage 9: Format citations
        citations = self._format_citations(valid_chunks)

        return QueryResponse(
            answer=answer,
            intent=intent_category,
            query_transformed=transformed_query,
            citations=citations,
            metadata={
                "chunks_retrieved": len(search_results),
                "chunks_used": len(valid_chunks),
                "hallucination_check": hallucination_check,
                "intent_confidence": intent_data.get("confidence", 0.0)
            }
        )

    def _get_refusal_message(self, refusal_check: Dict[str, Any]) -> str:
        """Generate appropriate refusal message."""
        category = refusal_check.get("category", "none")

        messages = {
            "pii": "I cannot process queries containing personal identifiable information (PII) for privacy and security reasons.",
            "legal": "I cannot provide legal advice. Please consult with a qualified attorney for legal matters.",
            "medical": "I cannot provide medical advice. Please consult with a qualified healthcare professional for medical concerns.",
            "harmful": "I cannot assist with requests that could be harmful or unethical."
        }

        return messages.get(category, "I cannot process this query. " + refusal_check.get("reason", ""))

    def _handle_non_search_intent(self, query: str, intent: str) -> str:
        """Handle queries that don't require knowledge base search."""
        if intent == "greeting":
            return "Hello! I'm here to help you find information from the knowledge base. What would you like to know?"

        elif intent == "chitchat":
            return "I'm a knowledge base assistant. I'm best at answering questions about the documents I have access to. What would you like to learn about?"

        else:
            return "I'm not sure how to help with that. Could you please ask a specific question about the information in the knowledge base?"

    def _generate_answer(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        intent: str,
        strict_mode: bool = False
    ) -> str:
        """
        Generate answer using LLM with retrieved chunks.

        Args:
            query: User query
            chunks: Retrieved chunks
            intent: Query intent
            strict_mode: Use stricter prompting to reduce hallucination

        Returns:
            Generated answer
        """
        # Build context from chunks
        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            source = chunk["source_file"]
            page = chunk.get("page_number", "N/A")
            text = chunk["text"]
            context_parts.append(f"[Source {i}: {source}, Page {page}]\n{text}")

        context = "\n\n".join(context_parts)

        # Choose template based on intent
        if strict_mode:
            template = self._get_strict_template()
        else:
            template = self._get_template_by_intent(intent)

        # Generate answer
        prompt = template.format(context=context, query=query)

        try:
            answer = self.llm_service.generate_text(
                prompt,
                temperature=0.3 if strict_mode else 0.7,
                max_tokens=1024
            )
            return answer

        except Exception as e:
            logger.error(f"Error generating answer: {str(e)}")
            return "I apologize, but I encountered an error while generating the answer. Please try again."

    def _get_template_by_intent(self, intent: str) -> str:
        """Get prompt template based on intent."""
        templates = {
            "knowledge_query": """You are a helpful assistant that answers questions based on the provided context.

Context from knowledge base:
{context}

User Question: {query}

Instructions:
- Answer the question using ONLY information from the context
- Be accurate and concise
- If the context doesn't fully answer the question, say so
- Cite sources by mentioning the document name when relevant
- Do not add information not present in the context

Answer:""",

            "list": """You are a helpful assistant that answers questions based on the provided context.

Context from knowledge base:
{context}

User Question: {query}

Instructions:
- Answer using ONLY information from the context
- Format your answer as a clear, organized list
- Use bullet points or numbered lists as appropriate
- Be concise and accurate

Answer:""",

            "comparison": """You are a helpful assistant that answers questions based on the provided context.

Context from knowledge base:
{context}

User Question: {query}

Instructions:
- Answer using ONLY information from the context
- Compare and contrast the relevant items
- Use a structured format (table or clear sections)
- Be objective and accurate

Answer:"""
        }

        return templates.get(intent, templates["knowledge_query"])

    def _get_strict_template(self) -> str:
        """Get strict template for hallucination prevention."""
        return """You are a careful assistant that answers questions based strictly on the provided context.

Context from knowledge base:
{context}

User Question: {query}

CRITICAL INSTRUCTIONS:
- Use ONLY information explicitly stated in the context above
- Do NOT add any external knowledge or assumptions
- If information is not in the context, clearly state that
- Quote directly from the context when possible
- Do not elaborate beyond what is written

Answer:"""

    def _format_citations(self, chunks: List[Dict[str, Any]]) -> List[Citation]:
        """Format chunks as citations."""
        citations = []

        for chunk in chunks:
            # Create text preview (first 200 chars)
            text_preview = chunk["text"][:200]
            if len(chunk["text"]) > 200:
                text_preview += "..."

            citation = Citation(
                chunk_id=chunk["chunk_id"],
                source_file=chunk["source_file"],
                page_number=chunk.get("page_number"),
                similarity_score=chunk["similarity_score"],
                text_preview=text_preview
            )
            citations.append(citation)

        return citations
