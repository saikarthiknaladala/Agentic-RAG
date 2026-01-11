"""LLM service for Groq + Ollama embeddings."""
import json
import logging
import os
from typing import List, Dict, Any, Optional

import requests


logger = logging.getLogger(__name__)


class OllamaService:
    """
    Service for interacting with Ollama (embeddings) and Groq (LLM).

    Models:
    - llama-3.1-8b-instant: For text generation via Groq
    - nomic-embed-text: For embeddings via Ollama
    """

    def __init__(
        self,
        ollama_base_url: str = "http://localhost:11434",
        llm_model: str = "llama-3.1-8b-instant",
        embedding_model: str = "nomic-embed-text:v1.5",
        groq_base_url: str = "https://api.groq.com/openai/v1",
        groq_api_key: Optional[str] = None
    ):
        """
        Initialize LLM service.

        Args:
            ollama_base_url: Ollama API base URL (embeddings)
            llm_model: Groq LLM model name
            embedding_model: Ollama embedding model name
            groq_base_url: Groq API base URL
            groq_api_key: Groq API key (optional, defaults to env var)
        """
        self.ollama_base_url = ollama_base_url
        self.llm_model = llm_model
        self.embedding_model = embedding_model
        self.groq_base_url = groq_base_url
        self.groq_api_key = groq_api_key or os.getenv("GROQ_API_KEY")

        self.embeddings_url = f"{ollama_base_url}/api/embeddings"
        self.embed_url = f"{ollama_base_url}/api/embed"
        self.groq_models_url = f"{groq_base_url}/models"
        self.groq_chat_url = f"{groq_base_url}/chat/completions"

    def _groq_headers(self) -> Dict[str, str]:
        if not self.groq_api_key:
            raise ValueError("GROQ_API_KEY is not set")
        return {
            "Authorization": f"Bearer {self.groq_api_key}",
            "Content-Type": "application/json"
        }

    def check_ollama(self) -> bool:
        """Check if Ollama is running and accessible."""
        try:
            response = requests.get(f"{self.ollama_base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Ollama health check failed: {str(e)}")
            return False

    def check_groq_model(self) -> bool:
        """Check if Groq is reachable and the configured model exists."""
        if not self.groq_api_key:
            logger.warning("GROQ_API_KEY is not set")
            return False

        try:
            response = requests.get(
                self.groq_models_url,
                headers=self._groq_headers(),
                timeout=10
            )
            response.raise_for_status()

            data = response.json().get("data", [])
            model_ids = [model.get("id") for model in data if model.get("id")]

            if self.llm_model in model_ids:
                return True

            logger.warning(
                "Groq model '%s' not found. Available: %s",
                self.llm_model,
                ", ".join(model_ids)
            )
            return False
        except Exception as e:
            logger.error(f"Groq model check failed: {str(e)}")
            return False

    def health_check(self) -> bool:
        """Check if Groq and Ollama are accessible."""
        return self.check_ollama() and self.check_groq_model()

    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for text using nomic-embed-text.

        Args:
            text: Input text

        Returns:
            Embedding vector
        """
        try:
            payload = {
                "model": self.embedding_model,
                "prompt": text
            }

            response = requests.post(
                self.embeddings_url,
                json=payload,
                timeout=1000
            )

            if response.status_code == 404:
                # Newer Ollama builds expose /api/embed instead of /api/embeddings.
                payload = {
                    "model": self.embedding_model,
                    "input": text
                }
                response = requests.post(
                    self.embed_url,
                    json=payload,
                    timeout=1000
                )

            response.raise_for_status()

            result = response.json()
            if "embedding" in result:
                return result["embedding"]
            if "embeddings" in result and result["embeddings"]:
                return result["embeddings"][0]
            raise ValueError("Ollama embedding response missing embedding data")

        except Exception as e:
            logger.error(f"Error generating embedding: {str(e)}")
            raise

    def generate_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts.

        Args:
            texts: List of input texts

        Returns:
            List of embedding vectors
        """
        embeddings = []
        for text in texts:
            embedding = self.generate_embedding(text)
            embeddings.append(embedding)
        return embeddings

    def generate_text(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048
    ) -> str:
        """
        Generate text using Groq.

        Args:
            prompt: User prompt
            system_prompt: System prompt (optional)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate

        Returns:
            Generated text
        """
        try:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            payload = {
                "model": self.llm_model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stream": False
            }

            response = requests.post(
                self.groq_chat_url,
                headers=self._groq_headers(),
                json=payload,
                timeout=60
            )
            response.raise_for_status()

            result = response.json()
            return result["choices"][0]["message"]["content"].strip()

        except Exception as e:
            logger.error(f"Error generating text: {str(e)}")
            raise

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 2048
    ) -> str:
        """
        Chat completion using Groq.

        Args:
            messages: List of message dicts with 'role' and 'content'
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate

        Returns:
            Generated response
        """
        try:
            payload = {
                "model": self.llm_model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stream": False
            }

            response = requests.post(
                self.groq_chat_url,
                headers=self._groq_headers(),
                json=payload,
                timeout=60
            )
            response.raise_for_status()

            result = response.json()
            return result["choices"][0]["message"]["content"].strip()

        except Exception as e:
            logger.error(f"Error in chat completion: {str(e)}")
            raise

    def detect_intent(self, query: str) -> Dict[str, Any]:
        """
        Detect query intent using LLM.

        Args:
            query: User query

        Returns:
            Intent classification result
        """
        prompt = f"""Analyze the following user query and classify its intent.

Query: "{query}"

Determine:
1. Is this a knowledge-seeking query that requires searching a knowledge base? (yes/no)
2. What is the primary intent category?
   - knowledge_query: Asking for information from documents
   - greeting: Simple greeting or pleasantry
   - chitchat: Casual conversation
   - unclear: Unclear or ambiguous intent

Respond ONLY with a JSON object in this exact format:
{{"requires_search": true/false, "intent_category": "category_name", "confidence": 0.0-1.0}}"""

        try:
            response = self.generate_text(
                prompt,
                temperature=0.3,
                max_tokens=100
            )

            # Extract JSON from response
            response = response.strip()
            if response.startswith("```json"):
                response = response.split("```json")[1].split("```")[0].strip()
            elif response.startswith("```"):
                response = response.split("```")[1].split("```")[0].strip()

            intent_data = json.loads(response)
            return intent_data

        except Exception as e:
            logger.error(f"Error detecting intent: {str(e)}")
            # Default to knowledge query on error
            return {
                "requires_search": True,
                "intent_category": "knowledge_query",
                "confidence": 0.5
            }

    def transform_query(self, query: str) -> str:
        """
        Transform query for better retrieval.

        Args:
            query: Original query

        Returns:
            Transformed query
        """
        prompt = f"""Transform the following query to improve information retrieval from a knowledge base.

Original query: "{query}"

Instructions:
- Expand abbreviations
- Add relevant synonyms or related terms
- Rephrase for clarity
- Keep it concise (1-2 sentences max)

Respond with ONLY the transformed query, nothing else."""

        try:
            transformed = self.generate_text(
                prompt,
                temperature=0.5,
                max_tokens=150
            )
            return transformed.strip().strip('"')

        except Exception as e:
            logger.error(f"Error transforming query: {str(e)}")
            return query

    def check_pii_and_refusal(self, query: str) -> Dict[str, Any]:
        """
        Check for PII and determine if query should be refused.

        Args:
            query: User query

        Returns:
            Refusal check result
        """
        prompt = f"""Analyze the following query for sensitive content that should be refused.

Query: "{query}"

Check for:
1. Personal Identifiable Information (PII) like SSN, credit card numbers, etc.
2. Requests for legal advice
3. Requests for medical advice
4. Harmful or unethical requests

Respond ONLY with a JSON object:
{{"should_refuse": true/false, "reason": "explanation if refused", "category": "pii/legal/medical/harmful/none"}}"""

        try:
            response = self.generate_text(
                prompt,
                temperature=0.2,
                max_tokens=150
            )

            # Extract JSON
            response = response.strip()
            if response.startswith("```json"):
                response = response.split("```json")[1].split("```")[0].strip()
            elif response.startswith("```"):
                response = response.split("```")[1].split("```")[0].strip()

            refusal_data = json.loads(response)
            return refusal_data

        except Exception as e:
            logger.error(f"Error checking refusal: {str(e)}")
            return {
                "should_refuse": False,
                "reason": "",
                "category": "none"
            }

    def check_hallucination(self, answer: str, evidence_chunks: List[str]) -> Dict[str, Any]:
        """
        Check if answer contains hallucinated information.

        Args:
            answer: Generated answer
            evidence_chunks: Retrieved evidence chunks

        Returns:
            Hallucination check result
        """
        evidence_text = "\n\n".join([f"Evidence {i+1}: {chunk}" for i, chunk in enumerate(evidence_chunks)])

        prompt = f"""You are a fact-checker. Analyze if the answer is fully supported by the provided evidence.

EVIDENCE:
{evidence_text}

ANSWER TO CHECK:
{answer}

Instructions:
- Check each claim in the answer against the evidence
- Identify any statements not supported by the evidence
- Be strict: even minor unsupported details count as hallucinations

Respond ONLY with a JSON object:
{{"is_supported": true/false, "unsupported_claims": ["claim1", "claim2"], "confidence": 0.0-1.0}}"""

        try:
            response = self.generate_text(
                prompt,
                temperature=0.2,
                max_tokens=300
            )

            # Extract JSON
            response = response.strip()
            if response.startswith("```json"):
                response = response.split("```json")[1].split("```")[0].strip()
            elif response.startswith("```"):
                response = response.split("```")[1].split("```")[0].strip()

            hallucination_data = json.loads(response)
            return hallucination_data

        except Exception as e:
            logger.error(f"Error checking hallucination: {str(e)}")
            return {
                "is_supported": True,
                "unsupported_claims": [],
                "confidence": 0.5
            }
