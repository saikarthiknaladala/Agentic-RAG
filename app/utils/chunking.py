"""Text chunking utilities with overlap for better context preservation."""
from typing import List, Tuple
import re


class TextChunker:
    """
    Intelligent text chunking with overlap.

    Considerations:
    1. Chunk size: 512 tokens (~400 words) balances context vs. precision
    2. Overlap: 128 tokens (~100 words) ensures context continuity across chunks
    3. Sentence boundaries: Prefer splitting at sentence ends to maintain coherence
    4. Metadata preservation: Track source file and page numbers for citations
    """

    def __init__(self, chunk_size: int = 512, overlap: int = 128):
        """
        Initialize chunker.

        Args:
            chunk_size: Target number of tokens per chunk
            overlap: Number of tokens to overlap between chunks
        """
        self.chunk_size = chunk_size
        self.overlap = overlap

    def estimate_tokens(self, text: str) -> int:
        """Rough token estimation: ~1.3 tokens per word for English."""
        words = text.split()
        return int(len(words) * 1.3)

    def split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences using regex."""
        # Handle common abbreviations
        text = re.sub(r'\b(Dr|Mr|Mrs|Ms|Prof|Sr|Jr)\.\s', r'\1<PERIOD> ', text)

        # Split on sentence boundaries
        sentences = re.split(r'(?<=[.!?])\s+', text)

        # Restore abbreviations
        sentences = [s.replace('<PERIOD>', '.') for s in sentences]

        return [s.strip() for s in sentences if s.strip()]

    def chunk_text(
        self,
        text: str,
        source_file: str,
        page_number: int = None
    ) -> List[Tuple[str, dict]]:
        """
        Chunk text with overlap at sentence boundaries.

        Args:
            text: Text to chunk
            source_file: Source filename for metadata
            page_number: Page number if available

        Returns:
            List of (chunk_text, metadata) tuples
        """
        sentences = self.split_into_sentences(text)
        chunks = []
        current_chunk = []
        current_tokens = 0

        for sentence in sentences:
            sentence_tokens = self.estimate_tokens(sentence)

            # If adding this sentence exceeds chunk_size, save current chunk
            if current_tokens + sentence_tokens > self.chunk_size and current_chunk:
                chunk_text = ' '.join(current_chunk)
                metadata = {
                    'source_file': source_file,
                    'page_number': page_number,
                    'chunk_index': len(chunks),
                    'token_count': current_tokens
                }
                chunks.append((chunk_text, metadata))

                # Start new chunk with overlap
                overlap_tokens = 0
                overlap_sentences = []

                # Take sentences from the end for overlap
                for s in reversed(current_chunk):
                    s_tokens = self.estimate_tokens(s)
                    if overlap_tokens + s_tokens <= self.overlap:
                        overlap_sentences.insert(0, s)
                        overlap_tokens += s_tokens
                    else:
                        break

                current_chunk = overlap_sentences
                current_tokens = overlap_tokens

            current_chunk.append(sentence)
            current_tokens += sentence_tokens

        # Add the last chunk
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            metadata = {
                'source_file': source_file,
                'page_number': page_number,
                'chunk_index': len(chunks),
                'token_count': current_tokens
            }
            chunks.append((chunk_text, metadata))

        return chunks

    def chunk_document(
        self,
        pages: List[Tuple[str, int]],
        source_file: str
    ) -> List[Tuple[str, dict]]:
        """
        Chunk an entire document page by page.

        Args:
            pages: List of (page_text, page_number) tuples
            source_file: Source filename

        Returns:
            List of (chunk_text, metadata) tuples
        """
        all_chunks = []

        for page_text, page_num in pages:
            page_chunks = self.chunk_text(page_text, source_file, page_num)
            all_chunks.extend(page_chunks)

        return all_chunks
