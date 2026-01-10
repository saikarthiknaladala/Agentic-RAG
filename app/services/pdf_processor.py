"""PDF processing service for text extraction and chunking."""
from typing import List, Tuple
import PyPDF2
from io import BytesIO
import logging

from app.utils.chunking import TextChunker


logger = logging.getLogger(__name__)


class PDFProcessor:
    """
    PDF text extraction and chunking processor.

    Handles:
    - Multi-page PDF extraction
    - Text cleaning and normalization
    - Intelligent chunking with overlap
    - Metadata preservation for citations
    """

    def __init__(self, chunk_size: int = 512, overlap: int = 128):
        """
        Initialize PDF processor.

        Args:
            chunk_size: Target tokens per chunk
            overlap: Overlapping tokens between chunks
        """
        self.chunker = TextChunker(chunk_size=chunk_size, overlap=overlap)

    def extract_text_from_pdf(self, pdf_content: bytes, filename: str) -> List[Tuple[str, int]]:
        """
        Extract text from PDF with page tracking.

        Args:
            pdf_content: PDF file content as bytes
            filename: Name of the PDF file

        Returns:
            List of (page_text, page_number) tuples
        """
        pages = []

        try:
            pdf_file = BytesIO(pdf_content)
            pdf_reader = PyPDF2.PdfReader(pdf_file)

            for page_num, page in enumerate(pdf_reader.pages, start=1):
                text = page.extract_text()

                if text and text.strip():
                    # Clean the text
                    text = self.clean_text(text)
                    pages.append((text, page_num))

            logger.info(f"Extracted {len(pages)} pages from {filename}")

        except Exception as e:
            logger.error(f"Error extracting text from {filename}: {str(e)}")
            raise

        return pages

    def clean_text(self, text: str) -> str:
        """
        Clean and normalize extracted text.

        Args:
            text: Raw extracted text

        Returns:
            Cleaned text
        """
        # Remove excessive whitespace
        text = ' '.join(text.split())

        # Remove common PDF artifacts
        text = text.replace('\x00', '')
        text = text.replace('\ufffd', '')

        # Normalize unicode quotation marks
        text = text.replace('\u201c', '"').replace('\u201d', '"')
        text = text.replace('\u2018', "'").replace('\u2019', "'")

        return text.strip()

    def process_pdf(
        self,
        pdf_content: bytes,
        filename: str
    ) -> List[Tuple[str, dict]]:
        """
        Process PDF into chunks with metadata.

        Args:
            pdf_content: PDF file bytes
            filename: Source filename

        Returns:
            List of (chunk_text, metadata) tuples
        """
        # Extract text page by page
        pages = self.extract_text_from_pdf(pdf_content, filename)

        if not pages:
            logger.warning(f"No text extracted from {filename}")
            return []

        # Chunk the document
        chunks = self.chunker.chunk_document(pages, filename)

        logger.info(f"Created {len(chunks)} chunks from {filename}")

        return chunks

    def process_multiple_pdfs(
        self,
        pdf_files: List[Tuple[bytes, str]]
    ) -> List[Tuple[str, dict]]:
        """
        Process multiple PDFs.

        Args:
            pdf_files: List of (pdf_content, filename) tuples

        Returns:
            All chunks from all PDFs
        """
        all_chunks = []

        for pdf_content, filename in pdf_files:
            try:
                chunks = self.process_pdf(pdf_content, filename)
                all_chunks.extend(chunks)
            except Exception as e:
                logger.error(f"Failed to process {filename}: {str(e)}")
                continue

        return all_chunks
