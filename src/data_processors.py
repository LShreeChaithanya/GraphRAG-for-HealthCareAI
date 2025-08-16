import os
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
import json

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
from langchain.schema import Document

from dotenv import load_dotenv
import PyPDF2
import pypdf
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class DataProcessor:
    """Enhanced class for processing text and PDF files with chunking and raw text modes."""
    
    def __init__(self, data_dir: str = "data", chunk_size: int = 200, chunk_overlap: int = 30):
        """
        Initialize the DataProcessor.
        
        Args:
            data_dir: Directory containing data files
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
        """
        self.data_dir = Path(data_dir)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        # Supported file extensions
        self.supported_extensions = {'.txt', '.pdf'}
    
    def load_text_file(self, file_path: Path) -> str:
        """Load text content from a .txt file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
            logger.info(f"Successfully loaded text file: {file_path}")
            return content
        except Exception as e:
            logger.error(f"Error loading text file {file_path}: {e}")
            return ""
    
    def load_pdf_file(self, file_path: Path) -> str:
        """Load text content from a PDF file using multiple methods."""
        try:
            # Try pypdf first (more modern)
            try:
                with open(file_path, 'rb') as file:
                    pdf_reader = pypdf.PdfReader(file)
                    text = ""
                    for page in pdf_reader.pages:
                        text += page.extract_text() + "\n"
                logger.info(f"Successfully loaded PDF with pypdf: {file_path}")
                return text
            except Exception as e:
                logger.warning(f"pypdf failed, trying PyPDF2: {e}")
                
                # Fallback to PyPDF2
                with open(file_path, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    text = ""
                    for page in pdf_reader.pages:
                        text += page.extract_text() + "\n"
                logger.info(f"Successfully loaded PDF with PyPDF2: {file_path}")
                return text
                
        except Exception as e:
            logger.error(f"Error loading PDF file {file_path}: {e}")
            return ""
    
    def load_file(self, file_path: Path) -> Optional[str]:
        """Load content from a file based on its extension."""
        if not file_path.exists():
            logger.error(f"File does not exist: {file_path}")
            return None
            
        file_extension = file_path.suffix.lower()
        
        if file_extension == '.txt':
            return self.load_text_file(file_path)
        elif file_extension == '.pdf':
            return self.load_pdf_file(file_path)
        else:
            logger.warning(f"Unsupported file extension: {file_extension}")
            return None
    
    def load_uploaded_file(self, uploaded_file) -> Optional[str]:
        """
        Load content from uploaded file (for Streamlit integration).
        
        Args:
            uploaded_file: Streamlit uploaded file object
            
        Returns:
            File content as string or None if failed
        """
        try:
            file_extension = Path(uploaded_file.name).suffix.lower()
            
            if file_extension == '.txt':
                content = str(uploaded_file.read(), "utf-8")
                logger.info(f"Successfully loaded uploaded text file: {uploaded_file.name}")
                return content
            elif file_extension == '.pdf':
                # For uploaded PDFs
                import io
                pdf_reader = pypdf.PdfReader(io.BytesIO(uploaded_file.read()))
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
                logger.info(f"Successfully loaded uploaded PDF: {uploaded_file.name}")
                return text
            else:
                logger.warning(f"Unsupported file extension: {file_extension}")
                return None
        except Exception as e:
            logger.error(f"Error loading uploaded file {uploaded_file.name}: {e}")
            return None
    
    def chunk_text(self, text: str, metadata: Dict[str, Any] = None) -> List[Document]:
        """Split text into chunks using the text splitter."""
        if not text.strip():
            return []
        
        try:
            chunks = self.text_splitter.split_text(text)
            documents = []
            
            for i, chunk in enumerate(chunks):
                chunk_metadata = metadata.copy() if metadata else {}
                chunk_metadata.update({
                    'chunk_id': i,
                    'chunk_size': len(chunk),
                    'total_chunks': len(chunks),
                    'processing_mode': 'chunked'
                })
                
                doc = Document(
                    page_content=chunk,
                    metadata=chunk_metadata
                )
                documents.append(doc)
            
            logger.info(f"Successfully chunked text into {len(documents)} chunks")
            return documents
            
        except Exception as e:
            logger.error(f"Error chunking text: {e}")
            return []
    
    def create_raw_document(self, text: str, metadata: Dict[str, Any] = None) -> Document:
        """
        Create a single document from raw text without chunking.
        
        Args:
            text: Raw text content
            metadata: Metadata to attach to document
            
        Returns:
            Single Document object
        """
        if not text.strip():
            return None
        
        try:
            doc_metadata = metadata.copy() if metadata else {}
            doc_metadata.update({
                'chunk_id': 0,
                'chunk_size': len(text),
                'total_chunks': 1,
                'processing_mode': 'raw'
            })
            
            doc = Document(
                page_content=text.strip(),
                metadata=doc_metadata
            )
            
            logger.info(f"Successfully created raw document (size: {len(text)} chars)")
            return doc
            
        except Exception as e:
            logger.error(f"Error creating raw document: {e}")
            return None
    
    def combine_all_texts(self, file_paths: List[Path]) -> str:
        """
        Combine all text content from multiple files into a single string.
        
        Args:
            file_paths: List of file paths to combine
            
        Returns:
            Combined text content
        """
        combined_text = ""
        
        for file_path in file_paths:
            content = self.load_file(file_path)
            if content:
                combined_text += f"\n\n--- Content from {file_path.name} ---\n\n"
                combined_text += content
        
        return combined_text.strip()
    
    def combine_uploaded_texts(self, uploaded_files) -> str:
        """
        Combine all text content from uploaded files into a single string.
        
        Args:
            uploaded_files: List of uploaded file objects
            
        Returns:
            Combined text content
        """
        combined_text = ""
        
        for uploaded_file in uploaded_files:
            content = self.load_uploaded_file(uploaded_file)
            if content:
                combined_text += f"\n\n--- Content from {uploaded_file.name} ---\n\n"
                combined_text += content
        
        return combined_text.strip()
    
    def process_file(self, file_path: Path, mode: str = "chunked") -> List[Document]:
        """
        Process a single file in either chunked or raw mode.
        
        Args:
            file_path: Path to the file to process
            mode: Processing mode - "chunked" or "raw"
            
        Returns:
            List of Document objects (single item for raw mode)
        """
        logger.info(f"Processing file: {file_path} in {mode} mode")
        
        # Load file content
        content = self.load_file(file_path)
        if not content:
            return []
        
        # Prepare metadata
        metadata = {
            'source_file': str(file_path),
            'file_name': file_path.name,
            'file_extension': file_path.suffix.lower(),
            'file_size': file_path.stat().st_size,
            'processing_timestamp': str(pd.Timestamp.now())
        }
        
        # Process based on mode
        if mode == "raw":
            doc = self.create_raw_document(content, metadata)
            return [doc] if doc else []
        else:  # chunked mode
            documents = self.chunk_text(content, metadata)
            return documents
    
    def process_uploaded_file(self, uploaded_file, mode: str = "chunked") -> List[Document]:
        """
        Process an uploaded file in either chunked or raw mode.
        
        Args:
            uploaded_file: Uploaded file object
            mode: Processing mode - "chunked" or "raw"
            
        Returns:
            List of Document objects (single item for raw mode)
        """
        logger.info(f"Processing uploaded file: {uploaded_file.name} in {mode} mode")
        
        # Load file content
        content = self.load_uploaded_file(uploaded_file)
        if not content:
            return []
        
        # Prepare metadata
        metadata = {
            'source_file': uploaded_file.name,
            'file_name': uploaded_file.name,
            'file_extension': Path(uploaded_file.name).suffix.lower(),
            'file_size': uploaded_file.size,
            'processing_timestamp': str(pd.Timestamp.now())
        }
        
        # Process based on mode
        if mode == "raw":
            doc = self.create_raw_document(content, metadata)
            return [doc] if doc else []
        else:  # chunked mode
            documents = self.chunk_text(content, metadata)
            return documents
    
    def process_directory(self, mode: str = "chunked") -> List[Document]:
        """
        Process all supported files in the data directory.
        
        Args:
            mode: Processing mode - "chunked" or "raw"
            
        Returns:
            List of all Document objects from all files
        """
        if not self.data_dir.exists():
            logger.error(f"Data directory does not exist: {self.data_dir}")
            return []
        
        all_documents = []
        
        if mode == "raw":
            # For raw mode, combine all files into a single document
            file_paths = [f for f in self.data_dir.iterdir() 
                         if f.is_file() and f.suffix.lower() in self.supported_extensions]
            
            if file_paths:
                combined_text = self.combine_all_texts(file_paths)
                if combined_text:
                    metadata = {
                        'source_file': 'combined_files',
                        'file_name': 'combined_documents',
                        'file_extension': '.combined',
                        'file_count': len(file_paths),
                        'source_files': [f.name for f in file_paths],
                        'processing_timestamp': str(pd.Timestamp.now())
                    }
                    
                    doc = self.create_raw_document(combined_text, metadata)
                    if doc:
                        all_documents = [doc]
        else:
            # For chunked mode, process each file separately
            for file_path in self.data_dir.iterdir():
                if file_path.is_file() and file_path.suffix.lower() in self.supported_extensions:
                    documents = self.process_file(file_path, mode)
                    all_documents.extend(documents)
        
        logger.info(f"Total documents processed in {mode} mode: {len(all_documents)}")
        return all_documents
    
    def process_multiple_uploaded_files(self, uploaded_files, mode: str = "chunked") -> List[Document]:
        """
        Process multiple uploaded files.
        
        Args:
            uploaded_files: List of uploaded file objects
            mode: Processing mode - "chunked" or "raw"
            
        Returns:
            List of Document objects
        """
        all_documents = []
        
        if mode == "raw":
            # For raw mode, combine all uploaded files into a single document
            combined_text = self.combine_uploaded_texts(uploaded_files)
            if combined_text:
                metadata = {
                    'source_file': 'combined_uploaded_files',
                    'file_name': 'combined_uploaded_documents',
                    'file_extension': '.combined',
                    'file_count': len(uploaded_files),
                    'source_files': [f.name for f in uploaded_files],
                    'processing_timestamp': str(pd.Timestamp.now())
                }
                
                doc = self.create_raw_document(combined_text, metadata)
                if doc:
                    all_documents = [doc]
        else:
            # For chunked mode, process each file separately
            for uploaded_file in uploaded_files:
                documents = self.process_uploaded_file(uploaded_file, mode)
                all_documents.extend(documents)
        
        logger.info(f"Total documents processed from uploads in {mode} mode: {len(all_documents)}")
        return all_documents
    
    def save_chunks_to_json(self, documents: List[Document], output_file: str = "processed_chunks.json"):
        """Save processed chunks to a JSON file for inspection."""
        try:
            chunks_data = []
            for doc in documents:
                chunk_data = {
                    'content': doc.page_content,
                    'metadata': doc.metadata
                }
                chunks_data.append(chunk_data)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(chunks_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Saved {len(documents)} chunks to {output_file}")
            
        except Exception as e:
            logger.error(f"Error saving chunks to JSON: {e}")
    
    def get_chunks_summary(self, documents: List[Document]) -> Dict[str, Any]:
        """Get a summary of the processed chunks."""
        if not documents:
            return {"total_chunks": 0, "files_processed": 0, "processing_modes": []}
        
        # Group by source file and processing mode
        files_processed = set()
        processing_modes = set()
        total_chunks = len(documents)
        
        for doc in documents:
            if 'source_file' in doc.metadata:
                files_processed.add(doc.metadata['source_file'])
            if 'processing_mode' in doc.metadata:
                processing_modes.add(doc.metadata['processing_mode'])
        
        return {
            "total_chunks": total_chunks,
            "files_processed": len(files_processed),
            "processing_modes": list(processing_modes),
            "average_chunk_size": sum(len(doc.page_content) for doc in documents) / total_chunks if total_chunks > 0 else 0,
            "file_types": list(set(doc.metadata.get('file_extension', '') for doc in documents if doc.metadata))
        }