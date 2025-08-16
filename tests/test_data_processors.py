#!/usr/bin/env python3
"""
Test suite for data_processors.py
Tests file loading, chunking, and processing functionality
"""

import unittest
import tempfile
import os
import sys
from pathlib import Path

# Add parent directory to path to import modules
sys.path.append(str(Path(__file__).parent.parent))

from src.data_processors import DataProcessor
from langchain_core.documents import Document


class TestDataProcessor(unittest.TestCase):
    """Test cases for DataProcessor class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.processor = DataProcessor()
        self.test_dir = tempfile.mkdtemp()
        
        # Create test files
        self.txt_file = os.path.join(self.test_dir, "test.txt")
        with open(self.txt_file, 'w', encoding='utf-8') as f:
            f.write("This is a test document about diabetes. It contains information about blood sugar levels and insulin.")
        
        self.large_txt_file = os.path.join(self.test_dir, "large_test.txt")
        with open(self.large_txt_file, 'w', encoding='utf-8') as f:
            # Create a larger text for chunking tests
            content = "Diabetes is a chronic condition. " * 50
            f.write(content)
    
    def tearDown(self):
        """Clean up test fixtures"""
        import shutil
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_initialization(self):
        """Test DataProcessor initialization"""
        processor = DataProcessor()
        self.assertIsNotNone(processor)
        self.assertEqual(processor.chunk_size, 1000)
        self.assertEqual(processor.chunk_overlap, 200)
    
    def test_initialization_with_custom_params(self):
        """Test DataProcessor initialization with custom parameters"""
        processor = DataProcessor(chunk_size=500, chunk_overlap=100)
        self.assertEqual(processor.chunk_size, 500)
        self.assertEqual(processor.chunk_overlap, 100)
    
    def test_load_text_file(self):
        """Test loading a text file"""
        documents = self.processor.load_text_file(self.txt_file)
        self.assertIsInstance(documents, list)
        self.assertEqual(len(documents), 1)
        self.assertIsInstance(documents[0], Document)
        self.assertIn("diabetes", documents[0].page_content.lower())
    
    def test_load_nonexistent_file(self):
        """Test loading a non-existent file"""
        with self.assertRaises(FileNotFoundError):
            self.processor.load_text_file("nonexistent.txt")
    
    def test_chunk_text(self):
        """Test text chunking functionality"""
        text = "This is a test document. " * 20  # Create long text
        chunks = self.processor.chunk_text(text)
        self.assertIsInstance(chunks, list)
        self.assertGreater(len(chunks), 1)  # Should be chunked
        
        # Test that chunks don't exceed chunk_size
        for chunk in chunks:
            self.assertLessEqual(len(chunk), self.processor.chunk_size)
    
    def test_process_file_chunked_mode(self):
        """Test processing file in chunked mode"""
        documents = self.processor.process_file(self.txt_file, mode="chunked")
        self.assertIsInstance(documents, list)
        self.assertGreater(len(documents), 0)
        
        for doc in documents:
            self.assertIsInstance(doc, Document)
            self.assertIn('source_file', doc.metadata)
            self.assertIn('chunk_id', doc.metadata)
    
    def test_process_file_raw_mode(self):
        """Test processing file in raw mode"""
        documents = self.processor.process_file(self.txt_file, mode="raw")
        self.assertIsInstance(documents, list)
        self.assertEqual(len(documents), 1)  # Should be one document
        
        doc = documents[0]
        self.assertIsInstance(doc, Document)
        self.assertIn('source_file', doc.metadata)
        self.assertNotIn('chunk_id', doc.metadata)  # Raw mode shouldn't have chunk_id
    
    def test_process_directory(self):
        """Test processing entire directory"""
        documents = self.processor.process_directory(self.test_dir, mode="chunked")
        self.assertIsInstance(documents, list)
        self.assertGreater(len(documents), 0)
        
        # Check that documents from different files are included
        source_files = set(doc.metadata.get('source_file') for doc in documents)
        self.assertGreaterEqual(len(source_files), 1)
    
    def test_combine_all_texts(self):
        """Test combining all texts from directory"""
        combined_doc = self.processor.combine_all_texts(self.test_dir)
        self.assertIsInstance(combined_doc, Document)
        self.assertIn("diabetes", combined_doc.page_content.lower())
        self.assertIn("chronic", combined_doc.page_content.lower())
    
    def test_create_raw_document(self):
        """Test creating raw document from text"""
        text = "This is test content about diabetes."
        doc = self.processor.create_raw_document(text, "test_source.txt")
        
        self.assertIsInstance(doc, Document)
        self.assertEqual(doc.page_content, text)
        self.assertEqual(doc.metadata.get('source_file'), "test_source.txt")
        self.assertNotIn('chunk_id', doc.metadata)
    
    def test_process_multiple_uploaded_files(self):
        """Test processing multiple uploaded files"""
        # Simulate uploaded files
        uploaded_files = [
            ("test1.txt", "Content about diabetes"),
            ("test2.txt", "Content about insulin")
        ]
        
        documents = self.processor.process_multiple_uploaded_files(uploaded_files, mode="chunked")
        self.assertIsInstance(documents, list)
        self.assertGreater(len(documents), 0)
        
        # Check that documents from both files are included
        source_files = set(doc.metadata.get('source_file') for doc in documents)
        self.assertEqual(len(source_files), 2)
    
    def test_combine_uploaded_texts(self):
        """Test combining uploaded texts"""
        uploaded_files = [
            ("test1.txt", "Content about diabetes"),
            ("test2.txt", "Content about insulin")
        ]
        
        combined_doc = self.processor.combine_uploaded_texts(uploaded_files)
        self.assertIsInstance(combined_doc, Document)
        self.assertIn("diabetes", combined_doc.page_content.lower())
        self.assertIn("insulin", combined_doc.page_content.lower())
    
    def test_invalid_mode(self):
        """Test processing with invalid mode"""
        with self.assertRaises(ValueError):
            self.processor.process_file(self.txt_file, mode="invalid_mode")
    
    def test_empty_file(self):
        """Test processing empty file"""
        empty_file = os.path.join(self.test_dir, "empty.txt")
        with open(empty_file, 'w') as f:
            pass
        
        documents = self.processor.process_file(empty_file, mode="raw")
        self.assertEqual(len(documents), 1)
        self.assertEqual(documents[0].page_content.strip(), "")


class TestDataProcessorIntegration(unittest.TestCase):
    """Integration tests for DataProcessor"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.processor = DataProcessor()
        self.test_dir = tempfile.mkdtemp()
        
        # Create multiple test files
        files_content = {
            "diabetes.txt": "Diabetes is a chronic disease affecting blood sugar levels.",
            "insulin.txt": "Insulin is a hormone that regulates blood glucose.",
            "treatment.txt": "Treatment includes diet, exercise, and medication."
        }
        
        for filename, content in files_content.items():
            filepath = os.path.join(self.test_dir, filename)
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
    
    def tearDown(self):
        """Clean up test fixtures"""
        import shutil
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_full_pipeline_chunked(self):
        """Test full processing pipeline in chunked mode"""
        documents = self.processor.process_directory(self.test_dir, mode="chunked")
        
        # Verify structure
        self.assertGreater(len(documents), 0)
        for doc in documents:
            self.assertIsInstance(doc, Document)
            self.assertIn('source_file', doc.metadata)
            self.assertIn('chunk_id', doc.metadata)
            self.assertIn('file_name', doc.metadata)
            self.assertIn('file_extension', doc.metadata)
    
    def test_full_pipeline_raw(self):
        """Test full processing pipeline in raw mode"""
        documents = self.processor.process_directory(self.test_dir, mode="raw")
        
        # Should have one document per file
        self.assertEqual(len(documents), 3)
        
        # Check content
        content = " ".join(doc.page_content for doc in documents)
        self.assertIn("diabetes", content.lower())
        self.assertIn("insulin", content.lower())
        self.assertIn("treatment", content.lower())


if __name__ == '__main__':
    unittest.main()
