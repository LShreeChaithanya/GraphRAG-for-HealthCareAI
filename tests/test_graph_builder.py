#!/usr/bin/env python3
"""
Test suite for Graph_builder.py
Tests knowledge graph creation, LLM integration, and serialization
"""

import unittest
import tempfile
import os
import sys
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Add parent directory to path to import modules
sys.path.append(str(Path(__file__).parent.parent))

from src.Graph_builder import EnhancedGraphBuilder
from src.data_processors import DataProcessor
from langchain_core.documents import Document


class TestEnhancedGraphBuilder(unittest.TestCase):
    """Test cases for EnhancedGraphBuilder class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.test_dir = tempfile.mkdtemp()
        
        # Create test files
        self.txt_file = os.path.join(self.test_dir, "test.txt")
        with open(self.txt_file, 'w', encoding='utf-8') as f:
            f.write("Diabetes is a chronic disease. Insulin helps control blood sugar.")
        
        # Mock LLM for testing
        self.mock_llm = Mock()
        self.mock_llm.invoke.return_value = "Mock LLM response"
        
        # Create graph builder with mock LLM
        self.graph_builder = EnhancedGraphBuilder(custom_llm=self.mock_llm)
    
    def tearDown(self):
        """Clean up test fixtures"""
        import shutil
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_initialization(self):
        """Test EnhancedGraphBuilder initialization"""
        builder = EnhancedGraphBuilder()
        self.assertIsNotNone(builder)
        self.assertIsNotNone(builder.graph_transformer)
        self.assertIsNotNone(builder.data_processor)
    
    def test_initialization_with_custom_llm(self):
        """Test initialization with custom LLM"""
        builder = EnhancedGraphBuilder(custom_llm=self.mock_llm)
        self.assertEqual(builder.llm, self.mock_llm)
    
    def test_create_knowledge_graph_from_files(self):
        """Test creating knowledge graph from files"""
        # Mock the LLMGraphTransformer
        with patch('Graph_builder.LLMGraphTransformer') as mock_transformer:
            mock_instance = Mock()
            mock_instance.convert_to_graph_documents.return_value = [
                Mock(nodes=[Mock(id="node1", type="Entity", properties={})],
                     relationships=[Mock(source="node1", target="node2", type="RELATES_TO", properties={})])
            ]
            mock_transformer.return_value = mock_instance
            
            builder = EnhancedGraphBuilder(custom_llm=self.mock_llm)
            result = builder.create_knowledge_graph_from_files([self.txt_file], mode="raw")
            
            self.assertIsInstance(result, dict)
            self.assertIn('success', result)
            self.assertTrue(result['success'])
    
    def test_build_graph_from_directory(self):
        """Test building graph from directory"""
        with patch('Graph_builder.LLMGraphTransformer') as mock_transformer:
            mock_instance = Mock()
            mock_instance.convert_to_graph_documents.return_value = [
                Mock(nodes=[Mock(id="node1", type="Entity", properties={})],
                     relationships=[Mock(source="node1", target="node2", type="RELATES_TO", properties={})])
            ]
            mock_transformer.return_value = mock_instance
            
            builder = EnhancedGraphBuilder(custom_llm=self.mock_llm)
            result = builder.build_graph_from_directory(self.test_dir, mode="chunked")
            
            self.assertIsInstance(result, dict)
            self.assertIn('success', result)
    
    def test_build_graph_from_uploaded_files(self):
        """Test building graph from uploaded files"""
        uploaded_files = [
            ("test1.txt", "Content about diabetes"),
            ("test2.txt", "Content about insulin")
        ]
        
        with patch('Graph_builder.LLMGraphTransformer') as mock_transformer:
            mock_instance = Mock()
            mock_instance.convert_to_graph_documents.return_value = [
                Mock(nodes=[Mock(id="node1", type="Entity", properties={})],
                     relationships=[Mock(source="node1", target="node2", type="RELATES_TO", properties={})])
            ]
            mock_transformer.return_value = mock_instance
            
            builder = EnhancedGraphBuilder(custom_llm=self.mock_llm)
            result = builder.build_graph_from_uploaded_files(uploaded_files, mode="raw")
            
            self.assertIsInstance(result, dict)
            self.assertIn('success', result)
    
    def test_serialize_nodes(self):
        """Test node serialization"""
        mock_node = Mock()
        mock_node.id = "test_node"
        mock_node.type = "Entity"
        mock_node.properties = {"name": "test"}
        
        builder = EnhancedGraphBuilder()
        serialized = builder._serialize_nodes([mock_node])
        
        self.assertIsInstance(serialized, list)
        self.assertEqual(len(serialized), 1)
        self.assertEqual(serialized[0]['id'], "test_node")
        self.assertEqual(serialized[0]['type'], "Entity")
        self.assertEqual(serialized[0]['properties'], {"name": "test"})
    
    def test_serialize_relationships(self):
        """Test relationship serialization"""
        mock_rel = Mock()
        mock_rel.source = "node1"
        mock_rel.target = "node2"
        mock_rel.type = "RELATES_TO"
        mock_rel.properties = {"weight": 1.0}
        
        builder = EnhancedGraphBuilder()
        serialized = builder._serialize_relationships([mock_rel])
        
        self.assertIsInstance(serialized, list)
        self.assertEqual(len(serialized), 1)
        self.assertEqual(serialized[0]['source'], "node1")
        self.assertEqual(serialized[0]['target'], "node2")
        self.assertEqual(serialized[0]['type'], "RELATES_TO")
        self.assertEqual(serialized[0]['properties'], {"weight": 1.0})
    
    def test_convert_documents_to_graph(self):
        """Test converting documents to graph"""
        documents = [
            Document(page_content="Diabetes is a disease", metadata={"source": "test.txt"}),
            Document(page_content="Insulin controls blood sugar", metadata={"source": "test.txt"})
        ]
        
        with patch('Graph_builder.LLMGraphTransformer') as mock_transformer:
            mock_instance = Mock()
            mock_instance.convert_to_graph_documents.return_value = [
                Mock(nodes=[Mock(id="node1", type="Entity", properties={})],
                     relationships=[Mock(source="node1", target="node2", type="RELATES_TO", properties={})])
            ]
            mock_transformer.return_value = mock_instance
            
            builder = EnhancedGraphBuilder(custom_llm=self.mock_llm)
            result = builder._convert_documents_to_graph(documents)
            
            self.assertIsInstance(result, list)
            self.assertGreater(len(result), 0)
    
    def test_save_results(self):
        """Test saving results to files"""
        test_results = {
            'success': True,
            'graphs': [
                {
                    'nodes': [{'id': 'node1', 'type': 'Entity', 'properties': {}}],
                    'relationships': [{'source': 'node1', 'target': 'node2', 'type': 'RELATES_TO', 'properties': {}}]
                }
            ]
        }
        
        builder = EnhancedGraphBuilder()
        with tempfile.TemporaryDirectory() as temp_dir:
            builder._save_results(test_results, temp_dir)
            
            # Check if files were created
            json_file = os.path.join(temp_dir, "knowledge_graph.json")
            csv_file = os.path.join(temp_dir, "knowledge_graph.csv")
            
            self.assertTrue(os.path.exists(json_file))
            self.assertTrue(os.path.exists(csv_file))
    
    def test_show_sample_data(self):
        """Test showing sample data"""
        test_results = {
            'success': True,
            'graphs': [
                {
                    'nodes': [{'id': 'node1', 'type': 'Entity', 'properties': {'name': 'test'}}],
                    'relationships': [{'source': 'node1', 'target': 'node2', 'type': 'RELATES_TO', 'properties': {}}]
                }
            ]
        }
        
        builder = EnhancedGraphBuilder()
        # This should not raise an exception
        builder._show_sample_data(test_results)


class TestGraphBuilderIntegration(unittest.TestCase):
    """Integration tests for GraphBuilder"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.test_dir = tempfile.mkdtemp()
        
        # Create test files with healthcare content
        files_content = {
            "diabetes.txt": "Diabetes is a chronic disease that affects how your body turns food into energy.",
            "insulin.txt": "Insulin is a hormone made by the pancreas that helps glucose enter cells.",
            "treatment.txt": "Treatment for diabetes includes diet, exercise, and medication."
        }
        
        for filename, content in files_content.items():
            filepath = os.path.join(self.test_dir, filename)
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
    
    def tearDown(self):
        """Clean up test fixtures"""
        import shutil
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    @patch('Graph_builder.LLMGraphTransformer')
    def test_full_pipeline_chunked_mode(self, mock_transformer):
        """Test full pipeline in chunked mode"""
        # Mock the transformer
        mock_instance = Mock()
        mock_instance.convert_to_graph_documents.return_value = [
            Mock(nodes=[Mock(id="diabetes", type="Disease", properties={"name": "Diabetes"}),
                       Mock(id="insulin", type="Hormone", properties={"name": "Insulin"})],
                 relationships=[Mock(source="diabetes", target="insulin", type="TREATED_BY", properties={})])
        ]
        mock_transformer.return_value = mock_instance
        
        # Mock LLM
        mock_llm = Mock()
        mock_llm.invoke.return_value = "Mock LLM response"
        
        builder = EnhancedGraphBuilder(custom_llm=mock_llm)
        result = builder.build_graph_from_directory(self.test_dir, mode="chunked")
        
        self.assertIsInstance(result, dict)
        self.assertIn('success', result)
        self.assertTrue(result['success'])
        self.assertIn('graphs', result)
    
    @patch('Graph_builder.LLMGraphTransformer')
    def test_full_pipeline_raw_mode(self, mock_transformer):
        """Test full pipeline in raw mode"""
        # Mock the transformer
        mock_instance = Mock()
        mock_instance.convert_to_graph_documents.return_value = [
            Mock(nodes=[Mock(id="diabetes", type="Disease", properties={"name": "Diabetes"})],
                 relationships=[])
        ]
        mock_transformer.return_value = mock_instance
        
        # Mock LLM
        mock_llm = Mock()
        mock_llm.invoke.return_value = "Mock LLM response"
        
        builder = EnhancedGraphBuilder(custom_llm=mock_llm)
        result = builder.build_graph_from_directory(self.test_dir, mode="raw")
        
        self.assertIsInstance(result, dict)
        self.assertIn('success', result)
        self.assertTrue(result['success'])


class TestGraphBuilderErrorHandling(unittest.TestCase):
    """Test error handling in GraphBuilder"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.test_dir = tempfile.mkdtemp()
        self.txt_file = os.path.join(self.test_dir, "test.txt")
        with open(self.txt_file, 'w', encoding='utf-8') as f:
            f.write("Test content")
    
    def tearDown(self):
        """Clean up test fixtures"""
        import shutil
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_empty_directory(self):
        """Test handling empty directory"""
        empty_dir = tempfile.mkdtemp()
        try:
            builder = EnhancedGraphBuilder()
            result = builder.build_graph_from_directory(empty_dir, mode="chunked")
            
            self.assertIsInstance(result, dict)
            self.assertIn('success', result)
            # Should handle empty directory gracefully
        finally:
            import shutil
            shutil.rmtree(empty_dir, ignore_errors=True)
    
    def test_invalid_file_type(self):
        """Test handling invalid file type"""
        invalid_file = os.path.join(self.test_dir, "test.invalid")
        with open(invalid_file, 'w') as f:
            f.write("test")
        
        builder = EnhancedGraphBuilder()
        result = builder.build_graph_from_directory(self.test_dir, mode="chunked")
        
        # Should handle invalid file types gracefully
        self.assertIsInstance(result, dict)
    
    @patch('Graph_builder.LLMGraphTransformer')
    def test_llm_error_handling(self, mock_transformer):
        """Test handling LLM errors"""
        # Mock transformer to raise exception
        mock_instance = Mock()
        mock_instance.convert_to_graph_documents.side_effect = Exception("LLM Error")
        mock_transformer.return_value = mock_instance
        
        builder = EnhancedGraphBuilder()
        result = builder.build_graph_from_directory(self.test_dir, mode="raw")
        
        self.assertIsInstance(result, dict)
        self.assertIn('success', result)
        self.assertFalse(result['success'])
        self.assertIn('error', result)


if __name__ == '__main__':
    unittest.main()