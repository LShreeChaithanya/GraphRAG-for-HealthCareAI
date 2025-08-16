#!/usr/bin/env python3
"""
Test suite for neo4j_utils.py
Tests Neo4j connection, CRUD operations, queries, and error handling
"""

import unittest
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Add parent directory to path to import modules
sys.path.append(str(Path(__file__).parent.parent))

from src.neo4j_utils import Neo4jProcessor
from langchain_core.documents import Document


class TestNeo4jProcessor(unittest.TestCase):
    """Test cases for Neo4jProcessor class"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Mock environment variables
        self.env_patcher = patch.dict(os.environ, {
            'NEO4J_URI': 'bolt://localhost:7687',
            'NEO4J_USERNAME': 'neo4j',
            'NEO4J_PASSWORD': 'password'
        })
        self.env_patcher.start()
    
    def tearDown(self):
        """Clean up test fixtures"""
        self.env_patcher.stop()
    
    @patch('neo4j_utils.Neo4jGraph')
    def test_initialization(self, mock_neo4j_graph):
        """Test Neo4jProcessor initialization"""
        mock_graph = Mock()
        mock_neo4j_graph.return_value = mock_graph
        
        processor = Neo4jProcessor()
        
        self.assertIsNotNone(processor)
        self.assertEqual(processor.uri, 'bolt://localhost:7687')
        self.assertEqual(processor.user, 'neo4j')
        self.assertEqual(processor.password, 'password')
        self.assertEqual(processor.database, 'neo4j')
        self.assertEqual(processor.graph, mock_graph)
    
    @patch('neo4j_utils.Neo4jGraph')
    def test_initialization_with_custom_params(self, mock_neo4j_graph):
        """Test initialization with custom parameters"""
        mock_graph = Mock()
        mock_neo4j_graph.return_value = mock_graph
        
        processor = Neo4jProcessor(
            uri='bolt://test:7687',
            user='testuser',
            password='testpass',
            database='testdb'
        )
        
        self.assertEqual(processor.uri, 'bolt://test:7687')
        self.assertEqual(processor.user, 'testuser')
        self.assertEqual(processor.password, 'testpass')
        self.assertEqual(processor.database, 'testdb')
    
    @patch('neo4j_utils.Neo4jGraph')
    def test_connection_failure(self, mock_neo4j_graph):
        """Test handling connection failure"""
        mock_neo4j_graph.side_effect = Exception("Connection failed")
        
        processor = Neo4jProcessor()
        
        self.assertIsNone(processor.graph)
    
    @patch('neo4j_utils.Neo4jGraph')
    def test_create_indexes(self, mock_neo4j_graph):
        """Test index creation"""
        mock_graph = Mock()
        mock_neo4j_graph.return_value = mock_graph
        
        processor = Neo4jProcessor()
        
        # Verify that _create_indexes was called
        self.assertEqual(mock_graph.query.call_count, 4)  # 4 indexes
    
    @patch('neo4j_utils.Neo4jGraph')
    def test_create_knowledge_graph_nodes(self, mock_neo4j_graph):
        """Test creating knowledge graph nodes"""
        mock_graph = Mock()
        mock_neo4j_graph.return_value = mock_graph
        
        processor = Neo4jProcessor()
        
        nodes = [
            {'id': 'node1', 'type': 'Entity', 'properties': {'name': 'Diabetes'}},
            {'id': 'node2', 'type': 'Disease', 'properties': {'name': 'Type 2 Diabetes'}}
        ]
        
        result = processor.create_knowledge_graph_nodes(nodes)
        
        self.assertTrue(result)
        self.assertEqual(mock_graph.query.call_count, 6)  # 4 indexes + 2 nodes
    
    @patch('neo4j_utils.Neo4jGraph')
    def test_create_knowledge_graph_relationships(self, mock_neo4j_graph):
        """Test creating knowledge graph relationships"""
        mock_graph = Mock()
        mock_neo4j_graph.return_value = mock_graph
        
        processor = Neo4jProcessor()
        
        relationships = [
            {'source': 'node1', 'target': 'node2', 'type': 'RELATES_TO', 'properties': {'weight': 1.0}},
            {'source': 'node2', 'target': 'node3', 'type': 'CAUSES', 'properties': {}}
        ]
        
        result = processor.create_knowledge_graph_relationships(relationships)
        
        self.assertTrue(result)
        # Should have 4 index calls + 2 relationship calls
        self.assertEqual(mock_graph.query.call_count, 6)
    
    @patch('neo4j_utils.Neo4jGraph')
    def test_create_full_knowledge_graph(self, mock_neo4j_graph):
        """Test creating full knowledge graph"""
        mock_graph = Mock()
        mock_neo4j_graph.return_value = mock_graph
        
        processor = Neo4jProcessor()
        
        graph_data = {
            'nodes': [
                {'id': 'node1', 'type': 'Entity', 'properties': {'name': 'Diabetes'}},
                {'id': 'node2', 'type': 'Disease', 'properties': {'name': 'Type 2 Diabetes'}}
            ],
            'relationships': [
                {'source': 'node1', 'target': 'node2', 'type': 'RELATES_TO', 'properties': {}}
            ]
        }
        
        result = processor.create_full_knowledge_graph(graph_data)
        
        self.assertTrue(result)
        # Should have 4 index calls + 2 node calls + 1 relationship call
        self.assertEqual(mock_graph.query.call_count, 7)
    
    @patch('neo4j_utils.Neo4jGraph')
    def test_create_document_nodes(self, mock_neo4j_graph):
        """Test creating document nodes"""
        mock_graph = Mock()
        mock_neo4j_graph.return_value = mock_graph
        
        processor = Neo4jProcessor()
        
        documents = [
            Document(
                page_content="Diabetes is a chronic disease",
                metadata={
                    'chunk_id': 1,
                    'source_file': 'test.txt',
                    'file_name': 'test.txt',
                    'file_extension': '.txt',
                    'chunk_size': 100,
                    'total_chunks': 1,
                    'processing_timestamp': '2024-01-01'
                }
            )
        ]
        
        result = processor.create_document_nodes(documents)
        
        self.assertTrue(result)
        # Should have 4 index calls + 1 document call
        self.assertEqual(mock_graph.query.call_count, 5)
    
    @patch('neo4j_utils.Neo4jGraph')
    def test_query_all_nodes(self, mock_neo4j_graph):
        """Test querying all nodes"""
        mock_graph = Mock()
        mock_neo4j_graph.return_value = mock_graph
        
        # Mock query result
        mock_graph.query.return_value = [
            {'n': {'id': 'node1', 'name': 'Diabetes'}},
            {'n': {'id': 'node2', 'name': 'Insulin'}}
        ]
        
        processor = Neo4jProcessor()
        result = processor.query_all_nodes(limit=10)
        
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]['id'], 'node1')
        self.assertEqual(result[1]['id'], 'node2')
    
    @patch('neo4j_utils.Neo4jGraph')
    def test_query_nodes_by_type(self, mock_neo4j_graph):
        """Test querying nodes by type"""
        mock_graph = Mock()
        mock_neo4j_graph.return_value = mock_graph
        
        mock_graph.query.return_value = [
            {'n': {'id': 'node1', 'type': 'Disease', 'name': 'Diabetes'}}
        ]
        
        processor = Neo4jProcessor()
        result = processor.query_nodes_by_type('Disease', limit=10)
        
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]['id'], 'node1')
        self.assertEqual(result[0]['type'], 'Disease')
    
    @patch('neo4j_utils.Neo4jGraph')
    def test_query_relationships(self, mock_neo4j_graph):
        """Test querying relationships"""
        mock_graph = Mock()
        mock_neo4j_graph.return_value = mock_graph
        
        # Mock relationship query result
        mock_source = Mock()
        mock_source.__iter__ = lambda x: iter({'id': 'node1', 'name': 'Diabetes'}.items())
        mock_target = Mock()
        mock_target.__iter__ = lambda x: iter({'id': 'node2', 'name': 'Insulin'}.items())
        mock_rel = Mock()
        mock_rel.__iter__ = lambda x: iter({'type': 'RELATES_TO', 'weight': 1.0}.items())
        
        mock_graph.query.return_value = [
            {'a': mock_source, 'r': mock_rel, 'b': mock_target}
        ]
        
        processor = Neo4jProcessor()
        result = processor.query_relationships(limit=10)
        
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 1)
        self.assertIn('source', result[0])
        self.assertIn('relationship', result[0])
        self.assertIn('target', result[0])
    
    @patch('neo4j_utils.Neo4jGraph')
    def test_query_by_keyword(self, mock_neo4j_graph):
        """Test querying by keyword"""
        mock_graph = Mock()
        mock_neo4j_graph.return_value = mock_graph
        
        mock_graph.query.return_value = [
            {'n': {'id': 'node1', 'name': 'Diabetes', 'description': 'A chronic disease'}}
        ]
        
        processor = Neo4jProcessor()
        result = processor.query_by_keyword('diabetes', limit=10)
        
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]['id'], 'node1')
    
    @patch('neo4j_utils.Neo4jGraph')
    def test_get_document_count(self, mock_neo4j_graph):
        """Test getting document count"""
        mock_graph = Mock()
        mock_neo4j_graph.return_value = mock_graph
        
        mock_graph.query.return_value = [{'count': 5}]
        
        processor = Neo4jProcessor()
        result = processor.get_document_count()
        
        self.assertEqual(result, 5)
    
    @patch('neo4j_utils.Neo4jGraph')
    def test_get_knowledge_graph_count(self, mock_neo4j_graph):
        """Test getting knowledge graph count"""
        mock_graph = Mock()
        mock_neo4j_graph.return_value = mock_graph
        
        # Mock different query results
        mock_graph.query.side_effect = [
            [{'count': 10}],  # Non-document nodes
            [{'count': 15}]   # Relationships
        ]
        
        processor = Neo4jProcessor()
        result = processor.get_knowledge_graph_count()
        
        self.assertEqual(result['nodes'], 10)
        self.assertEqual(result['relationships'], 15)
    
    @patch('neo4j_utils.Neo4jGraph')
    def test_clear_documents(self, mock_neo4j_graph):
        """Test clearing documents"""
        mock_graph = Mock()
        mock_neo4j_graph.return_value = mock_graph
        
        processor = Neo4jProcessor()
        result = processor.clear_documents()
        
        self.assertTrue(result)
        # Should have 4 index calls + 1 clear call
        self.assertEqual(mock_graph.query.call_count, 5)
    
    @patch('neo4j_utils.Neo4jGraph')
    def test_clear_knowledge_graph(self, mock_neo4j_graph):
        """Test clearing knowledge graph"""
        mock_graph = Mock()
        mock_neo4j_graph.return_value = mock_graph
        
        processor = Neo4jProcessor()
        result = processor.clear_knowledge_graph()
        
        self.assertTrue(result)
        # Should have 4 index calls + 1 clear call
        self.assertEqual(mock_graph.query.call_count, 5)
    
    @patch('neo4j_utils.Neo4jGraph')
    def test_delete_entire_graph(self, mock_neo4j_graph):
        """Test deleting entire graph"""
        mock_graph = Mock()
        mock_neo4j_graph.return_value = mock_graph
        
        processor = Neo4jProcessor()
        result = processor.delete_entire_graph()
        
        self.assertTrue(result)
        # Should have 4 index calls + 1 delete call
        self.assertEqual(mock_graph.query.call_count, 5)
    
    @patch('neo4j_utils.Neo4jGraph')
    def test_delete_by_source_file(self, mock_neo4j_graph):
        """Test deleting by source file"""
        mock_graph = Mock()
        mock_neo4j_graph.return_value = mock_graph
        
        processor = Neo4jProcessor()
        result = processor.delete_by_source_file('test.txt')
        
        self.assertTrue(result)
        # Should have 4 index calls + 2 delete calls (documents + knowledge graph)
        self.assertEqual(mock_graph.query.call_count, 6)
    
    @patch('neo4j_utils.Neo4jGraph')
    def test_execute_custom_query(self, mock_neo4j_graph):
        """Test executing custom query"""
        mock_graph = Mock()
        mock_neo4j_graph.return_value = mock_graph
        
        mock_graph.query.return_value = [
            {'name': 'Diabetes', 'count': 5},
            {'name': 'Insulin', 'count': 3}
        ]
        
        processor = Neo4jProcessor()
        result = processor.execute_custom_query(
            "MATCH (n) RETURN n.name as name, count(n) as count"
        )
        
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]['name'], 'Diabetes')
        self.assertEqual(result[0]['count'], 5)
    
    @patch('neo4j_utils.Neo4jGraph')
    def test_verify_graph_creation(self, mock_neo4j_graph):
        """Test verifying graph creation"""
        mock_graph = Mock()
        mock_neo4j_graph.return_value = mock_graph
        
        # Mock different query results
        mock_graph.query.side_effect = [
            [{'total': 10}],  # Total nodes
            [{'labels': ['Entity'], 'count': 5}, {'labels': ['Disease'], 'count': 5}],  # Labels
            [{'rel_type': 'RELATES_TO', 'count': 8}],  # Relationships
            [{'isolated_count': 2}],  # Isolated nodes
            [{'n': {'id': 'node1', 'name': 'Diabetes'}}]  # Sample nodes
        ]
        
        processor = Neo4jProcessor()
        result = processor.verify_graph_creation()
        
        self.assertIsInstance(result, dict)
        self.assertEqual(result['total_nodes'], 10)
        self.assertIn('node_labels', result)
        self.assertIn('relationship_types', result)
        self.assertIn('isolated_nodes', result)
        self.assertIn('sample_nodes', result)
    
    @patch('neo4j_utils.Neo4jGraph')
    def test_get_graph_statistics(self, mock_neo4j_graph):
        """Test getting graph statistics"""
        mock_graph = Mock()
        mock_neo4j_graph.return_value = mock_graph
        
        # Mock different query results
        mock_graph.query.side_effect = [
            [{'count': 10}],  # Total nodes
            [{'count': 15}],  # Total relationships
            [{'node_type': 'Entity', 'count': 5}, {'node_type': 'Disease', 'count': 5}],  # Node types
            [{'rel_type': 'RELATES_TO', 'count': 10}],  # Relationship types
            [{'total_docs': 3, 'unique_sources': 2, 'file_types': 1}]  # Document stats
        ]
        
        processor = Neo4jProcessor()
        result = processor.get_graph_statistics()
        
        self.assertIsInstance(result, dict)
        self.assertEqual(result['total_nodes'], 10)
        self.assertEqual(result['total_relationships'], 15)
        self.assertIn('node_types', result)
        self.assertIn('relationship_types', result)
        self.assertIn('document_stats', result)


class TestNeo4jProcessorErrorHandling(unittest.TestCase):
    """Test error handling in Neo4jProcessor"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.env_patcher = patch.dict(os.environ, {
            'NEO4J_URI': 'bolt://localhost:7687',
            'NEO4J_USERNAME': 'neo4j',
            'NEO4J_PASSWORD': 'password'
        })
        self.env_patcher.start()
    
    def tearDown(self):
        """Clean up test fixtures"""
        self.env_patcher.stop()
    
    @patch('neo4j_utils.Neo4jGraph')
    def test_no_graph_available(self, mock_neo4j_graph):
        """Test operations when graph is not available"""
        mock_neo4j_graph.side_effect = Exception("Connection failed")
        
        processor = Neo4jProcessor()
        
        # Test various operations
        self.assertFalse(processor.create_knowledge_graph_nodes([]))
        self.assertFalse(processor.create_knowledge_graph_relationships([]))
        self.assertEqual(processor.get_document_count(), 0)
        self.assertEqual(processor.query_all_nodes(), [])
    
    @patch('neo4j_utils.Neo4jGraph')
    def test_query_exception_handling(self, mock_neo4j_graph):
        """Test handling query exceptions"""
        mock_graph = Mock()
        mock_neo4j_graph.return_value = mock_graph
        
        # Mock query to raise exception
        mock_graph.query.side_effect = Exception("Query failed")
        
        processor = Neo4jProcessor()
        
        # Test that exceptions are handled gracefully
        self.assertEqual(processor.query_all_nodes(), [])
        self.assertEqual(processor.get_document_count(), 0)
        self.assertFalse(processor.create_knowledge_graph_nodes([]))


class TestNeo4jUtilityFunctions(unittest.TestCase):
    """Test utility functions in neo4j_utils"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.env_patcher = patch.dict(os.environ, {
            'NEO4J_URI': 'bolt://localhost:7687',
            'NEO4J_USERNAME': 'neo4j',
            'NEO4J_PASSWORD': 'password'
        })
        self.env_patcher.start()
    
    def tearDown(self):
        """Clean up test fixtures"""
        self.env_patcher.stop()
    
    @patch('neo4j_utils.Neo4jProcessor')
    def test_store_knowledge_graph_in_neo4j(self, mock_processor_class):
        """Test store_knowledge_graph_in_neo4j utility function"""
        mock_processor = Mock()
        mock_processor_class.return_value = mock_processor
        mock_processor.graph = Mock()
        mock_processor.create_full_knowledge_graph.return_value = True
        mock_processor.get_knowledge_graph_count.return_value = {'nodes': 5, 'relationships': 3}
        
        from src.neo4j_utils import store_knowledge_graph_in_neo4j
        
        graph_data = {
            'nodes': [{'id': 'node1', 'type': 'Entity'}],
            'relationships': [{'source': 'node1', 'target': 'node2', 'type': 'RELATES_TO'}]
        }
        
        result = store_knowledge_graph_in_neo4j(graph_data)
        
        self.assertTrue(result)
        mock_processor.create_full_knowledge_graph.assert_called_once_with(graph_data)
        mock_processor.close.assert_called_once()
    
    @patch('neo4j_utils.Neo4jProcessor')
    def test_verify_neo4j_graph(self, mock_processor_class):
        """Test verify_neo4j_graph utility function"""
        mock_processor = Mock()
        mock_processor_class.return_value = mock_processor
        mock_processor.graph = Mock()
        mock_processor.verify_graph_creation.return_value = {'total_nodes': 10}
        
        from src.neo4j_utils import verify_neo4j_graph
        
        result = verify_neo4j_graph()
        
        self.assertEqual(result, {'total_nodes': 10})
        mock_processor.verify_graph_creation.assert_called_once()
        mock_processor.close.assert_called_once()
    
    @patch('neo4j_utils.Neo4jProcessor')
    def test_query_neo4j_graph(self, mock_processor_class):
        """Test query_neo4j_graph utility function"""
        mock_processor = Mock()
        mock_processor_class.return_value = mock_processor
        mock_processor.graph = Mock()
        mock_processor.execute_custom_query.return_value = [{'name': 'Diabetes'}]
        
        from src.neo4j_utils import query_neo4j_graph
        
        result = query_neo4j_graph("MATCH (n) RETURN n.name as name")
        
        self.assertEqual(result, [{'name': 'Diabetes'}])
        mock_processor.execute_custom_query.assert_called_once_with("MATCH (n) RETURN n.name as name", None)
        mock_processor.close.assert_called_once()


if __name__ == '__main__':
    unittest.main()
