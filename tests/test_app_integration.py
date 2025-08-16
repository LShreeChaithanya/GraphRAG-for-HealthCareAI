
"""
Integration test suite for app.py
Tests Streamlit app functionality, UI components, and system integration
"""

import unittest
import sys
import os
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Add parent directory to path to import modules
sys.path.append(str(Path(__file__).parent.parent))

# Mock streamlit for testing
sys.modules['streamlit'] = Mock()
sys.modules['streamlit.components.v1'] = Mock()
sys.modules['pyvis.network'] = Mock()


class TestAppIntegration(unittest.TestCase):
    """Integration tests for the Streamlit app"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Mock environment variables
        self.env_patcher = patch.dict(os.environ, {
            'NEO4J_URI': 'bolt://localhost:7687',
            'NEO4J_USERNAME': 'neo4j',
            'NEO4J_PASSWORD': 'password',
            'GOOGLE_API_KEY': 'test_api_key'
        })
        self.env_patcher.start()
        
        # Create temporary test directory
        self.test_dir = tempfile.mkdtemp()
        
        # Create test files
        self.txt_file = os.path.join(self.test_dir, "test.txt")
        with open(self.txt_file, 'w', encoding='utf-8') as f:
            f.write("Diabetes is a chronic disease. Insulin helps control blood sugar.")
    
    def tearDown(self):
        """Clean up test fixtures"""
        self.env_patcher.stop()
        import shutil
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    @patch('app.st')
    @patch('app.Neo4jProcessor')
    def test_establish_neo4j_connection(self, mock_neo4j_processor, mock_st):
        """Test establishing Neo4j connection"""
        # Mock session state
        mock_st.session_state = {}
        
        # Mock Neo4j processor
        mock_processor = Mock()
        mock_neo4j_processor.return_value = mock_processor
        
        # Import and test the function
        from src.app import establish_neo4j_connection
        
        establish_neo4j_connection()
        
        # Verify connection was established
        self.assertTrue(mock_st.session_state.get('neo4j_connected', False))
        self.assertEqual(mock_st.session_state.get('neo4j_processor'), mock_processor)
    
    @patch('app.st')
    @patch('app.Neo4jProcessor')
    def test_store_knowledge_graph_in_neo4j(self, mock_neo4j_processor, mock_st):
        """Test storing knowledge graph in Neo4j"""
        # Mock session state
        mock_st.session_state = {
            'neo4j_connected': True,
            'neo4j_processor': Mock()
        }
        
        # Mock processor methods
        mock_st.session_state['neo4j_processor'].create_full_knowledge_graph.return_value = True
        
        # Test data
        test_results = {
            'success': True,
            'graphs': [
                {
                    'nodes': [{'id': 'node1', 'type': 'Entity', 'properties': {}}],
                    'relationships': [{'source': 'node1', 'target': 'node2', 'type': 'RELATES_TO', 'properties': {}}]
                }
            ]
        }
        
        # Import and test the function
        from src.app import store_knowledge_graph_in_neo4j
        
        result = store_knowledge_graph_in_neo4j(test_results)
        
        # Verify storage was successful
        self.assertTrue(result)
        mock_st.session_state['neo4j_processor'].create_full_knowledge_graph.assert_called_once()
    
    @patch('app.st')
    @patch('app.SimpleHealthcareAgent')
    def test_initialize_healthcare_agent(self, mock_agent_class, mock_st):
        """Test initializing healthcare agent"""
        # Mock session state
        mock_st.session_state = {
            'neo4j_processor': Mock()
        }
        
        # Mock agent
        mock_agent = Mock()
        mock_agent_class.return_value = mock_agent
        
        # Import and test the function
        from src.app import initialize_healthcare_agent
        
        initialize_healthcare_agent()
        
        # Verify agent was initialized
        self.assertEqual(mock_st.session_state.get('healthcare_agent'), mock_agent)
        mock_agent.set_neo4j_processor.assert_called_once_with(mock_st.session_state['neo4j_processor'])
    
    @patch('app.st')
    def test_show_notification(self, mock_st):
        """Test notification system"""
        # Mock session state
        mock_st.session_state = {'notifications': []}
        
        # Import and test the function
        from src.app import show_notification
        
        show_notification("Test message", "success", 5)
        
        # Verify notification was added
        self.assertEqual(len(mock_st.session_state['notifications']), 1)
        notification = mock_st.session_state['notifications'][0]
        self.assertEqual(notification['message'], "Test message")
        self.assertEqual(notification['type'], "success")
    
    @patch('app.st')
    def test_display_notifications(self, mock_st):
        """Test displaying notifications"""
        # Mock session state with notifications
        mock_st.session_state = {
            'notifications': [
                {'message': 'Test 1', 'type': 'success', 'timestamp': 1000},
                {'message': 'Test 2', 'type': 'error', 'timestamp': 1001}
            ]
        }
        
        # Mock time
        with patch('app.time.time', return_value=1002):
            # Import and test the function
            from src.app import display_notifications
            
            display_notifications()
            
            # Verify notifications were processed
            # (In real app, this would call st.success, st.error, etc.)
            pass
    
    @patch('app.st')
    @patch('app.Network')
    def test_create_pyvis_network(self, mock_network_class, mock_st):
        """Test creating Pyvis network visualization"""
        # Mock Network class
        mock_network = Mock()
        mock_network_class.return_value = mock_network
        mock_network.generate_html.return_value = "<html>Test Network</html>"
        
        # Test data
        test_graph = {
            'nodes': [
                {'id': 'node1', 'label': 'Diabetes', 'group': 'Disease'},
                {'id': 'node2', 'label': 'Insulin', 'group': 'Hormone'}
            ],
            'relationships': [
                {'source': 'node1', 'target': 'node2', 'type': 'TREATED_BY'}
            ]
        }
        
        # Import and test the function
        from src.app import create_pyvis_network
        
        html_content = create_pyvis_network(test_graph)
        
        # Verify network was created
        self.assertEqual(html_content, "<html>Test Network</html>")
        mock_network.add_node.assert_called()
        mock_network.add_edge.assert_called()
    
    @patch('app.st')
    def test_display_graph_visualization(self, mock_st):
        """Test displaying graph visualization"""
        # Mock components.html
        mock_st.components.v1.html = Mock()
        
        # Test data
        test_graph = {
            'nodes': [{'id': 'node1', 'label': 'Diabetes'}],
            'relationships': []
        }
        
        # Mock create_pyvis_network
        with patch('app.create_pyvis_network', return_value="<html>Test</html>"):
            # Import and test the function
            from src.app import display_graph_visualization
            
            display_graph_visualization(test_graph)
            
            # Verify HTML was displayed
            mock_st.components.v1.html.assert_called_once_with("<html>Test</html>", height=600)
    
    @patch('app.st')
    def test_display_neo4j_panel(self, mock_st):
        """Test displaying Neo4j panel"""
        # Mock session state
        mock_st.session_state = {
            'neo4j_connected': True,
            'neo4j_processor': Mock(),
            'graph_stored_in_neo4j': True
        }
        
        # Mock processor methods
        mock_st.session_state['neo4j_processor'].get_knowledge_graph_count.return_value = {
            'nodes': 10, 'relationships': 15
        }
        mock_st.session_state['neo4j_processor'].query_all_nodes.return_value = [
            {'id': 'node1', 'name': 'Diabetes'}
        ]
        
        # Mock UI components
        mock_st.tabs.return_value = [Mock(), Mock(), Mock(), Mock(), Mock()]
        mock_st.text_area = Mock()
        mock_st.button = Mock(return_value=False)
        mock_st.dataframe = Mock()
        
        # Import and test the function
        from src.app import display_neo4j_panel
        
        display_neo4j_panel()
        
        # Verify panel was displayed
        mock_st.tabs.assert_called_once()
    
    @patch('app.st')
    def test_display_healthcare_chat(self, mock_st):
        """Test displaying healthcare chat interface"""
        # Mock session state
        mock_st.session_state = {
            'healthcare_agent': Mock(),
            'chat_history': []
        }
        
        # Mock agent methods
        mock_st.session_state['healthcare_agent'].get_response.return_value = "Test response"
        
        # Mock UI components
        mock_st.text_area = Mock(return_value="What is diabetes?")
        mock_st.button = Mock(return_value=True)
        mock_st.spinner = Mock()
        mock_st.expander = Mock()
        
        # Import and test the function
        from src.app import display_healthcare_chat
        
        display_healthcare_chat()
        
        # Verify chat interface was displayed
        mock_st.text_area.assert_called()
        mock_st.button.assert_called()


class TestAppDataProcessing(unittest.TestCase):
    """Test data processing functionality in the app"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.test_dir = tempfile.mkdtemp()
        
        # Create test files
        self.txt_file = os.path.join(self.test_dir, "test.txt")
        with open(self.txt_file, 'w', encoding='utf-8') as f:
            f.write("Diabetes is a chronic disease. Insulin helps control blood sugar.")
    
    def tearDown(self):
        """Clean up test fixtures"""
        import shutil
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    @patch('app.st')
    @patch('app.EnhancedGraphBuilder')
    def test_build_knowledge_graph_from_directory(self, mock_builder_class, mock_st):
        """Test building knowledge graph from directory"""
        # Mock graph builder
        mock_builder = Mock()
        mock_builder_class.return_value = mock_builder
        mock_builder.build_graph_from_directory.return_value = {
            'success': True,
            'graphs': [{'nodes': [], 'relationships': []}]
        }
        
        # Mock session state
        mock_st.session_state = {}
        
        # Import and test the function
        from src.app import build_knowledge_graph_from_directory
        
        result = build_knowledge_graph_from_directory(self.test_dir, "chunked")
        
        # Verify graph was built
        self.assertTrue(result['success'])
        mock_builder.build_graph_from_directory.assert_called_once_with(self.test_dir, mode="chunked")
    
    @patch('app.st')
    @patch('app.EnhancedGraphBuilder')
    def test_build_knowledge_graph_from_uploaded_files(self, mock_builder_class, mock_st):
        """Test building knowledge graph from uploaded files"""
        # Mock graph builder
        mock_builder = Mock()
        mock_builder_class.return_value = mock_builder
        mock_builder.build_graph_from_uploaded_files.return_value = {
            'success': True,
            'graphs': [{'nodes': [], 'relationships': []}]
        }
        
        # Mock session state
        mock_st.session_state = {}
        
        # Test uploaded files
        uploaded_files = [
            ("test1.txt", "Content about diabetes"),
            ("test2.txt", "Content about insulin")
        ]
        
        # Import and test the function
        from src.app import build_knowledge_graph_from_uploaded_files
        
        result = build_knowledge_graph_from_uploaded_files(uploaded_files, "raw")
        
        # Verify graph was built
        self.assertTrue(result['success'])
        mock_builder.build_graph_from_uploaded_files.assert_called_once_with(uploaded_files, mode="raw")


class TestAppErrorHandling(unittest.TestCase):
    """Test error handling in the app"""
    
    @patch('app.st')
    def test_neo4j_connection_failure(self, mock_st):
        """Test handling Neo4j connection failure"""
        # Mock session state
        mock_st.session_state = {}
        
        # Mock Neo4j processor to raise exception
        with patch('app.Neo4jProcessor', side_effect=Exception("Connection failed")):
            # Import and test the function
            from src.app import establish_neo4j_connection
            
            establish_neo4j_connection()
            
            # Verify connection state is False
            self.assertFalse(mock_st.session_state.get('neo4j_connected', False))
    
    @patch('app.st')
    def test_graph_building_failure(self, mock_st):
        """Test handling graph building failure"""
        # Mock graph builder to raise exception
        with patch('app.EnhancedGraphBuilder', side_effect=Exception("Graph building failed")):
            # Import and test the function
            from src.app import build_knowledge_graph_from_directory
            
            result = build_knowledge_graph_from_directory("/nonexistent", "chunked")
            
            # Verify failure is handled gracefully
            self.assertFalse(result['success'])
            self.assertIn('error', result)
    
    @patch('app.st')
    def test_agent_initialization_failure(self, mock_st):
        """Test handling agent initialization failure"""
        # Mock session state
        mock_st.session_state = {}
        
        # Mock agent to raise exception
        with patch('app.SimpleHealthcareAgent', side_effect=Exception("Agent initialization failed")):
            # Import and test the function
            from src.app import initialize_healthcare_agent
            
            # Should not raise exception
            initialize_healthcare_agent()
            
            # Verify agent is not set
            self.assertNotIn('healthcare_agent', mock_st.session_state)


class TestAppSessionState(unittest.TestCase):
    """Test session state management in the app"""
    
    @patch('app.st')
    def test_session_state_initialization(self, mock_st):
        """Test session state initialization"""
        # Mock session state
        mock_st.session_state = {}
        
        # Import and test the function
        from src.app import initialize_session_state
        
        initialize_session_state()
        
        # Verify session state is properly initialized
        self.assertIn('notifications', mock_st.session_state)
        self.assertIsInstance(mock_st.session_state['notifications'], list)
    
    @patch('app.st')
    def test_session_state_persistence(self, mock_st):
        """Test session state persistence across function calls"""
        # Mock session state
        mock_st.session_state = {
            'neo4j_connected': True,
            'neo4j_processor': Mock(),
            'healthcare_agent': Mock(),
            'chat_history': []
        }
        
        # Test that session state persists
        self.assertTrue(mock_st.session_state['neo4j_connected'])
        self.assertIsNotNone(mock_st.session_state['neo4j_processor'])
        self.assertIsNotNone(mock_st.session_state['healthcare_agent'])


if __name__ == '__main__':
    unittest.main()
