#!/usr/bin/env python3
"""
Test suite for agent_simple.py
Tests healthcare agent functionality, knowledge graph querying, and conversation management
"""

import unittest
import sys
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Add parent directory to path to import modules
sys.path.append(str(Path(__file__).parent.parent))

from src.agent_simple import SimpleHealthcareAgent, create_simple_healthcare_agent


class TestSimpleHealthcareAgent(unittest.TestCase):
    """Test cases for SimpleHealthcareAgent class"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Mock LLM
        self.mock_llm = Mock()
        self.mock_llm.invoke.return_value = "Mock LLM response about diabetes"
        
        # Mock Neo4j processor
        self.mock_neo4j_processor = Mock()
        
        # Create agent with mocked dependencies
        self.agent = SimpleHealthcareAgent(neo4j_processor=self.mock_neo4j_processor)
        self.agent.llm = self.mock_llm
    
    def test_initialization(self):
        """Test SimpleHealthcareAgent initialization"""
        agent = SimpleHealthcareAgent()
        self.assertIsNotNone(agent)
        self.assertIsNotNone(agent.llm)
        self.assertEqual(agent.chat_history, [])
        self.assertIn("healthcare assistant", agent.prompt_template.lower())
    
    def test_initialization_with_neo4j_processor(self):
        """Test initialization with Neo4j processor"""
        agent = SimpleHealthcareAgent(neo4j_processor=self.mock_neo4j_processor)
        self.assertEqual(agent.neo4j_processor, self.mock_neo4j_processor)
    
    def test_set_neo4j_processor(self):
        """Test setting Neo4j processor"""
        new_processor = Mock()
        self.agent.set_neo4j_processor(new_processor)
        self.assertEqual(self.agent.neo4j_processor, new_processor)
    
    def test_query_knowledge_graph_with_processor(self):
        """Test querying knowledge graph when processor is available"""
        # Mock query results
        mock_nodes = [
            {'id': 'diabetes', 'name': 'Diabetes', 'description': 'A chronic disease'},
            {'id': 'insulin', 'name': 'Insulin', 'description': 'A hormone'}
        ]
        self.mock_neo4j_processor.query_by_keyword.return_value = mock_nodes
        
        context = self.agent.query_knowledge_graph("What is diabetes?")
        
        self.assertIsInstance(context, str)
        self.assertIn("diabetes", context.lower())
        self.assertIn("chronic disease", context.lower())
        self.mock_neo4j_processor.query_by_keyword.assert_called_once()
    
    def test_query_knowledge_graph_no_processor(self):
        """Test querying knowledge graph when no processor is available"""
        agent = SimpleHealthcareAgent(neo4j_processor=None)
        
        context = agent.query_knowledge_graph("What is diabetes?")
        
        self.assertEqual(context, "No knowledge graph available for context.")
    
    def test_query_knowledge_graph_no_graph(self):
        """Test querying knowledge graph when graph is not available"""
        self.mock_neo4j_processor.graph = None
        
        context = self.agent.query_knowledge_graph("What is diabetes?")
        
        self.assertEqual(context, "No knowledge graph available for context.")
    
    def test_query_knowledge_graph_medical_keywords(self):
        """Test that medical keywords are properly extracted and used"""
        # Mock query results
        mock_nodes = [
            {'id': 'diabetes', 'name': 'Diabetes', 'description': 'A chronic disease'}
        ]
        self.mock_neo4j_processor.query_by_keyword.return_value = mock_nodes
        
        context = self.agent.query_knowledge_graph("What is diabetes and how does it affect blood sugar?")
        
        # Should query for diabetes and blood sugar
        self.mock_neo4j_processor.query_by_keyword.assert_called()
        call_args = self.mock_neo4j_processor.query_by_keyword.call_args[0]
        self.assertIn("diabetes", call_args[0].lower())
    
    def test_get_response_success(self):
        """Test successful response generation"""
        # Mock knowledge graph query
        self.mock_neo4j_processor.query_by_keyword.return_value = [
            {'id': 'diabetes', 'name': 'Diabetes', 'description': 'A chronic disease'}
        ]
        
        response = self.agent.get_response("What is diabetes?")
        
        self.assertIsInstance(response, str)
        self.assertIn("diabetes", response.lower())
        self.mock_llm.invoke.assert_called_once()
        
        # Check that question was added to chat history
        self.assertEqual(len(self.agent.chat_history), 1)
        self.assertEqual(self.agent.chat_history[0]['question'], "What is diabetes?")
    
    def test_get_response_with_chat_history(self):
        """Test response generation with existing chat history"""
        # Add some chat history
        self.agent.chat_history = [
            {'question': 'What is diabetes?', 'response': 'Diabetes is a chronic disease.'},
            {'question': 'What are the symptoms?', 'response': 'Symptoms include frequent urination.'}
        ]
        
        # Mock knowledge graph query
        self.mock_neo4j_processor.query_by_keyword.return_value = [
            {'id': 'diabetes', 'name': 'Diabetes', 'description': 'A chronic disease'}
        ]
        
        response = self.agent.get_response("How is it treated?")
        
        self.assertIsInstance(response, str)
        self.assertEqual(len(self.agent.chat_history), 3)
        
        # Check that LLM was called with chat history
        llm_call_args = self.mock_llm.invoke.call_args[0][0]
        self.assertIn("What is diabetes?", llm_call_args)
        self.assertIn("Symptoms include frequent urination", llm_call_args)
    
    def test_get_response_exception_handling(self):
        """Test handling exceptions during response generation"""
        # Mock LLM to raise exception
        self.mock_llm.invoke.side_effect = Exception("LLM Error")
        
        response = self.agent.get_response("What is diabetes?")
        
        self.assertIsInstance(response, str)
        self.assertIn("error", response.lower())
        self.assertIn("apologize", response.lower())
    
    def test_get_chat_history(self):
        """Test getting chat history"""
        # Add some chat history
        self.agent.chat_history = [
            {'question': 'What is diabetes?', 'response': 'Diabetes is a chronic disease.'},
            {'question': 'What are the symptoms?', 'response': 'Symptoms include frequent urination.'}
        ]
        
        history = self.agent.get_chat_history()
        
        self.assertEqual(history, self.agent.chat_history)
        self.assertEqual(len(history), 2)
    
    def test_clear_memory(self):
        """Test clearing chat memory"""
        # Add some chat history
        self.agent.chat_history = [
            {'question': 'What is diabetes?', 'response': 'Diabetes is a chronic disease.'}
        ]
        
        self.agent.clear_memory()
        
        self.assertEqual(self.agent.chat_history, [])
    
    def test_prompt_template_formatting(self):
        """Test that prompt template is properly formatted"""
        # Mock knowledge graph query
        self.mock_neo4j_processor.query_by_keyword.return_value = [
            {'id': 'diabetes', 'name': 'Diabetes', 'description': 'A chronic disease'}
        ]
        
        self.agent.get_response("What is diabetes?")
        
        # Check that LLM was called with properly formatted prompt
        llm_call_args = self.mock_llm.invoke.call_args[0][0]
        self.assertIn("CURRENT QUESTION: What is diabetes?", llm_call_args)
        self.assertIn("CONTEXT FROM KNOWLEDGE GRAPH:", llm_call_args)
    
    def test_medical_keyword_extraction(self):
        """Test extraction of medical keywords from questions"""
        # Test various medical questions
        test_cases = [
            ("What is diabetes?", ["diabetes"]),
            ("How does insulin work?", ["insulin"]),
            ("What are the symptoms of high blood pressure?", ["blood pressure", "symptoms"]),
            ("Tell me about heart disease and cholesterol", ["heart disease", "cholesterol"]),
            ("What is the treatment for type 2 diabetes?", ["type 2 diabetes", "treatment"])
        ]
        
        for question, expected_keywords in test_cases:
            # Mock query results
            self.mock_neo4j_processor.query_by_keyword.return_value = []
            
            self.agent.query_knowledge_graph(question)
            
            # Check that query was called with appropriate keywords
            call_args = self.mock_neo4j_processor.query_by_keyword.call_args[0]
            query_keyword = call_args[0].lower()
            
            # At least one expected keyword should be in the query
            found_keyword = any(keyword in query_keyword for keyword in expected_keywords)
            self.assertTrue(found_keyword, f"Expected keywords {expected_keywords} not found in query '{query_keyword}'")


class TestCreateSimpleHealthcareAgent(unittest.TestCase):
    """Test cases for create_simple_healthcare_agent function"""
    
    def test_create_agent_with_processor(self):
        """Test creating agent with Neo4j processor"""
        mock_processor = Mock()
        agent = create_simple_healthcare_agent(mock_processor)
        
        self.assertIsInstance(agent, SimpleHealthcareAgent)
        self.assertEqual(agent.neo4j_processor, mock_processor)
    
    def test_create_agent_without_processor(self):
        """Test creating agent without Neo4j processor"""
        agent = create_simple_healthcare_agent()
        
        self.assertIsInstance(agent, SimpleHealthcareAgent)
        self.assertIsNone(agent.neo4j_processor)


class TestAgentIntegration(unittest.TestCase):
    """Integration tests for the healthcare agent"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.mock_llm = Mock()
        self.mock_llm.invoke.return_value = "Based on the knowledge graph, diabetes is a chronic disease that affects blood sugar levels."
        
        self.mock_neo4j_processor = Mock()
        
        self.agent = SimpleHealthcareAgent(neo4j_processor=self.mock_neo4j_processor)
        self.agent.llm = self.mock_llm
    
    def test_full_conversation_flow(self):
        """Test a full conversation flow with the agent"""
        # Mock knowledge graph responses
        self.mock_neo4j_processor.query_by_keyword.side_effect = [
            [{'id': 'diabetes', 'name': 'Diabetes', 'description': 'A chronic disease affecting blood sugar'}],
            [{'id': 'insulin', 'name': 'Insulin', 'description': 'A hormone that regulates blood glucose'}],
            [{'id': 'treatment', 'name': 'Treatment', 'description': 'Includes diet, exercise, and medication'}]
        ]
        
        # Simulate a conversation
        response1 = self.agent.get_response("What is diabetes?")
        response2 = self.agent.get_response("How does insulin work?")
        response3 = self.agent.get_response("What are the treatment options?")
        
        # Verify responses
        self.assertIsInstance(response1, str)
        self.assertIsInstance(response2, str)
        self.assertIsInstance(response3, str)
        
        # Verify chat history
        self.assertEqual(len(self.agent.chat_history), 3)
        self.assertEqual(self.agent.chat_history[0]['question'], "What is diabetes?")
        self.assertEqual(self.agent.chat_history[1]['question'], "How does insulin work?")
        self.assertEqual(self.agent.chat_history[2]['question'], "What are the treatment options?")
        
        # Verify LLM was called for each response
        self.assertEqual(self.mock_llm.invoke.call_count, 3)
    
    def test_context_aware_responses(self):
        """Test that responses are context-aware based on knowledge graph"""
        # Mock different knowledge graph responses
        self.mock_neo4j_processor.query_by_keyword.side_effect = [
            [{'id': 'diabetes', 'name': 'Diabetes', 'description': 'A chronic disease'}],
            []  # No results for second query
        ]
        
        # First question should get context from knowledge graph
        response1 = self.agent.get_response("What is diabetes?")
        
        # Second question should acknowledge lack of context
        response2 = self.agent.get_response("What is a rare genetic disorder?")
        
        # Verify that different contexts led to different LLM calls
        self.assertNotEqual(
            self.mock_llm.invoke.call_args_list[0][0][0],
            self.mock_llm.invoke.call_args_list[1][0][0]
        )
    
    def test_conversation_memory_persistence(self):
        """Test that conversation memory persists across multiple interactions"""
        # Mock knowledge graph responses
        self.mock_neo4j_processor.query_by_keyword.return_value = [
            {'id': 'diabetes', 'name': 'Diabetes', 'description': 'A chronic disease'}
        ]
        
        # First interaction
        self.agent.get_response("What is diabetes?")
        
        # Second interaction should include previous context
        self.agent.get_response("What are its symptoms?")
        
        # Verify that second LLM call includes chat history
        second_call_args = self.mock_llm.invoke.call_args_list[1][0][0]
        self.assertIn("What is diabetes?", second_call_args)
        self.assertIn("What are its symptoms?", second_call_args)


class TestAgentErrorHandling(unittest.TestCase):
    """Test error handling in the healthcare agent"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.mock_llm = Mock()
        self.mock_neo4j_processor = Mock()
        self.agent = SimpleHealthcareAgent(neo4j_processor=self.mock_neo4j_processor)
        self.agent.llm = self.mock_llm
    
    def test_neo4j_query_exception_handling(self):
        """Test handling Neo4j query exceptions"""
        # Mock Neo4j query to raise exception
        self.mock_neo4j_processor.query_by_keyword.side_effect = Exception("Neo4j Error")
        
        response = self.agent.get_response("What is diabetes?")
        
        # Should still generate a response even if Neo4j fails
        self.assertIsInstance(response, str)
        self.assertIn("diabetes", response.lower())
    
    def test_empty_question_handling(self):
        """Test handling empty questions"""
        response = self.agent.get_response("")
        
        self.assertIsInstance(response, str)
        # Should still generate a response
    
    def test_very_long_question_handling(self):
        """Test handling very long questions"""
        long_question = "What is diabetes? " * 100  # Very long question
        
        # Mock knowledge graph query
        self.mock_neo4j_processor.query_by_keyword.return_value = [
            {'id': 'diabetes', 'name': 'Diabetes', 'description': 'A chronic disease'}
        ]
        
        response = self.agent.get_response(long_question)
        
        self.assertIsInstance(response, str)
        # Should handle long questions gracefully


if __name__ == '__main__':
    unittest.main()
