#!/usr/bin/env python3
"""
Simplified Healthcare Q&A Agent using Neo4j Knowledge Graph
This agent queries the knowledge graph to provide context-aware healthcare responses.
"""

import os
import logging
from typing import List, Dict, Any, Optional
from gemini import llm

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleHealthcareAgent:
    """Simplified Healthcare Q&A Agent that uses Neo4j knowledge graph for context."""
    
    def __init__(self, neo4j_processor=None):
        """
        Initialize the healthcare agent.
        
        Args:
            neo4j_processor: Optional Neo4j processor instance
        """
        self.neo4j_processor = neo4j_processor
        # Use the pre-initialized LLM from gemini.py
        self.llm = llm
        self.chat_history = []
        
        # Define the prompt template for healthcare Q&A
        self.prompt_template = """
You are a knowledgeable healthcare assistant with access to a medical knowledge graph. 
Your role is to provide accurate, helpful, and safe healthcare information based on the context provided.

IMPORTANT DISCLAIMERS:
- This is for informational purposes only and should not replace professional medical advice
- Always consult with qualified healthcare professionals for medical decisions
- If you're unsure about something, recommend consulting a healthcare provider

CONTEXT FROM KNOWLEDGE GRAPH:
{context}

CHAT HISTORY:
{chat_history}

CURRENT QUESTION: {question}

Please provide a comprehensive, accurate, and helpful response based on the context from the knowledge graph. 

INSTRUCTIONS:
1. If the context contains relevant information, use it to provide a detailed, accurate response
2. If the context doesn't contain specific information about the question, acknowledge this clearly
3. Always be clear about what information comes from the knowledge graph vs. general medical knowledge
4. Include appropriate medical disclaimers when necessary
5. Encourage professional consultation for serious health concerns
6. Be empathetic, supportive, and easy to understand
7. If the knowledge graph has limited information, suggest consulting healthcare professionals for more detailed guidance

RESPONSE FORMAT:
- Start with a clear answer to the question
- Reference specific information from the knowledge graph when available
- Include relevant medical disclaimers
- End with a supportive note about consulting healthcare professionals if needed

RESPONSE:
"""
    
    def query_knowledge_graph(self, question: str) -> str:
        """
        Query the Neo4j knowledge graph for relevant context.
        
        Args:
            question: The user's healthcare question
            
        Returns:
            Context string from the knowledge graph
        """
        if not self.neo4j_processor or not hasattr(self.neo4j_processor, 'graph') or not self.neo4j_processor.graph:
            return "No knowledge graph available for context."
        
        try:
            # Extract key terms from the question for better search
            question_lower = question.lower()
            
            # Define medical keywords to search for
            medical_keywords = [
                'diabetes', 'blood sugar', 'insulin', 'glucose', 'hyperglycemia', 'hypoglycemia',
                'diet', 'nutrition', 'food', 'meal', 'carbohydrate', 'protein', 'fat',
                'exercise', 'physical activity', 'workout', 'fitness',
                'medication', 'drug', 'treatment', 'therapy', 'medicine',
                'symptom', 'condition', 'disease', 'illness', 'disorder',
                'prevention', 'management', 'control', 'monitoring',
                'testing', 'check', 'measure', 'blood test', 'a1c',
                'complication', 'risk', 'damage', 'organ', 'heart', 'kidney', 'eye', 'nerve'
            ]
            
            # Find relevant keywords in the question
            relevant_keywords = [kw for kw in medical_keywords if kw in question_lower]
            
            context_parts = []
            
            # First, try to find specific information about the question
            if relevant_keywords:
                for keyword in relevant_keywords[:5]:  # Limit to top 5 keywords
                    try:
                        # Search for nodes containing the keyword
                        if hasattr(self.neo4j_processor, 'query_by_keyword'):
                            nodes = self.neo4j_processor.query_by_keyword(
                                keyword=keyword,
                                limit=10
                            )
                            
                            if nodes:
                                context_parts.append(f"\n📋 Information about '{keyword}':")
                                for node in nodes:
                                    node_id = node.get('id', 'Unknown')
                                    node_type = node.get('type', 'Unknown')
                                    properties = node.get('properties', {})
                                    
                                    # Extract meaningful properties
                                    description = properties.get('description', '')
                                    name = properties.get('name', '')
                                    content = properties.get('content', '')
                                    text = properties.get('text', '')
                                    
                                    # Use the most relevant property
                                    info = description or name or content or text or str(node_id)
                                    if info and len(info) > 10:  # Only include meaningful content
                                        context_parts.append(f"• {node_type}: {info[:300]}...")
                    
                    except Exception as e:
                        logger.warning(f"Error searching for keyword '{keyword}': {e}")
                        continue
            
            # Search for relationships that might be relevant
            try:
                if hasattr(self.neo4j_processor, 'query_relationships'):
                    relationships = self.neo4j_processor.query_relationships(limit=20)
                    
                    relevant_rels = []
                    for rel in relationships:
                        source = rel.get('source', {}).get('id', '')
                        target = rel.get('target', {}).get('id', '')
                        rel_type = rel.get('relationship', {}).get('type', '')
                        
                        # Check if any part contains relevant keywords
                        rel_text = f"{source} {target} {rel_type}".lower()
                        if any(kw in rel_text for kw in relevant_keywords):
                            relevant_rels.append(rel)
                    
                    if relevant_rels:
                        context_parts.append(f"\n🔗 Related relationships:")
                        for rel in relevant_rels[:5]:
                            source = rel.get('source', {}).get('id', 'Unknown')
                            target = rel.get('target', {}).get('id', 'Unknown')
                            rel_type = rel.get('relationship', {}).get('type', 'RELATES_TO')
                            context_parts.append(f"• {source} --[{rel_type}]--> {target}")
            
            except Exception as e:
                logger.warning(f"Error searching for relationships: {e}")
            
            # If no specific keywords found, search for general medical concepts
            if not context_parts:
                try:
                    # Search for general medical concepts
                    if hasattr(self.neo4j_processor, 'query_nodes_by_type'):
                        for node_type in ["Concept", "Condition", "Treatment", "Symptom"]:
                            concepts = self.neo4j_processor.query_nodes_by_type(
                                node_type=node_type,
                                limit=5
                            )
                            
                            if concepts:
                                context_parts.append(f"\n📚 General {node_type.lower()} information:")
                                for concept in concepts:
                                    concept_id = concept.get('id', 'Unknown')
                                    properties = concept.get('properties', {})
                                    description = properties.get('description', '')
                                    name = properties.get('name', '')
                                    
                                    info = description or name or concept_id
                                    if info and len(info) > 10:
                                        context_parts.append(f"• {info[:200]}...")
                
                except Exception as e:
                    logger.warning(f"Error searching for general concepts: {e}")
            
            # Try to get some sample data to understand what's in the graph
            if not context_parts:
                try:
                    # Get a sample of all nodes to understand the data structure
                    if hasattr(self.neo4j_processor, 'query_all_nodes'):
                        sample_nodes = self.neo4j_processor.query_all_nodes(limit=10)
                        
                        if sample_nodes:
                            context_parts.append(f"\n📊 Available knowledge base contains:")
                            node_types = {}
                            for node in sample_nodes:
                                node_type = node.get('type', 'Unknown')
                                node_types[node_type] = node_types.get(node_type, 0) + 1
                            
                            for node_type, count in node_types.items():
                                context_parts.append(f"• {count} {node_type} nodes")
                
                except Exception as e:
                    logger.warning(f"Error getting sample data: {e}")
            
            if context_parts:
                return "\n".join(context_parts)
            else:
                return "No specific context found in the knowledge graph for this question. The knowledge base may not contain detailed information about this topic."
                
        except Exception as e:
            logger.error(f"Error querying knowledge graph: {e}")
            return f"Error accessing knowledge graph: {str(e)}"
    
    def get_response(self, question: str) -> str:
        """
        Get a response to a healthcare question using the knowledge graph.
        
        Args:
            question: The user's healthcare question
            
        Returns:
            The agent's response
        """
        try:
            # Query the knowledge graph for context
            context = self.query_knowledge_graph(question)
            
            # Format chat history
            chat_history_text = ""
            if self.chat_history:
                chat_history_text = "\n".join([
                    f"Q: {item['question']}\nA: {item['response']}" 
                    for item in self.chat_history[-3:]  # Last 3 exchanges
                ])
            
            # Create the full prompt
            full_prompt = self.prompt_template.format(
                question=question,
                context=context,
                chat_history=chat_history_text
            )
            
            # Generate response using the LLM
            response = self.llm.invoke(full_prompt)
            
            # Add to chat history
            self.chat_history.append({
                "question": question,
                "response": response
            })
            
            return response.strip()
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return f"I apologize, but I encountered an error while processing your question. Please try again or consult a healthcare professional for immediate assistance. Error: {str(e)}"
    
    def get_chat_history(self) -> List[Dict[str, str]]:
        """
        Get the conversation history.
        
        Returns:
            List of conversation turns
        """
        return self.chat_history
    
    def clear_memory(self):
        """Clear the conversation memory."""
        self.chat_history = []
    
    def set_neo4j_processor(self, processor):
        """
        Set the Neo4j processor for the agent.
        
        Args:
            processor: Neo4j processor instance
        """
        self.neo4j_processor = processor

# Utility functions for the agent
def create_simple_healthcare_agent(neo4j_processor=None):
    """
    Create a simplified healthcare agent instance.
    
    Args:
        neo4j_processor: Optional Neo4j processor instance
        
    Returns:
        SimpleHealthcareAgent instance
    """
    return SimpleHealthcareAgent(neo4j_processor)

def test_simple_healthcare_agent():
    """Test the simplified healthcare agent functionality."""
    print("Testing Simplified Healthcare Agent...")
    
    # Create agent without Neo4j for basic testing
    agent = create_simple_healthcare_agent()
    
    # Test questions
    test_questions = [
        "What is diabetes?",
        "How can I manage my blood sugar levels?",
        "What are the symptoms of diabetes?",
        "What diet is recommended for diabetes patients?"
    ]
    
    for question in test_questions:
        print(f"\nQuestion: {question}")
        try:
            response = agent.get_response(question)
            print(f"Response: {response[:200]}...")
        except Exception as e:
            print(f"Error: {e}")
    
    print("\nSimplified Healthcare Agent test completed!")

if __name__ == "__main__":
    test_simple_healthcare_agent()
