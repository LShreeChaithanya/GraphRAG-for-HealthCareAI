#!/usr/bin/env python3
"""
Enhanced Graph Builder using LLMGraphTransformer with support for both chunked and raw text modes.
"""

import logging
from pathlib import Path
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_core.documents import Document
from gemini import llm
from data_processors import DataProcessor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedGraphBuilder:
    """Enhanced Knowledge Graph Builder with mode selection."""
    
    def __init__(self, chunk_size: int = 200, chunk_overlap: int = 30):
        """
        Initialize the Enhanced Graph Builder.
        
        Args:
            chunk_size: Size of text chunks for chunked mode
            chunk_overlap: Overlap between chunks for chunked mode
        """
        self.processor = DataProcessor(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        
        # Initialize LLMGraphTransformer
        self.graph_transformer = LLMGraphTransformer(
            llm=llm,
            allowed_nodes=[
                "Concept", "Entity", "Topic", "Process", "Method", "Tool", 
                "Condition", "Symptom", "Treatment", "Person", "Organization", 
                "Location", "Event", "Technology", "Product", "Service"
            ],
            allowed_relationships=[
                "RELATES_TO", "IS_A", "PART_OF", "CAUSES", "TREATS", "REQUIRES", 
                "FOLLOWS", "CONTAINS", "SIMILAR_TO", "LOCATED_IN", "WORKS_FOR",
                "CREATES", "USES", "INFLUENCES", "DEPENDS_ON", "BELONGS_TO"
            ]
        )
    
    def build_graph_from_directory(self, data_dir: str = "data", mode: str = "chunked") -> dict:
        """
        Build knowledge graph from files in directory.
        
        Args:
            data_dir: Directory containing data files
            mode: Processing mode - "chunked" or "raw"
            
        Returns:
            Dictionary containing graph results
        """
        print("="*60)
        print(f"KNOWLEDGE GRAPH BUILDER - {mode.upper()} MODE")
        print("="*60)
        
        try:
            # Step 1: Process files
            print(f"\n1. Processing files in {mode} mode...")
            self.processor.data_dir = Path(data_dir) if isinstance(data_dir, str) else data_dir
            documents = self.processor.process_directory(mode=mode)
            
            if not documents:
                print("❌ No documents processed. Please check your data directory.")
                return {"success": False, "error": "No documents processed"}
            
            print(f"✅ Successfully processed {len(documents)} document(s)")
            
            # Step 2: Convert to graph
            return self._convert_documents_to_graph(documents, mode)
            
        except Exception as e:
            print(f"\n❌ Error in build_graph_from_directory: {e}")
            import traceback
            print(f"Traceback: {traceback.format_exc()}")
            return {"success": False, "error": str(e)}
    
    def build_graph_from_uploaded_files(self, uploaded_files, mode: str = "chunked") -> dict:
        """
        Build knowledge graph from uploaded files.
        
        Args:
            uploaded_files: List of uploaded file objects
            mode: Processing mode - "chunked" or "raw"
            
        Returns:
            Dictionary containing graph results
        """
        print("="*60)
        print(f"KNOWLEDGE GRAPH BUILDER - {mode.upper()} MODE (UPLOADED FILES)")
        print("="*60)
        
        try:
            # Step 1: Process uploaded files
            print(f"\n1. Processing {len(uploaded_files)} uploaded file(s) in {mode} mode...")
            documents = self.processor.process_multiple_uploaded_files(uploaded_files, mode=mode)
            
            if not documents:
                print("❌ No documents processed from uploaded files.")
                return {"success": False, "error": "No documents processed from uploaded files"}
            
            print(f"✅ Successfully processed {len(documents)} document(s)")
            
            # Step 2: Convert to graph
            return self._convert_documents_to_graph(documents, mode)
            
        except Exception as e:
            print(f"\n❌ Error in build_graph_from_uploaded_files: {e}")
            import traceback
            print(f"Traceback: {traceback.format_exc()}")
            return {"success": False, "error": str(e)}
    
    def _convert_documents_to_graph(self, documents: list, mode: str) -> dict:
        """
        Convert documents to knowledge graph.
        
        Args:
            documents: List of Document objects
            mode: Processing mode
            
        Returns:
            Dictionary containing graph results
        """
        print(f"\n2. Converting document(s) to knowledge graph...")
        print(f"Processing {len(documents)} document(s) in {mode} mode...")
        
        all_graphs = []
        
        if mode == "raw":
            # For raw mode, process the single combined document
            try:
                print("Processing combined document...")
                graph_documents = self.graph_transformer.convert_to_graph_documents(documents)
                
                if graph_documents and len(graph_documents) > 0:
                    for i, graph_doc in enumerate(graph_documents):
                        if hasattr(graph_doc, 'nodes') and hasattr(graph_doc, 'relationships'):
                            nodes = getattr(graph_doc, 'nodes', [])
                            relationships = getattr(graph_doc, 'relationships', [])
                            
                            if nodes or relationships:
                                # Convert to serializable format immediately
                                serializable_nodes = self._serialize_nodes(nodes)
                                serializable_relationships = self._serialize_relationships(relationships)
                                
                                graph_data = {
                                    'nodes': serializable_nodes,
                                    'relationships': serializable_relationships,
                                    'source_document': documents[i].metadata.get('file_name', 'Combined Document'),
                                    'processing_mode': 'raw',
                                    'source_files': documents[i].metadata.get('source_files', [])
                                }
                                all_graphs.append(graph_data)
                                print(f"  ✅ Combined Document: {len(serializable_nodes)} nodes, {len(serializable_relationships)} relationships")
                            else:
                                print(f"  ⚠️ Combined Document: No nodes or relationships extracted")
                else:
                    print(f"  ❌ No graph data generated from combined document")
                    
            except Exception as e:
                print(f"  ❌ Error processing combined document: {e}")
        
        else:  # chunked mode
            # Process documents in smaller batches to avoid token limits
            batch_size = 3
            
            for i in range(0, len(documents), batch_size):
                batch = documents[i:i+batch_size]
                print(f"Processing batch {i//batch_size + 1}: documents {i+1}-{min(i+batch_size, len(documents))}")
                
                try:
                    graph_documents = self.graph_transformer.convert_to_graph_documents(batch)
                    
                    if graph_documents and len(graph_documents) > 0:
                        for j, graph_doc in enumerate(graph_documents):
                            if hasattr(graph_doc, 'nodes') and hasattr(graph_doc, 'relationships'):
                                nodes = getattr(graph_doc, 'nodes', [])
                                relationships = getattr(graph_doc, 'relationships', [])
                                
                                if nodes or relationships:
                                    # Convert to serializable format immediately
                                    serializable_nodes = self._serialize_nodes(nodes)
                                    serializable_relationships = self._serialize_relationships(relationships)
                                    
                                    graph_data = {
                                        'nodes': serializable_nodes,
                                        'relationships': serializable_relationships,
                                        'source_document': batch[j].metadata.get('file_name', 'Unknown'),
                                        'processing_mode': 'chunked',
                                        'chunk_id': batch[j].metadata.get('chunk_id', 0)
                                    }
                                    all_graphs.append(graph_data)
                                    print(f"  ✅ Document {i+j+1}: {len(serializable_nodes)} nodes, {len(serializable_relationships)} relationships")
                                else:
                                    print(f"  ⚠️ Document {i+j+1}: No nodes or relationships extracted")
                            else:
                                print(f"  ⚠️ Document {i+j+1}: Missing nodes or relationships attributes")
                    else:
                        print(f"  ❌ Batch {i//batch_size + 1}: No graph data generated")
                        
                except Exception as e:
                    print(f"  ❌ Error processing batch {i//batch_size + 1}: {e}")
                    continue
        
        # Generate results
        return self._generate_results(all_graphs, mode)
    
    def _generate_results(self, all_graphs: list, mode: str) -> dict:
        """Generate and display results."""
        print("\n" + "="*60)
        print("KNOWLEDGE GRAPH RESULTS")
        print("="*60)
        
        if all_graphs:
            total_nodes = sum(len(g.get('nodes', [])) for g in all_graphs)
            total_relationships = sum(len(g.get('relationships', [])) for g in all_graphs)
            
            print(f"\n✅ Successfully created knowledge graph in {mode} mode!")
            print(f"Total graphs: {len(all_graphs)}")
            print(f"Total nodes: {total_nodes}")
            print(f"Total relationships: {total_relationships}")
            
            # Show details for each graph
            for i, graph in enumerate(all_graphs):
                source = graph.get('source_document', 'Unknown')
                if mode == "raw" and 'source_files' in graph:
                    source += f" (Combined from: {', '.join(graph['source_files'])})"
                elif mode == "chunked":
                    chunk_id = graph.get('chunk_id', 0)
                    source += f" (Chunk {chunk_id})"
                
                print(f"\nGraph {i+1} ({source}):")
                print(f"  Nodes: {len(graph.get('nodes', []))}")
                print(f"  Relationships: {len(graph.get('relationships', []))}")
                
                # Show sample nodes and relationships
                self._show_sample_data(graph)
            
            # Save results
            self._save_results(all_graphs, mode)
            
            return {
                "success": True,
                "graphs": all_graphs,
                "summary": {
                    "total_graphs": len(all_graphs),
                    "total_nodes": total_nodes,
                    "total_relationships": total_relationships,
                    "processing_mode": mode
                }
            }
            
        else:
            print(f"\n❌ No knowledge graphs were successfully created in {mode} mode.")
            print("This might be due to:")
            print("- LLM not responding properly")
            print("- Text content not suitable for entity extraction")
            print("- Token limits exceeded")
            print("- LLMGraphTransformer configuration issues")
            
            return {
                "success": False,
                "error": "No knowledge graphs created",
                "graphs": [],
                "summary": {
                    "total_graphs": 0,
                    "total_nodes": 0,
                    "total_relationships": 0,
                    "processing_mode": mode
                }
            }
    
    def _serialize_nodes(self, nodes: list) -> list:
        """Convert LangChain Node objects to serializable dictionaries."""
        serializable_nodes = []
        for node in nodes:
            if hasattr(node, 'id'):  # LangChain Node object
                serializable_nodes.append({
                    'id': getattr(node, 'id', ''),
                    'type': getattr(node, 'type', ''),
                    'properties': getattr(node, 'properties', {})
                })
            elif isinstance(node, dict):  # Already a dictionary
                serializable_nodes.append({
                    'id': node.get('id', ''),
                    'type': node.get('type', ''),
                    'properties': node.get('properties', {})
                })
            else:  # Fallback for other formats
                serializable_nodes.append({
                    'id': str(node),
                    'type': 'Unknown',
                    'properties': {}
                })
        return serializable_nodes
    
    def _serialize_relationships(self, relationships: list) -> list:
        """Convert LangChain Relationship objects to serializable dictionaries."""
        serializable_relationships = []
        for rel in relationships:
            if hasattr(rel, 'source'):  # LangChain Relationship object
                # Handle complex source/target objects
                source_obj = getattr(rel, 'source', {})
                target_obj = getattr(rel, 'target', {})
                
                # Extract source ID
                if hasattr(source_obj, 'get'):  # Dictionary-like
                    source = source_obj.get('id', str(source_obj))
                elif hasattr(source_obj, 'id'):  # Node object
                    source = getattr(source_obj, 'id', str(source_obj))
                else:
                    source = str(source_obj)
                
                # Extract target ID
                if hasattr(target_obj, 'get'):  # Dictionary-like
                    target = target_obj.get('id', str(target_obj))
                elif hasattr(target_obj, 'id'):  # Node object
                    target = getattr(target_obj, 'id', str(target_obj))
                else:
                    target = str(target_obj)
                
                serializable_relationships.append({
                    'source': source,
                    'target': target,
                    'type': getattr(rel, 'type', ''),
                    'properties': getattr(rel, 'properties', {})
                })
            elif isinstance(rel, dict):  # Already a dictionary
                serializable_relationships.append({
                    'source': rel.get('source', ''),
                    'target': rel.get('target', ''),
                    'type': rel.get('type', ''),
                    'properties': rel.get('properties', {})
                })
            else:  # Fallback for other formats
                serializable_relationships.append({
                    'source': str(rel),
                    'target': 'Unknown',
                    'type': 'RELATES_TO',
                    'properties': {}
                })
        return serializable_relationships
    
    def _show_sample_data(self, graph: dict):
        """Show sample nodes and relationships from a graph."""
        nodes = graph.get('nodes', [])
        relationships = graph.get('relationships', [])
        
        if nodes:
            print("  Sample nodes:")
            for node in nodes[:3]:  # Show first 3 nodes
                if isinstance(node, dict):  # Dictionary format
                    node_id = node.get('id', 'Unknown')
                    node_type = node.get('type', 'Unknown')
                else:  # Fallback
                    node_id = str(node)
                    node_type = 'Unknown'
                print(f"    - [{node_type}] {node_id}")
            if len(nodes) > 3:
                print(f"    ... and {len(nodes) - 3} more")
        
        if relationships:
            print("  Sample relationships:")
            for rel in relationships[:3]:  # Show first 3 relationships
                if isinstance(rel, dict):  # Dictionary format
                    source = rel.get('source', 'Unknown')
                    target = rel.get('target', 'Unknown')
                    rel_type = rel.get('type', 'RELATES_TO')
                else:  # Fallback
                    source = str(rel)
                    target = 'Unknown'
                    rel_type = 'RELATES_TO'
                
                print(f"    - {source} --{rel_type}--> {target}")
            if len(relationships) > 3:
                print(f"    ... and {len(relationships) - 3} more")
    
    def _save_results(self, all_graphs: list, mode: str):
        """Save results to JSON file."""
        import json
        
        # Graphs are already serializable from _convert_documents_to_graph
        output_data = {
            'graphs': all_graphs,
            'summary': {
                'total_graphs': len(all_graphs),
                'total_nodes': sum(len(g.get('nodes', [])) for g in all_graphs),
                'total_relationships': sum(len(g.get('relationships', [])) for g in all_graphs),
                'processing_mode': mode
            }
        }
        
        filename = f'knowledge_graph_{mode}_mode.json'
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        print(f"\n💾 Results saved to '{filename}'")


def main():
    """Main function with mode selection."""
    try:
        print("="*60)
        print("ENHANCED KNOWLEDGE GRAPH BUILDER")
        print("="*60)
        
        # Mode selection
        print("\nSelect processing mode:")
        print("1. Chunked mode - Process each file in chunks")
        print("2. Raw mode - Combine all files into single document")
        
        while True:
            choice = input("Enter your choice (1 or 2): ").strip()
            if choice == "1":
                mode = "chunked"
                break
            elif choice == "2":
                mode = "raw"
                break
            else:
                print("Invalid choice. Please enter 1 or 2.")
        
        # Initialize builder
        builder = EnhancedGraphBuilder()
        
        # Build graph
        result = builder.build_graph_from_directory(mode=mode)
        
        if result["success"]:
            print("\n🎉 Knowledge graph creation completed successfully!")
        else:
            print(f"\n❌ Failed to create knowledge graph: {result.get('error', 'Unknown error')}")
        
    except Exception as e:
        print(f"\n❌ Error in main function: {e}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")


if __name__ == "__main__":
    main()