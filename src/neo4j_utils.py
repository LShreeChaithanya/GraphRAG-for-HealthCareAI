import os
import logging
from typing import List, Dict, Any, Optional, Tuple
from langchain.schema import Document
from langchain_neo4j import Neo4jGraph
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class Neo4jProcessor:
    """Enhanced Neo4j processor with optimized queries and knowledge graph support."""
    
    def __init__(self, uri: str = None, user: str = None, password: str = None, database: str = "neo4j"):
        """
        Initialize Neo4j connection using langchain_neo4j.
        
        Args:
            uri: Neo4j connection URI
            user: Username
            password: Password
            database: Database name
        """
        self.uri = uri or os.getenv('NEO4J_URI', 'bolt://localhost:7687')
        self.user = user or os.getenv('NEO4J_USERNAME', 'neo4j')
        self.password = password or os.getenv('NEO4J_PASSWORD', 'password')
        self.database = database
        
        try:
            self.graph = Neo4jGraph(
                url=self.uri,
                username=self.user,
                password=self.password,
                database=self.database
            )
            logger.info("Neo4j connection established using langchain_neo4j")
            self._create_indexes()
        except ImportError:
            logger.error("langchain_neo4j package not installed. Install with: pip install langchain_neo4j")
            self.graph = None
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            self.graph = None
    
    def _create_indexes(self):
        """Create indexes for better performance."""
        if not self.graph:
            return
        
        try:
            # Create indexes for common properties
            indexes = [
                "CREATE INDEX node_id_index IF NOT EXISTS FOR (n) ON (n.id)",
                "CREATE INDEX node_type_index IF NOT EXISTS FOR (n) ON (n.type)",
                "CREATE INDEX document_source_index IF NOT EXISTS FOR (d:Document) ON (d.source_file)",
                "CREATE INDEX document_chunk_index IF NOT EXISTS FOR (d:Document) ON (d.chunk_id)"
            ]
            
            for index_query in indexes:
                try:
                    self.graph.query(index_query)
                    logger.debug(f"Created index: {index_query}")
                except Exception as e:
                    logger.debug(f"Index might already exist: {e}")
            
            logger.info("Database indexes created/verified")
            
        except Exception as e:
            logger.warning(f"Could not create indexes: {e}")
    
    def create_knowledge_graph_nodes(self, nodes: List[Dict[str, Any]]) -> bool:
        """
        Create knowledge graph nodes efficiently using MERGE to avoid duplicates.
        
        Args:
            nodes: List of node dictionaries with id, type, and properties
            
        Returns:
            True if successful, False otherwise
        """
        if not self.graph:
            logger.error("Neo4j graph not available")
            return False
        
        try:
            for node in nodes:
                node_id = node.get('id', '')
                node_type = node.get('type', 'Entity')
                properties = node.get('properties', {})
                
                if not node_id:
                    logger.warning(f"Skipping node without ID: {node}")
                    continue
                
                # Use MERGE to avoid duplicates and set properties efficiently
                query = f"""
                MERGE (n:{node_type} {{id: $node_id}})
                SET n += $properties
                SET n.type = $node_type
                """
                
                params = {
                    'node_id': node_id,
                    'node_type': node_type,
                    'properties': properties
                }
                
                self.graph.query(query, params)
            
            logger.info(f"Successfully created/updated {len(nodes)} knowledge graph nodes in Neo4j")
            return True
                
        except Exception as e:
            logger.error(f"Error creating knowledge graph nodes in Neo4j: {e}")
            return False
    
    def create_knowledge_graph_relationships(self, relationships: List[Dict[str, Any]]) -> bool:
        """
        Create knowledge graph relationships efficiently avoiding Cartesian products.
        
        Args:
            relationships: List of relationship dictionaries with source, target, type, and properties
            
        Returns:
            True if successful, False otherwise
        """
        if not self.graph:
            logger.error("Neo4j graph not available")
            return False
        
        try:
            for rel in relationships:
                source_id = rel.get('source', '')
                target_id = rel.get('target', '')
                rel_type = rel.get('type', 'RELATES_TO')
                properties = rel.get('properties', {})
                
                if not source_id or not target_id:
                    logger.warning(f"Skipping relationship with missing source/target: {rel}")
                    continue
                
                # OPTIMIZED: Use direct property matching to avoid Cartesian product
                query = f"""
                MATCH (a {{id: $source_id}}), (b {{id: $target_id}})
                MERGE (a)-[r:{rel_type}]->(b)
                SET r += $properties
                """
                
                params = {
                    'source_id': source_id,
                    'target_id': target_id,
                    'properties': properties
                }
                
                self.graph.query(query, params)
            
            logger.info(f"Successfully created {len(relationships)} relationships in Neo4j")
            return True
                
        except Exception as e:
            logger.error(f"Error creating relationships in Neo4j: {e}")
            return False
    
    def create_full_knowledge_graph(self, graph_data: Dict[str, Any]) -> bool:
        """
        Create a complete knowledge graph from nodes and relationships.
        
        Args:
            graph_data: Dictionary containing 'nodes' and 'relationships' lists
            
        Returns:
            True if successful, False otherwise
        """
        if not self.graph:
            logger.error("Neo4j graph not available")
            return False
        
        try:
            # Extract nodes and relationships
            nodes = graph_data.get('nodes', [])
            relationships = graph_data.get('relationships', [])
            
            # Convert LangChain objects to dictionaries if needed
            processed_nodes = []
            for node in nodes:
                if hasattr(node, 'id'):  # LangChain Node object
                    node_dict = {
                        'id': str(node.id),
                        'type': str(node.type) if hasattr(node, 'type') else 'Entity',
                        'properties': dict(node.properties) if hasattr(node, 'properties') else {}
                    }
                elif isinstance(node, dict):
                    node_dict = node
                else:
                    node_dict = {'id': str(node), 'type': 'Entity', 'properties': {}}
                processed_nodes.append(node_dict)
            
            processed_relationships = []
            for rel in relationships:
                if hasattr(rel, 'source'):  # LangChain Relationship object
                    # Handle complex source/target objects
                    source_obj = getattr(rel, 'source', {})
                    target_obj = getattr(rel, 'target', {})
                    
                    if hasattr(source_obj, 'id'):
                        source_id = str(source_obj.id)
                    elif hasattr(source_obj, 'get'):
                        source_id = source_obj.get('id', str(source_obj))
                    else:
                        source_id = str(source_obj)
                    
                    if hasattr(target_obj, 'id'):
                        target_id = str(target_obj.id)
                    elif hasattr(target_obj, 'get'):
                        target_id = target_obj.get('id', str(target_obj))
                    else:
                        target_id = str(target_obj)
                    
                    rel_dict = {
                        'source': source_id,
                        'target': target_id,
                        'type': str(getattr(rel, 'type', 'RELATES_TO')),
                        'properties': dict(getattr(rel, 'properties', {}))
                    }
                elif isinstance(rel, dict):
                    rel_dict = rel
                else:
                    rel_dict = {
                        'source': str(rel),
                        'target': 'Unknown',
                        'type': 'RELATES_TO',
                        'properties': {}
                    }
                processed_relationships.append(rel_dict)
            
            # Create nodes first
            if processed_nodes:
                nodes_success = self.create_knowledge_graph_nodes(processed_nodes)
                if not nodes_success:
                    logger.error("Failed to create nodes")
                    return False
            
            # Then create relationships
            if processed_relationships:
                rels_success = self.create_knowledge_graph_relationships(processed_relationships)
                if not rels_success:
                    logger.error("Failed to create relationships")
                    return False
            
            logger.info(f"Successfully created knowledge graph with {len(processed_nodes)} nodes and {len(processed_relationships)} relationships")
            return True
            
        except Exception as e:
            logger.error(f"Error creating full knowledge graph: {e}")
            return False
    
    def create_document_nodes(self, documents: List[Document]) -> bool:
        """
        Create document nodes in Neo4j from processed chunks using langchain_neo4j.
        
        Args:
            documents: List of Document objects
            
        Returns:
            True if successful, False otherwise
        """
        if not self.graph:
            logger.error("Neo4j graph not available")
            return False
        
        try:
            for doc in documents:
                # Use MERGE to avoid duplicates
                query = """
                MERGE (d:Document {
                    chunk_id: $chunk_id,
                    source_file: $source_file
                })
                SET d.content = $content,
                    d.file_name = $file_name,
                    d.file_extension = $file_extension,
                    d.chunk_size = $chunk_size,
                    d.total_chunks = $total_chunks,
                    d.processing_timestamp = $processing_timestamp
                """
                
                params = {
                    'content': doc.page_content,
                    'chunk_id': doc.metadata.get('chunk_id', 0),
                    'source_file': doc.metadata.get('source_file', ''),
                    'file_name': doc.metadata.get('file_name', ''),
                    'file_extension': doc.metadata.get('file_extension', ''),
                    'chunk_size': doc.metadata.get('chunk_size', 0),
                    'total_chunks': doc.metadata.get('total_chunks', 0),
                    'processing_timestamp': doc.metadata.get('processing_timestamp', '')
                }
                
                self.graph.query(query, params)
            
            logger.info(f"Successfully created {len(documents)} document nodes in Neo4j")
            return True
                
        except Exception as e:
            logger.error(f"Error creating document nodes in Neo4j: {e}")
            return False
    
    def batch_create_nodes(self, nodes: List[Dict[str, Any]], batch_size: int = 100) -> bool:
        """
        Create nodes in batches for better performance with large datasets.
        
        Args:
            nodes: List of node dictionaries
            batch_size: Number of nodes to process per batch
            
        Returns:
            True if successful, False otherwise
        """
        if not self.graph:
            logger.error("Neo4j graph not available")
            return False
        
        try:
            total_nodes = len(nodes)
            processed = 0
            
            for i in range(0, total_nodes, batch_size):
                batch = nodes[i:i + batch_size]
                
                # Create batch query
                query = """
                UNWIND $nodes as node
                CALL {
                    WITH node
                    CALL apoc.create.node([node.type], {id: node.id}) YIELD node as n
                    SET n += node.properties
                    RETURN n
                } IN TRANSACTIONS OF 50 ROWS
                """
                
                # Fallback query if APOC is not available
                fallback_query = """
                UNWIND $nodes as node
                MERGE (n {id: node.id})
                SET n += node.properties
                SET n:Entity
                """
                
                params = {'nodes': batch}
                
                try:
                    # Try APOC first
                    self.graph.query(query, params)
                except:
                    # Fallback to standard Cypher
                    self.graph.query(fallback_query, params)
                
                processed += len(batch)
                logger.info(f"Processed {processed}/{total_nodes} nodes")
            
            logger.info(f"Successfully batch created {total_nodes} nodes")
            return True
            
        except Exception as e:
            logger.error(f"Error in batch node creation: {e}")
            return False
    
    def batch_create_relationships(self, relationships: List[Dict[str, Any]], batch_size: int = 100) -> bool:
        """
        Create relationships in batches for better performance.
        
        Args:
            relationships: List of relationship dictionaries
            batch_size: Number of relationships to process per batch
            
        Returns:
            True if successful, False otherwise
        """
        if not self.graph:
            logger.error("Neo4j graph not available")
            return False
        
        try:
            total_rels = len(relationships)
            processed = 0
            
            for i in range(0, total_rels, batch_size):
                batch = relationships[i:i + batch_size]
                
                query = """
                UNWIND $relationships as rel
                MATCH (a {id: rel.source}), (b {id: rel.target})
                CALL apoc.create.relationship(a, rel.type, rel.properties, b) YIELD rel as r
                RETURN count(r)
                """
                
                # Fallback query if APOC is not available
                fallback_query = """
                UNWIND $relationships as rel
                MATCH (a {id: rel.source}), (b {id: rel.target})
                MERGE (a)-[r:RELATES_TO]->(b)
                SET r += rel.properties
                """
                
                params = {'relationships': batch}
                
                try:
                    # Try APOC first
                    self.graph.query(query, params)
                except:
                    # Fallback to standard Cypher
                    self.graph.query(fallback_query, params)
                
                processed += len(batch)
                logger.info(f"Processed {processed}/{total_rels} relationships")
            
            logger.info(f"Successfully batch created {total_rels} relationships")
            return True
            
        except Exception as e:
            logger.error(f"Error in batch relationship creation: {e}")
            return False
    
    def get_document_count(self) -> int:
        """
        Get the total count of Document nodes in Neo4j.
        
        Returns:
            Number of Document nodes
        """
        if not self.graph:
            return 0
        
        try:
            query = "MATCH (d:Document) RETURN count(d) as count"
            result = self.graph.query(query)
            if result and len(result) > 0:
                return result[0]['count']
            return 0
        except Exception as e:
            logger.error(f"Error getting document count: {e}")
            return 0
    
    def get_knowledge_graph_count(self) -> Dict[str, int]:
        """
        Get counts of knowledge graph nodes and relationships.
        
        Returns:
            Dictionary with node and relationship counts
        """
        if not self.graph:
            return {'nodes': 0, 'relationships': 0}
        
        try:
            # Count non-document nodes (knowledge graph nodes)
            nodes_query = """
            MATCH (n)
            WHERE NOT n:Document
            RETURN count(n) as count
            """
            nodes_result = self.graph.query(nodes_query)
            node_count = nodes_result[0]['count'] if nodes_result else 0
            
            # Count relationships
            rels_query = "MATCH ()-[r]->() RETURN count(r) as count"
            rels_result = self.graph.query(rels_query)
            rel_count = rels_result[0]['count'] if rels_result else 0
            
            return {'nodes': node_count, 'relationships': rel_count}
            
        except Exception as e:
            logger.error(f"Error getting knowledge graph count: {e}")
            return {'nodes': 0, 'relationships': 0}
    
    def clear_documents(self) -> bool:
        """
        Clear all Document nodes from Neo4j.
        
        Returns:
            True if successful, False otherwise
        """
        if not self.graph:
            logger.error("Neo4j graph not available")
            return False
        
        try:
            query = "MATCH (d:Document) DETACH DELETE d"
            self.graph.query(query)
            logger.info("Successfully cleared all Document nodes from Neo4j")
            return True
        except Exception as e:
            logger.error(f"Error clearing documents: {e}")
            return False
    
    def clear_knowledge_graph(self) -> bool:
        """
        Clear all knowledge graph nodes and relationships (keep documents).
        
        Returns:
            True if successful, False otherwise
        """
        if not self.graph:
            logger.error("Neo4j graph not available")
            return False
        
        try:
            # Delete all non-document nodes and their relationships
            query = """
            MATCH (n)
            WHERE NOT n:Document
            DETACH DELETE n
            """
            self.graph.query(query)
            logger.info("Successfully cleared knowledge graph from Neo4j")
            return True
        except Exception as e:
            logger.error(f"Error clearing knowledge graph: {e}")
            return False
    
    def get_documents_summary(self) -> Dict[str, Any]:
        """
        Get a summary of documents stored in Neo4j.
        
        Returns:
            Summary dictionary
        """
        if not self.graph:
            return {"total_documents": 0, "file_types": [], "sources": []}
        
        try:
            # Get total count
            count_query = "MATCH (d:Document) RETURN count(d) as count"
            count_result = self.graph.query(count_query)
            total_docs = count_result[0]['count'] if count_result else 0
            
            # Get file types
            types_query = "MATCH (d:Document) WHERE d.file_extension IS NOT NULL RETURN DISTINCT d.file_extension as file_type"
            types_result = self.graph.query(types_query)
            file_types = [r['file_type'] for r in types_result if r['file_type']]
            
            # Get source files
            sources_query = "MATCH (d:Document) WHERE d.source_file IS NOT NULL RETURN DISTINCT d.source_file as source"
            sources_result = self.graph.query(sources_query)
            sources = [r['source'] for r in sources_result if r['source']]
            
            return {
                "total_documents": total_docs,
                "file_types": file_types,
                "sources": sources
            }
            
        except Exception as e:
            logger.error(f"Error getting documents summary: {e}")
            return {"total_documents": 0, "file_types": [], "sources": []}
    
    def delete_entire_graph(self) -> bool:
        """
        Delete the entire knowledge graph from Neo4j (all nodes and relationships).
        
        Returns:
            True if successful, False otherwise
        """
        if not self.graph:
            logger.error("Neo4j graph not available")
            return False
        
        try:
            # Delete all nodes and relationships
            query = "MATCH (n) DETACH DELETE n"
            self.graph.query(query)
            logger.info("Successfully deleted entire knowledge graph from Neo4j")
            return True
        except Exception as e:
            logger.error(f"Error deleting entire graph: {e}")
            return False
    
    def verify_graph_creation(self) -> Dict[str, Any]:
        """
        Verify that the graph was created properly and return diagnostic info.
        
        Returns:
            Dictionary with diagnostic information
        """
        if not self.graph:
            return {"error": "Neo4j graph not available"}
        
        try:
            diagnostics = {}
            
            # Check total nodes
            total_query = "MATCH (n) RETURN count(n) as total"
            total_result = self.graph.query(total_query)
            diagnostics['total_nodes'] = total_result[0]['total'] if total_result else 0
            
            # Check node labels
            labels_query = """
            MATCH (n)
            RETURN DISTINCT labels(n) as labels, count(n) as count
            ORDER BY count DESC
            """
            labels_result = self.graph.query(labels_query)
            diagnostics['node_labels'] = {str(r['labels']): r['count'] for r in labels_result}
            
            # Check relationships
            rels_query = """
            MATCH ()-[r]->()
            RETURN type(r) as rel_type, count(r) as count
            ORDER BY count DESC
            """
            rels_result = self.graph.query(rels_query)
            diagnostics['relationship_types'] = {r['rel_type']: r['count'] for r in rels_result}
            
            # Check for isolated nodes
            isolated_query = """
            MATCH (n)
            WHERE NOT (n)--()
            RETURN count(n) as isolated_count
            """
            isolated_result = self.graph.query(isolated_query)
            diagnostics['isolated_nodes'] = isolated_result[0]['isolated_count'] if isolated_result else 0
            
            # Sample nodes for inspection
            sample_query = "MATCH (n) RETURN n LIMIT 5"
            sample_result = self.graph.query(sample_query)
            diagnostics['sample_nodes'] = [dict(r['n']) for r in sample_result]
            
            return diagnostics
            
        except Exception as e:
            logger.error(f"Error verifying graph creation: {e}")
            return {"error": str(e)}
    
    def get_graph_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive statistics about the knowledge graph.
        
        Returns:
            Dictionary containing graph statistics
        """
        if not self.graph:
            return {"error": "Neo4j graph not available"}
        
        try:
            stats = {}
            
            # Total nodes
            total_nodes_query = "MATCH (n) RETURN count(n) as count"
            total_nodes_result = self.graph.query(total_nodes_query)
            stats['total_nodes'] = total_nodes_result[0]['count'] if total_nodes_result else 0
            
            # Total relationships
            total_rels_query = "MATCH ()-[r]->() RETURN count(r) as count"
            total_rels_result = self.graph.query(total_rels_query)
            stats['total_relationships'] = total_rels_result[0]['count'] if total_rels_result else 0
            
            # Node types and counts
            node_types_query = """
            MATCH (n)
            RETURN labels(n)[0] as node_type, count(n) as count
            ORDER BY count DESC
            """
            node_types_result = self.graph.query(node_types_query)
            stats['node_types'] = {record['node_type']: record['count'] for record in node_types_result}
            
            # Relationship types and counts
            rel_types_query = """
            MATCH ()-[r]->()
            RETURN type(r) as rel_type, count(r) as count
            ORDER BY count DESC
            """
            rel_types_result = self.graph.query(rel_types_query)
            stats['relationship_types'] = {record['rel_type']: record['count'] for record in rel_types_result}
            
            # Document statistics
            try:
                doc_stats_query = """
                MATCH (d:Document)
                RETURN count(d) as total_docs,
                       count(DISTINCT CASE WHEN d.source_file IS NOT NULL THEN d.source_file END) as unique_sources,
                       count(DISTINCT CASE WHEN d.file_extension IS NOT NULL THEN d.file_extension END) as file_types
                """
                doc_stats_result = self.graph.query(doc_stats_query)
                if doc_stats_result and len(doc_stats_result) > 0:
                    stats['document_stats'] = dict(doc_stats_result[0])
                else:
                    stats['document_stats'] = {'total_docs': 0, 'unique_sources': 0, 'file_types': 0}
            except Exception as e:
                logger.warning(f"Error getting document statistics: {e}")
                stats['document_stats'] = {'total_docs': 0, 'unique_sources': 0, 'file_types': 0}
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting graph statistics: {e}")
            return {"error": str(e)}
    
    def query_all_nodes(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Query all nodes in the graph with optional limit.
        
        Args:
            limit: Maximum number of nodes to return
            
        Returns:
            List of node dictionaries
        """
        if not self.graph:
            return []
        
        try:
            query = f"MATCH (n) RETURN n LIMIT {limit}"
            result = self.graph.query(query)
            return [dict(record['n']) for record in result]
        except Exception as e:
            logger.error(f"Error querying all nodes: {e}")
            return []
    
    def query_nodes_by_type(self, node_type: str, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Query nodes by type.
        
        Args:
            node_type: Type of nodes to query
            limit: Maximum number of nodes to return
            
        Returns:
            List of node dictionaries
        """
        if not self.graph:
            return []
        
        try:
            query = f"MATCH (n:{node_type}) RETURN n LIMIT {limit}"
            result = self.graph.query(query)
            return [dict(record['n']) for record in result]
        except Exception as e:
            logger.error(f"Error querying nodes by type: {e}")
            return []
    
    def query_relationships(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Query all relationships in the graph.
        
        Args:
            limit: Maximum number of relationships to return
            
        Returns:
            List of relationship dictionaries
        """
        if not self.graph:
            return []
        
        try:
            query = f"""
            MATCH (a)-[r]->(b)
            RETURN a, r, b
            LIMIT {limit}
            """
            result = self.graph.query(query)
            relationships = []
            for record in result:
                # Extract properties safely from Neo4j objects
                source_node = record['a']
                target_node = record['b']
                relationship = record['r']
                
                # Convert to dictionaries safely
                try:
                    source_dict = dict(source_node) if hasattr(source_node, '__iter__') else {'id': str(source_node)}
                except:
                    source_dict = {'id': str(source_node)}
                
                try:
                    target_dict = dict(target_node) if hasattr(target_node, '__iter__') else {'id': str(target_node)}
                except:
                    target_dict = {'id': str(target_node)}
                
                try:
                    rel_dict = dict(relationship) if hasattr(relationship, '__iter__') else {'type': str(relationship)}
                except:
                    rel_dict = {'type': str(relationship)}
                
                rel_info = {
                    'source': source_dict,
                    'relationship': rel_dict,
                    'target': target_dict
                }
                relationships.append(rel_info)
            return relationships
        except Exception as e:
            logger.error(f"Error querying relationships: {e}")
            return []
    
    def query_relationships_by_type(self, rel_type: str, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Query relationships by type.
        
        Args:
            rel_type: Type of relationships to query
            limit: Maximum number of relationships to return
            
        Returns:
            List of relationship dictionaries
        """
        if not self.graph:
            return []
        
        try:
            query = f"""
            MATCH (a)-[r:{rel_type}]->(b)
            RETURN a, r, b
            LIMIT {limit}
            """
            result = self.graph.query(query)
            relationships = []
            for record in result:
                # Extract properties safely from Neo4j objects
                source_node = record['a']
                target_node = record['b']
                relationship = record['r']
                
                # Convert to dictionaries safely
                try:
                    source_dict = dict(source_node) if hasattr(source_node, '__iter__') else {'id': str(source_node)}
                except:
                    source_dict = {'id': str(source_node)}
                
                try:
                    target_dict = dict(target_node) if hasattr(target_node, '__iter__') else {'id': str(target_node)}
                except:
                    target_dict = {'id': str(target_node)}
                
                try:
                    rel_dict = dict(relationship) if hasattr(relationship, '__iter__') else {'type': str(relationship)}
                except:
                    rel_dict = {'type': str(relationship)}
                
                rel_info = {
                    'source': source_dict,
                    'relationship': rel_dict,
                    'target': target_dict
                }
                relationships.append(rel_info)
            return relationships
        except Exception as e:
            logger.error(f"Error querying relationships by type: {e}")
            return []
    
    def query_node_connections(self, node_id: str, depth: int = 2) -> List[Dict[str, Any]]:
        """
        Query node connections up to a specified depth.
        
        Args:
            node_id: ID of the node to query
            depth: Maximum depth of connections to explore
            
        Returns:
            List of connection dictionaries
        """
        if not self.graph:
            return []
        
        try:
            query = f"""
            MATCH path = (start {{id: $node_id}})-[*1..{depth}]-(connected)
            RETURN path
            LIMIT 100
            """
            result = self.graph.query(query, {'node_id': node_id})
            connections = []
            for record in result:
                try:
                    path_dict = dict(record['path']) if hasattr(record['path'], '__iter__') else {'path': str(record['path'])}
                except:
                    path_dict = {'path': str(record['path'])}
                connections.append(path_dict)
            return connections
        except Exception as e:
            logger.error(f"Error querying node connections: {e}")
            return []
    
    def query_by_keyword(self, keyword: str, node_types: List[str] = None, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Query nodes by keyword in their properties.
        
        Args:
            keyword: Keyword to search for
            node_types: List of node types to filter by
            limit: Maximum number of results to return
            
        Returns:
            List of matching node dictionaries
        """
        if not self.graph:
            return []
        
        try:
            if node_types:
                type_filter = " AND " + " OR ".join([f"n:{node_type}" for node_type in node_types])
            else:
                type_filter = ""
            
            query = f"""
            MATCH (n)
            WHERE ANY(prop IN keys(n) WHERE toString(n[prop]) CONTAINS $keyword){type_filter}
            RETURN n
            LIMIT {limit}
            """
            result = self.graph.query(query, {'keyword': keyword})
            return [dict(record['n']) for record in result]
        except Exception as e:
            logger.error(f"Error querying by keyword: {e}")
            return []
    
    def delete_knowledge_graph_nodes(self) -> bool:
        """
        Delete only knowledge graph nodes (keep Document nodes).
        
        Returns:
            True if successful, False otherwise
        """
        return self.clear_knowledge_graph()
    
    def delete_by_source_file(self, source_file: str) -> bool:
        """
        Delete nodes and relationships related to a specific source file.
        
        Args:
            source_file: Name of the source file
            
        Returns:
            True if successful, False otherwise
        """
        if not self.graph:
            logger.error("Neo4j graph not available")
            return False
        
        try:
            # Delete Document nodes with the specified source file
            doc_query = "MATCH (d:Document {source_file: $source_file}) DETACH DELETE d"
            self.graph.query(doc_query, {'source_file': source_file})
            
            # Delete any knowledge graph nodes that reference this source
            kg_query = """
            MATCH (n)
            WHERE n.source_file = $source_file OR n.source = $source_file
            DETACH DELETE n
            """
            self.graph.query(kg_query, {'source_file': source_file})
            
            logger.info(f"Successfully deleted data related to source file: {source_file}")
            return True
        except Exception as e:
            logger.error(f"Error deleting by source file: {e}")
            return False
    
    def execute_custom_query(self, query: str, parameters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Execute a custom Cypher query.
        
        Args:
            query: Cypher query string
            parameters: Query parameters
            
        Returns:
            Query results as list of dictionaries
        """
        if not self.graph:
            return []
        
        try:
            if parameters:
                result = self.graph.query(query, parameters)
            else:
                result = self.graph.query(query)
            
            # Convert result to list of dictionaries
            if result:
                # Handle different result formats
                if isinstance(result[0], dict):
                    return result
                else:
                    # Convert to dictionary format
                    return [dict(record) for record in result]
            return []
            
        except Exception as e:
            logger.error(f"Error executing custom query: {e}")
            return []
    
    def close(self):
        """Close the Neo4j connection."""
        if self.graph:
            # langchain_neo4j handles connection cleanup automatically
            logger.info("Neo4j connection closed")


# Enhanced utility functions
def store_knowledge_graph_in_neo4j(graph_data: Dict[str, Any],
                                   uri: str = None,
                                   user: str = None,
                                   password: str = None,
                                   database: str = "neo4j") -> bool:
    """
    Utility function to store a complete knowledge graph in Neo4j.
    
    Args:
        graph_data: Dictionary containing nodes and relationships
        uri: Neo4j connection URI
        user: Username
        password: Password
        database: Database name
        
    Returns:
        True if successful, False otherwise
    """
    processor = Neo4jProcessor(uri, user, password, database)
    
    try:
        if processor.graph:
            success = processor.create_full_knowledge_graph(graph_data)
            if success:
                counts = processor.get_knowledge_graph_count()
                logger.info(f"Successfully stored knowledge graph. Nodes: {counts['nodes']}, Relationships: {counts['relationships']}")
            return success
        else:
            logger.warning("Neo4j not available")
            return False
    finally:
        processor.close()


def verify_neo4j_graph(uri: str = None,
                       user: str = None,
                       password: str = None,
                       database: str = "neo4j") -> Dict[str, Any]:
    """
    Utility function to verify Neo4j graph creation.
    
    Args:
        uri: Neo4j connection URI
        user: Username
        password: Password
        database: Database name
        
    Returns:
        Diagnostic information dictionary
    """
    processor = Neo4jProcessor(uri, user, password, database)
    
    try:
        if processor.graph:
            return processor.verify_graph_creation()
        else:
            return {"error": "Neo4j not available"}
    finally:
        processor.close()


def get_neo4j_summary(uri: str = None, 
                     user: str = None, 
                     password: str = None,
                     database: str = "neo4j") -> Dict[str, Any]:
    """
    Utility function to get Neo4j documents summary.
    
    Args:
        uri: Neo4j connection URI
        user: Username
        password: Password
        database: Database name
        
    Returns:
        Summary dictionary
    """
    processor = Neo4jProcessor(uri, user, password, database)
    
    try:
        if processor.graph:
            return processor.get_documents_summary()
        else:
            return {"total_documents": 0, "file_types": [], "sources": []}
    finally:
        processor.close()


# Additional utility functions for the app
def query_neo4j_graph(query: str, parameters: Dict[str, Any] = None,
                     uri: str = None, user: str = None, password: str = None,
                     database: str = "neo4j") -> List[Dict[str, Any]]:
    """
    Utility function to execute a custom query on Neo4j.
    
    Args:
        query: Cypher query string
        parameters: Query parameters
        uri: Neo4j connection URI
        user: Username
        password: Password
        database: Database name
        
    Returns:
        Query results as list of dictionaries
    """
    processor = Neo4jProcessor(uri, user, password, database)
    
    try:
        if processor.graph:
            return processor.execute_custom_query(query, parameters)
        else:
            return []
    finally:
        processor.close()


def delete_neo4j_graph(uri: str = None, user: str = None, password: str = None,
                      database: str = "neo4j") -> bool:
    """
    Utility function to delete the entire Neo4j graph.
    
    Args:
        uri: Neo4j connection URI
        user: Username
        password: Password
        database: Database name
        
    Returns:
        True if successful, False otherwise
    """
    processor = Neo4jProcessor(uri, user, password, database)
    
    try:
        if processor.graph:
            return processor.delete_entire_graph()
        else:
            return False
    finally:
        processor.close()


def get_neo4j_graph_statistics(uri: str = None, user: str = None, password: str = None,
                              database: str = "neo4j") -> Dict[str, Any]:
    """
    Utility function to get Neo4j graph statistics.
    
    Args:
        uri: Neo4j connection URI
        user: Username
        password: Password
        database: Database name
        
    Returns:
        Graph statistics dictionary
    """
    processor = Neo4jProcessor(uri, user, password, database)
    
    try:
        if processor.graph:
            return processor.get_graph_statistics()
        else:
            return {"error": "Neo4j not available"}
    finally:
        processor.close()
