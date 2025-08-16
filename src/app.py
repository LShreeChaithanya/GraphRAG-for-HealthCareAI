import streamlit as st
import json
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import tempfile
import os
from io import BytesIO
import time
from pyvis.network import Network
import streamlit.components.v1 as components

# Import our custom modules
from Graph_builder import EnhancedGraphBuilder
from data_processors import DataProcessor
from neo4j_utils import (
    Neo4jProcessor, 
    query_neo4j_graph, 
    delete_neo4j_graph, 
    get_neo4j_graph_statistics
)
from agent_simple import create_simple_healthcare_agent, SimpleHealthcareAgent

# Configure page
st.set_page_config(
    page_title="Knowledge Graph Builder",
    page_icon="🕸️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2c3e50;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #dee2e6;
        margin-bottom: 1rem;
    }
    .success-message {
        background-color: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #c3e6cb;
    }
    .error-message {
        background-color: #f8d7da;
        color: #721c24;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #f5c6cb;
    }
    .warning-message {
        background-color: #fff3cd;
        color: #856404;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #ffeaa7;
    }
    .info-message {
        background-color: #d1ecf1;
        color: #0c5460;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #bee5eb;
    }
    .neo4j-panel {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border: 1px solid #dee2e6;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables."""
    if 'graph_results' not in st.session_state:
        st.session_state.graph_results = None
    if 'processing_complete' not in st.session_state:
        st.session_state.processing_complete = False
    if 'last_mode' not in st.session_state:
        st.session_state.last_mode = None
    if 'neo4j_processor' not in st.session_state:
        st.session_state.neo4j_processor = None
    if 'neo4j_connected' not in st.session_state:
        st.session_state.neo4j_connected = False
    if 'graph_stored_in_neo4j' not in st.session_state:
        st.session_state.graph_stored_in_neo4j = False
    if 'healthcare_agent' not in st.session_state:
        st.session_state.healthcare_agent = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'notifications' not in st.session_state:
        st.session_state.notifications = []

def show_notification(message, type="info", duration=5):
    """Show a notification that auto-dismisses after a specified duration."""
    notification = {
        "message": message,
        "type": type,
        "timestamp": time.time(),
        "duration": duration
    }
    st.session_state.notifications.append(notification)

def display_notifications():
    """Display and manage notifications."""
    current_time = time.time()
    
    # Filter out expired notifications
    st.session_state.notifications = [
        n for n in st.session_state.notifications 
        if current_time - n["timestamp"] < n["duration"]
    ]
    
    # Display active notifications
    for notification in st.session_state.notifications:
        if notification["type"] == "success":
            st.success(notification["message"])
        elif notification["type"] == "error":
            st.error(notification["message"])
        elif notification["type"] == "warning":
            st.warning(notification["message"])
        else:
            st.info(notification["message"])

def establish_neo4j_connection():
    """Establish and maintain Neo4j connection throughout the session."""
    if not st.session_state.neo4j_connected or not st.session_state.neo4j_processor:
        try:
            with st.spinner("🔌 Establishing Neo4j connection..."):
                processor = Neo4jProcessor()
                if processor.graph:
                    st.session_state.neo4j_processor = processor
                    st.session_state.neo4j_connected = True
                    show_notification("✅ Neo4j connection established!", "success", 3)
                    
                    # Create indexes for better performance
                    try:
                        create_neo4j_indexes(processor)
                    except Exception as index_error:
                        show_notification(f"⚠️ Could not create indexes: {str(index_error)}", "warning", 3)
                    
                    return True
                else:
                    show_notification("❌ Failed to establish Neo4j connection.", "error", 5)
                    return False
        except Exception as e:
            show_notification(f"❌ Error connecting to Neo4j: {str(e)}", "error", 5)
            return False
    return True

def create_neo4j_indexes(processor):
    """Create indexes for better Neo4j performance."""
    try:
        # Create indexes for common properties
        index_queries = [
            "CREATE INDEX node_id_index IF NOT EXISTS FOR (n) ON (n.id)",
            "CREATE INDEX node_type_index IF NOT EXISTS FOR (n) ON (n.type)",
            "CREATE INDEX relationship_type_index IF NOT EXISTS FOR ()-[r]-() ON (r.type)"
        ]
        
        for query in index_queries:
            try:
                processor.graph.query(query)
            except Exception as e:
                # Index might already exist, which is fine
                pass
                
    except Exception as e:
        # Index creation is optional, don't fail the connection
        pass

def close_neo4j_connection():
    """Close Neo4j connection and clean up."""
    if st.session_state.neo4j_processor:
        try:
            st.session_state.neo4j_processor.close()
            st.session_state.neo4j_processor = None
            st.session_state.neo4j_connected = False
            show_notification("🔌 Neo4j connection closed.", "info", 3)
        except Exception as e:
            show_notification(f"❌ Error closing Neo4j connection: {str(e)}", "error", 5)

def initialize_healthcare_agent():
    """Initialize the healthcare agent with Neo4j processor."""
    if st.session_state.neo4j_connected and st.session_state.neo4j_processor:
        try:
            with st.spinner("🤖 Initializing Healthcare Agent..."):
                agent = create_simple_healthcare_agent(st.session_state.neo4j_processor)
                st.session_state.healthcare_agent = agent
                show_notification("✅ Healthcare Agent initialized successfully!", "success", 3)
                return True
        except Exception as e:
            show_notification(f"❌ Error initializing Healthcare Agent: {str(e)}", "error", 5)
            return False
    else:
        show_notification("❌ Neo4j connection required to initialize Healthcare Agent.", "error", 5)
        return False

def display_header():
    """Display the main header."""
    st.markdown('<h1 class="main-header">🕸️ Knowledge Graph Builder</h1>', unsafe_allow_html=True)
    st.markdown("---")
    st.markdown(
        """
        <div style="text-align: center; margin-bottom: 2rem;">
            <p style="font-size: 1.2rem; color: #666;">
                Build knowledge graphs from your documents using AI-powered entity extraction
            </p>
        </div>
        """, 
        unsafe_allow_html=True
    )

def display_sidebar():
    """Display the sidebar with configuration options."""
    with st.sidebar:
        st.markdown('<h2 class="sub-header">⚙️ Configuration</h2>', unsafe_allow_html=True)
        
        # Processing mode selection
        st.markdown("### Processing Mode")
        mode = st.radio(
            "Choose how to process your documents:",
            options=["chunked", "raw"],
            format_func=lambda x: "📄 Chunked Mode (Process each file in chunks)" if x == "chunked" 
                                 else "📋 Raw Mode (Combine all files into one document)",
            help="""
            **Chunked Mode**: Each file is split into smaller chunks and processed separately. 
            Good for large documents and detailed analysis.
            
            **Raw Mode**: All files are combined into a single document before processing. 
            Good for getting an overall view of relationships across all documents.
            """
        )
        
        # Chunking parameters (only show for chunked mode)
        if mode == "chunked":
            st.markdown("### Chunking Parameters")
            chunk_size = st.slider(
                "Chunk Size",
                min_value=100,
                max_value=1000,
                value=200,
                step=50,
                help="Size of each text chunk in characters"
            )
            chunk_overlap = st.slider(
                "Chunk Overlap",
                min_value=0,
                max_value=100,
                value=30,
                step=10,
                help="Overlap between consecutive chunks in characters"
            )
        else:
            chunk_size = 200
            chunk_overlap = 30
        
        # Graph parameters
        st.markdown("### Graph Parameters")
        with st.expander("Advanced Settings", expanded=False):
            st.info("These settings are currently fixed but can be customized in the code:")
            st.write("**Node Types**: Concept, Entity, Topic, Process, Method, Tool, etc.")
            st.write("**Relationship Types**: RELATES_TO, IS_A, PART_OF, CAUSES, etc.")
        
        # Neo4j panel option
        st.markdown("---")
        st.markdown("### 🗄️ Neo4j Operations")
        
        # Connection status
        if st.session_state.neo4j_connected:
            st.success("🔌 Connected to Neo4j")
        else:
            st.error("🔌 Not connected to Neo4j")
        
        show_neo4j_panel = st.checkbox(
            "Show Neo4j Panel",
            help="Access Neo4j database operations (query, delete, statistics) without creating a new graph"
        )
        
        return mode, chunk_size, chunk_overlap, show_neo4j_panel

def process_uploaded_files(uploaded_files, mode, chunk_size, chunk_overlap):
    """Process uploaded files and create knowledge graph."""
    if not uploaded_files:
        st.warning("Please upload at least one file.")
        return None
    
    try:
        # Initialize the graph builder
        with st.spinner("Initializing Knowledge Graph Builder..."):
            builder = EnhancedGraphBuilder(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        
        # Create progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Process files
        status_text.text("Processing uploaded files...")
        progress_bar.progress(25)
        
        # Build the knowledge graph
        status_text.text("Building knowledge graph...")
        progress_bar.progress(50)
        
        # Reset file pointers
        for uploaded_file in uploaded_files:
            uploaded_file.seek(0)
        
        result = builder.build_graph_from_uploaded_files(uploaded_files, mode=mode)
        
        progress_bar.progress(75)
        status_text.text("Finalizing results...")
        
        progress_bar.progress(100)
        status_text.text("Complete!")
        
        # Clear progress indicators
        progress_bar.empty()
        status_text.empty()
        
        return result
        
    except Exception as e:
        st.error(f"Error processing files: {str(e)}")
        return None

def display_results(results):
    """Display the knowledge graph results."""
    if not results or not results.get("success"):
        st.error("Failed to create knowledge graph.")
        if results and "error" in results:
            st.error(f"Error: {results['error']}")
        return
    
    summary = results.get("summary", {})
    graphs = results.get("graphs", [])
    
    # Display success message
    st.markdown(
        f"""
        <div class="success-message">
            <h3>✅ Knowledge Graph Created Successfully!</h3>
            <p>Processing Mode: <strong>{summary.get('processing_mode', 'Unknown').title()}</strong></p>
        </div>
        """, 
        unsafe_allow_html=True
    )
    
    # Display summary metrics
    st.markdown('<h2 class="sub-header">📊 Summary</h2>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Graphs", summary.get("total_graphs", 0))
    
    with col2:
        st.metric("Total Nodes", summary.get("total_nodes", 0))
    
    with col3:
        st.metric("Total Relationships", summary.get("total_relationships", 0))
    
    with col4:
        avg_nodes = summary.get("total_nodes", 0) / max(summary.get("total_graphs", 1), 1)
        st.metric("Avg Nodes/Graph", f"{avg_nodes:.1f}")
    
    # Display individual graphs
    st.markdown('<h2 class="sub-header">🔍 Graph Details</h2>', unsafe_allow_html=True)
    
    for i, graph in enumerate(graphs):
        with st.expander(f"Graph {i+1}: {graph.get('source_document', 'Unknown')}", expanded=i==0):
            
            # Graph info
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Source**: {graph.get('source_document', 'Unknown')}")
                st.write(f"**Mode**: {graph.get('processing_mode', 'Unknown')}")
                if 'chunk_id' in graph:
                    st.write(f"**Chunk ID**: {graph.get('chunk_id')}")
                if 'source_files' in graph:
                    st.write(f"**Source Files**: {', '.join(graph.get('source_files', []))}")
            
            with col2:
                st.write(f"**Nodes**: {len(graph.get('nodes', []))}")
                st.write(f"**Relationships**: {len(graph.get('relationships', []))}")
            
            # Tabs for different views
            tab1, tab2, tab3 = st.tabs(["📝 Nodes", "🔗 Relationships", "📊 Visualization"])
            
            with tab1:
                display_nodes(graph.get('nodes', []))
            
            with tab2:
                display_relationships(graph.get('relationships', []))
            
            with tab3:
                display_graph_visualization(graph)

def display_nodes(nodes):
    """Display nodes in a table format."""
    if not nodes:
        st.write("No nodes found.")
        return
    
    # Convert nodes to DataFrame - handle both dict and object formats
    node_data = []
    for node in nodes:
        if hasattr(node, 'id') and not hasattr(node, 'get'):  # LangChain Node object (Pydantic model)
            node_data.append({
                'ID': str(node.id) if hasattr(node, 'id') else '',
                'Type': str(node.type) if hasattr(node, 'type') else '',
                'Properties': str(node.properties) if hasattr(node, 'properties') else '{}'
            })
        elif isinstance(node, dict):  # Dictionary format
            node_data.append({
                'ID': node.get('id', ''),
                'Type': node.get('type', ''),
                'Properties': str(node.get('properties', {}))
            })
        else:  # Fallback
            node_data.append({
                'ID': str(node),
                'Type': 'Unknown',
                'Properties': '{}'
            })
    
    df = pd.DataFrame(node_data)
    st.dataframe(df, use_container_width=True)
    
    # Node type distribution
    if len(nodes) > 1:
        type_counts = df['Type'].value_counts()
        fig = px.pie(
            values=type_counts.values,
            names=type_counts.index,
            title="Node Type Distribution"
        )
        st.plotly_chart(fig, use_container_width=True)

def display_relationships(relationships):
    """Display relationships in a table format."""
    if not relationships:
        st.write("No relationships found.")
        return
    
    # Convert relationships to DataFrame - handle both dict and object formats
    rel_data = []
    for rel in relationships:
        if hasattr(rel, 'source') and not hasattr(rel, 'get'):  # LangChain Relationship object (Pydantic model)
            # Handle Pydantic model objects
            source = str(rel.source.id) if hasattr(rel.source, 'id') else str(rel.source)
            target = str(rel.target.id) if hasattr(rel.target, 'id') else str(rel.target)
            rel_type = str(rel.type) if hasattr(rel, 'type') else 'RELATES_TO'
            properties = str(rel.properties) if hasattr(rel, 'properties') else '{}'
            
            rel_data.append({
                'Source': source,
                'Relationship': rel_type,
                'Target': target,
                'Properties': properties
            })
        elif isinstance(rel, dict):  # Dictionary format
            rel_data.append({
                'Source': rel.get('source', ''),
                'Relationship': rel.get('type', ''),
                'Target': rel.get('target', ''),
                'Properties': str(rel.get('properties', {}))
            })
        else:  # Fallback
            rel_data.append({
                'Source': str(rel),
                'Relationship': 'RELATES_TO',
                'Target': 'Unknown',
                'Properties': '{}'
            })
    
    df = pd.DataFrame(rel_data)
    st.dataframe(df, use_container_width=True)
    
    # Relationship type distribution
    if len(relationships) > 1:
        type_counts = df['Relationship'].value_counts()
        fig = px.bar(
            x=type_counts.index,
            y=type_counts.values,
            title="Relationship Type Distribution",
            labels={'x': 'Relationship Type', 'y': 'Count'}
        )
        fig.update_xaxes(tickangle=45)  # Fixed: changed from update_xaxis to update_xaxes
        st.plotly_chart(fig, use_container_width=True)

def create_pyvis_network(graph_data):
    """Create an interactive Pyvis network visualization."""
    try:
        # Create Pyvis network
        net = Network(height="600px", width="100%", bgcolor="#ffffff", font_color="#000000")
        
        # Configure physics for better layout
        net.set_options("""
        var options = {
          "physics": {
            "forceAtlas2Based": {
              "gravitationalConstant": -50,
              "centralGravity": 0.01,
              "springLength": 200,
              "springConstant": 0.08
            },
            "maxVelocity": 50,
            "minVelocity": 0.1,
            "solver": "forceAtlas2Based",
            "timestep": 0.35
          },
          "nodes": {
            "font": {
              "size": 14
            },
            "shape": "dot",
            "size": 25
          },
          "edges": {
            "color": {
              "color": "#848484",
              "highlight": "#848484",
              "hover": "#848484"
            },
            "font": {
              "size": 12
            },
            "smooth": {
              "type": "continuous"
            }
          }
        }
        """)
        
        # Add nodes
        nodes = graph_data.get('nodes', [])
        node_colors = {
            'Concept': '#ff7675', 'Entity': '#74b9ff', 'Topic': '#55a3ff',
            'Process': '#00b894', 'Method': '#fdcb6e', 'Tool': '#e17055',
            'Condition': '#fd79a8', 'Symptom': '#f39c12', 'Treatment': '#27ae60',
            'Person': '#8e44ad', 'Organization': '#2c3e50', 'Location': '#16a085',
            'Event': '#e74c3c', 'Technology': '#3498db', 'Product': '#9b59b6',
            'Service': '#1abc9c', 'Document': '#95a5a6', 'Unknown': '#bdc3c7'
        }
        
        for node in nodes:
            if isinstance(node, dict):
                node_id = str(node.get('id', ''))
                node_type = node.get('type', 'Unknown')
                properties = node.get('properties', {})
                
                # Get node label from properties or use ID
                label = properties.get('name', properties.get('description', node_id))
                if len(label) > 30:
                    label = label[:27] + "..."
                
                # Get color for node type
                color = node_colors.get(node_type, node_colors['Unknown'])
                
                # Create tooltip with properties
                tooltip = f"<b>{node_type}</b><br>ID: {node_id}<br>"
                for key, value in properties.items():
                    if key not in ['name', 'description']:
                        tooltip += f"{key}: {str(value)[:50]}<br>"
                
                net.add_node(node_id, label=label, title=tooltip, color=color, size=20)
        
        # Add relationships
        relationships = graph_data.get('relationships', [])
        for rel in relationships:
            if isinstance(rel, dict):
                source = str(rel.get('source', ''))
                target = str(rel.get('target', ''))
                rel_type = rel.get('type', 'RELATES_TO')
                properties = rel.get('properties', {})
                
                # Create tooltip for relationship
                tooltip = f"<b>{rel_type}</b><br>"
                for key, value in properties.items():
                    tooltip += f"{key}: {str(value)[:50]}<br>"
                
                net.add_edge(source, target, title=tooltip, label=rel_type)
        
        return net
    
    except Exception as e:
        st.error(f"Error creating Pyvis network: {str(e)}")
        return None

def display_graph_visualization(graph):
    """Display an interactive graph visualization using Pyvis."""
    nodes = graph.get('nodes', [])
    relationships = graph.get('relationships', [])
    
    if not nodes:
        st.info("No nodes found for visualization.")
        return
    
    if not relationships:
        st.info("No relationships found. Only nodes will be displayed.")
    
    # Create Pyvis network
    net = create_pyvis_network(graph)
    
    if net:
        # Generate HTML content directly
        html_content = net.generate_html()
        
        # Display the interactive graph
        st.markdown("### 🕸️ Interactive Knowledge Graph")
        st.markdown("**Instructions:** Drag nodes to reposition, zoom with mouse wheel, hover for details")
        
        # Display the graph
        components.html(html_content, height=600, scrolling=False)
        
        # Show graph statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Nodes", len(nodes))
        with col2:
            st.metric("Relationships", len(relationships))
        with col3:
            node_types = set()
            for node in nodes:
                if isinstance(node, dict):
                    node_types.add(node.get('type', 'Unknown'))
            st.metric("Node Types", len(node_types))
        
        # Show node type legend
        if node_types:
            st.markdown("**Node Type Legend:**")
            node_colors = {
                'Concept': '#ff7675', 'Entity': '#74b9ff', 'Topic': '#55a3ff',
                'Process': '#00b894', 'Method': '#fdcb6e', 'Tool': '#e17055',
                'Condition': '#fd79a8', 'Symptom': '#f39c12', 'Treatment': '#27ae60',
                'Person': '#8e44ad', 'Organization': '#2c3e50', 'Location': '#16a085',
                'Event': '#e74c3c', 'Technology': '#3498db', 'Product': '#9b59b6',
                'Service': '#1abc9c', 'Document': '#95a5a6', 'Unknown': '#bdc3c7'
            }
            
            cols = st.columns(min(len(node_types), 4))
            for i, node_type in enumerate(sorted(node_types)):
                with cols[i % len(cols)]:
                    color = node_colors.get(node_type, node_colors['Unknown'])
                    st.markdown(f'<span style="color: {color}; font-weight: bold;">●</span> {node_type}', 
                              unsafe_allow_html=True)

def download_results(results):
    """Provide download functionality for results."""
    if not results or not results.get("success"):
        return
    
    st.markdown('<h2 class="sub-header">💾 Download Results</h2>', unsafe_allow_html=True)
    
    # Convert results to JSON
    json_str = json.dumps(results, indent=2, ensure_ascii=False)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.download_button(
            label="📥 Download as JSON",
            data=json_str,
            file_name=f"knowledge_graph_{results['summary'].get('processing_mode', 'unknown')}.json",
            mime="application/json"
        )
    
    with col2:
        # Create CSV for nodes and relationships
        csv_data = create_csv_export(results)
        if csv_data:
            st.download_button(
                label="📊 Download as CSV",
                data=csv_data,
                file_name=f"knowledge_graph_{results['summary'].get('processing_mode', 'unknown')}.csv",
                mime="text/csv"
            )

def create_csv_export(results):
    """Create CSV export of the knowledge graph."""
    try:
        graphs = results.get("graphs", [])
        all_data = []
        
        for i, graph in enumerate(graphs):
            # Add nodes - data is already serialized as dictionaries
            for node in graph.get('nodes', []):
                if isinstance(node, dict):  # Dictionary format
                    node_id = node.get('id', '')
                    node_type = node.get('type', '')
                    node_properties = str(node.get('properties', {}))
                else:  # Fallback
                    node_id = str(node)
                    node_type = 'Unknown'
                    node_properties = '{}'
                
                all_data.append({
                    'Graph': i + 1,
                    'Source_Document': graph.get('source_document', 'Unknown'),
                    'Type': 'Node',
                    'ID': node_id,
                    'Category': node_type,
                    'Source': '',
                    'Target': '',
                    'Relationship': '',
                    'Properties': node_properties
                })
            
            # Add relationships - data is already serialized as dictionaries
            for rel in graph.get('relationships', []):
                if isinstance(rel, dict):  # Dictionary format
                    source = rel.get('source', '')
                    target = rel.get('target', '')
                    rel_type = rel.get('type', '')
                    rel_properties = str(rel.get('properties', {}))
                else:  # Fallback
                    source = str(rel)
                    target = 'Unknown'
                    rel_type = 'RELATES_TO'
                    rel_properties = '{}'
                
                all_data.append({
                    'Graph': i + 1,
                    'Source_Document': graph.get('source_document', 'Unknown'),
                    'Type': 'Relationship',
                    'ID': '',
                    'Category': '',
                    'Source': source,
                    'Target': target,
                    'Relationship': rel_type,
                    'Properties': rel_properties
                })
        
        df = pd.DataFrame(all_data)
        return df.to_csv(index=False)
    
    except Exception as e:
        st.error(f"Error creating CSV export: {str(e)}")
        return None

def store_knowledge_graph_in_neo4j(results):
    """Store the knowledge graph in Neo4j database using the improved method."""
    if not st.session_state.neo4j_connected or not st.session_state.neo4j_processor:
        show_notification("❌ Neo4j connection not available. Please establish connection first.", "error", 5)
        return False
    
    if not results or not results.get("success"):
        show_notification("❌ No valid knowledge graph to store.", "error", 5)
        return False
    
    try:
        with st.spinner("🗄️ Storing knowledge graph in Neo4j..."):
            processor = st.session_state.neo4j_processor
            graphs = results.get("graphs", [])
            
            total_nodes_created = 0
            total_relationships_created = 0
            
            for graph in graphs:
                # Use the improved create_full_knowledge_graph method
                try:
                    success = processor.create_full_knowledge_graph(graph)
                    if success:
                        # Get counts for this graph
                        nodes_count = len(graph.get('nodes', []))
                        relationships_count = len(graph.get('relationships', []))
                        total_nodes_created += nodes_count
                        total_relationships_created += relationships_count
                    else:
                        show_notification(f"⚠️ Warning: Failed to store graph from {graph.get('source_document', 'Unknown')}", "warning", 3)
                except Exception as graph_error:
                    show_notification(f"⚠️ Warning: Error storing graph: {str(graph_error)}", "warning", 3)
                    continue
            
            st.session_state.graph_stored_in_neo4j = True
            show_notification(f"✅ Knowledge graph stored successfully! Created {total_nodes_created} nodes and {total_relationships_created} relationships.", "success", 5)
            
            # Verify storage by checking statistics
            try:
                stats = processor.get_graph_statistics()
                if "error" not in stats:
                    show_notification(f"📊 Database now contains: {stats.get('total_nodes', 0)} nodes and {stats.get('total_relationships', 0)} relationships", "info", 4)
            except:
                pass
            
            return True
            
    except Exception as e:
        show_notification(f"❌ Error storing knowledge graph: {str(e)}", "error", 5)
        show_notification("💡 Please check your Neo4j connection and ensure the database is accessible.", "warning", 5)
        return False

def display_neo4j_panel():
    """Display the Neo4j operations panel."""
    st.markdown('<h2 class="sub-header">🗄️ Neo4j Database Operations</h2>', unsafe_allow_html=True)
    
    # Connection status and management
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        if st.session_state.neo4j_connected:
            st.success("✅ Neo4j Connection: Active")
        else:
            st.error("❌ Neo4j Connection: Not Connected")
    
    with col2:
        if not st.session_state.neo4j_connected:
            if st.button("🔌 Connect", type="primary"):
                establish_neo4j_connection()
        else:
            if st.button("🔌 Disconnect", type="secondary"):
                close_neo4j_connection()
    
    with col3:
        if st.session_state.neo4j_connected:
            if st.button("🔄 Refresh", type="secondary"):
                st.rerun()
    
    # Store knowledge graph button (if graph is available)
    if (st.session_state.processing_complete and 
        st.session_state.graph_results and 
        st.session_state.neo4j_connected and 
        not st.session_state.graph_stored_in_neo4j):
        
        st.markdown("---")
        st.markdown("### 💾 Store Knowledge Graph")
        
        col1, col2 = st.columns([3, 1])
        with col1:
            st.info("💡 You have a knowledge graph ready to be stored in Neo4j.")
        with col2:
            if st.button("🗄️ Store in Neo4j", type="primary"):
                store_knowledge_graph_in_neo4j(st.session_state.graph_results)
    
    elif st.session_state.graph_stored_in_neo4j:
        st.markdown("---")
        st.success("✅ Knowledge graph has been stored in Neo4j!")
        st.info("💡 **Troubleshooting**: If you can't see your data in Neo4j Browser, try the '🔧 Database Tools' tab to verify data storage and run diagnostic queries.")
    
    # Check if Neo4j is connected before proceeding
    if not st.session_state.neo4j_connected:
        st.error("❌ Please connect to Neo4j to access database operations.")
        st.info("💡 Make sure Neo4j is running and your connection details are correct in your environment variables.")
        st.markdown("""
        **Required Environment Variables:**
        - `NEO4J_URI` (default: bolt://localhost:7687)
        - `NEO4J_USERNAME` (default: neo4j)
        - `NEO4J_PASSWORD` (default: password)
        """)
        return
    
    # Get current statistics
    with st.spinner("📊 Loading graph statistics..."):
        stats = st.session_state.neo4j_processor.get_graph_statistics()
        if "error" not in stats:
            st.success("📊 Graph statistics loaded successfully!")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Nodes", stats.get('total_nodes', 0))
            with col2:
                st.metric("Total Relationships", stats.get('total_relationships', 0))
            with col3:
                st.metric("Node Types", len(stats.get('node_types', {})))
            with col4:
                st.metric("Relationship Types", len(stats.get('relationship_types', {})))
        else:
            st.error(f"❌ Error loading statistics: {stats.get('error')}")
            stats = {"error": "Failed to load statistics"}
    
    # Create tabs for different operations
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["🔍 Query Graph", "🗑️ Delete Operations", "📊 Statistics", "🏥 Healthcare Q&A", "🔧 Database Tools"])
    
    with tab1:
        display_query_interface()
    
    with tab2:
        display_delete_interface()
    
    with tab3:
        display_detailed_statistics(stats)
    
    with tab4:
        display_healthcare_chat()
    
    with tab5:
        display_database_tools()
        
    # Debug section for document statistics
    with st.expander("🔍 Debug Document Statistics", expanded=False):
        st.markdown("### Debug Information")
        if st.button("🔍 Check Document Statistics", key="debug_doc_stats"):
            if st.session_state.neo4j_processor:
                try:
                    # Get document count
                    doc_count = st.session_state.neo4j_processor.get_document_count()
                    st.write(f"**Document Count:** {doc_count}")
                    
                    # Get documents summary
                    doc_summary = st.session_state.neo4j_processor.get_documents_summary()
                    st.write(f"**Documents Summary:** {doc_summary}")
                    
                    # Try to query some actual documents
                    if doc_count > 0:
                        st.write("**Sample Documents:**")
                        sample_docs = st.session_state.neo4j_processor.query_nodes_by_type("Document", limit=3)
                        for i, doc in enumerate(sample_docs):
                            st.write(f"Document {i+1}: {doc}")
                    else:
                        st.write("**No documents found in database.**")
                        
                except Exception as e:
                    st.error(f"Error checking document statistics: {e}")
            else:
                st.error("Neo4j processor not available")

def display_query_interface():
    """Display the query interface."""
    st.markdown("### 🔍 Query Your Knowledge Graph")
    
    # Query type selection
    query_type = st.selectbox(
        "Select Query Type:",
        options=[
            "all_nodes",
            "nodes_by_type", 
            "relationships",
            "relationships_by_type",
            "by_keyword",
            "custom"
        ],
        format_func=lambda x: {
            "all_nodes": "All Nodes",
            "nodes_by_type": "Nodes by Type",
            "relationships": "All Relationships", 
            "relationships_by_type": "Relationships by Type",
            "by_keyword": "Search by Keyword",
            "custom": "Custom Cypher Query"
        }[x]
    )
    
    # Query parameters based on type
    params = {}
    
    if query_type == "nodes_by_type":
        node_type = st.selectbox(
            "Select Node Type:",
            options=["Concept", "Entity", "Topic", "Process", "Method", "Tool", 
                    "Condition", "Symptom", "Treatment", "Person", "Organization",
                    "Location", "Event", "Technology", "Product", "Service", "Document"]
        )
        params['node_type'] = node_type
    
    elif query_type == "relationships_by_type":
        rel_type = st.selectbox(
            "Select Relationship Type:",
            options=["RELATES_TO", "IS_A", "PART_OF", "CAUSES", "TREATS", "REQUIRES",
                    "FOLLOWS", "CONTAINS", "SIMILAR_TO", "LOCATED_IN", "WORKS_FOR",
                    "CREATES", "USES", "INFLUENCES", "DEPENDS_ON", "BELONGS_TO"]
        )
        params['rel_type'] = rel_type
    
    elif query_type == "by_keyword":
        keyword = st.text_input("Enter keyword to search:")
        if keyword:
            params['keyword'] = keyword
            node_types = st.multiselect(
                "Filter by Node Types (optional):",
                options=["Concept", "Entity", "Topic", "Process", "Method", "Tool",
                        "Condition", "Symptom", "Treatment", "Person", "Organization",
                        "Location", "Event", "Technology", "Product", "Service"]
            )
            if node_types:
                params['node_types'] = node_types
    
    elif query_type == "custom":
        st.markdown("**Custom Cypher Query**")
        st.info("💡 Write your own Cypher query to explore the graph.")
        custom_query = st.text_area(
            "Enter Cypher Query:",
            placeholder="MATCH (n) RETURN n LIMIT 10",
            height=100
        )
        if custom_query:
            params['query'] = custom_query
    
    # Limit parameter for most queries
    if query_type != "custom":
        limit = st.slider("Limit results:", min_value=1, max_value=1000, value=50, step=10)
        params['limit'] = limit
    
    # Execute query button
    if st.button("🚀 Execute Query", type="primary"):
        if query_type == "by_keyword" and not params.get('keyword'):
            st.error("❌ Please enter a keyword to search.")
        elif query_type == "custom" and not params.get('query'):
            st.error("❌ Please enter a Cypher query.")
        else:
            with st.spinner("🔍 Executing query..."):
                try:
                    processor = st.session_state.neo4j_processor
                    
                    if query_type == "all_nodes":
                        results = processor.query_all_nodes(**params)
                    elif query_type == "nodes_by_type":
                        results = processor.query_nodes_by_type(**params)
                    elif query_type == "relationships":
                        results = processor.query_relationships(**params)
                    elif query_type == "relationships_by_type":
                        results = processor.query_relationships_by_type(**params)
                    elif query_type == "by_keyword":
                        results = processor.query_by_keyword(**params)
                    elif query_type == "custom":
                        results = processor.execute_custom_query(**params)
                    else:
                        results = []
                    
                    if results:
                        show_notification(f"✅ Query executed successfully! Found {len(results)} results.", "success", 3)
                        display_query_results(results, query_type)
                    else:
                        show_notification("ℹ️ No results found for this query. Try adjusting your search criteria.", "info", 4)
                        
                except Exception as e:
                    show_notification(f"❌ Error executing query: {str(e)}", "error", 5)

def display_query_results(results, query_type):
    """Display query results in a formatted way."""
    st.markdown("### 📋 Query Results")
    
    if query_type == "custom":
        # For custom queries, display as JSON
        st.json(results)
    elif query_type in ["relationships", "relationships_by_type"]:
        # For relationships, display in a table format
        rel_data = []
        for rel in results:
            source = rel.get('source', {}).get('id', 'Unknown')
            target = rel.get('target', {}).get('id', 'Unknown')
            rel_type = rel.get('relationship', {}).get('type', 'Unknown')
            rel_data.append({
                'Source': source,
                'Relationship': rel_type,
                'Target': target
            })
        
        if rel_data:
            df = pd.DataFrame(rel_data)
            st.dataframe(df, use_container_width=True)
            
            # Download option
            csv = df.to_csv(index=False)
            st.download_button(
                label="📥 Download as CSV",
                data=csv,
                file_name=f"neo4j_query_results_{query_type}.csv",
                mime="text/csv"
            )
    else:
        # For node queries, display in a table format
        node_data = []
        for node in results:
            node_data.append({
                'ID': node.get('id', 'Unknown'),
                'Type': node.get('type', 'Unknown'),
                'Properties': str(node.get('properties', {}))
            })
        
        if node_data:
            df = pd.DataFrame(node_data)
            st.dataframe(df, use_container_width=True)
            
            # Download option
            csv = df.to_csv(index=False)
            st.download_button(
                label="📥 Download as CSV",
                data=csv,
                file_name=f"neo4j_query_results_{query_type}.csv",
                mime="text/csv"
            )

def display_delete_interface():
    """Display the delete operations interface."""
    st.markdown("### 🗑️ Delete Operations")
    st.warning("⚠️ **Warning**: Delete operations are irreversible. Please be careful!")
    
    # Delete type selection
    delete_type = st.selectbox(
        "Select Delete Operation:",
        options=["knowledge_nodes", "by_source", "entire"],
        format_func=lambda x: {
            "knowledge_nodes": "Delete Knowledge Graph Nodes Only",
            "by_source": "Delete by Source File",
            "entire": "Delete Entire Graph (ALL DATA)"
        }[x]
    )
    
    # Additional parameters for specific delete types
    if delete_type == "by_source":
        # Get available source files
        try:
            sources_query = "MATCH (d:Document) RETURN DISTINCT d.source_file as source"
            sources = st.session_state.neo4j_processor.execute_custom_query(sources_query)
            source_files = [s['source'] for s in sources if s.get('source')]
            
            if source_files:
                selected_source = st.selectbox("Select Source File:", source_files)
                source_file = selected_source
            else:
                st.info("No source files found in the database.")
                source_file = None
        except:
            source_file = st.text_input("Enter source file name:")
    else:
        source_file = None
    
    # Confirmation for dangerous operations
    if delete_type == "entire":
        st.error("🚨 **DANGER**: This will delete ALL nodes and relationships from the database!")
        confirmation = st.text_input("Type 'DELETE ALL' to confirm:")
        can_delete = confirmation == "DELETE ALL"
    else:
        can_delete = True
    
    # Delete button
    col1, col2 = st.columns([1, 3])
    with col1:
        if st.button("🗑️ Delete", type="secondary", disabled=not can_delete):
            if delete_type == "by_source" and not source_file:
                st.error("❌ Please select a source file.")
            else:
                with st.spinner("🗑️ Deleting data..."):
                    try:
                        processor = st.session_state.neo4j_processor
                        
                        if delete_type == "entire":
                            success = processor.delete_entire_graph()
                        elif delete_type == "knowledge_nodes":
                            success = processor.delete_knowledge_graph_nodes()
                        elif delete_type == "by_source":
                            success = processor.delete_by_source_file(source_file)
                        else:
                            success = False
                        
                        if success:
                            show_notification("✅ Delete operation completed successfully!", "success", 3)
                            st.balloons()  # Celebrate successful deletion
                            st.rerun()  # Refresh the page to update statistics
                        else:
                            show_notification("❌ Delete operation failed. Please check your Neo4j connection.", "error", 5)
                            
                    except Exception as e:
                        show_notification(f"❌ Error during delete operation: {str(e)}", "error", 5)
    
    with col2:
        if delete_type == "knowledge_nodes":
            st.info("💡 This will delete only the knowledge graph nodes (Concept, Entity, etc.) while keeping Document nodes.")
        elif delete_type == "by_source":
            st.info("💡 This will delete all nodes and relationships related to the selected source file.")
        elif delete_type == "entire":
            st.info("💡 This will delete ALL data from the Neo4j database.")

def display_detailed_statistics(stats):
    """Display detailed graph statistics."""
    st.markdown("### 📊 Detailed Statistics")
    
    if "error" in stats:
        st.error(f"Error retrieving statistics: {stats['error']}")
        return
    
    # Node types breakdown
    st.markdown("#### Node Types")
    if stats.get('node_types'):
        node_types_df = pd.DataFrame([
            {'Type': k, 'Count': v} for k, v in stats['node_types'].items()
        ])
        st.dataframe(node_types_df, use_container_width=True)
        
        # Pie chart
        fig = px.pie(node_types_df, values='Count', names='Type', title="Node Type Distribution")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No node types found.")
    
    # Relationship types breakdown
    st.markdown("#### Relationship Types")
    if stats.get('relationship_types'):
        rel_types_df = pd.DataFrame([
            {'Type': k, 'Count': v} for k, v in stats['relationship_types'].items()
        ])
        st.dataframe(rel_types_df, use_container_width=True)
        
        # Bar chart
        fig = px.bar(rel_types_df, x='Type', y='Count', title="Relationship Type Distribution")
        fig.update_xaxes(tickangle=45)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No relationship types found.")
    
    # Document statistics
    st.markdown("#### Document Statistics")
    if stats.get('document_stats'):
        doc_stats = stats['document_stats']
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Documents", doc_stats.get('total_docs', 0))
        with col2:
            st.metric("Unique Sources", doc_stats.get('unique_sources', 0))
        with col3:
            st.metric("File Types", doc_stats.get('file_types', 0))
    else:
        st.info("No document statistics available. This might be because no documents have been processed and stored in the database yet.")

def display_healthcare_chat():
    """Display the healthcare Q&A chat interface."""
    st.markdown('<h2 class="sub-header">🏥 Healthcare Q&A Assistant</h2>', unsafe_allow_html=True)
    
    # Check if Neo4j is connected
    if not st.session_state.neo4j_connected:
        st.error("❌ Neo4j connection required for Healthcare Agent.")
        st.info("💡 Please connect to Neo4j first to use the healthcare assistant.")
        return
    
    # Initialize agent if not already done
    if not st.session_state.healthcare_agent:
        col1, col2 = st.columns([3, 1])
        with col1:
            st.info("🤖 Healthcare Agent needs to be initialized to provide context-aware responses.")
        with col2:
            if st.button("🤖 Initialize Agent", type="primary", key="init_agent_btn"):
                initialize_healthcare_agent()
                # Don't use st.rerun() here
        return
    
    # Display agent status
    st.success("✅ Healthcare Agent is ready to answer your questions!")
    
    # Disclaimer
    st.warning("""
    ⚠️ **Medical Disclaimer**: This assistant provides general health information for educational purposes only. 
    It is not a substitute for professional medical advice, diagnosis, or treatment. 
    Always consult with qualified healthcare professionals for medical decisions.
    """)
    
    # Chat interface
    st.markdown("### 💬 Ask Your Healthcare Question")
    
    # Question input
    question = st.text_area(
        "Enter your healthcare question:",
        placeholder="e.g., What are the symptoms of diabetes? How can I manage blood sugar levels?",
        height=100,
        help="Ask any healthcare-related question and the agent will use the knowledge graph to provide context-aware responses."
    )
    
    # Submit button
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        if st.button("🚀 Ask Question", type="primary", disabled=not question.strip(), key="ask_question_btn"):
            if question.strip():
                with st.spinner("🤖 Processing your question..."):
                    try:
                        response = st.session_state.healthcare_agent.get_response(question.strip())
                        
                        # Add to chat history
                        st.session_state.chat_history.append({
                            "question": question.strip(),
                            "response": response,
                            "timestamp": pd.Timestamp.now()
                        })
                        
                        show_notification("✅ Response generated!", "success", 3)
                        # Don't use st.rerun() here - let the page update naturally
                        
                    except Exception as e:
                        show_notification(f"❌ Error generating response: {str(e)}", "error", 5)
    
    with col2:
        if st.button("🗑️ Clear Chat", type="secondary", key="clear_chat_btn"):
            st.session_state.chat_history = []
            st.session_state.healthcare_agent.clear_memory()
            show_notification("✅ Chat history cleared!", "success", 3)
            # Don't use st.rerun() here
    
    with col3:
        if st.button("🔄 Refresh Agent", type="secondary", key="refresh_agent_btn"):
            initialize_healthcare_agent()
            show_notification("✅ Agent refreshed!", "success", 3)
            # Don't use st.rerun() here
    
    # Display chat history
    if st.session_state.chat_history:
        st.markdown("### 📋 Conversation History")
        
        # Display in reverse chronological order (newest first)
        for i, chat in enumerate(reversed(st.session_state.chat_history)):
            with st.expander(f"💬 Q&A {len(st.session_state.chat_history) - i}", expanded=i==0):
                # Question section
                st.markdown("**❓ Question:**")
                st.info(chat["question"])
                
                # Response section
                st.markdown("**🤖 Response:**")
                st.success(chat["response"])
                
                # Timestamp
                if "timestamp" in chat:
                    st.caption(f"⏰ Asked on: {chat['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
                
                # Add a separator
                st.markdown("---")
    
    # Sample questions
    st.markdown("### 💡 Sample Questions")
    sample_questions = [
        "What is diabetes and how does it affect the body?",
        "What are the common symptoms of diabetes?",
        "How can I manage my blood sugar levels through diet?",
        "What exercises are recommended for diabetes patients?",
        "What medications are commonly used to treat diabetes?",
        "How often should I check my blood sugar levels?",
        "What are the complications of uncontrolled diabetes?",
        "How can I prevent diabetes complications?"
    ]
    
    cols = st.columns(2)
    for i, sample_q in enumerate(sample_questions):
        with cols[i % 2]:
            if st.button(f"💭 {sample_q[:50]}...", key=f"sample_{i}"):
                st.session_state.sample_question = sample_q
                # Don't use st.rerun() here
    
    # Handle sample question selection
    if hasattr(st.session_state, 'sample_question'):
        st.text_area("Selected question:", value=st.session_state.sample_question, disabled=True)
        if st.button("🚀 Ask This Question", type="primary", key="ask_sample_btn"):
            question = st.session_state.sample_question
            with st.spinner("🤖 Processing your question..."):
                try:
                    response = st.session_state.healthcare_agent.get_response(question)
                    
                    # Add to chat history
                    st.session_state.chat_history.append({
                        "question": question,
                        "response": response,
                        "timestamp": pd.Timestamp.now()
                    })
                    
                    show_notification("✅ Response generated!", "success", 3)
                    del st.session_state.sample_question
                    # Don't use st.rerun() here
                    
                except Exception as e:
                    show_notification(f"❌ Error generating response: {str(e)}", "error", 5)

def display_database_tools():
    """Display database maintenance and verification tools."""
    st.markdown("### 🔧 Database Tools")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📊 Quick Verification")
        
        if st.button("🔍 Verify Data Storage", type="primary"):
            try:
                processor = st.session_state.neo4j_processor
                
                # Check total nodes
                node_count = processor.execute_custom_query("MATCH (n) RETURN count(n) as count")
                total_nodes = node_count[0]['count'] if node_count else 0
                
                # Check total relationships
                rel_count = processor.execute_custom_query("MATCH ()-[r]->() RETURN count(r) as count")
                total_relationships = rel_count[0]['count'] if rel_count else 0
                
                # Check node types
                node_types = processor.execute_custom_query("MATCH (n) RETURN DISTINCT labels(n) as types")
                
                # Check relationship types
                rel_types = processor.execute_custom_query("MATCH ()-[r]->() RETURN DISTINCT type(r) as types")
                
                show_notification(f"✅ Database contains {total_nodes} nodes and {total_relationships} relationships", "success", 4)
                
                if node_types:
                    show_notification(f"Node types found: {[t['types'] for t in node_types]}", "info", 4)
                
                if rel_types:
                    show_notification(f"Relationship types found: {[t['types'] for t in rel_types]}", "info", 4)
                
                # Show sample data
                sample_nodes = processor.execute_custom_query("MATCH (n) RETURN n LIMIT 5")
                if sample_nodes:
                    st.markdown("**Sample Nodes:**")
                    for i, node in enumerate(sample_nodes):
                        st.json(node)
                
            except Exception as e:
                show_notification(f"❌ Error verifying data: {str(e)}", "error", 5)
        
        if st.button("🔧 Create Indexes", type="secondary"):
            try:
                create_neo4j_indexes(st.session_state.neo4j_processor)
                show_notification("✅ Indexes created successfully!", "success", 3)
            except Exception as e:
                show_notification(f"❌ Error creating indexes: {str(e)}", "error", 5)
    
    with col2:
        st.markdown("#### 🧹 Database Maintenance")
        
        if st.button("🗑️ Clear All Data", type="secondary"):
            st.warning("⚠️ This will delete ALL data from the database!")
            confirmation = st.text_input("Type 'DELETE ALL' to confirm:")
            
            if confirmation == "DELETE ALL":
                try:
                    processor = st.session_state.neo4j_processor
                    processor.execute_custom_query("MATCH (n) DETACH DELETE n")
                    show_notification("✅ All data cleared successfully!", "success", 3)
                    st.rerun()
                except Exception as e:
                    show_notification(f"❌ Error clearing data: {str(e)}", "error", 5)
        
        st.markdown("#### 📋 Useful Cypher Queries")
        
        useful_queries = {
            "Count all nodes": "MATCH (n) RETURN count(n) as total_nodes",
            "Count all relationships": "MATCH ()-[r]->() RETURN count(r) as total_relationships",
            "Show node types": "MATCH (n) RETURN DISTINCT labels(n) as node_types",
            "Show relationship types": "MATCH ()-[r]->() RETURN DISTINCT type(r) as relationship_types",
            "Sample nodes": "MATCH (n) RETURN n LIMIT 10",
            "Sample relationships": "MATCH (a)-[r]->(b) RETURN a, r, b LIMIT 10"
        }
        
        selected_query = st.selectbox("Select a query to run:", list(useful_queries.keys()))
        
        if st.button("🚀 Run Query", type="primary"):
            try:
                processor = st.session_state.neo4j_processor
                result = processor.execute_custom_query(useful_queries[selected_query])
                show_notification("✅ Query executed successfully!", "success", 3)
                st.json(result)
            except Exception as e:
                show_notification(f"❌ Error executing query: {str(e)}", "error", 5)

def main():
    """Main Streamlit application."""
    initialize_session_state()
    display_header()
    
    # Display notifications
    display_notifications()
    
    # Cleanup function for when session ends
    def cleanup():
        if st.session_state.neo4j_processor:
            st.session_state.neo4j_processor.close()
    
    # Register cleanup function
    import atexit
    atexit.register(cleanup)
    
    # Sidebar configuration
    mode, chunk_size, chunk_overlap, show_neo4j_panel = display_sidebar()
    
    # Main content area
    st.markdown('<h2 class="sub-header">📁 Upload Documents</h2>', unsafe_allow_html=True)
    
    uploaded_files = st.file_uploader(
        "Choose files to process",
        type=["txt", "pdf"],
        accept_multiple_files=True,
        help="Upload one or more text (.txt) or PDF (.pdf) files"
    )
    
    if uploaded_files:
        # Display uploaded files info
        st.markdown("### Uploaded Files")
        for file in uploaded_files:
            st.write(f"- **{file.name}** ({file.size:,} bytes)")
        
        # Process button
        if st.button("🚀 Build Knowledge Graph", type="primary", use_container_width=True):
            
            # Reset session state
            st.session_state.processing_complete = False
            st.session_state.graph_results = None
            st.session_state.graph_stored_in_neo4j = False
            
            # Process files
            results = process_uploaded_files(uploaded_files, mode, chunk_size, chunk_overlap)
            
            if results:
                st.session_state.graph_results = results
                st.session_state.processing_complete = True
                st.session_state.last_mode = mode
                st.rerun()
    
    # Display results if available
    if st.session_state.processing_complete and st.session_state.graph_results:
        display_results(st.session_state.graph_results)
        download_results(st.session_state.graph_results)
        
        # Add Neo4j panel after results are displayed
        st.markdown("---")
        display_neo4j_panel()
    
    # Standalone Neo4j panel (if requested from sidebar)
    elif show_neo4j_panel:
        st.markdown("---")
        display_neo4j_panel()
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style="text-align: center; color: #666; margin-top: 2rem;">
            <p>Built with Streamlit • Powered by LangChain & Google Gemini</p>
        </div>
        """, 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()