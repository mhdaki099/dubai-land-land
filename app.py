import streamlit as st
import pandas as pd
import json
import os
import re
import openai
from dotenv import load_dotenv
import numpy as np
import logging

# Try to import the agent system
try:
    from agents import FAQAgentSystem
    AGENTS_AVAILABLE = True
    logging.info("Agent system imported successfully")
except ImportError as e:
    AGENTS_AVAILABLE = False
    logging.warning(f"Agent system not available: {e}")

# Load environment variables
load_dotenv()

# Configure OpenAI API
openai.api_key = os.getenv("OPENAI_API_KEY")

# Configure logging
logging.basicConfig(level=logging.INFO)

# Page configuration
st.set_page_config(
    page_title="DLD FAQ Chatbot",
    page_icon="🏢",
    layout="wide"
)

# Custom CSS (same as before)
st.markdown("""
<style>
    .main-title {
        font-size: 2.8rem;
        color: #0f4c81;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: 700;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.1);
    }
    .sub-title {
        font-size: 1.6rem;
        color: #555;
        text-align: center;
        margin-bottom: 2.5rem;
        font-weight: 400;
    }
    .user-message {
        background-color: #e6f3ff;
        padding: 1.2rem;
        border-radius: 15px;
        margin: 1rem 0;
        border-left: 5px solid #0f4c81;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        max-width: 85%;
        margin-left: auto;
    }
    .bot-message {
        background-color: #f8f9fa;
        padding: 1.2rem;
        border-radius: 15px;
        margin: 1rem 0;
        border-left: 5px solid #555;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        max-width: 85%;
    }
    .source-info {
        font-size: 0.95rem;
        color: #0f4c81;
        margin-top: 1rem;
        font-style: italic;
        border-top: 1px dashed #ddd;
        padding-top: 0.8rem;
        background-color: #f8f9fa;
        padding: 0.8rem;
        border-radius: 8px;
    }
    .debug-info {
        font-size: 0.85rem;
        color: #666;
        margin-top: 0.8rem;
        border-top: 1px solid #eee;
        padding-top: 0.8rem;
        background-color: #fafafa;
        padding: 0.8rem;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state variables
if 'messages' not in st.session_state:
    st.session_state.messages = []
    
if 'agent_system' not in st.session_state:
    st.session_state.agent_system = None
    
if 'faq_data' not in st.session_state:
    st.session_state.faq_data = None
    
if 'topics' not in st.session_state:
    st.session_state.topics = ["All Topics"]
    
if 'selected_topic' not in st.session_state:
    st.session_state.selected_topic = "All Topics"
    
if 'debug_mode' not in st.session_state:
    st.session_state.debug_mode = False
    
if 'show_sources' not in st.session_state:
    st.session_state.show_sources = True

def initialize_agent_system():
    """Initialize the FAQ Agent System if available."""
    if AGENTS_AVAILABLE and st.session_state.agent_system is None:
        try:
            # Check if processed data exists
            if os.path.exists("processed_data/processed_faq_data.csv"):
                st.session_state.agent_system = FAQAgentSystem()
                return True
            else:
                st.error("Processed data not found. Please run the data processor first.")
                return False
        except Exception as e:
            st.error(f"Error initializing agent system: {str(e)}")
            return False
    return AGENTS_AVAILABLE

def load_faq_data_fallback():
    """Load FAQ data as fallback when agents are not available."""
    csv_path = os.path.join("processed_data", "dld_faq_data.csv")
    json_path = os.path.join("processed_data", "dld_faq_data.json")
    
    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)
            st.sidebar.success(f"Loaded {len(df)} FAQ items from CSV")
            
            # Get unique services as topics
            if 'Service' in df.columns:
                services = sorted(df["Service"].unique().tolist())
                topics = ["All Topics"] + services
            else:
                topics = ["All Topics"]
            
            return df, topics
            
        except Exception as e:
            st.error(f"Error loading CSV data: {str(e)}")
    
    elif os.path.exists(json_path):
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                faq_data = json.load(f)
                
            df = pd.DataFrame(faq_data)
            st.sidebar.success(f"Loaded {len(df)} FAQ items from JSON")
            
            # Get unique services as topics
            if 'Service' in df.columns:
                services = sorted(df["Service"].unique().tolist())
                topics = ["All Topics"] + services
            else:
                topics = ["All Topics"]
            
            return df, topics
            
        except Exception as e:
            st.error(f"Error loading JSON data: {str(e)}")
    
    # If no data files found
    st.error("No FAQ data files found. Please run the processor first.")
    return pd.DataFrame(), ["All Topics"]

def process_query_with_agents(query, selected_cluster=None):
    """Process query using the agent system."""
    try:
        result = st.session_state.agent_system.process_query(query, selected_cluster)
        return {
            'response': result['response'],
            'detected_language': result['detected_language'],
            'selected_cluster': result['selected_cluster'],
            'debug_info': f"Language: {result['detected_language']}, Cluster: {result['selected_cluster']}"
        }
    except Exception as e:
        st.error(f"Error processing query with agents: {str(e)}")
        return {
            'response': "Sorry, there was an error processing your request. Please try again.",
            'detected_language': 'english',
            'selected_cluster': selected_cluster,
            'debug_info': f"Error: {str(e)}"
        }

def process_query_fallback(query, df, selected_topic):
    """Fallback processing when agents are not available."""
    # Simple text-based search as fallback
    if df.empty:
        return {
            'response': "No FAQ data available. Please run the data processor first.",
            'debug_info': "No data available"
        }
    
    # Filter by topic if specified
    if selected_topic and selected_topic != "All Topics":
        if 'Service' in df.columns:
            filtered_df = df[df['Service'] == selected_topic]
            if len(filtered_df) == 0:
                filtered_df = df
        else:
            filtered_df = df
    else:
        filtered_df = df
    
    # Simple keyword search
    query_lower = query.lower()
    matches = []
    
    for idx, row in filtered_df.iterrows():
        question = str(row.get('Question (English)', '')).lower()
        answer = str(row.get('Answer (English)', ''))
        
        # Calculate simple relevance score
        score = 0
        if query_lower in question:
            score += 10
        
        # Check word overlap
        query_words = set(query_lower.split())
        question_words = set(question.split())
        overlap = len(query_words.intersection(question_words))
        score += overlap * 2
        
        if score > 0:
            matches.append({
                'score': score,
                'question': row.get('Question (English)', ''),
                'answer': answer,
                'service': row.get('Service', 'Unknown')
            })
    
    # Sort by relevance
    matches.sort(key=lambda x: x['score'], reverse=True)
    
    if matches:
        # Return the best match
        best_match = matches[0]
        return {
            'response': best_match['answer'],
            'debug_info': f"Found {len(matches)} matches, using best match from {best_match['service']}"
        }
    else:
        return {
            'response': "I'm sorry, I couldn't find an answer to your question. Could you rephrase your question or ask something else?",
            'debug_info': "No matches found"
        }

def main():
    # Main title
    st.markdown('<h1 class="main-title">DLD FAQ Assistant</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">Ask me anything about Dubai Land Department services</p>', unsafe_allow_html=True)
    
    # Try to initialize agent system or load data
    if AGENTS_AVAILABLE:
        agent_ready = initialize_agent_system()
        if agent_ready and st.session_state.agent_system:
            # Get available clusters from agent system
            topics = ["All Topics"] + st.session_state.agent_system.get_available_clusters()
            st.session_state.topics = topics
            data_status = "✅ Agent system ready"
        else:
            data_status = "❌ Agent system failed to load"
            # Fallback to direct data loading
            df, topics = load_faq_data_fallback()
            st.session_state.faq_data = df
            st.session_state.topics = topics
    else:
        # Load data directly
        if st.session_state.faq_data is None:
            with st.spinner("Loading FAQ data..."):
                df, topics = load_faq_data_fallback()
                st.session_state.faq_data = df
                st.session_state.topics = topics
        data_status = "⚠️ Using fallback mode (agents not available)"
    
    # Sidebar
    with st.sidebar:
        st.markdown("### Topic Selection")
        selected_topic = st.selectbox(
            "Select a topic for your question",
            st.session_state.topics,
            index=st.session_state.topics.index(st.session_state.selected_topic) if st.session_state.selected_topic in st.session_state.topics else 0
        )
        st.session_state.selected_topic = selected_topic
        
        st.markdown("### Settings")
        col1, col2 = st.columns(2)
        with col1:
            st.session_state.debug_mode = st.checkbox("Debug Mode", value=st.session_state.debug_mode)
        with col2:
            st.session_state.show_sources = st.checkbox("Show Sources", value=st.session_state.show_sources)
        
        st.markdown("### System Status")
        st.info(data_status)
        
        if AGENTS_AVAILABLE and st.session_state.agent_system:
            st.success("🤖 Advanced agent system active")
        elif not st.session_state.faq_data.empty if hasattr(st.session_state, 'faq_data') and st.session_state.faq_data is not None else False:
            st.info(f"📚 {len(st.session_state.faq_data)} FAQ items loaded")
        else:
            st.error("❌ No data available")
        
        st.markdown("---")
        if st.button("🔄 Reset Conversation", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
    
    # Quick tips
    with st.expander("💡 Quick Tips", expanded=False):
        st.markdown("""
        - Type your question in English or Arabic
        - Select a specific topic for more accurate answers
        - Use clear and specific language for best results
        - The system uses advanced AI to understand your questions
        """)
    
    # Display chat messages
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.messages:
            if message["role"] == "user":
                st.markdown(f'<div class="user-message">👤 {message["content"]}</div>', unsafe_allow_html=True)
            else:
                content = message["content"]
                debug_info = message.get("debug_info", "")
                
                # Display the message
                st.markdown(f'<div class="bot-message">🏢 {content}</div>', unsafe_allow_html=True)
                
                # Show debug info if enabled
                if st.session_state.debug_mode and debug_info:
                    st.markdown(f'<div class="debug-info">{debug_info}</div>', unsafe_allow_html=True)
    
    # User input
    user_query = st.chat_input("Type your question here...")
    
    # Process user query
    if user_query:
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": user_query})
        st.rerun()
    
    # If there's a user message without a response, generate a response
    if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
        with st.spinner("Finding the answer..."):
            user_query = st.session_state.messages[-1]["content"]
            
            # Process query
            if AGENTS_AVAILABLE and st.session_state.agent_system:
                # Use agent system
                result = process_query_with_agents(user_query, st.session_state.selected_topic)
            else:
                # Use fallback method
                result = process_query_fallback(user_query, st.session_state.faq_data, st.session_state.selected_topic)
            
            # Add bot response to chat history
            st.session_state.messages.append({
                "role": "assistant",
                "content": result['response'],
                "debug_info": result.get('debug_info', '')
            })
            
            st.rerun()

if __name__ == "__main__":
    main()
