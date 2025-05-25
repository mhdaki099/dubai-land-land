import streamlit as st
import pandas as pd
import os
import re
import openai

# =============================================================================
# STREAMLIT CLOUD CONFIGURATION
# =============================================================================

def setup_openai():
    """Setup OpenAI API."""
    try:
        if hasattr(st, 'secrets'):
            if 'openai' in st.secrets and 'OPENAI_API_KEY' in st.secrets['openai']:
                api_key = st.secrets['openai']['OPENAI_API_KEY']
            elif 'OPENAI_API_KEY' in st.secrets:
                api_key = st.secrets['OPENAI_API_KEY']
            else:
                st.error("❌ No OpenAI API key found in secrets!")
                return None
        else:
            st.error("❌ No secrets found!")
            return None
        
        openai.api_key = api_key
        return api_key
        
    except Exception as e:
        st.error(f"❌ Error setting up OpenAI: {str(e)}")
        return None

# =============================================================================
# SIMPLE DATA LOADER
# =============================================================================

@st.cache_data
def load_excel_data():
    """Load Excel data simply and efficiently."""
    data_dir = "data"
    
    if not os.path.exists(data_dir):
        st.error(f"❌ Data directory '{data_dir}' not found!")
        return pd.DataFrame(), []
    
    excel_files = [f for f in os.listdir(data_dir) if f.endswith('.xlsx')]
    
    if not excel_files:
        st.error(f"❌ No Excel files found in '{data_dir}' directory!")
        return pd.DataFrame(), []
    
    all_data = []
    
    # Simple progress indicator
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for idx, filename in enumerate(excel_files):
        try:
            status_text.text(f"Loading {filename}...")
            progress_bar.progress((idx + 1) / len(excel_files))
            
            file_path = os.path.join(data_dir, filename)
            service_name = filename.replace('FAQs.xlsx', '').replace('.xlsx', '').strip()
            
            # Try to read Excel with different header positions
            df = None
            for header_row in [1, 2, 0]:
                try:
                    temp_df = pd.read_excel(file_path, header=header_row)
                    temp_df = temp_df.dropna(how='all')
                    
                    # Look for question and answer columns
                    question_col = None
                    answer_col = None
                    
                    for col in temp_df.columns:
                        col_str = str(col).lower()
                        if 'question' in col_str and 'eng' in col_str:
                            question_col = col
                        elif 'answer' in col_str and 'eng' in col_str:
                            answer_col = col
                    
                    if question_col and answer_col:
                        df = temp_df
                        break
                        
                except Exception:
                    continue
            
            if df is not None and question_col and answer_col:
                # Clean and prepare data
                clean_df = df[[question_col, answer_col]].copy()
                clean_df.columns = ['Question', 'Answer']
                clean_df['Service'] = service_name
                
                # Remove empty rows
                clean_df = clean_df.dropna(subset=['Question', 'Answer'])
                clean_df = clean_df[clean_df['Question'].str.len() > 5]
                clean_df = clean_df[clean_df['Answer'].str.len() > 10]
                
                if len(clean_df) > 0:
                    all_data.append(clean_df)
                    
        except Exception as e:
            st.warning(f"⚠️ Could not process {filename}: {str(e)}")
            continue
    
    # Clear progress indicators
    progress_bar.empty()
    status_text.empty()
    
    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)
        combined_df = combined_df.drop_duplicates(subset=['Question'], keep='first')
        
        services = sorted(combined_df['Service'].unique().tolist())
        topics = ["All Topics"] + services
        
        st.success(f"✅ Loaded {len(combined_df)} FAQ items from {len(all_data)} files")
        return combined_df, topics
    else:
        st.error("❌ No valid data found in Excel files")
        return pd.DataFrame(), []

# =============================================================================
# SIMPLE SEARCH FUNCTION
# =============================================================================

def simple_search(query, df, topic=None, top_k=3):
    """Simple text-based search without embeddings."""
    if df.empty or not query:
        return []
    
    # Filter by topic if specified
    search_df = df
    if topic and topic != "All Topics":
        search_df = df[df['Service'] == topic]
        if search_df.empty:
            search_df = df
    
    query_lower = query.lower()
    results = []
    
    for idx, row in search_df.iterrows():
        question = row['Question'].lower()
        answer = row['Answer']
        service = row['Service']
        
        # Calculate simple relevance score
        score = 0
        
        # Exact phrase match
        if query_lower in question:
            score += 20
        
        # Word overlap
        query_words = set(query_lower.split())
        question_words = set(question.split())
        common_words = query_words.intersection(question_words)
        score += len(common_words) * 5
        
        # Keyword matching
        important_keywords = ['register', 'registration', 'mortgage', 'property', 'document', 'fee', 'time', 'ejari']
        for keyword in important_keywords:
            if keyword in query_lower and keyword in question:
                score += 10
        
        if score > 0:
            results.append({
                'question': row['Question'],
                'answer': answer,
                'service': service,
                'score': score
            })
    
    # Sort by score and return top results
    results.sort(key=lambda x: x['score'], reverse=True)
    return results[:top_k]

# =============================================================================
# SIMPLE RESPONSE GENERATION
# =============================================================================

def generate_simple_response(query, search_results):
    """Generate response using OpenAI with search results."""
    if not search_results:
        return "I'm sorry, I couldn't find an answer to your question. Could you please rephrase it or ask about Dubai Land Department services?"
    
    # If we have a single, highly relevant result, return it directly
    if len(search_results) == 1 and search_results[0]['score'] > 15:
        return search_results[0]['answer']
    
    # Prepare context from search results
    context = ""
    for i, result in enumerate(search_results):
        context += f"\nQ: {result['question']}\nA: {result['answer']}\nService: {result['service']}\n"
    
    try:
        system_prompt = """You are a helpful assistant for the Dubai Land Department. 
        Based on the provided FAQ information, give a clear and helpful answer to the user's question.
        Use a professional but friendly tone. Only use information from the provided FAQs."""
        
        response = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"User question: {query}\n\nRelevant FAQs:\n{context}"}
            ],
            max_tokens=500
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        st.error(f"Error generating response: {str(e)}")
        # Fallback to best match
        return search_results[0]['answer']

# =============================================================================
# STREAMLIT APP
# =============================================================================

def main():
    # Page config
    st.set_page_config(
        page_title="DLD FAQ Chatbot",
        page_icon="🏢",
        layout="wide"
    )
    
    # Custom CSS
    st.markdown("""
    <style>
        .main-title {
            font-size: 2.5rem;
            color: #0f4c81;
            text-align: center;
            margin-bottom: 1rem;
            font-weight: 700;
        }
        .user-message {
            background-color: #e6f3ff;
            padding: 1rem;
            border-radius: 10px;
            margin: 10px 0;
            border-left: 5px solid #0f4c81;
        }
        .bot-message {
            background-color: #f8f9fa;
            padding: 1rem;
            border-radius: 10px;
            margin: 10px 0;
            border-left: 5px solid #555;
        }
        .source-info {
            font-size: 0.9rem;
            color: #0f4c81;
            margin-top: 10px;
            font-style: italic;
            border-top: 1px dashed #ddd;
            padding-top: 5px;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Title
    st.markdown('<h1 class="main-title">🏢 DLD FAQ Assistant</h1>', unsafe_allow_html=True)
    st.markdown("### Ask me anything about Dubai Land Department services")
    
    # Setup OpenAI
    api_key = setup_openai()
    if not api_key:
        st.error("Please configure your OpenAI API key in Streamlit secrets.")
        st.info("""
        **To configure secrets:**
        1. Go to your Streamlit Cloud dashboard
        2. Click Settings → Secrets
        3. Add:
        ```
        [openai]
        OPENAI_API_KEY = "sk-your-api-key-here"
        ```
        """)
        st.stop()
    
    # Initialize session state
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'faq_data' not in st.session_state:
        st.session_state.faq_data = None
    if 'topics' not in st.session_state:
        st.session_state.topics = ["All Topics"]
    
    # Load data
    if st.session_state.faq_data is None:
        with st.spinner("Loading FAQ data..."):
            df, topics = load_excel_data()
            st.session_state.faq_data = df
            st.session_state.topics = topics
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 📚 Topic Selection")
        selected_topic = st.selectbox(
            "Choose a topic:",
            st.session_state.topics
        )
        
        st.markdown("### ℹ️ System Status")
        if not st.session_state.faq_data.empty:
            num_items = len(st.session_state.faq_data)
            num_topics = len(st.session_state.topics) - 1
            st.success(f"✅ {num_items} FAQ items across {num_topics} topics")
        else:
            st.error("❌ No data loaded")
        
        if st.button("🔄 Reset Chat"):
            st.session_state.messages = []
            st.rerun()
    
    # Display chat messages
    for message in st.session_state.messages:
        if message["role"] == "user":
            st.markdown(f'<div class="user-message">👤 {message["content"]}</div>', unsafe_allow_html=True)
        else:
            content = message["content"]
            sources = message.get("sources", "")
            
            st.markdown(f'<div class="bot-message">🏢 {content}</div>', unsafe_allow_html=True)
            
            if sources:
                st.markdown(f'<div class="source-info">{sources}</div>', unsafe_allow_html=True)
    
    # User input
    if st.session_state.faq_data.empty:
        st.warning("Please wait for data to load.")
        return
    
    user_input = st.chat_input("Ask your question about DLD services...")
    
    if user_input:
        # Add user message
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        with st.spinner("Finding answer..."):
            # Search for relevant FAQs
            search_results = simple_search(user_input, st.session_state.faq_data, selected_topic)
            
            # Generate response
            response = generate_simple_response(user_input, search_results)
            
            # Create source information
            sources = ""
            if search_results:
                if len(search_results) == 1:
                    sources = f"📚 Source: {search_results[0]['service']} - {search_results[0]['question']}"
                else:
                    sources = f"📚 Sources: Based on {len(search_results)} FAQ items from {', '.join(set(r['service'] for r in search_results))}"
            
            # Add bot response
            st.session_state.messages.append({
                "role": "assistant",
                "content": response,
                "sources": sources
            })
        
        st.rerun()

if __name__ == "__main__":
    main()
