import streamlit as st
import pandas as pd
import json
import os
import re
import openai
import numpy as np
import logging
import pickle
import faiss
from sklearn.cluster import KMeans
import hashlib

# =============================================================================
# STREAMLIT CLOUD CONFIGURATION
# =============================================================================

def setup_openai_for_cloud():
    """Setup OpenAI API for Streamlit Cloud deployment."""
    try:
        # Try Streamlit secrets first (for cloud deployment)
        if hasattr(st, 'secrets'):
            if 'openai' in st.secrets and 'OPENAI_API_KEY' in st.secrets['openai']:
                api_key = st.secrets['openai']['OPENAI_API_KEY']
                st.sidebar.success("✅ Using Streamlit secrets")
            elif 'OPENAI_API_KEY' in st.secrets:
                api_key = st.secrets['OPENAI_API_KEY']
                st.sidebar.success("✅ Using Streamlit secrets")
            else:
                # Fallback to environment variables (for local development)
                from dotenv import load_dotenv
                load_dotenv()
                api_key = os.getenv("OPENAI_API_KEY")
                if api_key:
                    st.sidebar.info("ℹ️ Using environment variables")
                else:
                    st.sidebar.error("❌ No OpenAI API key found!")
                    return None
        else:
            # Fallback to environment variables
            try:
                from dotenv import load_dotenv
                load_dotenv()
            except:
                pass
            api_key = os.getenv("OPENAI_API_KEY")
            if api_key:
                st.sidebar.info("ℹ️ Using environment variables")
            else:
                st.sidebar.error("❌ No OpenAI API key found!")
                return None
        
        # Set the API key using the old format for compatibility
        openai.api_key = api_key
        return api_key
        
    except Exception as e:
        st.sidebar.error(f"❌ Error setting up OpenAI: {str(e)}")
        return None

# =============================================================================
# OPTIMIZED DATA PROCESSOR WITH CACHING
# =============================================================================

@st.cache_data(show_spinner=False)
def get_data_hash(data_dir):
    """Create hash of data directory to detect changes."""
    if not os.path.exists(data_dir):
        return None
    
    files = [f for f in os.listdir(data_dir) if f.endswith('.xlsx')]
    if not files:
        return None
    
    # Create hash based on file names and modification times
    file_info = []
    for f in sorted(files):
        file_path = os.path.join(data_dir, f)
        mtime = os.path.getmtime(file_path)
        file_info.append(f"{f}:{mtime}")
    
    return hashlib.md5("".join(file_info).encode()).hexdigest()

@st.cache_data(show_spinner=False)
def load_cached_data():
    """Load cached processed data if it exists."""
    cache_dir = "processed_cache"
    
    if not os.path.exists(cache_dir):
        return None, None, None, None, None
    
    try:
        # Load cached DataFrame
        df_path = os.path.join(cache_dir, "faq_data.csv")
        if not os.path.exists(df_path):
            return None, None, None, None, None
        
        df = pd.read_csv(df_path)
        
        # Load embeddings
        embeddings_path = os.path.join(cache_dir, "embeddings.npy")
        if os.path.exists(embeddings_path):
            embeddings = np.load(embeddings_path)
        else:
            return None, None, None, None, None
        
        # Load FAISS index
        index_path = os.path.join(cache_dir, "faiss_index.bin")
        if os.path.exists(index_path):
            index = faiss.read_index(index_path)
        else:
            return None, None, None, None, None
        
        # Load metadata
        meta_path = os.path.join(cache_dir, "metadata.json")
        if os.path.exists(meta_path):
            with open(meta_path, 'r') as f:
                metadata = json.load(f)
        else:
            metadata = {}
        
        # Load topics
        topics_path = os.path.join(cache_dir, "topics.json")
        if os.path.exists(topics_path):
            with open(topics_path, 'r') as f:
                topics = json.load(f)
        else:
            topics = ["All Topics"]
        
        return df, index, embeddings, metadata, topics
        
    except Exception as e:
        st.error(f"Error loading cached data: {str(e)}")
        return None, None, None, None, None

def save_processed_data(df, index, embeddings, metadata, topics):
    """Save processed data to cache."""
    cache_dir = "processed_cache"
    os.makedirs(cache_dir, exist_ok=True)
    
    try:
        # Save DataFrame
        df.to_csv(os.path.join(cache_dir, "faq_data.csv"), index=False)
        
        # Save embeddings
        np.save(os.path.join(cache_dir, "embeddings.npy"), embeddings)
        
        # Save FAISS index
        faiss.write_index(index, os.path.join(cache_dir, "faiss_index.bin"))
        
        # Save metadata
        with open(os.path.join(cache_dir, "metadata.json"), 'w') as f:
            json.dump(metadata, f)
        
        # Save topics
        with open(os.path.join(cache_dir, "topics.json"), 'w') as f:
            json.dump(topics, f)
        
        return True
    except Exception as e:
        st.error(f"Error saving data: {str(e)}")
        return False

def get_embedding_batch(texts, batch_size=20):
    """Get embeddings in small batches to avoid rate limits."""
    all_embeddings = []
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        
        try:
            response = openai.Embedding.create(
                model="text-embedding-3-small",
                input=batch_texts
            )
            batch_embeddings = [item["embedding"] for item in response["data"]]
            all_embeddings.extend(batch_embeddings)
            
            # Show progress
            progress = (i + batch_size) / len(texts)
            st.session_state.progress_bar.progress(min(progress, 1.0))
            st.session_state.status_text.text(f"Creating embeddings... {i + batch_size}/{len(texts)}")
            
        except Exception as e:
            st.error(f"Error in embedding batch: {str(e)}")
            # Add zero vectors as fallback
            for _ in range(len(batch_texts)):
                all_embeddings.append([0.0] * 1536)
    
    return np.array(all_embeddings).astype('float32')

def process_excel_files():
    """Process Excel files quickly and efficiently."""
    data_dir = "data"
    
    if not os.path.exists(data_dir):
        st.error(f"Data directory '{data_dir}' not found!")
        return None, None, None, None, None
    
    excel_files = [f for f in os.listdir(data_dir) if f.endswith('.xlsx')]
    
    if not excel_files:
        st.error(f"No Excel files found in '{data_dir}' directory!")
        return None, None, None, None, None
    
    all_dataframes = []
    
    # Process files with progress
    for idx, filename in enumerate(excel_files):
        progress = idx / len(excel_files)
        st.session_state.progress_bar.progress(progress)
        st.session_state.status_text.text(f"Processing {filename} ({idx+1}/{len(excel_files)})")
        
        try:
            file_path = os.path.join(data_dir, filename)
            service_name = filename.replace('FAQs.xlsx', '').replace('.xlsx', '').strip()
            
            # Try different header configurations
            df = None
            for header_row in [1, 2, 0]:
                try:
                    df_temp = pd.read_excel(file_path, header=header_row)
                    df_temp = df_temp.dropna(how='all')
                    
                    # Check for required columns
                    has_question = any('question' in str(col).lower() and 'eng' in str(col).lower() for col in df_temp.columns)
                    has_answer = any('answer' in str(col).lower() and 'eng' in str(col).lower() for col in df_temp.columns)
                    
                    if has_question and has_answer:
                        df = df_temp
                        break
                except:
                    continue
            
            if df is None:
                continue
            
            # Quick column mapping
            col_mapping = {}
            for col in df.columns:
                col_str = str(col).lower()
                if 'question' in col_str and ('eng' in col_str or 'en' in col_str):
                    col_mapping[col] = 'Question (English)'
                elif 'answer' in col_str and ('eng' in col_str or 'en' in col_str):
                    col_mapping[col] = 'Answer (English)'
                elif 'module' in col_str:
                    col_mapping[col] = 'Module'
            
            if col_mapping:
                df = df.rename(columns=col_mapping)
            
            # Basic data cleaning
            if 'Question (English)' in df.columns and 'Answer (English)' in df.columns:
                df['Service'] = service_name
                if 'Module' not in df.columns:
                    df['Module'] = service_name
                
                # Clean data
                df = df[df['Question (English)'].notna() & df['Answer (English)'].notna()]
                df = df[df['Question (English)'].str.len() > 10]  # Remove very short questions
                
                if len(df) > 0:
                    all_dataframes.append(df[['Question (English)', 'Answer (English)', 'Service', 'Module']])
                    
        except Exception as e:
            continue  # Skip problematic files
    
    if not all_dataframes:
        st.error("No valid data found in Excel files")
        return None, None, None, None, None
    
    # Combine all data
    df_combined = pd.concat(all_dataframes, ignore_index=True)
    
    # Remove duplicates
    df_combined.drop_duplicates(subset=['Question (English)'], keep='first', inplace=True)
    
    st.session_state.status_text.text("Creating embeddings...")
    st.session_state.progress_bar.progress(0)
    
    # Create embeddings
    embeddings = get_embedding_batch(df_combined['Question (English)'].tolist())
    
    # Create FAISS index
    st.session_state.status_text.text("Building search index...")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    
    # Get topics
    services = sorted(df_combined["Service"].unique().tolist())
    topics = ["All Topics"] + services
    
    # Metadata
    metadata = {
        'total_faqs': len(df_combined),
        'total_services': len(services),
        'processed_date': pd.Timestamp.now().isoformat()
    }
    
    return df_combined, index, embeddings, metadata, topics

# Page configuration
st.set_page_config(
    page_title="DLD FAQ Chatbot",
    page_icon="🏢",
    layout="wide"
)

# Custom CSS
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
    
    .sidebar-header {
        font-size: 1.3rem;
        font-weight: 600;
        color: #0f4c81;
        margin: 1.5rem 0 0.8rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #e6f3ff;
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

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'faq_data' not in st.session_state:
    st.session_state.faq_data = None
if 'search_index' not in st.session_state:
    st.session_state.search_index = None
if 'embeddings' not in st.session_state:
    st.session_state.embeddings = None
if 'topics' not in st.session_state:
    st.session_state.topics = ["All Topics"]
if 'selected_topic' not in st.session_state:
    st.session_state.selected_topic = "All Topics"
if 'debug_mode' not in st.session_state:
    st.session_state.debug_mode = False
if 'show_sources' not in st.session_state:
    st.session_state.show_sources = True
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False

# Chat functions
def detect_language(text):
    """Detect if text is in Arabic or English."""
    arabic_pattern = re.compile(r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]+')
    if arabic_pattern.search(text):
        return "arabic"
    return "english"

def translate_text(text, target_language):
    """Translate text between English and Arabic."""
    if not text:
        return ""
    try:
        system_prompt = f"Translate the following text to {target_language}. Keep the translation natural and accurate."
        
        response = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text}
            ],
            temperature=0.3
        )
        
        return response.choices[0].message.content
    except Exception as e:
        return text

def search_faqs_optimized(query, df, index, topic=None, top_k=3):
    """Optimized FAQ search using FAISS."""
    if df.empty or not query or index is None:
        return []
    
    try:
        # Get query embedding
        response = openai.Embedding.create(
            model="text-embedding-3-small",
            input=query
        )
        query_embedding = np.array(response["data"][0]["embedding"]).reshape(1, -1).astype('float32')
        
        # Search FAISS index
        distances, indices = index.search(query_embedding, min(top_k * 3, len(df)))
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(df):
                row = df.iloc[idx]
                
                # Filter by topic if specified
                if topic and topic != "All Topics" and row['Service'] != topic:
                    continue
                
                results.append({
                    'question': row['Question (English)'],
                    'answer': row['Answer (English)'],
                    'service': row['Service'],
                    'module': row['Module'] if not pd.isna(row['Module']) else "",
                    'relevance': 1 - (distances[0][i] / 10),
                    'debug_info': {
                        'idx': idx,
                        'distance': distances[0][i],
                        'rank': len(results) + 1
                    }
                })
                
                if len(results) >= top_k:
                    break
        
        return results
        
    except Exception as e:
        st.error(f"Search error: {str(e)}")
        return []

def generate_response(query, relevant_faqs, language):
    """Generate response based on relevant FAQs."""
    if not relevant_faqs:
        if language == "arabic":
            return "عذراً، لم أتمكن من العثور على إجابة لسؤالك. هل يمكنك إعادة صياغة السؤال أو طرح سؤال آخر؟"
        else:
            return "I'm sorry, I couldn't find an answer to your question. Could you rephrase your question or ask something else?"
    
    # For single highly relevant FAQ, return directly
    if len(relevant_faqs) == 1 and relevant_faqs[0]['relevance'] > 0.8:
        answer = relevant_faqs[0]['answer']
        if language == "arabic":
            try:
                answer = translate_text(answer, "arabic")
            except:
                pass
        return answer
    
    # Prepare context for multiple FAQs
    faq_context = ""
    for i, faq in enumerate(relevant_faqs):
        faq_context += f"\nItem {i+1}:\n"
        faq_context += f"Question: {faq['question']}\n"
        faq_context += f"Answer: {faq['answer']}\n"
        faq_context += f"Service: {faq['service']}\n"
    
    try:
        system_prompt = """
        You are a helpful assistant for the Dubai Land Department. Your task is to provide accurate and helpful responses to user queries about DLD services.
        
        Guidelines:
        1. Base your response on the provided FAQ items, prioritizing the most relevant ones.
        2. Provide a complete, clear answer that directly addresses the user's question.
        3. If multiple FAQ items are relevant, synthesize the information into a cohesive response.
        4. Use a professional but friendly tone appropriate for a government service.
        5. If the FAQ items don't fully answer the question, acknowledge this and provide the best available information.
        6. Don't make up information that isn't in the provided FAQ items.
        
        Format your response as a complete answer without mentioning that it came from FAQs or referring to "FAQ items" in your response.
        """
        
        response = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"User query: {query}\n\nRelevant FAQ information:{faq_context}"}
            ]
        )
        
        answer = response.choices[0].message.content
        
        if language == "arabic":
            answer = translate_text(answer, "arabic")
        
        return answer
    
    except Exception as e:
        st.error(f"Error generating response: {str(e)}")
        
        if language == "arabic":
            return "عذراً، حدث خطأ أثناء معالجة طلبك. يرجى المحاولة مرة أخرى."
        else:
            return "Sorry, there was an error processing your request. Please try again."

def create_source_explanation(query, relevant_faqs, language="english"):
    """Create explanation of sources used."""
    if not relevant_faqs:
        return ""
    
    if language == "english":
        if len(relevant_faqs) == 1:
            explanation = f"📚 Source: This answer is based on the FAQ question '{relevant_faqs[0]['question']}' from the {relevant_faqs[0]['service']} service."
        else:
            explanation = "📚 Sources: This answer is based on the following FAQ questions:\n"
            for i, faq in enumerate(relevant_faqs):
                explanation += f"{i+1}. '{faq['question']}' from {faq['service']}"
                if i < len(relevant_faqs) - 1:
                    explanation += "\n"
    else:  # Arabic
        if len(relevant_faqs) == 1:
            base_explanation = f"📚 المصدر: تستند هذه الإجابة إلى سؤال الأسئلة الشائعة '{relevant_faqs[0]['question']}' من خدمة {relevant_faqs[0]['service']}."
            explanation = translate_text(base_explanation, "arabic")
        else:
            base_explanation = "📚 المصادر: تستند هذه الإجابة إلى أسئلة الأسئلة الشائعة التالية:\n"
            for i, faq in enumerate(relevant_faqs):
                base_explanation += f"{i+1}. '{faq['question']}' من {faq['service']}"
                if i < len(relevant_faqs) - 1:
                    base_explanation += "\n"
            explanation = translate_text(base_explanation, "arabic")
    
    return explanation

# Main application
def main():
    # Setup OpenAI
    api_key = setup_openai_for_cloud()
    if not api_key:
        st.error("❌ Cannot proceed without OpenAI API key. Please configure it in Streamlit secrets.")
        st.info("""
        **To configure secrets:**
        1. Go to your Streamlit Cloud dashboard
        2. Click on your app → Settings → Secrets
        3. Add:
        ```
        [openai]
        OPENAI_API_KEY = "sk-your-api-key-here"
        ```
        """)
        st.stop()
    
    # Main title
    st.markdown('<h1 class="main-title">DLD FAQ Assistant</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">Ask me anything about Dubai Land Department services</p>', unsafe_allow_html=True)
    
    # Load or process data
    if not st.session_state.data_loaded:
        # Check current data hash
        current_hash = get_data_hash("data")
        
        if current_hash is None:
            st.error("❌ No Excel files found in 'data' directory!")
            st.stop()
        
        # Try to load cached data first
        st.info("🔍 Checking for cached data...")
        cached_df, cached_index, cached_embeddings, cached_metadata, cached_topics = load_cached_data()
        
        # Check if cached data is valid
        cache_valid = (
            cached_df is not None and
            cached_metadata is not None and
            cached_metadata.get('data_hash') == current_hash
        )
        
        if cache_valid:
            # Use cached data
            st.success("✅ Loaded cached data instantly!")
            st.session_state.faq_data = cached_df
            st.session_state.search_index = cached_index
            st.session_state.embeddings = cached_embeddings
            st.session_state.topics = cached_topics
            st.session_state.data_loaded = True
        else:
            # Process data fresh
            st.info("🚀 Processing Excel files (first time or data changed)...")
            
            # Create progress indicators
            progress_container = st.container()
            with progress_container:
                st.session_state.progress_bar = st.progress(0)
                st.session_state.status_text = st.empty()
            
            df, index, embeddings, metadata, topics = process_excel_files()
            
            if df is not None:
                # Add data hash to metadata
                metadata['data_hash'] = current_hash
                
                # Save to cache
                if save_processed_data(df, index, embeddings, metadata, topics):
                    st.success("✅ Data processed and cached successfully!")
                
                # Update session state
                st.session_state.faq_data = df
                st.session_state.search_index = index
                st.session_state.embeddings = embeddings
                st.session_state.topics = topics
                st.session_state.data_loaded = True
                
                # Clear progress indicators
                progress_container.empty()
                st.rerun()
            else:
                st.error("❌ Failed to process data")
                st.stop()
    
    # Sidebar
    with st.sidebar:
        st.markdown('<div class="sidebar-header">Topic Selection</div>', unsafe_allow_html=True)
        selected_topic = st.selectbox(
            "Select a topic for your question",
            st.session_state.topics,
            index=st.session_state.topics.index(st.session_state.selected_topic) if st.session_state.selected_topic in st.session_state.topics else 0
        )
        st.session_state.selected_topic = selected_topic
        
        st.markdown('<div class="sidebar-header">Settings</div>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            st.session_state.debug_mode = st.checkbox("Debug Mode", value=st.session_state.debug_mode)
        with col2:
            st.session_state.show_sources = st.checkbox("Show Sources", value=st.session_state.show_sources)
        
        st.markdown('<div class="sidebar-header">System Status</div>', unsafe_allow_html=True)
        if st.session_state.faq_data is not None and not st.session_state.faq_data.empty:
            num_topics = len(st.session_state.topics) - 1
            st.info(f"📚 {len(st.session_state.faq_data)} FAQ items across {num_topics} topics")
            st.success("✅ Real FAQ data with embeddings loaded")
        else:
            st.error("❌ No FAQ data loaded")
        
        st.markdown("---")
        if st.button("🔄 Reset Conversation", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
        
        if st.button("🔄 Rebuild Cache", use_container_width=True):
            st.session_state.data_loaded = False
            # Clear cache
            cache_dir = "processed_cache"
            if os.path.exists(cache_dir):
                import shutil
                shutil.rmtree(cache_dir)
            st.rerun()
    
    # Display chat messages
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.messages:
            if message["role"] == "user":
                st.markdown(f'<div class="user-message">👤 {message["content"]}</div>', unsafe_allow_html=True)
            else:
                content = message["content"]
                source_info = message.get("source_info", "")
                debug_info = message.get("debug_info", "")
                
                # Display the message
                st.markdown(f'<div class="bot-message">🏢 {content}</div>', unsafe_allow_html=True)
                
                # Show source info if enabled
                if st.session_state.show_sources and source_info:
                    st.markdown(f'<div class="source-info">{source_info}</div>', unsafe_allow_html=True)
                
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
            
            # Detect language
            language = detect_language(user_query)
            
            # Translate query to English if needed
            if language == "arabic":
                english_query = translate_text(user_query, "english")
            else:
                english_query = user_query
            
            # Search for relevant FAQs
            relevant_faqs = search_faqs_optimized(
                english_query,
                st.session_state.faq_data,
                st.session_state.search_index,
                st.session_state.selected_topic
            )
            
            # Generate response
            response = generate_response(english_query, relevant_faqs, language)
            
            # Create source explanation if needed
            source_info = ""
            if st.session_state.show_sources and relevant_faqs:
                source_info = create_source_explanation(english_query, relevant_faqs, language)
            
            # Prepare debug information
            debug_info = ""
            if st.session_state.debug_mode:
                debug_info = f"Language: {language}\nRelevant FAQs found: {len(relevant_faqs)}"
                if relevant_faqs:
                    debug_info += f"\nTop match: {relevant_faqs[0]['service']}"
                    debug_info += f"\nRelevance score: {relevant_faqs[0]['relevance']:.3f}"
            
            # Add bot response to chat history
            st.session_state.messages.append({
                "role": "assistant",
                "content": response,
                "source_info": source_info,
                "debug_info": debug_info
            })
            
            st.rerun()

if __name__ == "__main__":
    main()
