import streamlit as st
import pandas as pd
import os
import re
import openai
import time
import numpy as np
import json

# =============================================================================
# OPENAI v1.0+ API SETUP AND TESTING
# =============================================================================

def setup_openai():
    """Setup OpenAI API v1.0+ with testing."""
    try:
        if hasattr(st, 'secrets'):
            if 'openai' in st.secrets and 'OPENAI_API_KEY' in st.secrets['openai']:
                api_key = st.secrets['openai']['OPENAI_API_KEY']
            elif 'OPENAI_API_KEY' in st.secrets:
                api_key = st.secrets['OPENAI_API_KEY']
            else:
                st.sidebar.error("❌ No OpenAI API key found in secrets!")
                return None
        else:
            st.sidebar.error("❌ No secrets found!")
            return None
        
        # Initialize OpenAI client with v1.0+ syntax
        client = openai.OpenAI(api_key=api_key)
        print(f"[DEBUG] OpenAI client initialized with key: {api_key[:10]}...")
        
        return client
        
    except Exception as e:
        st.sidebar.error(f"❌ Error setting up OpenAI: {str(e)}")
        print(f"[DEBUG] OpenAI setup error: {str(e)}")
        return None

def test_openai_api(client):
    """Test OpenAI API v1.0+ with both ChatCompletion and Embedding."""
    print("[DEBUG] Testing OpenAI API v1.0+...")
    
    test_results = {
        'client_initialized': False,
        'chat_completion_works': False,
        'embedding_works': False,
        'errors': []
    }
    
    try:
        if not client:
            test_results['errors'].append("No OpenAI client available")
            return test_results
        
        test_results['client_initialized'] = True
        print("[DEBUG] OpenAI client available")
        
        # Test ChatCompletion with v1.0+ syntax
        try:
            print("[DEBUG] Testing chat.completions.create...")
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": "Say 'API test successful'"}],
                max_tokens=10
            )
            test_results['chat_completion_works'] = True
            print("[DEBUG] ChatCompletion test successful")
        except Exception as e:
            error_msg = f"ChatCompletion failed: {str(e)}"
            test_results['errors'].append(error_msg)
            print(f"[DEBUG] {error_msg}")
        
        # Test Embedding with v1.0+ syntax
        try:
            print("[DEBUG] Testing embeddings.create...")
            response = client.embeddings.create(
                model="text-embedding-3-small",
                input="test embedding"
            )
            test_results['embedding_works'] = True
            print("[DEBUG] Embedding test successful")
        except Exception as e:
            error_msg = f"Embedding failed: {str(e)}"
            test_results['errors'].append(error_msg)
            print(f"[DEBUG] {error_msg}")
            
    except Exception as e:
        error_msg = f"General API error: {str(e)}"
        test_results['errors'].append(error_msg)
        print(f"[DEBUG] {error_msg}")
    
    return test_results

# =============================================================================
# DATA LOADER
# =============================================================================

@st.cache_data
def load_excel_data():
    """Load Excel data with detailed tracking."""
    print("[DEBUG] Starting Excel data loading...")
    data_dir = "data"
    
    if not os.path.exists(data_dir):
        print(f"[DEBUG] Data directory '{data_dir}' not found!")
        st.error(f"❌ Data directory '{data_dir}' not found!")
        return pd.DataFrame(), [], {}
    
    excel_files = [f for f in os.listdir(data_dir) if f.endswith('.xlsx')]
    print(f"[DEBUG] Found {len(excel_files)} Excel files: {excel_files}")
    
    if not excel_files:
        print("[DEBUG] No Excel files found!")
        st.error(f"❌ No Excel files found in '{data_dir}' directory!")
        return pd.DataFrame(), [], {}
    
    all_data = []
    file_stats = {}
    
    # Progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for idx, filename in enumerate(excel_files):
        print(f"[DEBUG] Processing file {idx+1}/{len(excel_files)}: {filename}")
        try:
            status_text.text(f"Loading {filename}...")
            progress_bar.progress((idx + 1) / len(excel_files))
            
            file_path = os.path.join(data_dir, filename)
            service_name = filename.replace('FAQs.xlsx', '').replace('.xlsx', '').strip()
            
            # Try different header positions
            df = None
            header_used = None
            
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
                        header_used = header_row
                        break
                        
                except Exception:
                    continue
            
            if df is not None and question_col and answer_col:
                # Clean and prepare data
                clean_df = df[[question_col, answer_col]].copy()
                clean_df.columns = ['Question', 'Answer']
                clean_df['Service'] = service_name
                clean_df['Source_File'] = filename
                
                # Remove empty rows
                original_count = len(clean_df)
                clean_df = clean_df.dropna(subset=['Question', 'Answer'])
                clean_df = clean_df[clean_df['Question'].str.len() > 5]
                clean_df = clean_df[clean_df['Answer'].str.len() > 10]
                final_count = len(clean_df)
                
                # Track statistics
                file_stats[filename] = {
                    'service': service_name,
                    'header_row': header_used,
                    'original_rows': original_count,
                    'final_rows': final_count,
                    'question_col': question_col,
                    'answer_col': answer_col
                }
                
                if final_count > 0:
                    all_data.append(clean_df)
                    
            else:
                file_stats[filename] = {
                    'service': service_name,
                    'error': 'Could not find question/answer columns'
                }
                
        except Exception as e:
            file_stats[filename] = {
                'service': service_name,
                'error': str(e)
            }
            continue
    
    # Clear progress indicators
    progress_bar.empty()
    status_text.empty()
    
    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)
        
        # Remove duplicates and track
        original_total = len(combined_df)
        combined_df = combined_df.drop_duplicates(subset=['Question'], keep='first')
        final_total = len(combined_df)
        
        services = sorted(combined_df['Service'].unique().tolist())
        topics = ["All Topics"] + services
        
        # Add processing stats
        file_stats['_summary'] = {
            'total_files_processed': len([f for f in file_stats if not f.startswith('_') and 'error' not in file_stats[f]]),
            'total_files_failed': len([f for f in file_stats if not f.startswith('_') and 'error' in file_stats[f]]),
            'original_total_rows': original_total,
            'final_total_rows': final_total,
            'duplicates_removed': original_total - final_total
        }
        
        print(f"[DEBUG] Data loading complete: {final_total} FAQ items from {len(all_data)} files")
        st.success(f"✅ Loaded {final_total} FAQ items from {len(all_data)} files")
        return combined_df, topics, file_stats
    else:
        print("[DEBUG] No valid data found in any Excel files")
        st.error("❌ No valid data found in Excel files")
        return pd.DataFrame(), [], file_stats

# =============================================================================
# EMBEDDING SYSTEM (v1.0+ API)
# =============================================================================

def check_embeddings_exist():
    """Check if embeddings are already cached."""
    embeddings_file = "embeddings_cache.npy"
    metadata_file = "embeddings_metadata.json"
    exists = os.path.exists(embeddings_file) and os.path.exists(metadata_file)
    print(f"[DEBUG] Embeddings exist: {exists}")
    return exists

def load_cached_embeddings():
    """Load cached embeddings if they exist."""
    try:
        print("[DEBUG] Loading cached embeddings...")
        embeddings = np.load("embeddings_cache.npy")
        
        with open("embeddings_metadata.json", 'r') as f:
            metadata = json.load(f)
        
        print(f"[DEBUG] Loaded {len(embeddings)} cached embeddings with shape {embeddings.shape}")
        return embeddings, metadata
    except Exception as e:
        print(f"[DEBUG] Error loading cached embeddings: {str(e)}")
        return None, None

def save_embeddings(embeddings, metadata):
    """Save embeddings to cache."""
    try:
        print("[DEBUG] Saving embeddings to cache...")
        np.save("embeddings_cache.npy", embeddings)
        
        with open("embeddings_metadata.json", 'w') as f:
            json.dump(metadata, f)
        
        print("[DEBUG] Embeddings saved successfully")
        return True
    except Exception as e:
        print(f"[DEBUG] Error saving embeddings: {str(e)}")
        return False

def create_embeddings_for_df(df, client):
    """Create embeddings for all questions using v1.0+ API."""
    print(f"[DEBUG] Starting embedding creation for {len(df)} questions")
    
    questions = df['Question'].tolist()
    embeddings = []
    
    # Create progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    batch_size = 5  # Small batches to avoid rate limits
    
    total_batches = (len(questions) + batch_size - 1) // batch_size
    print(f"[DEBUG] Processing {total_batches} batches of {batch_size} questions each")
    
    for i in range(0, len(questions), batch_size):
        batch_num = i // batch_size + 1
        batch_questions = questions[i:i+batch_size]
        
        print(f"[DEBUG] Processing batch {batch_num}/{total_batches}")
        status_text.text(f"Creating embeddings... Batch {batch_num}/{total_batches}")
        progress_bar.progress(batch_num / total_batches)
        
        try:
            print(f"[DEBUG] Calling client.embeddings.create for batch {batch_num}")
            # Using v1.0+ API syntax
            response = client.embeddings.create(
                model="text-embedding-3-small",
                input=batch_questions
            )
            
            batch_embeddings = [item.embedding for item in response.data]
            embeddings.extend(batch_embeddings)
            
            print(f"[DEBUG] Batch {batch_num} completed successfully - got {len(batch_embeddings)} embeddings")
            
            # Small delay to respect rate limits
            time.sleep(0.2)
            
        except Exception as e:
            print(f"[DEBUG] Error in batch {batch_num}: {str(e)}")
            st.error(f"Error in batch {batch_num}: {str(e)}")
            
            # Add zero vectors as fallback
            for _ in range(len(batch_questions)):
                embeddings.append([0.0] * 1536)
    
    progress_bar.progress(1.0)
    status_text.text("✅ Embeddings creation complete!")
    
    embeddings_array = np.array(embeddings).astype('float32')
    print(f"[DEBUG] Created embeddings array with shape: {embeddings_array.shape}")
    
    # Save metadata
    metadata = {
        'total_questions': len(questions),
        'embedding_dimension': embeddings_array.shape[1],
        'model': 'text-embedding-3-small',
        'created_at': time.time()
    }
    
    # Save to cache
    save_embeddings(embeddings_array, metadata)
    
    # Clear progress indicators
    progress_bar.empty()
    status_text.empty()
    
    return embeddings_array, metadata

def semantic_search_with_embeddings(query, df, embeddings, client, top_k=3):
    """Perform semantic search using embeddings with v1.0+ API."""
    print(f"[DEBUG] Starting semantic search for: '{query}'")
    print(f"[DEBUG] DataFrame shape: {df.shape}")
    print(f"[DEBUG] Embeddings shape: {embeddings.shape}")
    
    try:
        # Get query embedding using v1.0+ API
        print("[DEBUG] Getting query embedding with client.embeddings.create...")
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=query
        )
        query_embedding = np.array(response.data[0].embedding).astype('float32')
        print(f"[DEBUG] Query embedding shape: {query_embedding.shape}")
        
        # Calculate similarities (cosine similarity)
        print("[DEBUG] Calculating cosine similarities...")
        # Normalize embeddings for cosine similarity
        embeddings_norm = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        query_norm = query_embedding / np.linalg.norm(query_embedding)
        
        similarities = np.dot(embeddings_norm, query_norm)
        print(f"[DEBUG] Similarities calculated, shape: {similarities.shape}")
        print(f"[DEBUG] Similarity range: {similarities.min():.3f} to {similarities.max():.3f}")
        
        # Get top results
        top_indices = np.argsort(similarities)[::-1][:top_k]
        print(f"[DEBUG] Top {top_k} indices: {top_indices}")
        print(f"[DEBUG] Top {top_k} similarities: {similarities[top_indices]}")
        
        results = []
        for i, idx in enumerate(top_indices):
            similarity_score = similarities[idx]
            row = df.iloc[idx]
            
            result = {
                'question': row['Question'],
                'answer': row['Answer'],
                'service': row['Service'],
                'source_file': row['Source_File'],
                'similarity_score': float(similarity_score),
                'rank': i + 1,
                'method': 'semantic_embedding'
            }
            results.append(result)
            
            print(f"[DEBUG] Result {i+1}: {similarity_score:.3f} - {row['Question'][:50]}...")
        
        print(f"[DEBUG] Semantic search completed successfully with {len(results)} results")
        return results
        
    except Exception as e:
        print(f"[DEBUG] Error in semantic search: {str(e)}")
        st.error(f"Semantic search error: {str(e)}")
        return []

# =============================================================================
# TEXT-BASED SEARCH (FALLBACK)
# =============================================================================

def text_based_search(query, df, topic=None, top_k=3):
    """Text-based search as fallback."""
    print(f"[DEBUG] Performing text-based search for: '{query}'")
    
    if df.empty or not query:
        return []
    
    # Filter by topic if specified
    search_df = df
    if topic and topic != "All Topics":
        search_df = df[df['Service'] == topic]
        if search_df.empty:
            search_df = df
    
    print(f"[DEBUG] Searching through {len(search_df)} documents")
    
    query_lower = query.lower()
    query_words = set(query_lower.split())
    results = []
    
    # Important keywords for DLD services
    important_keywords = {
        'register': ['register', 'registration', 'apply'],
        'property': ['property', 'real estate', 'land'],
        'mortgage': ['mortgage', 'loan', 'finance'],
        'document': ['document', 'papers', 'certificate'],
        'fee': ['fee', 'cost', 'charge', 'price'],
        'time': ['time', 'duration', 'fast', 'quick', 'long'],
        'ejari': ['ejari', 'rental', 'lease'],
        'broker': ['broker', 'agent', 'intermediary'],
        'payment': ['payment', 'pay', 'transaction']
    }
    
    for idx, row in search_df.iterrows():
        question = row['Question']
        question_lower = question.lower()
        question_words = set(question_lower.split())
        
        # Calculate score
        score = 0
        
        # Exact phrase matching
        if query_lower in question_lower:
            score += 20
        
        # Word overlap
        common_words = query_words.intersection(question_words)
        score += len(common_words) * 5
        
        # Keyword matching
        for category, keywords in important_keywords.items():
            query_has = any(kw in query_lower for kw in keywords)
            question_has = any(kw in question_lower for kw in keywords)
            if query_has and question_has:
                score += 10
        
        if score > 0:
            results.append({
                'question': question,
                'answer': row['Answer'],
                'service': row['Service'],
                'source_file': row['Source_File'],
                'similarity_score': score / 100.0,  # Normalize to 0-1 range
                'rank': 0,  # Will be set after sorting
                'method': 'text_matching'
            })
    
    # Sort and rank
    results.sort(key=lambda x: x['similarity_score'], reverse=True)
    for i, result in enumerate(results[:top_k]):
        result['rank'] = i + 1
    
    print(f"[DEBUG] Text search found {len(results)} results")
    return results[:top_k]

# =============================================================================
# RESPONSE GENERATION (v1.0+ API)
# =============================================================================

def generate_response(query, search_results, search_method, client):
    """Generate response using v1.0+ API."""
    print(f"[DEBUG] Generating response using {search_method} with {len(search_results)} results")
    
    if not search_results:
        print("[DEBUG] No search results, returning default message")
        return "I'm sorry, I couldn't find an answer to your question. Could you please rephrase it or ask about Dubai Land Department services?"
    
    # If single high-confidence result, return directly
    if len(search_results) == 1 and search_results[0]['similarity_score'] > 0.8:
        print("[DEBUG] Using direct answer from single high-confidence result")
        return search_results[0]['answer']
    
    # Prepare context for OpenAI
    context = ""
    for i, result in enumerate(search_results):
        context += f"\nFAQ {i+1}:\n"
        context += f"Question: {result['question']}\n"
        context += f"Answer: {result['answer']}\n"
        context += f"Service: {result['service']}\n"
        context += f"Confidence: {result['similarity_score']:.3f}\n"
    
    try:
        print("[DEBUG] Calling client.chat.completions.create for response generation")
        system_prompt = """You are a helpful assistant for the Dubai Land Department. 
        Based on the provided FAQ information, give a clear and helpful answer to the user's question.
        Use a professional but friendly tone. Only use information from the provided FAQs.
        
        Guidelines:
        1. Synthesize information from multiple FAQs if relevant
        2. Be specific and actionable
        3. Mention relevant services or processes
        4. If information is incomplete, acknowledge it"""
        
        # Using v1.0+ API syntax
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"User question: {query}\n\nRelevant FAQs:\n{context}"}
            ],
            max_tokens=500,
            temperature=0.3
        )
        
        print("[DEBUG] OpenAI response generated successfully")
        return response.choices[0].message.content
        
    except Exception as e:
        print(f"[DEBUG] OpenAI generation error: {str(e)}")
        # Fallback to best result
        return search_results[0]['answer']

# =============================================================================
# STREAMLIT APP
# =============================================================================

def main():
    print("[DEBUG] Starting main application")
    
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
            padding: 8px;
            background-color: #f8f9fa;
            border-radius: 5px;
        }
        .debug-info {
            font-size: 0.8rem;
            color: #666;
            margin-top: 10px;
            border-top: 1px solid #eee;
            padding: 8px;
            background-color: #fafafa;
            border-radius: 5px;
        }
        .embedding-status {
            padding: 10px;
            border-radius: 5px;
            margin: 10px 0;
        }
        .embedding-enabled {
            background-color: #d4edda;
            color: #155724;
            border: 1px solid #c3e6cb;
        }
        .embedding-disabled {
            background-color: #f8d7da;
            color: #721c24;
            border: 1px solid #f5c6cb;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Title
    st.markdown('<h1 class="main-title">🏢 DLD FAQ Assistant</h1>', unsafe_allow_html=True)
    st.markdown("### Ask me anything about Dubai Land Department services")
    
    # Setup OpenAI v1.0+ client
    client = setup_openai()
    if not client:
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
    if 'file_stats' not in st.session_state:
        st.session_state.file_stats = {}
    if 'debug_mode' not in st.session_state:
        st.session_state.debug_mode = False
    if 'embeddings' not in st.session_state:
        st.session_state.embeddings = None
    if 'embeddings_metadata' not in st.session_state:
        st.session_state.embeddings_metadata = None
    if 'use_embeddings' not in st.session_state:
        st.session_state.use_embeddings = False
    if 'openai_client' not in st.session_state:
        st.session_state.openai_client = client
    
    # Load data (fast startup)
    if st.session_state.faq_data is None:
        with st.spinner("Loading FAQ data..."):
            print("[DEBUG] Loading FAQ data...")
            df, topics, file_stats = load_excel_data()
            st.session_state.faq_data = df
            st.session_state.topics = topics
            st.session_state.file_stats = file_stats
            print("[DEBUG] FAQ data loaded into session state")
    
    # Check for existing embeddings
    if st.session_state.embeddings is None:
        print("[DEBUG] Checking for existing embeddings...")
        if check_embeddings_exist():
            print("[DEBUG] Found existing embeddings, loading...")
            embeddings, metadata = load_cached_embeddings()
            if embeddings is not None:
                st.session_state.embeddings = embeddings
                st.session_state.embeddings_metadata = metadata
                st.session_state.use_embeddings = True
                print("[DEBUG] Embeddings loaded from cache and set to session state")
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 📚 Topic Selection")
        selected_topic = st.selectbox(
            "Choose a topic:",
            st.session_state.topics
        )
        
        st.markdown("### ⚙️ Settings")
        st.session_state.debug_mode = st.checkbox("🔍 Debug Mode", value=st.session_state.debug_mode)
        show_sources = st.checkbox("📚 Show Sources", value=True)
        
        # API Test Button
        if st.button("Quick test for our Agents"):
            with st.spinner("Testing API..."):
                test_results = test_openai_api(client)
                if test_results['chat_completion_works'] and test_results['embedding_works']:
                    st.success("✅ All API functions working!")
                else:
                    st.error("❌ API issues detected:")
                    for error in test_results['errors']:
                        st.error(f"• {error}")
        
        # Embedding controls
        st.markdown("### 🧠 AI Search Mode")
        
        if st.session_state.embeddings is not None:
            st.markdown('<div class="embedding-status embedding-enabled">✅ Semantic Search Active</div>', unsafe_allow_html=True)
            st.write(f"📊 {len(st.session_state.embeddings)} embeddings ready")
            if st.session_state.embeddings_metadata:
                # st.write(f"🤖 Model: {st.session_state.embeddings_metadata.get('model', 'unknown')}")
            
            use_semantic = st.radio(
                "Search method:",
                ["🧠 Semantic Search (AI)", "📝 Text Search (Fast)"],
                index=0 if st.session_state.use_embeddings else 1
            )
            st.session_state.use_embeddings = (use_semantic == "🧠 Semantic Search (AI)")
            
        else:
            st.markdown('<div class="embedding-status embedding-disabled">📝 Text Search Mode</div>', unsafe_allow_html=True)
            
            if not st.session_state.faq_data.empty:
                if st.button("🚀 Create AI Embeddings", use_container_width=True):
                    with st.spinner("Creating embeddings... This will take 2-3 minutes."):
                        embeddings, metadata = create_embeddings_for_df(st.session_state.faq_data, client)
                        st.session_state.embeddings = embeddings
                        st.session_state.embeddings_metadata = metadata
                        st.session_state.use_embeddings = True
                        st.success("✅ Embeddings created! Semantic search is now active.")
                        st.rerun()
                
                st.info("💡 Create embeddings for AI-powered semantic search that understands meaning, not just keywords!")
        
        st.markdown("### ℹ️ System Status")
        if not st.session_state.faq_data.empty:
            num_items = len(st.session_state.faq_data)
            num_topics = len(st.session_state.topics) - 1
            st.success(f"✅ {num_items} FAQ items across {num_topics} topics")
            
            if st.session_state.debug_mode:
                with st.expander("📊 Data Processing Stats"):
                    if '_summary' in st.session_state.file_stats:
                        summary = st.session_state.file_stats['_summary']
                        st.write(f"**Files processed**: {summary['total_files_processed']}")
                        st.write(f"**Files failed**: {summary['total_files_failed']}")
                        st.write(f"**Total rows**: {summary['final_total_rows']}")
                        st.write(f"**Duplicates removed**: {summary['duplicates_removed']}")
        else:
            st.error("❌ No FAQ data loaded")
            st.info("Please add Excel files to the 'data' directory")
        
        if st.button("🔄 Reset Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
    
    # Main chat interface
    st.markdown("---")
    
    # Display chat messages
    for message in st.session_state.messages:
        if message["role"] == "user":
            st.markdown(f'<div class="user-message">👤 **You:** {message["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="bot-message">🏢 **DLD Assistant:** {message["content"]}</div>', unsafe_allow_html=True)
            
            # Show sources if available
            if show_sources and "sources" in message:
                st.markdown('<div class="source-info">', unsafe_allow_html=True)
                st.markdown("📚 **Sources:**")
                for i, source in enumerate(message["sources"], 1):
                    with st.expander(f"Source {i}: {source['question'][:60]}..." if len(source['question']) > 60 else f"Source {i}: {source['question']}", expanded=False):
                        st.markdown(f"**📋 Service:** {source['service']}")
                        st.markdown(f"**❓ Question:** {source['question']}")
                        st.markdown(f"**✅ Answer:** {source['answer']}")
                        st.markdown(f"**📄 File:** {source['source_file']}")
                        st.markdown(f"**🎯 Relevance Score:** {source['similarity_score']:.3f}")
                        st.markdown(f"**🔍 Search Method:** {source['method']}")
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Show debug info if enabled
            if st.session_state.debug_mode and "debug_info" in message:
                debug_text = f"🔧 **Debug:** {message['debug_info']}"
                st.markdown(f'<div class="debug-info">{debug_text}</div>', unsafe_allow_html=True)
    
    # Chat input
    if st.session_state.faq_data.empty:
        st.warning("Please add Excel files to the 'data' directory to start chatting")
    else:
        user_input = st.chat_input("Ask me about Dubai Land Department services...")
        
        if user_input:
            # Add user message
            st.session_state.messages.append({"role": "user", "content": user_input})
            
            # Process the query
            with st.spinner("Searching for answer..."):
                # Choose search method
                if st.session_state.use_embeddings and st.session_state.embeddings is not None:
                    print("[DEBUG] Using semantic search with embeddings")
                    search_results = semantic_search_with_embeddings(
                        user_input, 
                        st.session_state.faq_data, 
                        st.session_state.embeddings, 
                        client
                    )
                    search_method = "Semantic Search"
                else:
                    print("[DEBUG] Using text-based search")
                    search_results = text_based_search(
                        user_input, 
                        st.session_state.faq_data, 
                        selected_topic
                    )
                    search_method = "Text Search"
                
                # Generate response
                response = generate_response(user_input, search_results, search_method, client)
                
                # Prepare message data
                message_data = {
                    "role": "assistant", 
                    "content": response
                }
                
                # Add sources
                if search_results:
                    message_data["sources"] = search_results
                
                # Add debug info
                if st.session_state.debug_mode:
                    debug_info = f"Method: {search_method}, Results: {len(search_results)}"
                    if search_results:
                        debug_info += f", Top score: {search_results[0]['similarity_score']:.3f}"
                    message_data["debug_info"] = debug_info
                
                # Add assistant message
                st.session_state.messages.append(message_data)
            
            # Rerun to show the new messages
            st.rerun()

if __name__ == "__main__":
    main()
