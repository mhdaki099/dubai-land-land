import streamlit as st
import pandas as pd
import os
import re
import openai
import time

# =============================================================================
# STREAMLIT CLOUD CONFIGURATION
# =============================================================================

def setup_openai():
    """Setup OpenAI API."""
    try:
        if hasattr(st, 'secrets'):
            if 'openai' in st.secrets and 'OPENAI_API_KEY' in st.secrets['openai']:
                api_key = st.secrets['openai']['OPENAI_API_KEY']
                st.sidebar.success("✅ Using Streamlit secrets")
            elif 'OPENAI_API_KEY' in st.secrets:
                api_key = st.secrets['OPENAI_API_KEY']
                st.sidebar.success("✅ Using Streamlit secrets")
            else:
                st.sidebar.error("❌ No OpenAI API key found in secrets!")
                return None
        else:
            st.sidebar.error("❌ No secrets found!")
            return None
        
        openai.api_key = api_key
        return api_key
        
    except Exception as e:
        st.sidebar.error(f"❌ Error setting up OpenAI: {str(e)}")
        return None

# =============================================================================
# DATA LOADER WITH DETAILED TRACKING
# =============================================================================

@st.cache_data
def load_excel_data():
    """Load Excel data with detailed tracking."""
    data_dir = "data"
    
    if not os.path.exists(data_dir):
        st.error(f"❌ Data directory '{data_dir}' not found!")
        return pd.DataFrame(), [], {}
    
    excel_files = [f for f in os.listdir(data_dir) if f.endswith('.xlsx')]
    
    if not excel_files:
        st.error(f"❌ No Excel files found in '{data_dir}' directory!")
        return pd.DataFrame(), [], {}
    
    all_data = []
    file_stats = {}
    
    # Progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for idx, filename in enumerate(excel_files):
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
        
        st.success(f"✅ Loaded {final_total} FAQ items from {len(all_data)} files")
        return combined_df, topics, file_stats
    else:
        st.error("❌ No valid data found in Excel files")
        return pd.DataFrame(), [], file_stats

# =============================================================================
# ENHANCED SEARCH WITH DETAILED SCORING
# =============================================================================

def detailed_search(query, df, topic=None, top_k=3):
    """Enhanced search with detailed scoring information."""
    if df.empty or not query:
        return [], {}
    
    start_time = time.time()
    
    # Filter by topic if specified
    search_df = df
    if topic and topic != "All Topics":
        search_df = df[df['Service'] == topic]
        if search_df.empty:
            search_df = df
    
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
        answer = row['Answer']
        service = row['Service']
        
        # Calculate detailed scores
        scoring_details = {
            'exact_phrase': 0,
            'word_overlap': 0,
            'keyword_matches': 0,
            'question_patterns': 0
        }
        
        # 1. Exact phrase matching
        if query_lower in question_lower:
            scoring_details['exact_phrase'] = 20
        
        # 2. Word overlap
        common_words = query_words.intersection(question_words)
        scoring_details['word_overlap'] = len(common_words) * 5
        
        # 3. Important keyword matching
        keyword_score = 0
        matched_keywords = []
        for category, keywords in important_keywords.items():
            query_has = any(kw in query_lower for kw in keywords)
            question_has = any(kw in question_lower for kw in keywords)
            if query_has and question_has:
                keyword_score += 10
                matched_keywords.append(category)
        scoring_details['keyword_matches'] = keyword_score
        
        # 4. Question pattern matching
        question_patterns = ['how', 'what', 'when', 'where', 'why', 'can', 'do', 'is']
        pattern_score = 0
        for pattern in question_patterns:
            if pattern in query_lower and pattern in question_lower:
                pattern_score += 3
        scoring_details['question_patterns'] = pattern_score
        
        # Total score
        total_score = sum(scoring_details.values())
        
        if total_score > 0:
            results.append({
                'question': question,
                'answer': answer,
                'service': service,
                'source_file': row.get('Source_File', 'Unknown'),
                'total_score': total_score,
                'scoring_details': scoring_details,
                'matched_keywords': matched_keywords,
                'common_words': list(common_words)
            })
    
    # Sort by score
    results.sort(key=lambda x: x['total_score'], reverse=True)
    
    search_time = time.time() - start_time
    
    search_stats = {
        'query': query,
        'search_time': search_time,
        'total_documents_searched': len(search_df),
        'results_found': len(results),
        'top_score': results[0]['total_score'] if results else 0,
        'topic_filter': topic
    }
    
    return results[:top_k], search_stats

# =============================================================================
# ENHANCED RESPONSE GENERATION
# =============================================================================

def generate_enhanced_response(query, search_results, show_debug=False):
    """Generate response with optional debug information."""
    if not search_results:
        return "I'm sorry, I couldn't find an answer to your question. Could you please rephrase it or ask about Dubai Land Department services?", {}
    
    start_time = time.time()
    
    # If we have a single, highly relevant result, return it directly
    if len(search_results) == 1 and search_results[0]['total_score'] > 15:
        response = search_results[0]['answer']
        generation_stats = {
            'method': 'direct_answer',
            'generation_time': time.time() - start_time,
            'openai_used': False
        }
        return response, generation_stats
    
    # Prepare context from search results
    context = ""
    for i, result in enumerate(search_results):
        context += f"\nFAQ {i+1}:\n"
        context += f"Question: {result['question']}\n"
        context += f"Answer: {result['answer']}\n"
        context += f"Service: {result['service']}\n"
        context += f"Relevance Score: {result['total_score']}\n"
    
    try:
        system_prompt = """You are a helpful assistant for the Dubai Land Department. 
        Based on the provided FAQ information, give a clear and helpful answer to the user's question.
        Use a professional but friendly tone. Only use information from the provided FAQs.
        
        Guidelines:
        1. Synthesize information from multiple FAQs if relevant
        2. Be specific and actionable
        3. Mention relevant services or processes
        4. If information is incomplete, acknowledge it"""
        
        openai_start = time.time()
        
        response = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"User question: {query}\n\nRelevant FAQs:\n{context}"}
            ],
            max_tokens=500,
            temperature=0.3
        )
        
        openai_time = time.time() - openai_start
        
        generation_stats = {
            'method': 'openai_synthesis',
            'generation_time': time.time() - start_time,
            'openai_time': openai_time,
            'openai_used': True,
            'model': 'gpt-4o',
            'context_length': len(context)
        }
        
        return response.choices[0].message.content, generation_stats
        
    except Exception as e:
        # Fallback to best match
        generation_stats = {
            'method': 'fallback',
            'generation_time': time.time() - start_time,
            'openai_used': False,
            'error': str(e)
        }
        return search_results[0]['answer'], generation_stats

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
            background-color: #f8f9fa;
            padding: 8px;
            border-radius: 5px;
        }
        .debug-info {
            font-size: 0.8rem;
            color: #666;
            margin-top: 10px;
            border-top: 1px solid #eee;
            padding-top: 5px;
            background-color: #fafafa;
            padding: 8px;
            border-radius: 5px;
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
    if 'file_stats' not in st.session_state:
        st.session_state.file_stats = {}
    if 'debug_mode' not in st.session_state:
        st.session_state.debug_mode = False
    
    # Load data
    if st.session_state.faq_data is None:
        with st.spinner("Loading FAQ data..."):
            df, topics, file_stats = load_excel_data()
            st.session_state.faq_data = df
            st.session_state.topics = topics
            st.session_state.file_stats = file_stats
    
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
            st.error("❌ No data loaded")
        
        if st.button("🔄 Reset Chat"):
            st.session_state.messages = []
            st.rerun()
        
        if st.session_state.debug_mode:
            with st.expander("🔧 File Details"):
                for filename, stats in st.session_state.file_stats.items():
                    if not filename.startswith('_'):
                        if 'error' in stats:
                            st.error(f"❌ {filename}: {stats['error']}")
                        else:
                            st.success(f"✅ {filename}: {stats['final_rows']} rows")
    
    # Display chat messages
    for message in st.session_state.messages:
        if message["role"] == "user":
            st.markdown(f'<div class="user-message">👤 {message["content"]}</div>', unsafe_allow_html=True)
        else:
            content = message["content"]
            sources = message.get("sources", "")
            debug_info = message.get("debug_info", "")
            
            st.markdown(f'<div class="bot-message">🏢 {content}</div>', unsafe_allow_html=True)
            
            if show_sources and sources:
                st.markdown(f'<div class="source-info">{sources}</div>', unsafe_allow_html=True)
            
            if st.session_state.debug_mode and debug_info:
                st.markdown(f'<div class="debug-info">{debug_info}</div>', unsafe_allow_html=True)
    
    # User input
    if st.session_state.faq_data.empty:
        st.warning("Please wait for data to load.")
        return
    
    user_input = st.chat_input("Ask your question about DLD services...")
    
    if user_input:
        # Add user message
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        with st.spinner("Finding answer..."):
            # Search for relevant FAQs with detailed tracking
            search_results, search_stats = detailed_search(user_input, st.session_state.faq_data, selected_topic)
            
            # Generate response with stats
            response, generation_stats = generate_enhanced_response(user_input, search_results, st.session_state.debug_mode)
            
            # Create source information
            sources = ""
            debug_info = ""
            
            if search_results:
                if len(search_results) == 1:
                    result = search_results[0]
                    sources = f"📚 **Source**: {result['service']} | **File**: {result['source_file']}\n"
                    sources += f"**Question**: {result['question']}\n"
                    sources += f"**Relevance Score**: {result['total_score']}/100"
                else:
                    sources = f"📚 **Sources**: Based on {len(search_results)} FAQ items\n"
                    for i, result in enumerate(search_results):
                        sources += f"{i+1}. {result['service']} (Score: {result['total_score']}) | {result['source_file']}\n"
                
                # Debug information
                if st.session_state.debug_mode:
                    debug_info = f"🔍 **Search Stats**:\n"
                    debug_info += f"• Query: '{search_stats['query']}'\n"
                    debug_info += f"• Search time: {search_stats['search_time']:.3f}s\n"
                    debug_info += f"• Documents searched: {search_stats['total_documents_searched']}\n"
                    debug_info += f"• Results found: {search_stats['results_found']}\n"
                    debug_info += f"• Top score: {search_stats['top_score']}/100\n"
                    debug_info += f"• Topic filter: {search_stats['topic_filter']}\n\n"
                    
                    debug_info += f"🤖 **Generation Stats**:\n"
                    debug_info += f"• Method: {generation_stats['method']}\n"
                    debug_info += f"• Generation time: {generation_stats['generation_time']:.3f}s\n"
                    debug_info += f"• OpenAI used: {generation_stats['openai_used']}\n"
                    
                    if search_results:
                        debug_info += f"\n📊 **Top Result Scoring**:\n"
                        top_result = search_results[0]
                        scoring = top_result['scoring_details']
                        debug_info += f"• Exact phrase match: {scoring['exact_phrase']} points\n"
                        debug_info += f"• Word overlap: {scoring['word_overlap']} points\n"
                        debug_info += f"• Keyword matches: {scoring['keyword_matches']} points\n"
                        debug_info += f"• Question patterns: {scoring['question_patterns']} points\n"
                        debug_info += f"• **Total**: {top_result['total_score']} points\n"
                        
                        if top_result['matched_keywords']:
                            debug_info += f"• Matched keywords: {', '.join(top_result['matched_keywords'])}\n"
                        if top_result['common_words']:
                            debug_info += f"• Common words: {', '.join(top_result['common_words'])}\n"
            
            # Add bot response
            st.session_state.messages.append({
                "role": "assistant",
                "content": response,
                "sources": sources,
                "debug_info": debug_info
            })
        
        st.rerun()

if __name__ == "__main__":
    main()
