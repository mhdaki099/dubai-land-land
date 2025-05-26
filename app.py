import streamlit as st
import pandas as pd
import json
import os
import re
import openai
from dotenv import load_dotenv
import numpy as np

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
                st.sidebar.success("✅ Using Streamlit secrets (openai.OPENAI_API_KEY)")
            elif 'OPENAI_API_KEY' in st.secrets:
                api_key = st.secrets['OPENAI_API_KEY']
                st.sidebar.success("✅ Using Streamlit secrets (OPENAI_API_KEY)")
            else:
                # Fallback to environment variables (for local development)
                load_dotenv()
                api_key = os.getenv("OPENAI_API_KEY")
                if api_key:
                    st.sidebar.info("ℹ️ Using environment variables")
                else:
                    st.sidebar.error("❌ No OpenAI API key found!")
                    return None
        else:
            # Fallback to environment variables
            load_dotenv()
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
# DIRECT EXCEL PROCESSING (NEW)
# =============================================================================

@st.cache_data
def load_excel_data_directly():
    """Load Excel data directly from the data directory with detailed tracking."""
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
        st.warning(f"⚠️ No Excel files found in '{data_dir}' directory!")
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
            question_col = None
            answer_col = None
            
            for header_row in [1, 2, 0]:
                try:
                    temp_df = pd.read_excel(file_path, header=header_row)
                    temp_df = temp_df.dropna(how='all')
                    
                    # Look for question and answer columns
                    question_col = None
                    answer_col = None
                    
                    for col in temp_df.columns:
                        col_str = str(col).lower()
                        if 'question' in col_str and ('eng' in col_str or 'english' in col_str):
                            question_col = col
                        elif 'answer' in col_str and ('eng' in col_str or 'english' in col_str):
                            answer_col = col
                    
                    if question_col and answer_col:
                        df = temp_df
                        header_used = header_row
                        break
                        
                except Exception as e:
                    print(f"[DEBUG] Error trying header row {header_row} for {filename}: {str(e)}")
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
                    'answer_col': answer_col,
                    'status': 'success'
                }
                
                if final_count > 0:
                    all_data.append(clean_df)
                    print(f"[DEBUG] Successfully processed {filename}: {final_count} Q&A pairs")
                else:
                    file_stats[filename]['status'] = 'failed'
                    file_stats[filename]['error'] = 'No valid Q&A pairs after cleaning'
                    
            else:
                error_msg = "Could not find question/answer columns"
                if question_col and not answer_col:
                    error_msg = f"Found question column '{question_col}' but no answer column"
                elif answer_col and not question_col:
                    error_msg = f"Found answer column '{answer_col}' but no question column"
                elif not question_col and not answer_col:
                    available_cols = list(df.columns) if df is not None else "Could not read file"
                    error_msg = f"No question/answer columns found. Available columns: {available_cols}"
                
                file_stats[filename] = {
                    'service': service_name,
                    'error': error_msg,
                    'status': 'failed'
                }
                print(f"[DEBUG] Failed to process {filename}: {error_msg}")
                
        except Exception as e:
            file_stats[filename] = {
                'service': service_name,
                'error': f"File processing error: {str(e)}",
                'status': 'failed'
            }
            print(f"[DEBUG] Exception processing {filename}: {str(e)}")
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
            'total_files_found': len(excel_files),
            'total_files_processed': len([f for f in file_stats if not f.startswith('_') and file_stats[f].get('status') == 'success']),
            'total_files_failed': len([f for f in file_stats if not f.startswith('_') and file_stats[f].get('status') == 'failed']),
            'original_total_rows': original_total,
            'final_total_rows': final_total,
            'duplicates_removed': original_total - final_total
        }
        
        print(f"[DEBUG] Data loading complete: {final_total} FAQ items from {len(all_data)} files")
        st.success(f"✅ Loaded {final_total} FAQ items from {len(all_data)} files")
        return combined_df, topics, file_stats
    else:
        # Add summary even if no data
        file_stats['_summary'] = {
            'total_files_found': len(excel_files),
            'total_files_processed': 0,
            'total_files_failed': len([f for f in file_stats if not f.startswith('_') and file_stats[f].get('status') == 'failed']),
            'original_total_rows': 0,
            'final_total_rows': 0,
            'duplicates_removed': 0
        }
        
        print("[DEBUG] No valid data found in any Excel files")
        st.error("❌ No valid data found in Excel files")
        return pd.DataFrame(), ["All Topics"], file_stats

# Load environment variables (fallback for local development)
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="DLD FAQ Chatbot",
    page_icon="🏢",
    layout="wide"
)

# Custom CSS (same as your original)
st.markdown("""
<style>
    /* Main container styling */
    .main {
        padding: 2rem;
        max-width: 1200px;
        margin: 0 auto;
    }
    
    /* Title styling */
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
    
    /* Chat message styling */
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
    
    /* Sidebar styling */
    .sidebar-header {
        font-size: 1.3rem;
        font-weight: 600;
        color: #0f4c81;
        margin: 1.5rem 0 0.8rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #e6f3ff;
    }
    
    /* Info boxes styling */
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
    
    .enhanced-answer {
        margin-top: 1.2rem;
        background-color: #f8f9fa;
        padding: 1.2rem;
        border-radius: 10px;
        border-left: 4px solid #0f4c81;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    /* Button styling */
    .stButton>button {
        background-color: #0f4c81;
        color: white;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        border: none;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        background-color: #0d3d6b;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state variables
if 'messages' not in st.session_state:
    st.session_state.messages = []
    
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
    
if 'enhanced_answers' not in st.session_state:
    st.session_state.enhanced_answers = True

if 'file_stats' not in st.session_state:
    st.session_state.file_stats = {}

# Function to detect language
def detect_language(text):
    """Detect if text is in Arabic or English."""
    arabic_pattern = re.compile(r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]+')
    if arabic_pattern.search(text):
        return "arabic"
    return "english"

# Function to translate text - FIXED FOR CLOUD COMPATIBILITY
def translate_text(text, target_language):
    """Translate text between English and Arabic."""
    if not text:
        return ""
        
    try:
        system_prompt = f"Translate the following text to {target_language}. Keep the translation natural and accurate."
        
        # Using CONSISTENT older OpenAI API syntax for cloud compatibility
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
        st.error(f"Translation error: {str(e)}")
        return text  # Return original answer if translation fails

# Function to load the FAQ data - UPDATED TO USE DIRECT EXCEL PROCESSING
def load_faq_data():
    """Load the FAQ data from Excel files or processed data."""
    # First try to load from processed data
    csv_path = os.path.join("processed_data", "dld_faq_data.csv")
    json_path = os.path.join("processed_data", "dld_faq_data.json")
    
    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)
            st.sidebar.success(f"Loaded {len(df)} FAQ items from processed CSV")
            
            # Get unique services as topics
            if 'Service' in df.columns:
                services = sorted(df["Service"].unique().tolist())
                topics = ["All Topics"] + services
            else:
                topics = ["All Topics"]
            
            return df, topics, {}
            
        except Exception as e:
            st.error(f"Error loading CSV data: {str(e)}")
    
    elif os.path.exists(json_path):
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                faq_data = json.load(f)
                
            df = pd.DataFrame(faq_data)
            st.sidebar.success(f"Loaded {len(df)} FAQ items from processed JSON")
            
            # Get unique services as topics
            if 'Service' in df.columns:
                services = sorted(df["Service"].unique().tolist())
                topics = ["All Topics"] + services
            else:
                topics = ["All Topics"]
            
            return df, topics, {}
            
        except Exception as e:
            st.error(f"Error loading JSON data: {str(e)}")
    
    # If no processed data found, try to load directly from Excel files
    st.info("📂 No processed data found. Loading directly from Excel files...")
    return load_excel_data_directly()

# Improved search function - FIXED FOR CLOUD COMPATIBILITY
def search_faqs_improved(query, df, topic=None, top_k=5):
    """Enhanced search function that combines text matching with OpenAI semantic search."""
    if df.empty or not query:
        return []
    
    # Filter by topic if specified
    if topic and topic != "All Topics":
        if 'Service' in df.columns:
            filtered_df = df[df['Service'] == topic]
            if len(filtered_df) == 0:  # If no items in selected topic
                filtered_df = df
        else:
            filtered_df = df
    else:
        filtered_df = df
    
    # Make sure we have the required columns
    if 'Question' not in filtered_df.columns:
        st.error("❌ Question column not found in data")
        return []
    
    if 'Answer' not in filtered_df.columns:
        st.error("❌ Answer column not found in data")
        return []
    
    # First, try to use OpenAI to find the most semantically similar questions
    try:
        # Create a list of questions for OpenAI to compare against
        max_questions = min(200, len(filtered_df))  # Limit to reduce token count
        # Create a list of questions with their indices
        questions_with_indices = [(i, q) for i, q in enumerate(filtered_df['Question'].head(max_questions))]
        
        # Format the questions for OpenAI
        questions_text = "\n".join([f"{i+1}. {q}" for i, (_, q) in enumerate(questions_with_indices)])
        
        system_prompt = """
        You are a search engine that finds questions in a FAQ database that are semantically similar 
        to a user's query, even if they use different wording.
        
        Given a user query and a list of FAQ questions, identify the top 3 questions that are most similar 
        in meaning to the user query. Look for questions that are asking for the same information, 
        even if they use different words or phrasing.
        
        Return ONLY the numbers of the 3 most semantically similar questions, separated by commas.
        Example output: 5, 12, 8
        
        If none of the questions are semantically similar, return "NONE".
        """
        
        # Using CONSISTENT older OpenAI API syntax for cloud compatibility
        response = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"User query: {query}\n\nFAQ Questions:\n{questions_text}"}
            ],
            temperature=0.2
        )
        
        result = response.choices[0].message.content.strip()
        
        # Get the similar questions
        semantic_indices = []
        if result != "NONE":
            # Extract the question numbers and map them back to dataframe indices
            question_numbers = [int(num.strip()) for num in re.findall(r'\d+', result)]
            for num in question_numbers:
                if 1 <= num <= len(questions_with_indices):
                    # Convert 1-based to 0-based and get the original dataframe index
                    semantic_indices.append(questions_with_indices[num-1][0])
        
        # Also do a simple text-based search as a backup
        # Clean query
        query_lower = query.lower().strip()
        query_words = set(query_lower.split())
        
        # Keywords for similar concepts
        time_keywords = {'time', 'fast', 'quick', 'duration', 'long', 'take', 'minutes', 'hours', 'days', 'complete', 'finish'}
        registration_keywords = {'register', 'registration', 'signup', 'sign-up', 'enroll', 'join', 'account', 'create'}
        
        # Score each FAQ item based on text matching
        scores = []
        for idx, row in filtered_df.iterrows():
            question = row['Question'].lower()
            question_words = set(question.split())
            
            # Calculate a relevance score
            score = 0
            
            # Check for exact substring match
            if query_lower in question:
                score += 20
                
            # Count word overlap
            common_words = query_words.intersection(question_words)
            score += len(common_words) * 8
            
            # Check for conceptual similarity (e.g., "how fast" vs "how long")
            if any(word in time_keywords for word in query_words) and any(word in time_keywords for word in question_words):
                score += 15
                
            # Check for registration-related terms
            if any(word in registration_keywords for word in query_words) and any(word in registration_keywords for word in question_words):
                score += 10
            
            # Check for common question patterns
            q_patterns = {
                'how': ['how'],
                'what': ['what'],
                'can': ['can', 'able', 'possible'],
                'where': ['where', 'location'],
                'when': ['when', 'time', 'date'],
                'why': ['why', 'reason'],
                'who': ['who'],
                'is': ['is', 'are', 'am'],
                'do': ['do', 'does', 'did']
            }
            
            for query_word in query_words:
                for pattern, variations in q_patterns.items():
                    if query_word in variations and pattern in question_words:
                        score += 5
                        break
            
            scores.append((score, idx))
        
        # Sort by score and take top matches
        scores.sort(reverse=True)
        text_indices = [idx for score, idx in scores[:top_k] if score > 0]
        
        # Combine the semantic and text indices, prioritizing semantic matches
        combined_indices = []
        for idx in semantic_indices:
            if idx not in combined_indices:
                combined_indices.append(idx)
                
        for idx in text_indices:
            if idx not in combined_indices and len(combined_indices) < top_k:
                combined_indices.append(idx)
        
        # If we still don't have enough results, add the top text matches
        while len(combined_indices) < min(top_k, 3) and len(text_indices) > len(combined_indices):
            next_idx = text_indices[len(combined_indices)]
            if next_idx not in combined_indices:
                combined_indices.append(next_idx)
        
        # Prepare the results
        results = []
        for i, idx in enumerate(combined_indices):
            # Check if the index is valid
            if 0 <= idx < len(filtered_df):
                row = filtered_df.iloc[idx]
                
                # Get the match source (semantic or text)
                match_type = "Semantic" if idx in semantic_indices else "Text"
                
                # Get the question score from text matching for debugging
                text_score = 0
                for score, score_idx in scores:
                    if score_idx == idx:
                        text_score = score
                        break
                
                results.append({
                    'question': row['Question'],
                    'answer': row['Answer'],
                    'service': row['Service'] if 'Service' in row else 'Unknown',
                    'source_file': row['Source_File'] if 'Source_File' in row else 'Unknown',
                    'relevance': 1.0 - (0.1 * i),  # Assign relevance based on position
                    'debug_info': {
                        'match_type': match_type,
                        'text_score': text_score,
                        'idx': idx,
                        'rank': i+1
                    }
                })
        
        return results
        
    except Exception as e:
        st.error(f"Error in semantic search: {str(e)}")
        # Fall back to basic text search
        return []

# Function to generate response - FIXED FOR CLOUD COMPATIBILITY
def generate_response(query, relevant_faqs, language):
    """Generate a response based on the relevant FAQs."""
    if not relevant_faqs:
        if language == "arabic":
            return "عذراً، لم أتمكن من العثور على إجابة لسؤالك. هل يمكنك إعادة صياغة السؤال أو طرح سؤال آخر؟"
        else:
            return "I'm sorry, I couldn't find an answer to your question. Could you rephrase your question or ask something else?"
    
    # For a single, highly relevant FAQ, just return the answer directly
    if len(relevant_faqs) == 1 and relevant_faqs[0]['relevance'] > 0.9:
        answer = relevant_faqs[0]['answer']
        if language == "arabic":
            try:
                answer = translate_text(answer, "arabic")
            except:
                pass  # Use original answer if translation fails
        return answer
    
    # Prepare the context from relevant FAQs
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
        
        # Using CONSISTENT older OpenAI API syntax for cloud compatibility
        response = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"User query: {query}\n\nRelevant FAQ information:{faq_context}"}
            ]
        )
        
        answer = response.choices[0].message.content
        
        # Translate if needed
        if language == "arabic":
            answer = translate_text(answer, "arabic")
            
        return answer
    
    except Exception as e:
        st.error(f"Error generating response: {str(e)}")
        
        if language == "arabic":
            return "عذراً، حدث خطأ أثناء معالجة طلبك. يرجى المحاولة مرة أخرى."
        else:
            return "Sorry, there was an error processing your request. Please try again."

# Function to create source explanation
def create_source_explanation(query, relevant_faqs, language="english"):
    """Create an explanation of the sources used to answer the query."""
    if not relevant_faqs:
        return ""
    
    # Enhanced source display with expandable sections
    sources_info = ""
    if language == "english":
        sources_info = "📚 **Sources used to answer your question:**"
    else:
        sources_info = "📚 **المصادر المستخدمة للإجابة على سؤالك:**"
    
    return sources_info

# Main application
def main():
    # Setup OpenAI for cloud deployment
    api_key = setup_openai_for_cloud()
    if not api_key:
        st.error("❌ Cannot proceed without OpenAI API key. Please configure it in Streamlit secrets.")
        st.stop()
    
    # Load data
    if st.session_state.faq_data is None:
        with st.spinner("Loading FAQ data..."):
            df, topics, file_stats = load_faq_data()
            st.session_state.faq_data = df
            st.session_state.topics = topics
            st.session_state.file_stats = file_stats
    
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
        st.session_state.enhanced_answers = st.checkbox("Enhanced Answers", value=st.session_state.enhanced_answers)
        
        st.markdown('<div class="sidebar-header">FAQ Data</div>', unsafe_allow_html=True)
        if not st.session_state.faq_data.empty:
            st.info(f"📚 {len(st.session_state.faq_data)} FAQ items across {len(st.session_state.topics)-1} topics")
            
            # ENHANCED DEBUG INFORMATION WITH FAILED FILES DETAILS
            if st.session_state.debug_mode and st.session_state.file_stats:
                with st.expander("📊 Data Processing Stats"):
                    if '_summary' in st.session_state.file_stats:
                        summary = st.session_state.file_stats['_summary']
                        st.write(f"**Files found**: {summary.get('total_files_found', 'Unknown')}")
                        st.write(f"**Files processed**: {summary.get('total_files_processed', 0)}")
                        st.write(f"**Files failed**: {summary.get('total_files_failed', 0)}")
                        st.write(f"**Total rows**: {summary.get('final_total_rows', 0)}")
                        st.write(f"**Duplicates removed**: {summary.get('duplicates_removed', 0)}")
                        
                        # Show detailed file processing results
                        st.markdown("---")
                        st.markdown("### 📋 Detailed File Processing Results")
                        
                        # Successful files
                        successful_files = [f for f in st.session_state.file_stats 
                                          if not f.startswith('_') and st.session_state.file_stats[f].get('status') == 'success']
                        if successful_files:
                            st.markdown("#### ✅ **Successfully Processed Files:**")
                            for filename in successful_files:
                                stats = st.session_state.file_stats[filename]
                                with st.expander(f"✅ {filename}", expanded=False):
                                    st.success(f"**Service:** {stats['service']}")
                                    st.write(f"**Header row used:** {stats['header_row']}")
                                    st.write(f"**Question column:** `{stats['question_col']}`")
                                    st.write(f"**Answer column:** `{stats['answer_col']}`")
                                    st.write(f"**Valid Q&A pairs:** {stats['final_rows']} (from {stats['original_rows']} rows)")
                        
                        # Failed files
                        failed_files = [f for f in st.session_state.file_stats 
                                      if not f.startswith('_') and st.session_state.file_stats[f].get('status') == 'failed']
                        if failed_files:
                            st.markdown("#### ❌ **Failed Files:**")
                            for filename in failed_files:
                                stats = st.session_state.file_stats[filename]
                                with st.expander(f"❌ {filename}", expanded=True):
                                    st.error(f"**Service:** {stats['service']}")
                                    st.error(f"**⚠️ Error:** {stats['error']}")
                                    
                                    # Add troubleshooting suggestions based on error type
                                    if 'Could not find question/answer columns' in stats['error']:
                                        st.info("""
                                        **💡 Troubleshooting Tips:**
                                        - Column headers should contain "question" + "eng" (or "english")
                                        - Column headers should contain "answer" + "eng" (or "english") 
                                        - Try renaming columns to: "Question (English)" and "Answer (English)"
                                        - Make sure headers are in row 1, 2, or 3
                                        """)
                                    elif 'No question/answer columns found' in stats['error']:
                                        st.info(f"""
                                        **💡 Available columns in this file:**
                                        {stats['error'].split('Available columns:')[1] if 'Available columns:' in stats['error'] else 'Unknown'}
                                        
                                        **Expected column patterns:**
                                        - Question column: contains "question" AND ("eng" OR "english")
                                        - Answer column: contains "answer" AND ("eng" OR "english")
                                        """)
                                    elif 'File processing error' in stats['error']:
                                        st.info(f"""
                                        **💡 File seems corrupted or unreadable:**
                                        - Try re-saving the file as .xlsx format
                                        - Check if file is password protected
                                        - Verify file is not corrupted
                                        """)
                                    else:
                                        st.info(f"**💡 Suggestion:** Check the file format and structure for {filename}")
                        
                        # Show column detection details
                        st.markdown("---")
                        st.markdown("### 🔍 **Column Detection Rules**")
                        st.info("""
                        **The app looks for columns containing these keywords:**
                        
                        **For Questions:**
                        - Must contain: "question" 
                        - AND contain: "eng" or "english"
                        - Examples: "Question (English)", "Question_Eng", "English Question"
                        
                        **For Answers:**
                        - Must contain: "answer"
                        - AND contain: "eng" or "english"  
                        - Examples: "Answer (English)", "Answer_Eng", "English Answer"
                        
                        **Header Detection:**
                        - Tries row 1, then row 2, then row 3 as headers
                        - Skips empty rows automatically
                        """)
                        
                        # Export failed files list
                        if failed_files:
                            st.markdown("---")
                            failed_files_text = "Failed Files Report\n" + "="*50 + "\n\n"
                            for filename in failed_files:
                                stats = st.session_state.file_stats[filename]
                                failed_files_text += f"❌ {filename}\n"
                                failed_files_text += f"   Service: {stats['service']}\n"
                                failed_files_text += f"   Error: {stats['error']}\n\n"
                            
                            st.download_button(
                                "📥 Download Failed Files Report",
                                failed_files_text,
                                "failed_files_report.txt",
                                "text/plain",
                                help="Download a detailed report of all failed files"
                            )
        else:
            st.error("❌ No FAQ data loaded")
            st.info("Please add Excel files to the 'data' directory")
        
        st.markdown("---")
        if st.button("🔄 Reset Conversation", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
    
    # Main content
    st.markdown('<h1 class="main-title">DLD FAQ Assistant</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">Ask me anything about Dubai Land Department services</p>', unsafe_allow_html=True)
    
    # Quick tips section
    with st.expander("💡 Quick Tips", expanded=False):
        st.markdown("""
        - Type your question in English or Arabic
        - Select a specific topic for more accurate answers
        - Enable 'Enhanced Answers' for detailed explanations
        - Use 'Show Sources' to see where the information comes from
        - Enable 'Debug Mode' to see detailed file processing information
        """)
    
    # Display chat messages
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.messages:
            if message["role"] == "user":
                st.markdown(f'<div class="user-message">👤 {message["content"]}</div>', unsafe_allow_html=True)
            else:
                content = message["content"]
                source_info = message.get("source_info", "")
                enhanced_answer = message.get("enhanced_answer", "")
                debug_info = message.get("debug_info", "")
                sources = message.get("sources", [])
                
                # Display the message
                st.markdown(f'<div class="bot-message">🏢 {content}</div>', unsafe_allow_html=True)
                
                # Show enhanced answer if available and enabled
                if st.session_state.enhanced_answers and enhanced_answer:
                    st.markdown(f'<div class="enhanced-answer">{enhanced_answer}</div>', unsafe_allow_html=True)
                
                # Show detailed sources if enabled
                if st.session_state.show_sources and sources:
                    st.markdown('<div class="source-info">', unsafe_allow_html=True)
                    st.markdown("📚 **Sources:**")
                    for i, source in enumerate(sources, 1):
                        with st.expander(f"Source {i}: {source['question'][:60]}..." if len(source['question']) > 60 else f"Source {i}: {source['question']}", expanded=False):
                            st.markdown(f"**📋 Service:** {source['service']}")
                            st.markdown(f"**❓ Question:** {source['question']}")
                            st.markdown(f"**✅ Answer:** {source['answer']}")
                            if 'source_file' in source:
                                st.markdown(f"**📄 File:** {source['source_file']}")
                            st.markdown(f"**🎯 Relevance Score:** {source.get('relevance', 0):.3f}")
                            if 'debug_info' in source:
                                st.markdown(f"**🔍 Search Method:** {source['debug_info'].get('match_type', 'Unknown')}")
                    st.markdown('</div>', unsafe_allow_html=True)
                
                # Show debug info if enabled
                if st.session_state.debug_mode and debug_info:
                    st.markdown(f'<div class="debug-info">{debug_info}</div>', unsafe_allow_html=True)
    
    # No FAQ data warning
    if st.session_state.faq_data.empty:
        st.warning("No FAQ data available. Please add Excel files to the 'data' directory.")
        return
    
    # User input with placeholder and help text
    user_query = st.chat_input(
        "Type your question here...",
        help="You can ask questions in English or Arabic. For best results, be specific and clear in your question."
    )
    
    # Process user query
    if user_query:
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": user_query})
        
        # Display updated chat with user's message
        st.rerun()
    
    # If there's a user message without a response, generate a response
    if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
        with st.spinner("Finding the answer..."):
            # Get the user's last query
            user_query = st.session_state.messages[-1]["content"]
            
            # Detect language
            language = detect_language(user_query)
            
            # Translate query to English if it's in Arabic
            if language == "arabic":
                english_query = translate_text(user_query, "english")
            else:
                english_query = user_query
            
            # Search for relevant FAQs using the improved search
            relevant_faqs = search_faqs_improved(
                english_query, 
                st.session_state.faq_data, 
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
            
            # Add bot response to chat history
            st.session_state.messages.append({
                "role": "assistant",
                "content": response,
                "source_info": source_info,
                "sources": relevant_faqs,  # Store full source details
                "debug_info": debug_info
            })
            
            # Display updated chat with bot's response
            st.rerun()

if __name__ == "__main__":
    main()
