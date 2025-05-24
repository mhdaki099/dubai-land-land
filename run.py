import streamlit as st
import pandas as pd
import json
import os
import re
import openai
from dotenv import load_dotenv
import numpy as np

# Load environment variables
load_dotenv()

# Configure OpenAI API - Try Streamlit secrets first, then env vars
try:
    openai.api_key = st.secrets["OPENAI_API_KEY"]
except:
    openai.api_key = os.getenv("OPENAI_API_KEY")

# Create OpenAI client for new API
if openai.api_key:
    client = openai.OpenAI(api_key=openai.api_key)
else:
    client = None

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
    }
    
    .sub-title {
        font-size: 1.6rem;
        color: #555;
        text-align: center;
        margin-bottom: 2.5rem;
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
        background-color: #f8f9fa;
        padding: 0.8rem;
        border-radius: 8px;
    }
    
    .debug-info {
        font-size: 0.85rem;
        color: #666;
        margin-top: 0.8rem;
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
if 'topics' not in st.session_state:
    st.session_state.topics = ["All Topics"]
if 'selected_topic' not in st.session_state:
    st.session_state.selected_topic = "All Topics"
if 'debug_mode' not in st.session_state:
    st.session_state.debug_mode = False
if 'show_sources' not in st.session_state:
    st.session_state.show_sources = True
if 'data_processed' not in st.session_state:
    st.session_state.data_processed = False

def detect_language(text):
    """Detect if text is in Arabic or English."""
    arabic_pattern = re.compile(r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]+')
    if arabic_pattern.search(text):
        return "arabic"
    return "english"

def translate_text(text, target_language):
    """Translate text between English and Arabic."""
    if not text or not client:
        return text
        
    try:
        system_prompt = f"Translate the following text to {target_language}. Keep the translation natural and accurate."
        
        response = client.chat.completions.create(
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
        return text

@st.cache_data
def process_excel_files():
    """Process Excel files from the data directory."""
    data_dir = "data"
    all_qa_pairs = []
    
    if not os.path.exists(data_dir):
        st.error(f"Data directory '{data_dir}' not found!")
        return pd.DataFrame(), ["All Topics"]
    
    # Get all Excel files
    excel_files = [f for f in os.listdir(data_dir) if f.endswith('.xlsx')]
    
    if not excel_files:
        st.error(f"No Excel files found in '{data_dir}' directory!")
        return pd.DataFrame(), ["All Topics"]
    
    st.info(f"Found {len(excel_files)} Excel files: {', '.join(excel_files[:5])}{'...' if len(excel_files) > 5 else ''}")
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, filename in enumerate(excel_files):
        try:
            file_path = os.path.join(data_dir, filename)
            service_name = filename.replace('FAQs.xlsx', '').replace('.xlsx', '').strip()
            
            status_text.text(f"Processing {filename}...")
            progress_bar.progress((i + 1) / len(excel_files))
            
            # Try different header rows to find the right format
            df = None
            for header_row in [0, 1, 2]:
                try:
                    df_temp = pd.read_excel(file_path, header=header_row)
                    
                    # Check if this looks like the right format
                    expected_cols = ['Question (English)', 'Answer (English)']
                    if all(col in df_temp.columns for col in expected_cols):
                        df = df_temp
                        break
                    
                    # Also check for similar column names
                    cols_lower = [str(col).lower() for col in df_temp.columns]
                    if any('question' in col and ('english' in col or 'eng' in col) for col in cols_lower):
                        df = df_temp
                        # Rename columns to standard format
                        column_mapping = {}
                        for col in df.columns:
                            col_str = str(col).lower()
                            if 'question' in col_str and ('eng' in col_str or 'english' in col_str):
                                column_mapping[col] = 'Question (English)'
                            elif 'answer' in col_str and ('eng' in col_str or 'english' in col_str):
                                column_mapping[col] = 'Answer (English)'
                            elif 'question' in col_str and ('ar' in col_str or 'arabic' in col_str):
                                column_mapping[col] = 'Question (Arabic)'
                            elif 'answer' in col_str and ('ar' in col_str or 'arabic' in col_str):
                                column_mapping[col] = 'Answer (Arabic)'
                            elif 'module' in col_str or 'section' in col_str:
                                column_mapping[col] = 'Module'
                            elif 'keyword' in col_str:
                                column_mapping[col] = 'Keywords'
                        
                        if column_mapping:
                            df = df.rename(columns=column_mapping)
                        break
                except Exception as e:
                    continue
            
            if df is None:
                st.warning(f"Could not find proper format in {filename}")
                continue
            
            # Extract Q&A pairs
            qa_pairs = []
            current_module = service_name
            
            for idx, row in df.iterrows():
                # Skip empty rows
                if pd.isna(row.get('Question (English)')) or pd.isna(row.get('Answer (English)')):
                    continue
                
                question = str(row['Question (English)']).strip()
                answer = str(row['Answer (English)']).strip()
                
                if not question or not answer or question == 'nan' or answer == 'nan':
                    continue
                
                # Update module if available
                if 'Module' in row and not pd.isna(row['Module']):
                    current_module = str(row['Module']).strip()
                
                qa_pair = {
                    'Question (English)': question,
                    'Answer (English)': answer,
                    'Question (Arabic)': str(row.get('Question (Arabic)', '')).strip() if not pd.isna(row.get('Question (Arabic)')) else '',
                    'Answer (Arabic)': str(row.get('Answer (Arabic)', '')).strip() if not pd.isna(row.get('Answer (Arabic)')) else '',
                    'Service': service_name,
                    'Module': current_module,
                    'Keywords': str(row.get('Keywords', '')).strip() if not pd.isna(row.get('Keywords')) else ''
                }
                
                qa_pairs.append(qa_pair)
            
            all_qa_pairs.extend(qa_pairs)
            if qa_pairs:
                st.success(f"✅ Extracted {len(qa_pairs)} Q&A pairs from {filename}")
            
        except Exception as e:
            st.error(f"❌ Error processing {filename}: {str(e)}")
            continue
    
    progress_bar.empty()
    status_text.empty()
    
    if not all_qa_pairs:
        st.error("No Q&A pairs were extracted from any files!")
        return pd.DataFrame(), ["All Topics"]
    
    # Create DataFrame
    df = pd.DataFrame(all_qa_pairs)
    
    # Remove duplicates
    df = df.drop_duplicates(subset=['Question (English)'], keep='first')
    
    # Get unique services as topics
    services = sorted(df['Service'].unique().tolist())
    topics = ["All Topics"] + services
    
    st.success(f"🎉 Successfully processed {len(df)} unique Q&A pairs from {len(excel_files)} files!")
    
    return df, topics

def search_faqs_simple(query, df, topic=None, top_k=3):
    """Simple text-based search through FAQs."""
    if df.empty or not query:
        return []
    
    # Filter by topic if specified
    if topic and topic != "All Topics":
        filtered_df = df[df['Service'] == topic]
        if len(filtered_df) == 0:
            filtered_df = df
    else:
        filtered_df = df
    
    query_lower = query.lower().strip()
    query_words = set(query_lower.split())
    
    # Score each FAQ item
    scores = []
    for idx, row in filtered_df.iterrows():
        question = str(row['Question (English)']).lower()
        answer = str(row['Answer (English)']).lower()
        
        score = 0
        
        # Exact substring matches
        if query_lower in question:
            score += 20
        if query_lower in answer:
            score += 10
        
        # Word overlap
        question_words = set(question.split())
        answer_words = set(answer.split())
        
        common_q_words = query_words.intersection(question_words)
        common_a_words = query_words.intersection(answer_words)
        
        score += len(common_q_words) * 5
        score += len(common_a_words) * 2
        
        # Keyword matching
        if 'Keywords' in row and not pd.isna(row['Keywords']):
            keywords = str(row['Keywords']).lower()
            if query_lower in keywords:
                score += 15
            keyword_words = set(keywords.split())
            common_k_words = query_words.intersection(keyword_words)
            score += len(common_k_words) * 3
        
        if score > 0:
            scores.append((score, idx))
    
    # Sort by score and return top results
    scores.sort(reverse=True, key=lambda x: x[0])
    
    results = []
    for i, (score, idx) in enumerate(scores[:top_k]):
        row = filtered_df.iloc[idx]
        results.append({
            'question': row['Question (English)'],
            'answer': row['Answer (English)'],
            'service': row['Service'],
            'module': row.get('Module', ''),
            'relevance': score / 100.0,  # Normalize score
            'debug_info': {
                'match_type': 'Text',
                'text_score': score,
                'idx': idx,
                'rank': i+1
            }
        })
    
    return results

def search_faqs_improved(query, df, topic=None, top_k=3):
    """Enhanced search using OpenAI if available, otherwise fallback to simple search."""
    if not client:
        return search_faqs_simple(query, df, topic, top_k)
    
    if df.empty or not query:
        return []
    
    # Filter by topic
    if topic and topic != "All Topics":
        filtered_df = df[df['Service'] == topic]
        if len(filtered_df) == 0:
            filtered_df = df
    else:
        filtered_df = df
    
    try:
        # Limit questions for OpenAI to avoid token limits
        max_questions = min(100, len(filtered_df))
        questions_sample = filtered_df['Question (English)'].head(max_questions).tolist()
        
        # Format questions for OpenAI
        questions_text = "\n".join([f"{i+1}. {q}" for i, q in enumerate(questions_sample)])
        
        system_prompt = """
        You are a search engine for Dubai Land Department FAQs. Find the most relevant questions for the user's query.
        Return only the numbers of the 3 most relevant questions, separated by commas.
        If no questions are relevant, return "NONE".
        Focus on semantic similarity and intent matching.
        """
        
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"User query: {query}\n\nFAQ Questions:\n{questions_text}"}
            ],
            temperature=0.1
        )
        
        result = response.choices[0].message.content.strip()
        
        # Parse OpenAI response
        results = []
        if result != "NONE":
            question_numbers = [int(num.strip()) for num in re.findall(r'\d+', result)]
            for i, num in enumerate(question_numbers[:top_k]):
                if 1 <= num <= len(questions_sample):
                    idx = num - 1  # Convert to 0-based
                    row = filtered_df.iloc[idx]
                    results.append({
                        'question': row['Question (English)'],
                        'answer': row['Answer (English)'],
                        'service': row['Service'],
                        'module': row.get('Module', ''),
                        'relevance': 1.0 - (0.1 * i),
                        'debug_info': {
                            'match_type': 'Semantic',
                            'text_score': 0,
                            'idx': idx,
                            'rank': i+1
                        }
                    })
        
        # If no semantic results, fallback to simple search
        if not results:
            results = search_faqs_simple(query, df, topic, top_k)
        
        return results
        
    except Exception as e:
        st.warning(f"OpenAI search failed: {str(e)}")
        return search_faqs_simple(query, df, topic, top_k)

def generate_response(query, relevant_faqs, language):
    """Generate a response based on relevant FAQs."""
    if not relevant_faqs:
        if language == "arabic":
            return "عذراً، لم أتمكن من العثور على إجابة لسؤالك. هل يمكنك إعادة صياغة السؤال؟"
        else:
            return "I'm sorry, I couldn't find an answer to your question. Could you rephrase your question?"
    
    # If no OpenAI client, return first FAQ answer
    if not client:
        answer = relevant_faqs[0]['answer']
        if language == "arabic":
            return f"(AI features unavailable without OpenAI API key)\n\n{answer}"
        return answer
    
    # For single highly relevant FAQ, return answer directly
    if len(relevant_faqs) == 1 and relevant_faqs[0]['relevance'] > 0.8:
        answer = relevant_faqs[0]['answer']
        if language == "arabic":
            try:
                answer = translate_text(answer, "arabic")
            except:
                pass
        return answer
    
    # Generate enhanced response using OpenAI
    try:
        faq_context = ""
        for i, faq in enumerate(relevant_faqs[:3]):  # Limit to top 3 to avoid token limits
            faq_context += f"\nQ: {faq['question']}\nA: {faq['answer']}\nService: {faq['service']}\n"
        
        system_prompt = """
        You are a helpful assistant for the Dubai Land Department. 
        Provide accurate, professional responses based on the FAQ information provided.
        Be concise but comprehensive. Use a friendly, professional tone.
        If the information doesn't fully answer the question, acknowledge this.
        """
        
        response = client.chat.completions.create(
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
        # Fallback to first FAQ answer
        return relevant_faqs[0]['answer']

def load_existing_data():
    """Try to load existing processed data."""
    csv_path = os.path.join("processed_data", "dld_faq_data.csv")
    json_path = os.path.join("processed_data", "dld_faq_data.json")
    
    # Try to load from processed files first
    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)
            services = sorted(df["Service"].unique().tolist())
            topics = ["All Topics"] + services
            st.success(f"✅ Loaded existing data: {len(df)} FAQ items")
            return df, topics
        except Exception as e:
            st.warning(f"Error loading existing CSV: {str(e)}")
    
    elif os.path.exists(json_path):
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                faq_data = json.load(f)
            df = pd.DataFrame(faq_data)
            services = sorted(df["Service"].unique().tolist())
            topics = ["All Topics"] + services
            st.success(f"✅ Loaded existing data: {len(df)} FAQ items")
            return df, topics
        except Exception as e:
            st.warning(f"Error loading existing JSON: {str(e)}")
    
    return None, None

def main():
    # Try to load existing data first
    if st.session_state.faq_data is None and not st.session_state.data_processed:
        with st.spinner("🔍 Checking for existing data..."):
            existing_df, existing_topics = load_existing_data()
            
            if existing_df is not None:
                st.session_state.faq_data = existing_df
                st.session_state.topics = existing_topics
                st.session_state.data_processed = True
            else:
                st.info("💾 No existing data found. Processing Excel files...")
                with st.spinner("🔄 Processing Excel files from data directory..."):
                    df, topics = process_excel_files()
                    st.session_state.faq_data = df
                    st.session_state.topics = topics
                    st.session_state.data_processed = True
    
    # Sidebar
    with st.sidebar:
        st.markdown('<div class="sidebar-header">🏢 DLD FAQ Assistant</div>', unsafe_allow_html=True)
        
        # API Status
        if client:
            st.success("✅ OpenAI API Connected")
        else:
            st.warning("⚠️ OpenAI API Not Connected")
            st.info("Add your API key in Streamlit secrets for enhanced features")
        
        # Data Status
        if st.session_state.faq_data is not None and not st.session_state.faq_data.empty:
            st.success(f"✅ Data Ready")
            st.info(f"📚 {len(st.session_state.faq_data)} FAQ items")
            st.info(f"🏷️ {len(st.session_state.topics)-1} services")
            
            # Show service breakdown
            if st.checkbox("Show Service Details"):
                service_counts = st.session_state.faq_data['Service'].value_counts()
                for service, count in service_counts.head(10).items():
                    st.write(f"• **{service}**: {count} items")
        else:
            st.error("❌ No data available")
        
        # Topic Selection
        if st.session_state.topics:
            st.markdown('<div class="sidebar-header">Topic Selection</div>', unsafe_allow_html=True)
            selected_topic = st.selectbox(
                "Select a service",
                st.session_state.topics,
                index=st.session_state.topics.index(st.session_state.selected_topic)
            )
            st.session_state.selected_topic = selected_topic
        
        # Settings
        st.markdown('<div class="sidebar-header">Settings</div>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            st.session_state.debug_mode = st.checkbox("Debug", value=st.session_state.debug_mode)
        with col2:
            st.session_state.show_sources = st.checkbox("Sources", value=st.session_state.show_sources)
        
        # Control buttons
        st.markdown("---")
        if st.button("🔄 Reprocess Data", use_container_width=True):
            st.session_state.faq_data = None
            st.session_state.data_processed = False
            st.cache_data.clear()
            st.rerun()
        
        if st.button("💬 Reset Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
    
    # Main content
    st.markdown('<h1 class="main-title">DLD FAQ Assistant</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">Ask me anything about Dubai Land Department services</p>', unsafe_allow_html=True)
    
    # Show data processing status
    if st.session_state.faq_data is None or st.session_state.faq_data.empty:
        st.error("⚠️ No FAQ data available. Please check that Excel files are in the 'data' directory.")
        st.stop()
    
    # Display chat messages
    for message in st.session_state.messages:
        if message["role"] == "user":
            st.markdown(f'<div class="user-message">👤 {message["content"]}</div>', unsafe_allow_html=True)
        else:
            content = message["content"]
            source_info = message.get("source_info", "")
            debug_info = message.get("debug_info", "")
            
            st.markdown(f'<div class="bot-message">🏢 {content}</div>', unsafe_allow_html=True)
            
            if st.session_state.show_sources and source_info:
                st.markdown(f'<div class="source-info">{source_info}</div>', unsafe_allow_html=True)
            
            if st.session_state.debug_mode and debug_info:
                st.markdown(f'<div class="debug-info">{debug_info}</div>', unsafe_allow_html=True)
    
    # User input
    user_query = st.chat_input("Type your question here...")
    
    if user_query:
        # Add user message
        st.session_state.messages.append({"role": "user", "content": user_query})
        
        with st.spinner("🔍 Finding the answer..."):
            # Detect language
            language = detect_language(user_query)
            
            # Translate to English if needed
            english_query = user_query
            if language == "arabic" and client:
                try:
                    english_query = translate_text(user_query, "english")
                except:
                    pass
            
            # Search FAQs
            relevant_faqs = search_faqs_improved(
                english_query, 
                st.session_state.faq_data, 
                st.session_state.selected_topic
            )
            
            # Generate response
            response = generate_response(english_query, relevant_faqs, language)
            
            # Create source info
            source_info = ""
            if st.session_state.show_sources and relevant_faqs:
                if len(relevant_faqs) == 1:
                    source_info = f"📚 Source: '{relevant_faqs[0]['question']}' from {relevant_faqs[0]['service']}"
                else:
                    services = list(set([faq['service'] for faq in relevant_faqs[:3]]))
                    source_info = f"📚 Based on {len(relevant_faqs)} FAQ items from: {', '.join(services)}"
            
            # Create debug info
            debug_info = ""
            if st.session_state.debug_mode and relevant_faqs:
                debug_info = f"Language: {language} | FAQs found: {len(relevant_faqs)} | Match type: {relevant_faqs[0]['debug_info']['match_type']}"
            
            # Add bot response
            st.session_state.messages.append({
                "role": "assistant",
                "content": response,
                "source_info": source_info,
                "debug_info": debug_info
            })
        
        st.rerun()

if __name__ == "__main__":
    main()
