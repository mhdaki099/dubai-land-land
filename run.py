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
# CLOUD-COMPATIBLE DATA PROCESSOR
# =============================================================================

class CloudDataProcessor:
    def __init__(self, data_dir: str = "data", num_clusters: int = 12):
        """Initialize the cloud data processor."""
        self.data_dir = data_dir
        self.num_clusters = num_clusters
        self.df_combined = None
        self.cluster_names = {
            0: "DLD Website",
            1: "MyDLD App", 
            2: "Dubai REST API",
            3: "Ejari",
            4: "Property Registration",
            5: "Broker Services",
            6: "Payments",
            7: "Property Survey",
            8: "Property Trustee",
            9: "Rental Disputes",
            10: "Inspection System",
            11: "General Services"
        }
        
    def get_embedding(self, text: str) -> list:
        """Get embedding for text using OpenAI API."""
        try:
            response = openai.Embedding.create(
                model="text-embedding-3-small",
                input=text
            )
            return response["data"][0]["embedding"]
        except Exception as e:
            st.error(f"Error getting embedding: {str(e)}")
            return [0.0] * 1536
    
    def get_embeddings_batch(self, texts: list, batch_size: int = 50) -> np.ndarray:
        """Get embeddings for multiple texts in batches."""
        all_embeddings = []
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            batch_num = i//batch_size + 1
            total_batches = (len(texts)-1)//batch_size + 1
            
            status_text.text(f"Processing embedding batch {batch_num}/{total_batches}")
            progress_bar.progress(i / len(texts))
            
            try:
                response = openai.Embedding.create(
                    model="text-embedding-3-small",
                    input=batch_texts
                )
                batch_embeddings = [item["embedding"] for item in response["data"]]
                all_embeddings.extend(batch_embeddings)
            except Exception as e:
                st.error(f"Error in batch {batch_num}: {str(e)}")
                # Add zero vectors as fallback
                for _ in range(len(batch_texts)):
                    all_embeddings.append([0.0] * 1536)
        
        progress_bar.progress(1.0)
        status_text.text("✅ Embedding generation complete!")
        
        return np.array(all_embeddings).astype('float32')
    
    def load_excel_files(self) -> pd.DataFrame:
        """Load and process all Excel files."""
        all_dataframes = []
        
        if not os.path.exists(self.data_dir):
            st.error(f"Data directory '{self.data_dir}' not found!")
            return pd.DataFrame()
        
        excel_files = [f for f in os.listdir(self.data_dir) if f.endswith('.xlsx')]
        
        if not excel_files:
            st.error(f"No Excel files found in '{self.data_dir}' directory!")
            return pd.DataFrame()
        
        st.info(f"Found {len(excel_files)} Excel files to process")
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for idx, filename in enumerate(excel_files):
            status_text.text(f"Processing {filename} ({idx+1}/{len(excel_files)})")
            progress_bar.progress(idx / len(excel_files))
            
            try:
                file_path = os.path.join(self.data_dir, filename)
                service_name = filename.replace('FAQs.xlsx', '').replace('.xlsx', '').strip()
                
                # Try different header rows
                df = None
                for header_row in [1, 2, 0]:
                    try:
                        df_temp = pd.read_excel(file_path, header=header_row)
                        df_temp = df_temp.dropna(how='all')
                        
                        # Check if we have the expected columns
                        has_question = any('question' in str(col).lower() and 'eng' in str(col).lower() for col in df_temp.columns)
                        has_answer = any('answer' in str(col).lower() and 'eng' in str(col).lower() for col in df_temp.columns)
                        
                        if has_question and has_answer:
                            df = df_temp
                            break
                    except:
                        continue
                
                if df is None:
                    st.warning(f"Could not process {filename} - skipping")
                    continue
                
                # Map columns
                col_mapping = {}
                for col in df.columns:
                    col_str = str(col).lower()
                    if 'module' in col_str or 'section' in col_str:
                        col_mapping[col] = 'Module'
                    elif 'question' in col_str and ('eng' in col_str or 'en' in col_str):
                        col_mapping[col] = 'Question (English)'
                    elif 'answer' in col_str and ('eng' in col_str or 'en' in col_str):
                        col_mapping[col] = 'Answer (English)'
                    elif 'question' in col_str and ('ar' in col_str or 'arabic' in col_str):
                        col_mapping[col] = 'Question (Arabic)'
                    elif 'answer' in col_str and ('ar' in col_str or 'arabic' in col_str):
                        col_mapping[col] = 'Answer (Arabic)'
                    elif 'key' in col_str and 'word' in col_str:
                        col_mapping[col] = 'Keywords'
                
                if col_mapping:
                    df = df.rename(columns=col_mapping)
                
                # Ensure required columns exist
                if 'Question (English)' in df.columns and 'Answer (English)' in df.columns:
                    df['Service'] = service_name
                    
                    if 'Module' not in df.columns:
                        df['Module'] = service_name
                    
                    # Clean data
                    df = df[df['Question (English)'].notna() & df['Answer (English)'].notna()]
                    
                    # Add missing columns
                    for col in ['Question (Arabic)', 'Answer (Arabic)', 'Keywords']:
                        if col not in df.columns:
                            df[col] = ''
                    
                    all_dataframes.append(df)
                    st.success(f"✅ Processed {len(df)} FAQ items from {filename}")
                else:
                    st.warning(f"⚠️ Could not find required columns in {filename}")
                    
            except Exception as e:
                st.error(f"❌ Error processing {filename}: {str(e)}")
        
        progress_bar.progress(1.0)
        status_text.text("✅ File processing complete!")
        
        if all_dataframes:
            self.df_combined = pd.concat(all_dataframes, ignore_index=True)
            st.success(f"🎉 Successfully loaded {len(self.df_combined)} FAQ items from {len(all_dataframes)} files")
            return self.df_combined
        else:
            st.error("❌ No valid data found in Excel files")
            return pd.DataFrame()
    
    def clean_text(self, text: str) -> str:
        """Clean and preprocess text data."""
        if not isinstance(text, str):
            return ""
        text = re.sub(r'[^\w\s\.\?\!]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def preprocess_data(self) -> pd.DataFrame:
        """Preprocess the combined dataframe."""
        if self.df_combined is None or self.df_combined.empty:
            return pd.DataFrame()
        
        st.info("🔄 Preprocessing data...")
        
        # Clean text
        self.df_combined['Question_Clean'] = self.df_combined['Question (English)'].apply(self.clean_text)
        self.df_combined['Answer_Clean'] = self.df_combined['Answer (English)'].apply(self.clean_text)
        
        # Fill missing values
        self.df_combined['Question (Arabic)'].fillna('', inplace=True)
        self.df_combined['Answer (Arabic)'].fillna('', inplace=True)
        
        # Remove duplicates
        self.df_combined.drop_duplicates(subset=['Question_Clean'], keep='first', inplace=True)
        
        # Create embedding text
        self.df_combined['Embed_Text'] = self.df_combined['Question_Clean']
        if 'Keywords' in self.df_combined.columns:
            self.df_combined['Embed_Text'] += ' ' + self.df_combined['Keywords'].fillna('')
        
        st.success(f"✅ Preprocessed {len(self.df_combined)} unique FAQ items")
        return self.df_combined
    
    def create_clusters(self) -> pd.DataFrame:
        """Create topic clusters from the questions."""
        if self.df_combined is None or self.df_combined.empty:
            return pd.DataFrame()
        
        st.info("🤖 Creating embeddings and clusters...")
        
        # Generate embeddings
        embeddings = self.get_embeddings_batch(self.df_combined['Embed_Text'].tolist())
        
        # Perform clustering
        n_clusters = min(self.num_clusters, len(self.df_combined))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        self.df_combined['Cluster'] = kmeans.fit_predict(embeddings)
        
        # Map cluster IDs to names
        available_names = list(self.cluster_names.values())[:n_clusters]
        cluster_mapping = {i: available_names[i] if i < len(available_names) else f"Topic {i+1}" 
                          for i in range(n_clusters)}
        
        self.df_combined['Cluster_Name'] = self.df_combined['Cluster'].map(cluster_mapping)
        
        # Show cluster distribution
        cluster_counts = self.df_combined['Cluster_Name'].value_counts()
        st.info("📊 Cluster distribution:")
        for cluster, count in cluster_counts.items():
            st.write(f"  • {cluster}: {count} questions")
        
        return self.df_combined
    
    def create_faiss_index(self) -> tuple:
        """Create FAISS index for similarity search."""
        if self.df_combined is None or self.df_combined.empty:
            return None, None
        
        st.info("🔍 Creating search index...")
        
        # Generate embeddings for search
        texts = self.df_combined['Embed_Text'].tolist()
        embeddings = self.get_embeddings_batch(texts)
        
        # Create FAISS index
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings)
        
        st.success(f"✅ Created search index with {len(texts)} vectors")
        return index, embeddings
    
    def process_all(self):
        """Process all data and return results."""
        # Load Excel files
        df = self.load_excel_files()
        if df.empty:
            return None, None, None, None
        
        # Preprocess
        df = self.preprocess_data()
        if df.empty:
            return None, None, None, None
        
        # Create clusters
        df = self.create_clusters()
        
        # Create search index
        index, embeddings = self.create_faiss_index()
        
        return df, index, embeddings, self.cluster_names

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
if 'data_processed' not in st.session_state:
    st.session_state.data_processed = False

# Functions for chat functionality
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
        st.error(f"Translation error: {str(e)}")
        return text

def search_faqs_with_faiss(query, df, index, embeddings, topic=None, top_k=5):
    """Search FAQs using FAISS index."""
    if df.empty or not query or index is None:
        return []
    
    # Filter by topic if specified
    if topic and topic != "All Topics":
        filtered_df = df[df['Service'] == topic]
        if len(filtered_df) == 0:
            filtered_df = df
    else:
        filtered_df = df
    
    try:
        # Get query embedding
        processor = CloudDataProcessor()
        query_embedding = np.array(processor.get_embedding(query)).reshape(1, -1).astype('float32')
        
        # Search FAISS index
        distances, indices = index.search(query_embedding, min(top_k * 2, len(df)))
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(df):
                row = df.iloc[idx]
                
                # Skip if topic filter doesn't match
                if topic and topic != "All Topics" and row['Service'] != topic:
                    continue
                
                results.append({
                    'question': row['Question (English)'],
                    'answer': row['Answer (English)'],
                    'service': row['Service'],
                    'module': row['Module'] if 'Module' in row and not pd.isna(row['Module']) else "",
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
            for i, faq in enumerate(relevant_faqs[:3]):
                explanation += f"{i+1}. '{faq['question']}' from {faq['service']}"
                if i < len(relevant_faqs[:3]) - 1:
                    explanation += "\n"
    else:  # Arabic
        if len(relevant_faqs) == 1:
            base_explanation = f"📚 المصدر: تستند هذه الإجابة إلى سؤال الأسئلة الشائعة '{relevant_faqs[0]['question']}' من خدمة {relevant_faqs[0]['service']}."
            explanation = translate_text(base_explanation, "arabic")
        else:
            base_explanation = "📚 المصادر: تستند هذه الإجابة إلى أسئلة الأسئلة الشائعة التالية:\n"
            for i, faq in enumerate(relevant_faqs[:3]):
                base_explanation += f"{i+1}. '{faq['question']}' من {faq['service']}"
                if i < len(relevant_faqs[:3]) - 1:
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
    
    # Check if data needs to be processed
    if not st.session_state.data_processed:
        st.info("🚀 **First-time setup**: Processing your Excel files to create embeddings and search index...")
        
        with st.spinner("Processing data... This may take a few minutes."):
            processor = CloudDataProcessor()
            df, index, embeddings, cluster_names = processor.process_all()
            
            if df is not None and not df.empty:
                st.session_state.faq_data = df
                st.session_state.search_index = index
                st.session_state.embeddings = embeddings
                
                # Get topics from services
                services = sorted(df["Service"].unique().tolist())
                st.session_state.topics = ["All Topics"] + services
                
                st.session_state.data_processed = True
                st.success("🎉 Data processing complete! You can now ask questions.")
                st.rerun()
            else:
                st.error("❌ Failed to process data. Please check your Excel files.")
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
            num_topics = len(st.session_state.topics) - 1  # Exclude "All Topics"
            st.info(f"📚 {len(st.session_state.faq_data)} FAQ items across {num_topics} topics")
            st.success("✅ Real FAQ data with embeddings loaded")
        else:
            st.error("❌ No FAQ data loaded")
        
        st.markdown("---")
        if st.button("🔄 Reset Conversation", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
        
        if st.button("🔄 Reprocess Data", use_container_width=True):
            st.session_state.data_processed = False
            st.session_state.faq_data = None
            st.session_state.search_index = None
            st.session_state.embeddings = None
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
    
    # No FAQ data warning
    if st.session_state.faq_data is None or st.session_state.faq_data.empty:
        st.warning("Please wait for data processing to complete.")
        return
    
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
            
            # Search for relevant FAQs using FAISS
            relevant_faqs = search_faqs_with_faiss(
                english_query,
                st.session_state.faq_data,
                st.session_state.search_index,
                st.session_state.embeddings,
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
