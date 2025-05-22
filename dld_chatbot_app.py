import streamlit as st
import pandas as pd
import json
import os
import re
import openai
import logging
from dotenv import load_dotenv
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)

# Load environment variables
load_dotenv()

# Configure OpenAI API
openai.api_key = os.getenv("OPENAI_API_KEY")

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
        font-size: 2.5rem;
        color: #0f4c81;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-title {
        font-size: 1.5rem;
        color: #555;
        text-align: center;
        margin-bottom: 2rem;
    }
    .user-message {
        background-color: #e6f3ff;
        padding: 10px;
        border-radius: 10px;
        margin: 10px 0;
        border-left: 5px solid #0f4c81;
    }
    .bot-message {
        background-color: #f0f0f0;
        padding: 10px;
        border-radius: 10px;
        margin: 10px 0;
        border-left: 5px solid #555;
    }
    .sidebar-header {
        font-size: 1.2rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    .debug-info {
        font-size: 0.8rem;
        color: #666;
        margin-top: 5px;
        border-top: 1px solid #eee;
        padding-top: 5px;
    }
    .source-info {
        font-size: 0.9rem;
        color: #0f4c81;
        margin-top: 10px;
        font-style: italic;
        border-top: 1px dashed #ddd;
        padding-top: 5px;
    }
    .enhanced-answer {
        margin-top: 15px;
        background-color: #f9f9f9;
        padding: 10px;
        border-radius: 5px;
        border-left: 3px solid #0f4c81;
    }
    .highlight-point {
        font-weight: bold;
        color: #0f4c81;
    }
    .explanation {
        color: #555;
        font-style: italic;
        margin-left: 10px;
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

# Function to detect language
def detect_language(text):
    """Detect if text is in Arabic or English."""
    arabic_pattern = re.compile(r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]+')
    if arabic_pattern.search(text):
        return "arabic"
    return "english"

# Function to translate text
def translate_text(text, target_language):
    """Translate text between English and Arabic."""
    if not text:
        return ""
        
    try:
        system_prompt = f"Translate the following text to {target_language}. Keep the translation natural and accurate."
        
        # Using the newer OpenAI API syntax
        # response = openai.chat.completions.create(
        #     model="gpt-4",
        #     messages=[
        #         {"role": "system", "content": system_prompt},
        #         {"role": "user", "content": text}
        #     ],
        #     temperature=0.3
        # )

        response = openai.ChatCompletion.create(model="gpt-4", messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": f"User query: {text}"}])
        
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"Translation error: {str(e)}")
        return text  # Return original answer if translation fails

# Function to load the actual FAQ data from CSV or JSON
def load_faq_data():
    """Load the FAQ data directly from CSV or JSON file."""
    csv_path = os.path.join("processed_data", "dld_faq_data.csv")
    json_path = os.path.join("processed_data", "dld_faq_data.json")
    
    st.sidebar.write("Debug: Checking for data files...")
    st.sidebar.write(f"CSV path exists: {os.path.exists(csv_path)}")
    st.sidebar.write(f"JSON path exists: {os.path.exists(json_path)}")
    
    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)
            st.sidebar.success(f"Loaded {len(df)} FAQ items from CSV")
            st.sidebar.write(f"Debug: CSV columns: {df.columns.tolist()}")
            
            # Get unique services as topics
            services = sorted(df["Service"].unique().tolist())
            topics = ["All Topics"] + services
            
            return df, topics
            
        except Exception as e:
            st.error(f"Error loading CSV data: {str(e)}")
            st.sidebar.error(f"Debug: CSV loading error details: {str(e)}")
    
    elif os.path.exists(json_path):
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                faq_data = json.load(f)
                
            df = pd.DataFrame(faq_data)
            st.sidebar.success(f"Loaded {len(df)} FAQ items from JSON")
            st.sidebar.write(f"Debug: JSON columns: {df.columns.tolist()}")
            
            # Get unique services as topics
            services = sorted(df["Service"].unique().tolist())
            topics = ["All Topics"] + services
            
            return df, topics
            
        except Exception as e:
            st.error(f"Error loading JSON data: {str(e)}")
            st.sidebar.error(f"Debug: JSON loading error details: {str(e)}")
    
    # If no data files found
    st.error("No FAQ data files found. Please run the processor first.")
    return pd.DataFrame(), ["All Topics"]

# Improved search function using semantic matching with OpenAI
def search_faqs_improved(query, df, topic=None, top_k=5):
    """Enhanced search function that combines text matching with OpenAI semantic search."""
    if df.empty or not query:
        return []
    
    # Filter by topic if specified
    if topic and topic != "All Topics":
        filtered_df = df[df['Service'] == topic]
        if len(filtered_df) == 0:  # If no items in selected topic
            filtered_df = df
    else:
        filtered_df = df
    
    # First, try to use OpenAI to find the most semantically similar questions
    try:
        # Create a list of questions for OpenAI to compare against
        max_questions = min(200, len(filtered_df))  # Limit to reduce token count
        # Create a list of questions with their indices
        questions_with_indices = [(i, q) for i, q in enumerate(filtered_df['Question (English)'].head(max_questions))]
        
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
        
        # Using the newer OpenAI API syntax
        response = openai.chat.completions.create(
            model="gpt-4",
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
            question = row['Question (English)'].lower()
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
                    'question': row['Question (English)'],
                    'answer': row['Answer (English)'],
                    'service': row['Service'],
                    'module': row['Module'] if 'Module' in row and not pd.isna(row['Module']) else "",
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
        try:
            # Clean query
            query_lower = query.lower().strip()
            query_words = set(query_lower.split())
            
            # Score each FAQ item based on text matching
            scores = []
            for idx, row in filtered_df.iterrows():
                question = row['Question (English)'].lower()
                question_words = set(question.split())
                
                # Calculate a relevance score
                score = 0
                
                # Check for exact substring match
                if query_lower in question:
                    score += 20
                    
                # Count word overlap
                common_words = query_words.intersection(question_words)
                score += len(common_words) * 8
                
                scores.append((score, idx))
            
            # Sort by score and take top matches
            scores.sort(reverse=True)
            text_indices = [idx for score, idx in scores[:top_k] if score > 0]
            
            # Prepare the results
            results = []
            for i, idx in enumerate(text_indices):
                row = filtered_df.iloc[idx]
                results.append({
                    'question': row['Question (English)'],
                    'answer': row['Answer (English)'],
                    'service': row['Service'],
                    'module': row['Module'] if 'Module' in row and not pd.isna(row['Module']) else "",
                    'relevance': 1.0 - (0.1 * i),
                    'debug_info': {
                        'match_type': 'Text',
                        'text_score': scores[i][0],
                        'idx': idx,
                        'rank': i+1
                    }
                })
            
            return results
        except Exception as e:
            st.error(f"Error in fallback text search: {str(e)}")
            return []

# Function to generate response
def generate_response(query, relevant_faqs, language):
    """Generate a response based on the relevant FAQs."""
    if not relevant_faqs:
        if language == "arabic":
            return "عذراً، لم أتمكن من العثور على إجابة لسؤالك. هل يمكنك إعادة صياغة السؤال أو طرح سؤال آخر؟"
        else:
            return "I'm sorry, I couldn't find an answer to your question. Could you rephrase your question or ask something else?"
    
    # For a single, highly relevant FAQ, just return the answer directly
    if len(relevant_faqs) == 1 and relevant_faqs[
(Content truncated due to size limit. Use line ranges to read in chunks)
