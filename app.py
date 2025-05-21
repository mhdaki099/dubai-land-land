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
    
    /* Highlight styling */
    .highlight-point {
        font-weight: 600;
        color: #0f4c81;
        background-color: #e6f3ff;
        padding: 0.2rem 0.5rem;
        border-radius: 4px;
    }
    
    .explanation {
        color: #555;
        font-style: italic;
        margin-left: 1rem;
        padding-left: 1rem;
        border-left: 2px solid #e6f3ff;
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
    
    /* Input styling */
    .stTextInput>div>div>input {
        border-radius: 8px;
        border: 2px solid #e6f3ff;
        padding: 0.8rem;
    }
    
    /* Selectbox styling */
    .stSelectbox>div>div>select {
        border-radius: 8px;
        border: 2px solid #e6f3ff;
        padding: 0.5rem;
    }
    
    /* Checkbox styling */
    .stCheckbox>div>label {
        font-weight: 500;
        color: #555;
    }
    
    /* Spinner styling */
    .stSpinner>div {
        border-color: #0f4c81;
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
        response = openai.chat.completions.create(
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

# Function to load the actual FAQ data from CSV or JSON
def load_faq_data():
    """Load the FAQ data directly from CSV or JSON file."""
    csv_path = os.path.join("processed_data", "dld_faq_data.csv")
    json_path = os.path.join("processed_data", "dld_faq_data.json")
    
    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)
            st.sidebar.success(f"Loaded {len(df)} FAQ items from CSV")
            
            # Get unique services as topics
            services = sorted(df["Service"].unique().tolist())
            topics = ["All Topics"] + services
            
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
            services = sorted(df["Service"].unique().tolist())
            topics = ["All Topics"] + services
            
            return df, topics
            
        except Exception as e:
            st.error(f"Error loading JSON data: {str(e)}")
    
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
        if 'module' in faq and faq['module']:
            faq_context += f"Module: {faq['module']}\n"
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
        
        # Using the newer OpenAI API syntax
        response = openai.chat.completions.create(
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
    
    # Format the explanation based on the matched FAQs
    if language == "english":
        if len(relevant_faqs) == 1:
            explanation = f"📚 Source: This answer is based on the FAQ question '{relevant_faqs[0]['question']}' from the {relevant_faqs[0]['service']} service."
        else:
            explanation = "📚 Sources: This answer is based on the following FAQ questions:\n"
            for i, faq in enumerate(relevant_faqs[:3]):  # Limit to top 3
                explanation += f"{i+1}. '{faq['question']}' from {faq['service']}"
                if i < len(relevant_faqs[:3]) - 1:
                    explanation += "\n"
    else:  # Arabic
        # We'll need to translate this
        if len(relevant_faqs) == 1:
            base_explanation = f"📚 المصدر: تستند هذه الإجابة إلى سؤال الأسئلة الشائعة '{relevant_faqs[0]['question']}' من خدمة {relevant_faqs[0]['service']}."
            explanation = translate_text(base_explanation, "arabic")
        else:
            base_explanation = "📚 المصادر: تستند هذه الإجابة إلى أسئلة الأسئلة الشائعة التالية:\n"
            for i, faq in enumerate(relevant_faqs[:3]):  # Limit to top 3
                base_explanation += f"{i+1}. '{faq['question']}' من {faq['service']}"
                if i < len(relevant_faqs[:3]) - 1:
                    base_explanation += "\n"
            explanation = translate_text(base_explanation, "arabic")
    
    return explanation

# Function to enhance answers with highlighted key points and explanations
def enhance_answer(query, answer, relevant_faqs, language="english"):
    """Generate an enhanced version of the answer with highlights and explanations."""
    if not answer or not relevant_faqs:
        return ""
    
    # Get original FAQ answer text to compare
    original_answers = [faq['answer'] for faq in relevant_faqs]
    
    try:
        if language == "english":
            system_prompt = """
            You are an expert at explaining information clearly. Given a user's question and an answer,
            identify 2-3 key points from the answer and provide brief explanations for any terms or 
            concepts that might be ambiguous or need clarification.
            
            Format your response as follows:
            
            Key Points:
            • [Key Point 1]: [Brief explanation if needed]
            • [Key Point 2]: [Brief explanation if needed]
            • [Key Point 3]: [Brief explanation if needed]
            
            Your key points should be direct and actionable, focusing on the most important information.
            Your explanations should be concise but helpful, especially for technical terms or procedures.
            """
            
            # Using the newer OpenAI API syntax
            response = openai.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"User question: {query}\n\nAnswer: {answer}\n\nOriginal FAQ answer(s): {' '.join(original_answers)}"}
                ],
                temperature=0.2
            )
            
            enhanced = response.choices[0].message.content
        else:  # Arabic
            # First create the English version, then translate
            system_prompt = """
            You are an expert at explaining information clearly. Given a user's question and an answer,
            identify 2-3 key points from the answer and provide brief explanations for any terms or 
            concepts that might be ambiguous or need clarification.
            
            Format your response as follows:
            
            Key Points:
            • [Key Point 1]: [Brief explanation if needed]
            • [Key Point 2]: [Brief explanation if needed]
            • [Key Point 3]: [Brief explanation if needed]
            
            Your key points should be direct and actionable, focusing on the most important information.
            Your explanations should be concise but helpful, especially for technical terms or procedures.
            """
            
            # Create English version first
            response = openai.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"User question: {query}\n\nAnswer: {answer}\n\nOriginal FAQ answer(s): {' '.join(original_answers)}"}
                ],
                temperature=0.2
            )
            
            english_enhanced = response.choices[0].message.content
            
            # Then translate to Arabic
            enhanced = translate_text(english_enhanced, "arabic")
        
        return enhanced
        
    except Exception as e:
        st.error(f"Error enhancing answer: {str(e)}")
        return ""  # Return empty string if enhancement fails

# Main application
def main():
    # Load data
    if st.session_state.faq_data is None:
        with st.spinner("Loading FAQ data..."):
            df, topics = load_faq_data()
            st.session_state.faq_data = df
            st.session_state.topics = topics
    
    # Sidebar
    with st.sidebar:
        st.markdown('<div class="sidebar-header">Topic Selection</div>', unsafe_allow_html=True)
        selected_topic = st.selectbox(
            "Select a topic for your question",
            st.session_state.topics,
            index=st.session_state.topics.index(st.session_state.selected_topic)
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
        else:
            st.error("❌ No FAQ data loaded")
            st.info("Please run the data processor script first to extract FAQ data from Excel files.")
        
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
                
                # Display the message
                st.markdown(f'<div class="bot-message">🏢 {content}</div>', unsafe_allow_html=True)
                
                # Show enhanced answer if available and enabled
                if st.session_state.enhanced_answers and enhanced_answer:
                    st.markdown(f'<div class="enhanced-answer">{enhanced_answer}</div>', unsafe_allow_html=True)
                
                # Show source info if enabled
                if st.session_state.show_sources and source_info:
                    st.markdown(f'<div class="source-info">{source_info}</div>', unsafe_allow_html=True)
                
                # Show debug info if enabled
                if st.session_state.debug_mode and debug_info:
                    st.markdown(f'<div class="debug-info">{debug_info}</div>', unsafe_allow_html=True)
    
    # No FAQ data warning
    if st.session_state.faq_data.empty:
        st.warning("No FAQ data available. Please run the data processor script first to extract data from Excel files.")
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
            
            # Create enhanced answer if needed
            enhanced_answer = ""
            if st.session_state.enhanced_answers and relevant_faqs:
                enhanced_answer = enhance_answer(english_query, response, relevant_faqs, language)
            
            # Prepare debug information
            debug_info = ""
            if st.session_state.debug_mode:
                debug_info = f"Language: {language}\nRelevant FAQs found: {len(relevant_faqs)}"
            
            # Add bot response to chat history
            st.session_state.messages.append({
                "role": "assistant",
                "content": response,
                "source_info": source_info,
                "enhanced_answer": enhanced_answer,
                "debug_info": debug_info
            })
            
            # Display updated chat with bot's response
            st.rerun()

if __name__ == "__main__":
    main()