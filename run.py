import streamlit as st
import pandas as pd
import json
import os
import re
import openai
import numpy as np
import logging

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
# TRY TO IMPORT PROCESSORS (OPTIONAL FOR CLOUD)
# =============================================================================

try:
    from dld_faq_processor import DLDFAQProcessor
    PROCESSOR_AVAILABLE = True
except ImportError:
    PROCESSOR_AVAILABLE = False
    st.sidebar.warning("⚠️ Data processor not available in cloud environment")

# =============================================================================
# EMBEDDED FAQ DATA FOR CLOUD DEPLOYMENT
# =============================================================================

# Since processed data files might not exist in cloud, embed sample data
SAMPLE_FAQ_DATA = [
    {
        "Question (English)": "How can I register a property sale with an initial mortgage?",
        "Answer (English)": "To register a property sale with an initial mortgage, you need to apply via either the 'Registration Trustee' or 'Oqood' service depending on the details of your transaction. The documents you'll need to provide include a sale and purchase contract, a valid Emirates ID (or passport for non-residents), and a bank letter detailing the mortgage value, date, and three mortgage contracts. It's important that you ensure all these documents are presented in order to successfully register your property sale. Don't hesitate to reach out if you need further clarification or assistance.",
        "Service": "Property Registration",
        "Module": "Registration Trustee"
    },
    {
        "Question (English)": "What documents do I need for property registration?",
        "Answer (English)": "For property registration, you typically need: 1) Sale and purchase contract, 2) Valid Emirates ID or passport, 3) Bank letter for mortgage details (if applicable), 4) No Objection Certificate (NOC) from developer, 5) Property title deed, 6) Payment receipts for registration fees. Additional documents may be required depending on the specific type of transaction.",
        "Service": "Property Registration", 
        "Module": "Documentation"
    },
    {
        "Question (English)": "How long does property registration take?",
        "Answer (English)": "Property registration typically takes 1-3 business days for standard transactions. Complex cases involving mortgages or multiple parties may take 5-7 business days. You can track your application status through the DLD website or MyDLD app. Processing times may vary during peak periods or for special cases requiring additional verification.",
        "Service": "Property Registration",
        "Module": "Processing Times"
    },
    {
        "Question (English)": "What are the fees for property registration?",
        "Answer (English)": "Property registration fees are typically 4% of the property value, plus administrative fees. There may be additional charges for mortgage registration (approximately 0.25% of mortgage amount). Exact fees depend on property type and transaction details. You can get a fee estimate through the DLD website calculator.",
        "Service": "Property Registration",
        "Module": "Fees and Charges"
    },
    {
        "Question (English)": "How can I check my property ownership status?",
        "Answer (English)": "You can check your property ownership status through: 1) DLD website portal, 2) MyDLD mobile app, 3) Visit DLD customer service centers, 4) Call DLD customer service hotline. You'll need your Emirates ID and property details. The online services are available 24/7 for your convenience.",
        "Service": "Property Inquiry",
        "Module": "Ownership Verification"
    },
    {
        "Question (English)": "What is Ejari and do I need it?",
        "Answer (English)": "Ejari is the online rental registration system in Dubai. It's mandatory for all rental contracts and serves as proof of residence. You need Ejari for: opening bank accounts, getting utility connections, school admissions, and visa applications. You can register through the DLD website or authorized typing centers.",
        "Service": "Ejari",
        "Module": "Rental Registration"
    },
    {
        "Question (English)": "How do I use the MyDLD app?",
        "Answer (English)": "MyDLD is DLD's official mobile app available on iOS and Android. You can use it to: check property ownership, view transaction history, pay fees, track application status, and access various DLD services. Download from your app store and register with your Emirates ID.",
        "Service": "MyDLD App",
        "Module": "Mobile Services"
    },
    {
        "Question (English)": "What services does a property trustee provide?",
        "Answer (English)": "A Property Trustee facilitates property transactions by ensuring all legal requirements are met. Services include: document verification, escrow services, legal compliance checks, transaction coordination, and dispute resolution. Trustees are licensed professionals who protect both buyer and seller interests.",
        "Service": "Property Trustee",
        "Module": "Professional Services"
    }
]

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
        return text

# Function to load FAQ data (cloud-compatible)
def load_faq_data():
    """Load FAQ data from files or use embedded data."""
    # Try to load from processed files first
    csv_path = os.path.join("processed_data", "dld_faq_data.csv")
    json_path = os.path.join("processed_data", "dld_faq_data.json")
    
    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)
            st.sidebar.success(f"✅ Loaded {len(df)} FAQ items from CSV")
            
            # Get unique services as topics
            if 'Service' in df.columns:
                services = sorted(df["Service"].unique().tolist())
                topics = ["All Topics"] + services
            else:
                topics = ["All Topics"]
            
            return df, topics
            
        except Exception as e:
            st.sidebar.error(f"Error loading CSV: {str(e)}")
    
    elif os.path.exists(json_path):
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                faq_data = json.load(f)
                
            df = pd.DataFrame(faq_data)
            st.sidebar.success(f"✅ Loaded {len(df)} FAQ items from JSON")
            
            # Get unique services as topics
            if 'Service' in df.columns:
                services = sorted(df["Service"].unique().tolist())
                topics = ["All Topics"] + services
            else:
                topics = ["All Topics"]
            
            return df, topics
            
        except Exception as e:
            st.sidebar.error(f"Error loading JSON: {str(e)}")
    
    # Fall back to embedded sample data
    st.sidebar.warning("⚠️ Using embedded sample data (processed files not found)")
    df = pd.DataFrame(SAMPLE_FAQ_DATA)
    services = sorted(df["Service"].unique().tolist())
    topics = ["All Topics"] + services
    
    return df, topics

# Improved search function
def search_faqs_improved(query, df, topic=None, top_k=5):
    """Enhanced search function that combines text matching with OpenAI semantic search."""
    if df.empty or not query:
        return []
    
    # Filter by topic if specified
    if topic and topic != "All Topics":
        if 'Service' in df.columns:
            filtered_df = df[df['Service'] == topic]
            if len(filtered_df) == 0:
                filtered_df = df
        else:
            filtered_df = df
    else:
        filtered_df = df
    
    # Try OpenAI semantic search first
    try:
        # Create a list of questions for OpenAI to compare against
        max_questions = min(50, len(filtered_df))  # Reduced for cloud efficiency
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
        
        # Using CONSISTENT older OpenAI API syntax
        response = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"User query: {query}\n\nFAQ Questions:\n{questions_text}"}
            ],
            temperature=0.2
        )
        
        result = response.choices[0].message.content.strip()
        
        # Get semantic matches
        semantic_indices = []
        if result != "NONE":
            question_numbers = [int(num.strip()) for num in re.findall(r'\d+', result)]
            for num in question_numbers:
                if 1 <= num <= len(questions_with_indices):
                    semantic_indices.append(questions_with_indices[num-1][0])
        
        # Also do text-based search as backup
        query_lower = query.lower().strip()
        query_words = set(query_lower.split())
        
        scores = []
        for idx, row in filtered_df.iterrows():
            question = row['Question (English)'].lower()
            question_words = set(question.split())
            
            score = 0
            if query_lower in question:
                score += 20
            
            common_words = query_words.intersection(question_words)
            score += len(common_words) * 8
            
            # Boost for registration-related terms
            if any(word in ['register', 'registration'] for word in query_words) and any(word in ['register', 'registration'] for word in question_words):
                score += 10
            
            scores.append((score, idx))
        
        scores.sort(reverse=True)
        text_indices = [idx for score, idx in scores[:top_k] if score > 0]
        
        # Combine semantic and text results
        combined_indices = []
        for idx in semantic_indices:
            if idx not in combined_indices:
                combined_indices.append(idx)
                
        for idx in text_indices:
            if idx not in combined_indices and len(combined_indices) < top_k:
                combined_indices.append(idx)
        
        # Prepare results
        results = []
        for i, idx in enumerate(combined_indices):
            if 0 <= idx < len(filtered_df):
                row = filtered_df.iloc[idx]
                results.append({
                    'question': row['Question (English)'],
                    'answer': row['Answer (English)'],
                    'service': row['Service'] if 'Service' in row else 'Unknown',
                    'module': row['Module'] if 'Module' in row and not pd.isna(row['Module']) else "",
                    'relevance': 1.0 - (0.1 * i),
                    'debug_info': {
                        'match_type': "Semantic" if idx in semantic_indices else "Text",
                        'idx': idx,
                        'rank': i+1
                    }
                })
        
        return results
        
    except Exception as e:
        st.error(f"Error in search: {str(e)}")
        return []

# Function to generate response
def generate_response(query, relevant_faqs, language):
    """Generate a response based on the relevant FAQs."""
    if not relevant_faqs:
        if language == "arabic":
            return "عذراً، لم أتمكن من العثور على إجابة لسؤالك. هل يمكنك إعادة صياغة السؤال أو طرح سؤال آخر؟"
        else:
            return "I'm sorry, I couldn't find an answer to your question. Could you rephrase your question or ask something else?"
    
    # For a single, highly relevant FAQ, return directly
    if len(relevant_faqs) == 1 and relevant_faqs[0]['relevance'] > 0.9:
        answer = relevant_faqs[0]['answer']
        if language == "arabic":
            try:
                answer = translate_text(answer, "arabic")
            except:
                pass
        return answer
    
    # Prepare context from multiple FAQs
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
        
        # Using CONSISTENT older OpenAI API syntax
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

# Function to create source explanation
def create_source_explanation(query, relevant_faqs, language="english"):
    """Create an explanation of the sources used to answer the query."""
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
        
        st.markdown('<div class="sidebar-header">System Status</div>', unsafe_allow_html=True)
        if not st.session_state.faq_data.empty:
            st.info(f"📚 {len(st.session_state.faq_data)} FAQ items across {len(st.session_state.topics)-1} topics")
        else:
            st.error("❌ No FAQ data loaded")
        
        st.markdown("---")
        if st.button("🔄 Reset Conversation", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
    
    # Main content
    st.markdown('<h1 class="main-title">DLD FAQ Assistant</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">Ask me anything about Dubai Land Department services</p>', unsafe_allow_html=True)
    
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
    if st.session_state.faq_data.empty:
        st.warning("No FAQ data available.")
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
            
            # Search for relevant FAQs
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
                "debug_info": debug_info
            })
            
            st.rerun()

if __name__ == "__main__":
    main()
