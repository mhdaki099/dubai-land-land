import os
from typing import List, Dict, Any
import pandas as pd
import numpy as np
import faiss
from dotenv import load_dotenv
import pickle
from crewai import Agent, Task, Crew, Process
from langchain_openai import ChatOpenAI
import logging
from langchain.tools import Tool
import re
import openai

# Load environment variables
load_dotenv()

# Configure OpenAI API
openai.api_key = os.getenv("OPENAI_API_KEY")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class FAQAgentSystem:
    def __init__(self, data_path: str = "processed_data"):
        """Initialize the FAQ Agent System."""
        self.data_path = data_path
        self.llm = ChatOpenAI(model="gpt-4o")  # Use GPT-4o
        
        # Load processed data
        self.load_data()
        
        # Create agents
        self.setup_agents()
        
    def load_data(self) -> None:
        """Load processed data and models."""
        try:
            # Load the dataframe
            self.df = pd.read_csv(os.path.join(self.data_path, 'processed_faq_data.csv'))
            
            # Load FAISS index and embeddings
            self.index = faiss.read_index(os.path.join(self.data_path, 'faiss_index.bin'))
            self.embeddings = np.load(os.path.join(self.data_path, 'embeddings.npy'))
            
            # Load embedding dimension
            with open(os.path.join(self.data_path, 'embedding_dimension.txt'), 'r') as f:
                self.embedding_dimension = int(f.read().strip())
            
            # Load cluster mappings
            with open(os.path.join(self.data_path, 'cluster_names.pkl'), 'rb') as f:
                self.cluster_names = pickle.load(f)
                
            logging.info("Successfully loaded processed data and models")
        except Exception as e:
            logging.error(f"Error loading data: {str(e)}")
            raise
            
    def detect_language(self, text: str) -> str:
        """Detect if text is in Arabic or English."""
        # Simple detection based on character ranges
        arabic_pattern = re.compile(r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]+')
        if arabic_pattern.search(text):
            return "arabic"
        return "english"
    
    def get_embedding(self, text: str) -> List[float]:
        """Get embedding for a text using OpenAI's embeddings API."""
        try:
            # Using the older OpenAI API syntax
            response = openai.Embedding.create(
                model="text-embedding-3-small",
                input=text
            )
            return response["data"][0]["embedding"]
        except Exception as e:
            logging.error(f"Error getting embedding: {str(e)}")
            # Return a zero vector as fallback
            return [0.0] * self.embedding_dimension
        
    def search_faqs(self, query: str, cluster: str = None, top_k: int = 5) -> List[Dict]:
        """Search the FAQ database for relevant answers."""
        # Generate embedding for the query
        query_embedding = np.array(self.get_embedding(query)).reshape(1, -1).astype('float32')
        
        # Search the FAISS index
        distances, indices = self.index.search(query_embedding, top_k * 3)  # Get more results to filter by cluster
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.df):
                result = {
                    'question_en': self.df.iloc[idx]['Question (English)'],
                    'answer_en': self.df.iloc[idx]['Answer (English)'],
                    'question_ar': self.df.iloc[idx]['Question (Arabic)'],
                    'answer_ar': self.df.iloc[idx]['Answer (Arabic)'],
                    'cluster': self.df.iloc[idx]['Cluster_Name'],
                    'score': 1 - (distances[0][i] / 10)  # Normalize similarity score
                }
                
                # Filter by cluster if specified
                if cluster and result['cluster'] != cluster:
                    continue
                    
                results.append(result)
                
                # Break once we have enough results
                if len(results) >= top_k:
                    break
                    
        return results
        
    def setup_agents(self) -> None:
        """Set up the CrewAI agents for the chatbot."""
        # Create a tool for FAQ search
        search_tool = Tool(
            name="search_faqs",
            func=self.search_faqs,
            description="Search for relevant FAQ items. Input should be a query string."
        )
        
        # Create language detection tool
        language_tool = Tool(
            name="detect_language",
            func=self.detect_language,
            description="Detect if text is in Arabic or English."
        )
        
        # Query Understanding Agent
        self.query_agent = Agent(
            role="Query Analyzer",
            goal="Understand user queries and route them to appropriate clusters",
            backstory="I am an expert at understanding user intent and categorizing questions.",
            verbose=True,
            allow_delegation=True,
            tools=[language_tool],
            llm=self.llm
        )
        
        # Knowledge Retrieval Agent
        self.retrieval_agent = Agent(
            role="Knowledge Expert",
            goal="Find the most relevant information from the FAQ database",
            backstory="I am a specialist in retrieving accurate information from the knowledge base.",
            verbose=True,
            allow_delegation=True,
            tools=[search_tool],
            llm=self.llm
        )
        
        # Response Generation Agent
        self.response_agent = Agent(
            role="Response Crafter",
            goal="Create helpful, accurate, and natural-sounding responses",
            backstory="I am skilled at crafting clear and concise responses based on retrieved information.",
            verbose=True,
            allow_delegation=False,
            llm=self.llm
        )
        
        # Translation Agent
        self.translation_agent = Agent(
            role="Language Translator",
            goal="Accurately translate content between English and Arabic",
            backstory="I am fluent in both English and Arabic and ensure high-quality translations.",
            verbose=True,
            allow_delegation=False,
            tools=[language_tool],
            llm=self.llm
        )
        
    def get_available_clusters(self) -> List[str]:
        """Get list of available topic clusters."""
        return list(self.cluster_names.values())
        
    def translate_text(self, text: str, target_language: str) -> str:
        """Translate text using OpenAI's API."""
        if target_language.lower() == "arabic":
            system_prompt = "You are a professional translator from English to Arabic. Translate the following text to Arabic, maintaining the same meaning, tone, and level of formality."
        else:
            system_prompt = "You are a professional translator from Arabic to English. Translate the following text to English, maintaining the same meaning, tone, and level of formality."
            
        try:
            # Using the older OpenAI API syntax
            response = openai.ChatCompletion.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": text}
                ]
            )
            return response["choices"][0]["message"]["content"]
        except Exception as e:
            logging.error(f"Translation error: {str(e)}")
            return text  # Return original text if translation fails
        
    def process_query(self, query: str, selected_cluster: str = None) -> Dict:
        """Process a user query through the agent system."""
        # Detect language
        language = self.detect_language(query)
        logging.info(f"Detected language: {language}")
        
        # Translate query if needed
        working_query = query
        if language == "arabic":
            logging.info("Translating query from Arabic to English")
            working_query = self.translate_text(query, "english")
            logging.info(f"Translated query: {working_query}")
        
        # Determine cluster if not provided
        if not selected_cluster or selected_cluster == "All Topics":
            logging.info("No cluster selected, determining best cluster")
            
            # Create a system prompt for cluster determination
            system_prompt = f"""
            You are an expert at categorizing questions into predefined topics.
            Analyze the query and determine which topic cluster it belongs to.
            
            Available clusters:
            {', '.join(self.get_available_clusters())}
            
            Return ONLY the name of the single most relevant cluster, nothing else.
            """
            
            try:
                # Using the older OpenAI API syntax
                response = openai.ChatCompletion.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": working_query}
                    ],
                    temperature=0.3
                )
                selected_cluster = response["choices"][0]["message"]["content"].strip()
                logging.info(f"Determined cluster: {selected_cluster}")
                
                # Check if determined cluster is in our list
                if selected_cluster not in self.get_available_clusters():
                    logging.warning(f"Determined cluster '{selected_cluster}' not in available clusters")
                    selected_cluster = None  # Use all clusters if determined one is invalid
            except Exception as e:
                logging.error(f"Error determining cluster: {str(e)}")
                selected_cluster = None  # Use all clusters if there's an error
        
        # Retrieve relevant FAQ items
        logging.info(f"Searching FAQs for query: '{working_query}' in cluster: {selected_cluster}")
        faq_results = self.search_faqs(working_query, selected_cluster)
        
        if not faq_results:
            logging.warning("No FAQ items found")
            
            if language == "arabic":
                response = "عذراً، لم أتمكن من العثور على إجابة لسؤالك. هل يمكنك إعادة صياغة السؤال أو طرح سؤال آخر؟"
            else:
                response = "I'm sorry, I couldn't find an answer to your question. Could you rephrase your question or ask something else?"
                
            return {
                "query": query,
                "detected_language": language,
                "selected_cluster": selected_cluster,
                "response": response
            }
        
        # Prepare FAQ results for response generation
        faq_context = ""
        for i, result in enumerate(faq_results):
            faq_context += f"\nItem {i+1}:\n"
            faq_context += f"Question: {result['question_en']}\n"
            faq_context += f"Answer: {result['answer_en']}\n"
            faq_context += f"Topic: {result['cluster']}\n"
            faq_context += f"Relevance Score: {result['score']:.2f}\n"
        
        # Generate response
        logging.info("Generating response")
        
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
        
        try:
            # Using the older OpenAI API syntax
            response = openai.ChatCompletion.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"User query: {working_query}\n\nRelevant FAQ information:{faq_context}"}
                ]
            )
            
            final_response = response["choices"][0]["message"]["content"]
            logging.info("Generated response (English)")
            
            # Translate response if needed
            if language == "arabic":
                logging.info("Translating response to Arabic")
                final_response = self.translate_text(final_response, "arabic")
                logging.info("Translated response to Arabic")
        except Exception as e:
            logging.error(f"Error generating response: {str(e)}")
            
            if language == "arabic":
                final_response = "عذراً، حدث خطأ أثناء معالجة طلبك. يرجى المحاولة مرة أخرى."
            else:
                final_response = "Sorry, there was an error processing your request. Please try again."
        
        return {
            "query": query,
            "detected_language": language,
            "selected_cluster": selected_cluster,
            "response": final_response
        }