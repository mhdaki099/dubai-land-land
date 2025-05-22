import os
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import faiss
import pickle
import re
from typing import List, Dict, Tuple, Any
import logging
import openai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure OpenAI API
openai.api_key = os.getenv("OPENAI_API_KEY")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DataProcessor:
    def __init__(self, data_dir: str, num_clusters: int = 12):
        """Initialize the data processor with data directory and number of clusters."""
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
        
    def load_excel_files(self) -> pd.DataFrame:
        """Load and combine all Excel files in the data directory, handling the specific format."""
        all_dataframes = []
        
        for filename in os.listdir(self.data_dir):
            if filename.endswith('.xlsx'):
                file_path = os.path.join(self.data_dir, filename)
                try:
                    # Extract service name from filename
                    service_name = filename.replace('FAQs.xlsx', '').strip()
                    
                    logging.info(f"Processing {file_path}")
                    
                    # Read the Excel file - skipping the first row which contains the title
                    # The second row contains the column headers
                    df = pd.read_excel(file_path, header=1)
                    
                    # Clean up the DataFrame
                    df = df.dropna(how='all')  # Drop rows where all columns are NaN
                    
                    # Look for the expected column names
                    expected_cols = ['Module', 'Question (English)', 'Answer (English)', 
                                    'Question (Arabic)', 'Answer (Arabic)', 'Keywords']
                    
                    # Check if columns exist or need renaming
                    col_mapping = {}
                    
                    # Find if columns exist or need mapping
                    for col in df.columns:
                        # Find column matches based on keywords
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
                    
                    # If there's a column with '#' or numbered index, map it to an index column
                    for col in df.columns:
                        if col == '#' or str(col).lower() == 'id' or str(col).lower() == 'index':
                            col_mapping[col] = 'Index'
                    
                    # Apply column mapping if needed
                    if col_mapping:
                        df = df.rename(columns=col_mapping)
                    
                    # Make sure we have the essential columns
                    if 'Question (English)' not in df.columns:
                        # If module column exists, look for cell values that might be questions
                        if 'Module' in df.columns:
                            for i, col in enumerate(df.columns):
                                if col not in col_mapping.values():
                                    # First unmapped column after Module might be questions
                                    if i > list(df.columns).index('Module'):
                                        col_mapping[col] = 'Question (English)'
                                        break
                        # Otherwise, assume first unmapped column is questions
                        else:
                            for col in df.columns:
                                if col not in col_mapping.values():
                                    col_mapping[col] = 'Question (English)'
                                    break
                    
                    if 'Answer (English)' not in df.columns:
                        # Find first unmapped column after Question (English)
                        if 'Question (English)' in df.columns:
                            found_question = False
                            for col in df.columns:
                                if col == 'Question (English)':
                                    found_question = True
                                elif found_question and col not in col_mapping.values():
                                    col_mapping[col] = 'Answer (English)'
                                    break
                    
                    # Apply any additional column mapping
                    if col_mapping:
                        df = df.rename(columns=col_mapping)
                    
                    # Ensure we have the minimum required columns
                    if 'Question (English)' in df.columns and 'Answer (English)' in df.columns:
                        # Add service name column
                        df['Service'] = service_name
                        
                        # If no Module column, add it with default service name
                        if 'Module' not in df.columns:
                            df['Module'] = service_name
                            
                        # Filter out rows without questions or answers
                        df = df[df['Question (English)'].notna() & df['Answer (English)'].notna()]
                        
                        # Add missing columns with empty values if needed
                        for col in expected_cols:
                            if col not in df.columns:
                                df[col] = ''
                        
                        all_dataframes.append(df)
                        logging.info(f"Processed {len(df)} FAQ items from {filename}")
                    else:
                        logging.error(f"Required columns not found in {filename} after mapping")
                        
                except Exception as e:
                    logging.error(f"Error processing {filename}: {str(e)}")
        
        if not all_dataframes:
            raise ValueError("No valid data found in Excel files")
            
        self.df_combined = pd.concat(all_dataframes, ignore_index=True)
        logging.info(f"Loaded {len(self.df_combined)} FAQ items from {len(all_dataframes)} files")
        
        return self.df_combined
    
    def clean_text(self, text: str) -> str:
        """Clean and preprocess text data."""
        if not isinstance(text, str):
            return ""
        
        # Remove special characters and extra whitespace
        text = re.sub(r'[^\w\s\.\?\!]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def preprocess_data(self) -> pd.DataFrame:
        """Preprocess the combined dataframe."""
        if self.df_combined is None:
            self.load_excel_files()
        
        # Clean the questions and answers
        self.df_combined['Question_Clean'] = self.df_combined['Question (English)'].apply(self.clean_text)
        self.df_combined['Answer_Clean'] = self.df_combined['Answer (English)'].apply(self.clean_text)
        
        # Fill missing values for Arabic if needed
        if 'Question (Arabic)' in self.df_combined.columns:
            self.df_combined['Question (Arabic)'].fillna('', inplace=True)
        else:
            self.df_combined['Question (Arabic)'] = ''
            
        if 'Answer (Arabic)' in self.df_combined.columns:
            self.df_combined['Answer (Arabic)'].fillna('', inplace=True)
        else:
            self.df_combined['Answer (Arabic)'] = ''
            
        # Remove duplicate questions
        self.df_combined.drop_duplicates(subset=['Question_Clean'], keep='first', inplace=True)
        
        # Create combined text for embedding (question + keywords if available)
        self.df_combined['Embed_Text'] = self.df_combined['Question_Clean']
        if 'Keywords' in self.df_combined.columns:
            self.df_combined['Embed_Text'] += ' ' + self.df_combined['Keywords'].fillna('')
            
        logging.info(f"Preprocessed data: {len(self.df_combined)} unique FAQ items")
        return self.df_combined
    
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
            # Return a zero vector as fallback (1536 is the dimension for text-embedding-3-small)
            return [0.0] * 1536
    
    def get_embeddings_batch(self, texts: List[str], batch_size: int = 100) -> np.ndarray:
        """Get embeddings for multiple texts in batches."""
        all_embeddings = []
        
        # Process in batches to avoid rate limits
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            logging.info(f"Processing embedding batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}")
            
            try:
                # Using the older OpenAI API syntax
                response = openai.Embedding.create(
                    model="text-embedding-3-small",
                    input=batch_texts
                )
                batch_embeddings = [item["embedding"] for item in response["data"]]
                all_embeddings.extend(batch_embeddings)
            except Exception as e:
                logging.error(f"Error in batch {i//batch_size + 1}: {str(e)}")
                # Add zero vectors as fallback
                for _ in range(len(batch_texts)):
                    all_embeddings.append([0.0] * 1536)
        
        return np.array(all_embeddings).astype('float32')
    
    def create_clusters(self) -> pd.DataFrame:
        """Create topic clusters from the questions."""
        if 'Embed_Text' not in self.df_combined.columns:
            self.preprocess_data()
            
        # Create embeddings for questions
        logging.info("Generating embeddings for clustering...")
        embeddings = self.get_embeddings_batch(self.df_combined['Embed_Text'].tolist())
        
        # Perform K-means clustering
        logging.info(f"Performing K-means clustering with {self.num_clusters} clusters...")
        kmeans = KMeans(n_clusters=self.num_clusters, random_state=42, n_init=10)
        self.df_combined['Cluster'] = kmeans.fit_predict(embeddings)
        
        # Map cluster IDs to names
        self.df_combined['Cluster_Name'] = self.df_combined['Cluster'].map(self.cluster_names)
        
        # Count questions per cluster
        cluster_counts = self.df_combined['Cluster_Name'].value_counts()
        logging.info("Cluster distribution:")
        for cluster, count in cluster_counts.items():
            logging.info(f"  {cluster}: {count} questions")
            
        return self.df_combined
    
    def create_faiss_index(self) -> Tuple[Any, np.ndarray]:
        """Create a FAISS index for fast similarity search."""
        if 'Cluster' not in self.df_combined.columns:
            self.create_clusters()
            
        logging.info("Creating FAISS index for similarity search...")
        # Generate embeddings for all questions
        texts = self.df_combined['Embed_Text'].tolist()
        embeddings = self.get_embeddings_batch(texts)
        
        # Create FAISS index
        dimension = embeddings.shape[1]  # Size of each embedding
        index = faiss.IndexFlatL2(dimension)
        
        # Add vectors to the index
        index.add(embeddings)
        
        logging.info(f"Created FAISS index with {len(texts)} vectors of dimension {dimension}")
        return index, embeddings
    
    def save_processed_data(self, output_dir: str) -> None:
        """Save processed data and models for later use."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save the dataframe
        self.df_combined.to_csv(os.path.join(output_dir, 'processed_faq_data.csv'), index=False)
        
        # Create and save FAISS index
        index, embeddings = self.create_faiss_index()
        faiss.write_index(index, os.path.join(output_dir, 'faiss_index.bin'))
        np.save(os.path.join(output_dir, 'embeddings.npy'), embeddings)
        
        # Save the embedding dimension
        with open(os.path.join(output_dir, 'embedding_dimension.txt'), 'w') as f:
            f.write(str(embeddings.shape[1]))
        
        # Save cluster mappings
        with open(os.path.join(output_dir, 'cluster_names.pkl'), 'wb') as f:
            pickle.dump(self.cluster_names, f)
            
        logging.info(f"Saved processed data and models to {output_dir}")

if __name__ == "__main__":
    # Example usage
    processor = DataProcessor(data_dir="data")
    processor.load_excel_files()
    processor.preprocess_data()
    processor.create_clusters()
    processor.save_processed_data(output_dir="processed_data")