import os
import pandas as pd
import numpy as np
import faiss
import pickle
import re
import json
import logging
import openai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure OpenAI API
openai.api_key = os.getenv("OPENAI_API_KEY")

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DLDFAQProcessor:
    """Process DLD FAQ Excel files with the specific structure identified."""
    
    def __init__(self, data_dir="data", output_dir="processed_data"):
        """Initialize the processor."""
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.all_qa_pairs = []
        
    def process_excel_files(self):
        """Process all Excel files in the data directory."""
        # Create output directory if it doesn't exist
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        if not os.path.exists(self.data_dir):
            logging.error(os.listdir())
            logging.error(f"Data directory {self.data_dir} does not exist")
            return False
        
        # Get all Excel files
        excel_files = [f for f in os.listdir(self.data_dir) if f.endswith('.xlsx')]
        
        if not excel_files:
            logging.error(f"No Excel files found in {self.data_dir}")
            return False
            
        logging.info(f"Found {len(excel_files)} Excel files to process")
        
        # Process each file
        for filename in excel_files:
            try:
                file_path = os.path.join(self.data_dir, filename)
                service_name = filename.replace('FAQs.xlsx', '').strip()
                
                logging.info(f"Processing {filename}")
                
                # Read Excel file with header at row 2 (0-based index)
                df = pd.read_excel(file_path, header=2)
                
                # Check if the file has the expected columns
                expected_cols = ['Question (English)', 'Answer (English)']
                if not all(col in df.columns for col in expected_cols):
                    logging.warning(f"File {filename} doesn't have the expected columns. Columns found: {df.columns.tolist()}")
                    continue
                
                # Process Q&A pairs
                qa_pairs = []
                current_module = service_name
                current_question = None
                current_answer = ""
                
                for idx, row in df.iterrows():
                    # Skip completely empty rows
                    if row.isna().all():
                        if current_question and current_answer:
                            # Add the completed Q&A pair
                            qa_pairs.append({
                                'Module': current_module,
                                'Question (English)': current_question,
                                'Answer (English)': current_answer.strip(),
                                'Service': service_name
                            })
                            # Reset for next pair
                            current_question = None
                            current_answer = ""
                        continue
                    
                    # Check for module
                    if not pd.isna(row['Module']):
                        current_module = row['Module']
                    
                    # Check for question - if we already have a question, this is a new pair
                    if not pd.isna(row['Question (English)']) and row['Question (English)'].strip():
                        # If we already have a question-answer pair, save it
                        if current_question and current_answer:
                            qa_pairs.append({
                                'Module': current_module,
                                'Question (English)': current_question,
                                'Answer (English)': current_answer.strip(),
                                'Service': service_name
                            })
                        
                        # Start new Q&A pair
                        current_question = row['Question (English)'].strip()
                        current_answer = "" if pd.isna(row['Answer (English)']) else row['Answer (English)'].strip()
                    # Append to the current answer if there's no new question but there is answer text
                    elif current_question and not pd.isna(row['Answer (English)']) and row['Answer (English)'].strip():
                        if current_answer:
                            current_answer += " " + row['Answer (English)'].strip()
                        else:
                            current_answer = row['Answer (English)'].strip()
                
                # Add the last Q&A pair if pending
                if current_question and current_answer:
                    qa_pairs.append({
                        'Module': current_module,
                        'Question (English)': current_question,
                        'Answer (English)': current_answer.strip(),
                        'Service': service_name
                    })
                
                # Add Arabic if available
                for qa_pair in qa_pairs:
                    # Find the corresponding row with the same question
                    question_rows = df[df['Question (English)'] == qa_pair['Question (English)']]
                    if not question_rows.empty:
                        # Get the first matching row
                        row = question_rows.iloc[0]
                        # Add Arabic question and answer if available
                        if 'Question (Arabic)' in df.columns and not pd.isna(row['Question (Arabic)']):
                            qa_pair['Question (Arabic)'] = row['Question (Arabic)']
                        else:
                            qa_pair['Question (Arabic)'] = ""
                            
                        if 'Answer (Arabic)' in df.columns and not pd.isna(row['Answer (Arabic)']):
                            qa_pair['Answer (Arabic)'] = row['Answer (Arabic)']
                        else:
                            qa_pair['Answer (Arabic)'] = ""
                            
                        # Add keywords if available
                        if 'Keywords' in df.columns and not pd.isna(row['Keywords']):
                            qa_pair['Keywords'] = row['Keywords']
                        else:
                            qa_pair['Keywords'] = ""
                
                # Extend the overall collection
                self.all_qa_pairs.extend(qa_pairs)
                
                logging.info(f"Extracted {len(qa_pairs)} Q&A pairs from {filename}")
                
            except Exception as e:
                logging.error(f"Error processing {filename}: {str(e)}")
        
        # Save the results
        if self.all_qa_pairs:
            self.save_data()
            logging.info(f"Processed a total of {len(self.all_qa_pairs)} Q&A pairs")
            return True
        else:
            logging.error("No Q&A pairs were extracted")
            return False
    
    def save_data(self):
        """Save the extracted Q&A pairs to files."""
        # Create DataFrame
        df = pd.DataFrame(self.all_qa_pairs)
        
        # Save as CSV
        csv_path = os.path.join(self.output_dir, 'dld_faq_data.csv')
        df.to_csv(csv_path, index=False)
        
        # Save as JSON for easier reading
        json_path = os.path.join(self.output_dir, 'dld_faq_data.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(self.all_qa_pairs, f, ensure_ascii=False, indent=2)
        
        # Create search data format
        search_data = {
            "questions": [item["Question (English)"] for item in self.all_qa_pairs],
            "answers": [item["Answer (English)"] for item in self.all_qa_pairs],
            "services": [item["Service"] for item in self.all_qa_pairs],
            "modules": [item["Module"] for item in self.all_qa_pairs]
        }
        
        # Save search data
        search_path = os.path.join(self.output_dir, 'search_data.json')
        with open(search_path, 'w', encoding='utf-8') as f:
            json.dump(search_data, f, ensure_ascii=False, indent=2)
            
        logging.info(f"Saved data to {csv_path}, {json_path}, and {search_path}")
    
    def create_embeddings(self):
        """Create embeddings for the extracted Q&A pairs."""
        if not self.all_qa_pairs:
            logging.error("No Q&A pairs to create embeddings for")
            return False
        
        try:
            logging.info("Creating embeddings for Q&A pairs...")
            
            # Prepare texts for embedding
            texts = []
            for qa_pair in self.all_qa_pairs:
                # Combine question and keywords if available
                text = qa_pair["Question (English)"]
                if "Keywords" in qa_pair and qa_pair["Keywords"]:
                    text += " " + qa_pair["Keywords"]
                texts.append(text)
            
            # Generate embeddings in batches
            batch_size = 100
            all_embeddings = []
            
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
            
            # Convert to numpy array
            embeddings = np.array(all_embeddings).astype('float32')
            
            # Create FAISS index
            dimension = embeddings.shape[1]
            index = faiss.IndexFlatL2(dimension)
            index.add(embeddings)
            
            # Save embeddings and index
            embedding_path = os.path.join(self.output_dir, 'embeddings.npy')
            np.save(embedding_path, embeddings)
            
            index_path = os.path.join(self.output_dir, 'faiss_index.bin')
            faiss.write_index(index, index_path)
            
            # Save embedding dimension
            with open(os.path.join(self.output_dir, 'embedding_dimension.txt'), 'w') as f:
                f.write(str(dimension))
                
            logging.info(f"Created and saved embeddings for {len(texts)} Q&A pairs")
            return True
            
        except Exception as e:
            logging.error(f"Error creating embeddings: {str(e)}")
            return False
    
    def process_all(self):
        """Process all Excel files and create embeddings."""
        success = self.process_excel_files()
        if success:
            self.create_embeddings()
            return True
        return False

if __name__ == "__main__":
    processor = DLDFAQProcessor()
    processor.process_all()
