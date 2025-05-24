import os
from data_processor_header_format import DataProcessor
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def process_data():
    """Process FAQ data files with the specialized header format."""
    logging.info("Starting data processing...")
    
    # Check if data directory exists
    if not os.path.exists("data"):
        logging.error("Data directory not found. Please create a 'data' directory and add your Excel files.")
        return False
        
    # Check if there are Excel files in the data directory
    excel_files = [f for f in os.listdir("data") if f.endswith('.xlsx')]
    if not excel_files:
        logging.error("No Excel files found in 'data' directory. Please add your FAQ files.")
        return False
    
    # Create processed_data directory if it doesn't exist
    if not os.path.exists("processed_data"):
        os.makedirs("processed_data")
    
    # Process the data
    try:
        processor = DataProcessor(data_dir="data")
        processor.load_excel_files()
        processor.preprocess_data()
        processor.create_clusters()
        processor.save_processed_data(output_dir="processed_data")
        logging.info("Data processing complete!")
        return True
    except Exception as e:
        logging.error(f"Error processing data: {str(e)}")
        return False

if __name__ == "__main__":  # Fixed: changed **name** to __name__
    process_data()
