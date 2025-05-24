import os
import argparse
import logging
from data_processor_header_format import DataProcessor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def setup_environment():
    """Create necessary directories and check environment."""
    # Check for .env file
    if not os.path.exists('.env'):
        logging.warning("No .env file found. Creating template...")
        with open('.env', 'w') as f:
            f.write("OPENAI_API_KEY=your_api_key_here\n")
        logging.info("Created .env template. Please add your OpenAI API key.")
        return False
    
    # Create data directory if it doesn't exist
    if not os.path.exists('data'):
        os.makedirs('data')
        logging.info("Created 'data' directory. Please place your Excel files here.")
        return False
    
    # Check if data directory has Excel files
    excel_files = [f for f in os.listdir('data') if f.endswith('.xlsx')]
    if not excel_files:
        logging.warning("No Excel files found in 'data' directory. Please add your FAQ files.")
        return False
    
    # Create processed_data directory
    if not os.path.exists('processed_data'):
        os.makedirs('processed_data')
    
    return True

def process_data(num_clusters=12):
    """Process FAQ data files."""
    logging.info("Starting data processing...")
    processor = DataProcessor(data_dir="data", num_clusters=num_clusters)
    processor.load_excel_files()
    processor.preprocess_data()
    processor.create_clusters()
    processor.save_processed_data(output_dir="processed_data")
    logging.info("Data processing complete!")

def start_app():
    """Start the Streamlit application."""
    logging.info("Starting Streamlit application...")
    os.system("streamlit run app.py")  # Fixed: removed S/ prefix

if __name__ == "__main__":  # Fixed: changed **name** to __name__
    parser = argparse.ArgumentParser(description="DLD FAQ Chatbot")
    parser.add_argument('--process', action='store_true', help='Process data files')
    parser.add_argument('--run', action='store_true', help='Run the Streamlit app')
    parser.add_argument('--clusters', type=int, default=12, help='Number of clusters for topic modeling')
    
    args = parser.parse_args()
    
    env_ready = setup_environment()
    
    if args.process:
        if env_ready:
            process_data(num_clusters=args.clusters)
        else:
            logging.error("Environment not ready. Please fix the issues mentioned above.")
    
    if args.run:
        if os.path.exists('processed_data/processed_faq_data.csv'):
            start_app()
        else:
            logging.error("Processed data not found. Run with --process flag first.")
    
    if not args.process and not args.run:
        print("Usage:")
        print("  python main.py --process  # Process the Excel files")
        print("  python main.py --run      # Run the Streamlit app")
        print("  python main.py --process --run  # Process files and run the app")
