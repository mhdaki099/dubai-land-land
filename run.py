import os
import sys
import logging
from dld_faq_processor import DLDFAQProcessor

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def check_environment():
    """Check if the environment is properly set up."""
    # Check for .env file with OpenAI API key
    if not os.path.exists('.env'):
        logging.warning("No .env file found. Creating template...")
        with open('.env', 'w') as f:
            f.write("OPENAI_API_KEY=your_api_key_here\n")
        logging.info("Created .env template. Please add your OpenAI API key.")
        return False
    
    # Check if data directory exists
    if not os.path.exists('data'):
        logging.error("Data directory not found. Please create a 'data' directory with your Excel files.")
        return False
    
    # Check if Excel files exist in data directory
    excel_files = [f for f in os.listdir('data') if f.endswith('.xlsx')]
    if not excel_files:
        logging.error("No Excel files found in 'data' directory. Please add your FAQ files.")
        return False
    
    # Create processed_data directory if it doesn't exist
    if not os.path.exists('processed_data'):
        os.makedirs('processed_data')
    
    return True

def process_data():
    """Process the Excel files and create a dataset."""
    logging.info("Starting data processing...")
    
    try:
        processor = DLDFAQProcessor()
        success = processor.process_all()
        
        if success:
            logging.info("Data processing complete!")
            return True
        else:
            logging.error("Data processing failed.")
            return False
    except Exception as e:
        logging.error(f"Error during data processing: {str(e)}")
        return False

def run_app():
    """Run the Streamlit application."""
    if not os.path.exists('processed_data/search_data.json'):
        logging.error("Processed data not found. Please run with --process first.")
        return False
    
    logging.info("Starting Streamlit application...")
    os.system("streamlit run S/dld_chatbot_app.py")
    return True

def print_usage():
    """Print usage instructions."""
    print("\nUsage:")
    print("  python run.py --process  # Process Excel files")
    print("  python run.py --run      # Run the chatbot")
    print("  python run.py --all      # Process files and run the chatbot")
    print("\nOptions:")
    print("  --process     Process Excel files and create dataset")
    print("  --run         Run the Streamlit chatbot application")
    print("  --all         Process files and run the chatbot")
    print("  --help        Show this help message and exit")

if __name__ == "__main__":

    process_data()
    run_app()
    # Parse command line arguments
    if len(sys.argv) < 2 or "--help" in sys.argv:
        print_usage()
        sys.exit(0)


    
    # Check environment
    if not check_environment():
        print_usage()
        sys.exit(1)
    
    # Process data if requested
    if "--process" in sys.argv or "--all" in sys.argv:
        if not process_data():
            sys.exit(1)
    
    # Run app if requested
    if "--run" in sys.argv or "--all" in sys.argv:
        run_app()