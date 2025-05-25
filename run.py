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
    
    logging.info(f"Found {len(excel_files)} Excel files: {excel_files}")
    
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
            # Check what files were created
            if os.path.exists('processed_data'):
                files = os.listdir('processed_data')
                logging.info(f"Created files: {files}")
            return True
        else:
            logging.error("Data processing failed.")
            return False
    except Exception as e:
        logging.error(f"Error during data processing: {str(e)}")
        return False

def run_app():
    """Run the Streamlit application."""
    # Check for required data files
    required_files = [
        'processed_data/dld_faq_data.csv',
        'processed_data/dld_faq_data.json'
    ]
    
    missing_files = [f for f in required_files if not os.path.exists(f)]
    if missing_files:
        logging.error(f"Required data files not found: {missing_files}")
        logging.error("Please run with --process first.")
        return False
    
    logging.info("Starting Streamlit application...")
    # Use the correct app file - change this to match your actual app file
    if os.path.exists('app.py'):
        os.system("streamlit run app.py")
    elif os.path.exists('dld_chatbot_app.py'):
        os.system("streamlit run dld_chatbot_app.py")
    else:
        logging.error("No Streamlit app file found (app.py or dld_chatbot_app.py)")
        return False
    return True

def print_usage():
    """Print usage instructions."""
    print("\nDLD FAQ Chatbot - Usage Instructions")
    print("=" * 50)
    print("\nUsage:")
    print("  python run.py --process  # Process Excel files")
    print("  python run.py --run      # Run the chatbot")
    print("  python run.py --all      # Process files and run the chatbot")
    print("\nOptions:")
    print("  --process     Process Excel files and create dataset")
    print("  --run         Run the Streamlit chatbot application")
    print("  --all         Process files and run the chatbot")
    print("  --help        Show this help message and exit")
    print("\nSetup Instructions:")
    print("1. Create a 'data' folder and add your Excel FAQ files")
    print("2. Create a '.env' file with your OpenAI API key")
    print("3. Run 'python run.py --process' to process the data")
    print("4. Run 'python run.py --run' to start the chatbot")

def debug_environment():
    """Print debug information about the environment."""
    print("\nEnvironment Debug Information:")
    print("=" * 40)
    print(f"Current directory: {os.getcwd()}")
    print(f"Python executable: {sys.executable}")
    
    # Check for data directory
    if os.path.exists('data'):
        excel_files = [f for f in os.listdir('data') if f.endswith('.xlsx')]
        print(f"Data directory: Found {len(excel_files)} Excel files")
        for f in excel_files:
            print(f"  - {f}")
    else:
        print("Data directory: NOT FOUND")
    
    # Check for processed data
    if os.path.exists('processed_data'):
        processed_files = os.listdir('processed_data')
        print(f"Processed data: Found {len(processed_files)} files")
        for f in processed_files:
            print(f"  - {f}")
    else:
        print("Processed data directory: NOT FOUND")
    
    # Check for .env file
    if os.path.exists('.env'):
        print(".env file: EXISTS")
    else:
        print(".env file: NOT FOUND")
    
    # Check for app files
    app_files = ['app.py', 'dld_chatbot_app.py', 'agents.py']
    for app_file in app_files:
        if os.path.exists(app_file):
            print(f"{app_file}: EXISTS")
        else:
            print(f"{app_file}: NOT FOUND")

if __name__ == "__main__":
    # Parse command line arguments
    if len(sys.argv) < 2 or "--help" in sys.argv:
        print_usage()
        sys.exit(0)
    
    # Debug mode
    if "--debug" in sys.argv:
        debug_environment()
        sys.exit(0)
    
    # Check environment
    if not check_environment():
        print("\nEnvironment check failed. Please fix the issues above.")
        debug_environment()
        sys.exit(1)
    
    # Process data if requested
    if "--process" in sys.argv or "--all" in sys.argv:
        if not process_data():
            sys.exit(1)
    
    # Run app if requested
    if "--run" in sys.argv or "--all" in sys.argv:
        if not run_app():
            sys.exit(1)
