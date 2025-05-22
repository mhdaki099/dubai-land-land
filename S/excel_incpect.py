import pandas as pd
import os
import sys
import re
import glob
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def inspect_excel_file_raw(file_path):
    """Inspect the Excel file at a very basic level, printing raw data and skipping header interpretations."""
    try:
        print(f"\n===== RAW INSPECTION: {os.path.basename(file_path)} =====")
        
        # Read the Excel file with no header inference
        df_raw = pd.read_excel(file_path, header=None)
        
        # Print dimensions
        print(f"Shape: {df_raw.shape}")
        
        # Print first 10 rows raw
        print("\nFirst 10 rows (raw, no header interpretation):")
        for i in range(min(10, df_raw.shape[0])):
            row_values = []
            for j in range(df_raw.shape[1]):
                cell = df_raw.iloc[i, j]
                if pd.isna(cell):
                    row_values.append("NaN")
                else:
                    # Truncate long cell values
                    cell_str = str(cell)
                    if len(cell_str) > 50:
                        cell_str = cell_str[:47] + "..."
                    row_values.append(cell_str)
            print(f"Row {i}: {row_values}")
        
        # Look for potential headers by analyzing row patterns
        print("\nSearching for potential header rows...")
        header_candidates = []
        for i in range(min(10, df_raw.shape[0])):
            row = df_raw.iloc[i]
            non_null_count = row.count()  # Count non-null values
            
            # Check if the row contains words like "question", "answer", "module"
            header_keywords = ["question", "answer", "module", "english", "arabic", "keywords"]
            keyword_matches = 0
            for j in range(df_raw.shape[1]):
                cell = df_raw.iloc[i, j]
                if isinstance(cell, str):
                    cell_lower = cell.lower()
                    for keyword in header_keywords:
                        if keyword in cell_lower:
                            keyword_matches += 1
                            break
            
            print(f"Row {i}: Non-null cells: {non_null_count}/{df_raw.shape[1]}, Keyword matches: {keyword_matches}")
            
            if keyword_matches >= 2:  # If at least 2 keywords found, might be a header
                header_candidates.append(i)
                print(f"  Potential header row detected at index {i}")
        
        # Try to find potential Q&A structure
        print("\nAnalyzing structure to detect Q&A patterns...")
        for header_row in header_candidates:
            print(f"\nTrying with header at row {header_row}:")
            try:
                df = pd.read_excel(file_path, header=header_row)
                print(f"Columns after using row {header_row} as header: {df.columns.tolist()}")
                
                # Detect question-answer columns based on column names
                q_col = None
                a_col = None
                m_col = None
                
                for col in df.columns:
                    col_str = str(col).lower()
                    if 'question' in col_str and ('eng' in col_str or 'en' in col_str):
                        q_col = col
                    elif 'answer' in col_str and ('eng' in col_str or 'en' in col_str):
                        a_col = col
                    elif 'module' in col_str or 'section' in col_str:
                        m_col = col
                
                if q_col and a_col:
                    print(f"Found potential Q&A columns: Questions: '{q_col}', Answers: '{a_col}'")
                    
                    # Sample some Q&A pairs
                    print("\nSample Q&A pairs:")
                    qa_count = 0
                    for i in range(min(df.shape[0], 30)):  # Check first 30 rows
                        q_val = df[q_col].iloc[i] if not pd.isna(df[q_col].iloc[i]) else None
                        a_val = df[a_col].iloc[i] if not pd.isna(df[a_col].iloc[i]) else None
                        
                        if q_val and a_val:  # Both question and answer exist
                            qa_count += 1
                            print(f"\nQ&A Pair {qa_count}:")
                            print(f"Q: {str(q_val)[:100]}")
                            print(f"A: {str(a_val)[:100]}")
                            
                            if qa_count >= 3:  # Show just 3 samples
                                break
                    
                    print(f"\nFound {qa_count} valid Q&A pairs in the first 30 rows")
                else:
                    print("Could not identify clear Question and Answer columns from headers")
                    
                    # Try a different approach: look for patterns in the data
                    print("\nLooking for patterns in data format...")
                    
                    # Check if first column might contain questions
                    if df.shape[1] >= 2:  # Need at least 2 columns for Q&A
                        first_col = df.columns[0]
                        second_col = df.columns[1]
                        
                        # Check if the first column looks like it contains questions
                        # (questions often end with a question mark)
                        q_marks = sum([1 for cell in df[first_col] if isinstance(cell, str) and '?' in cell])
                        print(f"First column contains {q_marks} cells with question marks")
                        
                        # See if second column might contain answers
                        # (answers are often longer than questions)
                        q_lengths = [len(str(cell)) for cell in df[first_col] if not pd.isna(cell)]
                        a_lengths = [len(str(cell)) for cell in df[second_col] if not pd.isna(cell)]
                        
                        if q_lengths and a_lengths:
                            avg_q_len = sum(q_lengths) / len(q_lengths)
                            avg_a_len = sum(a_lengths) / len(a_lengths)
                            print(f"Average length - First column: {avg_q_len:.1f}, Second column: {avg_a_len:.1f}")
                            
                            if avg_a_len > avg_q_len * 1.5:  # If second column values are much longer
                                print("Second column values tend to be longer, might be answers")
                                
                                # Show some sample pairs
                                print("\nPotential Q&A pairs based on column position:")
                                qa_count = 0
                                for i in range(min(df.shape[0], 30)):
                                    q_val = df[first_col].iloc[i] if not pd.isna(df[first_col].iloc[i]) else None
                                    a_val = df[second_col].iloc[i] if not pd.isna(df[second_col].iloc[i]) else None
                                    
                                    if q_val and a_val:
                                        qa_count += 1
                                        print(f"\nPotential Pair {qa_count}:")
                                        print(f"Q?: {str(q_val)[:100]}")
                                        print(f"A?: {str(a_val)[:100]}")
                                        
                                        if qa_count >= 3:
                                            break
            except Exception as e:
                print(f"Error analyzing with header at row {header_row}: {str(e)}")
        
        # Try to look for data patterns without any header
        print("\nAnalyzing raw data patterns directly...")
        # Look for rows that might begin with a question (text ending with ?)
        question_rows = []
        for i in range(min(50, df_raw.shape[0])):
            first_cell = df_raw.iloc[i, 0]
            if isinstance(first_cell, str) and first_cell.strip() and '?' in first_cell:
                question_rows.append(i)
        
        if question_rows:
            print(f"Found {len(question_rows)} rows that might contain questions (first column, contains '?')")
            print("Sample potential questions:")
            for i in question_rows[:3]:  # Show just 3 samples
                print(f"Row {i}: {str(df_raw.iloc[i, 0])[:100]}")
                if df_raw.shape[1] > 1:  # If there's a second column, show it as potential answer
                    print(f"Potential answer: {str(df_raw.iloc[i, 1])[:100] if not pd.isna(df_raw.iloc[i, 1]) else 'NaN'}")
        
        print("\nINSPECTION COMPLETE")
        
    except Exception as e:
        print(f"Error inspecting file: {str(e)}")

def inspect_all_excel_files(directory):
    """Inspect all Excel files in the specified directory."""
    excel_files = glob.glob(os.path.join(directory, "*.xlsx"))
    
    if not excel_files:
        print(f"No Excel files found in {directory}")
        return
    
    print(f"Found {len(excel_files)} Excel files in {directory}")
    
    for file_path in excel_files:
        inspect_excel_file_raw(file_path)

if __name__ == "__main__":
    # Check if a directory is provided
    if len(sys.argv) > 1:
        directory = sys.argv[1]
    else:
        directory = "data"  # Default directory
    
    inspect_all_excel_files(directory)