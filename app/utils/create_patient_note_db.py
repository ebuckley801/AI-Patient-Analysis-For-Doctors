import pandas as pd
import os
import json
import time
from supabase import create_client, Client
from dotenv import load_dotenv
from tqdm import tqdm

def create_patient_note_table():
    """
    Creates the Supabase table for patient notes and loads data from CSV with batching and resume capability.
    """
    load_dotenv()
    
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_KEY")
    
    if not supabase_url or not supabase_key:
        raise ValueError("SUPABASE_URL and SUPABASE_KEY must be set in .env file")
    
    csv_file_path = os.getenv("ROOT_DIR") + "/PMC_Patients_clean.csv"
    progress_file = os.getenv("ROOT_DIR") + "/patient_notes_progress.json"
    
    if not csv_file_path:
        print("Please set the csv_file_path variable to the path of your patient notes CSV file")
        return
    
    batch_size = 500
    start_from = 0
    
    try:
        # Check for existing progress
        if os.path.exists(progress_file):
            with open(progress_file, 'r') as f:
                progress_data = json.load(f)
                start_from = progress_data.get('last_processed', 0)
                print(f"Resuming from record {start_from}")
        
        # Initialize Supabase client
        supabase: Client = create_client(supabase_url, supabase_key)
        
        # Read CSV file
        print("Reading CSV file...")
        df = pd.read_csv(csv_file_path)
        total_records = len(df)
        
        expected_columns = ["patient_id", "patient_uid", "patient_note", "age", "gender"]
        if df.columns.tolist() != expected_columns:
            print(f"Warning: CSV columns should be: {expected_columns}")
            print(f"Found columns: {df.columns.tolist()}")
        
        print(f"Total records to process: {total_records}")
        print(f"Starting from record: {start_from}")
        print(f"Batch size: {batch_size}")
        
        # Performance warning for large datasets
        if total_records > 10000:
            print("\n⚠️  LARGE DATASET DETECTED:")
            print("   - Connection may drop after thousands of records")
            print("   - Script will automatically retry connection errors")
            print("   - Progress is saved - you can resume if interrupted\n")
        
        # Process in batches with progress bar
        processed_count = start_from
        
        with tqdm(total=total_records, initial=start_from, desc="Inserting records") as pbar:
            for i in range(start_from, total_records, batch_size):
                end_idx = min(i + batch_size, total_records)
                batch_df = df.iloc[i:end_idx]
                batch_records = batch_df.to_dict('records')
                
                retry_count = 0
                max_retries = 3
                
                while retry_count < max_retries:
                    try:
                        result = supabase.table("patient_notes").insert(batch_records).execute()
                        processed_count = end_idx
                        
                        # Update progress file
                        with open(progress_file, 'w') as f:
                            json.dump({'last_processed': processed_count, 'total': total_records}, f)
                        
                        pbar.update(len(batch_records))
                        print(f"Batch {i//batch_size + 1}: Inserted {len(batch_records)} records (Total: {processed_count}/{total_records})")
                        break
                        
                    except Exception as batch_error:
                        retry_count += 1
                        wait_time = 2 ** retry_count
                        error_msg = str(batch_error)
                        
                        # Check for connection errors specifically
                        if "connectionterminated" in error_msg.lower() or "connection" in error_msg.lower():
                            print(f"CONNECTION ERROR - Batch {i//batch_size + 1} failed (attempt {retry_count}/{max_retries})")
                            print("CAUSE: Network connection dropped or server overloaded")
                            print("SOLUTIONS: 1) Reduce batch size, 2) Check network stability, 3) Retry during off-peak hours")
                        elif "timeout" in error_msg.lower():
                            print(f"TIMEOUT ERROR - Batch {i//batch_size + 1} failed (attempt {retry_count}/{max_retries})")
                            print("CAUSE: Database performance issue or large batch size")
                            print("SOLUTIONS: 1) Reduce batch size to 250 or 100")
                        else:
                            print(f"Batch failed (attempt {retry_count}/{max_retries}): {batch_error}")
                        
                        if retry_count < max_retries:
                            print(f"Retrying in {wait_time} seconds...")
                            time.sleep(wait_time)
                        else:
                            print(f"Failed to insert batch after {max_retries} attempts. Progress saved at record {processed_count}")
                            print("\nTROUBLESHOOTING:")
                            print("1. Try reducing batch_size to 250 or 100")
                            print("2. Check your internet connection stability")
                            print("3. Run script during off-peak hours")
                            print("4. Consider upgrading Supabase plan for better performance")
                            return
                
                # Small delay between batches to avoid overwhelming the connection
                time.sleep(0.2)
        
        # Clean up progress file on successful completion
        if os.path.exists(progress_file):
            os.remove(progress_file)
        
        print(f"Successfully inserted all {total_records} records into patient_notes table")
        
    except Exception as e:
        print(f"An error occurred: {e}")
        print("Progress saved. You can resume by running the script again.")

if __name__ == "__main__":
    create_patient_note_table()