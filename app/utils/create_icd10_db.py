
import pandas as pd
import os
import json
import time
from supabase import create_client, Client
from dotenv import load_dotenv
from tqdm import tqdm

def create_icd10_table():
    """
    Creates the Supabase table for ICD-10 codes and loads data from CSV with batching and resume capability.
    """
    load_dotenv()
    
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_KEY")
    
    if not supabase_url or not supabase_key:
        raise ValueError("SUPABASE_URL and SUPABASE_KEY must be set in .env file")
    
    csv_file_path = os.getenv("ROOT_DIR")+"/icd_10_codes.csv"
    progress_file = os.getenv("ROOT_DIR")+"/icd10_progress.json"
    
    if not csv_file_path:
        print("Please set the csv_file_path variable to the path of your ICD-10 CSV file")
        return
    
    # Start with smaller batches to avoid timeouts as table grows
    batch_size = 250
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
        
        # Read CSV file in chunks for memory efficiency
        print("Reading CSV file...")
        df = pd.read_csv(csv_file_path)
        total_records = len(df)
        
        if df.columns.tolist() != ["embedded_description", "icd_10_code", "description"]:
            print("Warning: CSV columns should be: embedded_description, icd_10_code, description")
            print(f"Found columns: {df.columns.tolist()}")
        
        # Convert string representation of arrays to actual arrays
        print("Converting embedded_description strings to arrays...")
        import ast
        df['embedded_description'] = df['embedded_description'].apply(
            lambda x: ast.literal_eval(x) if isinstance(x, str) and x.startswith('[') else x
        )
        
        print(f"Total records to process: {total_records}")
        print(f"Starting from record: {start_from}")
        print(f"Batch size: {batch_size}")
        
        # Performance warning for large datasets
        if total_records > 50000:
            print("\n⚠️  LARGE DATASET DETECTED:")
            print("   - Database performance may degrade as table grows")
            print("   - Expect slower inserts after ~10k records")
            print("   - Script will automatically retry timeouts")
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
                        # Use insert instead of upsert for better performance on new data
                        result = supabase.table("icd_10_codes").insert(batch_records).execute()
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
                        
                        # Check for timeout specifically
                        if "timeout" in error_msg.lower() or "57014" in error_msg:
                            print(f"TIMEOUT ERROR - Batch {i//batch_size + 1} failed (attempt {retry_count}/{max_retries})")
                            print("CAUSE: Database performance degrades as table grows larger")
                            print("SOLUTIONS: 1) Reduce batch size, 2) Add database indexes, 3) Use faster instance")
                        else:
                            print(f"Batch failed (attempt {retry_count}/{max_retries}): {batch_error}")
                        
                        if retry_count < max_retries:
                            print(f"Retrying in {wait_time} seconds...")
                            time.sleep(wait_time)
                        else:
                            print(f"Failed to insert batch after {max_retries} attempts. Progress saved at record {processed_count}")
                            print("\nTROUBLESHOOTING:")
                            print("1. Try reducing batch_size to 250 or 100")
                            print("2. Add index on embedded_description if doing vector searches")
                            print("3. Consider upgrading Supabase plan for better performance")
                            print("4. Run script during off-peak hours")
                            return
                
                # Small delay between batches to avoid rate limiting
                time.sleep(0.1)
        
        # Clean up progress file on successful completion
        if os.path.exists(progress_file):
            os.remove(progress_file)
        
        print(f"Successfully inserted all {total_records} records into icd_10_codes table")
        
    except Exception as e:
        print(f"An error occurred: {e}")
        print(f"Progress saved. You can resume by running the script again.")

if __name__ == "__main__":
    create_icd10_table()
