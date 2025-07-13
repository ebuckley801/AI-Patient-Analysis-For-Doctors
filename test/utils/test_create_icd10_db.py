import unittest
from unittest.mock import patch, MagicMock, mock_open, call
import pandas as pd
import os
import sys
import json

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from app.utils.create_icd10_db import create_icd10_table

class TestCreateICD10DB(unittest.TestCase):
    
    def setUp(self):
        self.sample_csv_data = """embedded_description,icd_10_code,description
"[0.1, 0.2, 0.3]",A00,Cholera
"[0.4, 0.5, 0.6]",A01,Typhoid fever"""
        
        self.expected_records = [
            {
                'embedded_description': '[0.1, 0.2, 0.3]',
                'icd_10_code': 'A00',
                'description': 'Cholera'
            },
            {
                'embedded_description': '[0.4, 0.5, 0.6]',
                'icd_10_code': 'A01',
                'description': 'Typhoid fever'
            }
        ]

    @patch('app.utils.create_icd10_db.os.path.exists')
    @patch('app.utils.create_icd10_db.os.remove')
    @patch('app.utils.create_icd10_db.tqdm')
    @patch('app.utils.create_icd10_db.load_dotenv')
    @patch('app.utils.create_icd10_db.os.getenv')
    @patch('app.utils.create_icd10_db.create_client')
    @patch('app.utils.create_icd10_db.pd.read_csv')
    @patch('builtins.open', new_callable=mock_open)
    def test_successful_execution_with_batching(self, mock_file, mock_read_csv, mock_create_client, 
                                                mock_getenv, mock_load_dotenv, mock_tqdm, mock_remove, mock_exists):
        mock_getenv.side_effect = lambda key: {
            'SUPABASE_URL': 'https://test.supabase.co',
            'SUPABASE_KEY': 'test-key',
            'ROOT_DIR': '/test/path'
        }.get(key)
        
        mock_exists.return_value = False  # No existing progress file
        
        # Create mock DataFrame with 2000 records to test batching
        mock_df = MagicMock()
        mock_df.columns.tolist.return_value = ['embedded_description', 'icd_10_code', 'description']
        mock_df.__len__.return_value = 2000
        mock_df.iloc.__getitem__.side_effect = lambda slice_obj: mock_df  # Return self for slicing
        mock_df.to_dict.return_value = self.expected_records * 1000  # Simulate large batch
        mock_read_csv.return_value = mock_df
        
        mock_supabase = MagicMock()
        mock_table = MagicMock()
        mock_supabase.table.return_value = mock_table
        mock_table.upsert.return_value = mock_table
        mock_table.execute.return_value = MagicMock()
        mock_create_client.return_value = mock_supabase
        
        # Mock tqdm progress bar
        mock_pbar = MagicMock()
        mock_tqdm.return_value.__enter__.return_value = mock_pbar
        
        with patch('builtins.print') as mock_print:
            create_icd10_table()
            
        mock_load_dotenv.assert_called_once()
        mock_create_client.assert_called_once_with('https://test.supabase.co', 'test-key')
        mock_read_csv.assert_called_once_with('/test/path/icd_10_codes.csv')
        
        # Should process in batches - expect multiple upsert calls
        assert mock_table.upsert.call_count >= 2  # At least 2 batches for 2000 records
        mock_print.assert_any_call('Total records to process: 2000')
        mock_print.assert_any_call('Successfully inserted all 2000 records into icd_10_codes table')

    @patch('app.utils.create_icd10_db.load_dotenv')
    @patch('app.utils.create_icd10_db.os.getenv')
    def test_missing_supabase_url(self, mock_getenv, mock_load_dotenv):
        mock_getenv.side_effect = lambda key: {
            'SUPABASE_URL': None,
            'SUPABASE_KEY': 'test-key'
        }.get(key)
        
        with self.assertRaises(ValueError) as context:
            create_icd10_table()
        
        self.assertEqual(str(context.exception), 'SUPABASE_URL and SUPABASE_KEY must be set in .env file')

    @patch('app.utils.create_icd10_db.load_dotenv')
    @patch('app.utils.create_icd10_db.os.getenv')
    def test_missing_supabase_key(self, mock_getenv, mock_load_dotenv):
        mock_getenv.side_effect = lambda key: {
            'SUPABASE_URL': 'https://test.supabase.co',
            'SUPABASE_KEY': None
        }.get(key)
        
        with self.assertRaises(ValueError) as context:
            create_icd10_table()
        
        self.assertEqual(str(context.exception), 'SUPABASE_URL and SUPABASE_KEY must be set in .env file')

    @patch('app.utils.create_icd10_db.load_dotenv')
    @patch('app.utils.create_icd10_db.os.getenv')
    def test_missing_root_dir(self, mock_getenv, mock_load_dotenv):
        mock_getenv.side_effect = lambda key: {
            'SUPABASE_URL': 'https://test.supabase.co',
            'SUPABASE_KEY': 'test-key',
            'ROOT_DIR': None
        }.get(key)
        
        with patch('builtins.print') as mock_print:
            create_icd10_table()
            
        mock_print.assert_called_with('Please set the csv_file_path variable to the path of your ICD-10 CSV file')

    @patch('app.utils.create_icd10_db.load_dotenv')
    @patch('app.utils.create_icd10_db.os.getenv')
    @patch('app.utils.create_icd10_db.create_client')
    @patch('app.utils.create_icd10_db.pd.read_csv')
    def test_wrong_csv_columns(self, mock_read_csv, mock_create_client, mock_getenv, mock_load_dotenv):
        mock_getenv.side_effect = lambda key: {
            'SUPABASE_URL': 'https://test.supabase.co',
            'SUPABASE_KEY': 'test-key',
            'ROOT_DIR': '/test/path'
        }.get(key)
        
        mock_df = MagicMock()
        mock_df.columns.tolist.return_value = ['wrong', 'column', 'names']
        mock_df.to_dict.return_value = []
        mock_read_csv.return_value = mock_df
        
        mock_supabase = MagicMock()
        mock_create_client.return_value = mock_supabase
        
        with patch('builtins.print') as mock_print:
            create_icd10_table()
            
        mock_print.assert_any_call('Warning: CSV columns should be: embedded_description, icd_10_code, description')
        mock_print.assert_any_call('Found columns: [\'wrong\', \'column\', \'names\']')

    @patch('app.utils.create_icd10_db.load_dotenv')
    @patch('app.utils.create_icd10_db.os.getenv')
    @patch('app.utils.create_icd10_db.create_client')
    @patch('app.utils.create_icd10_db.pd.read_csv')
    def test_csv_read_error(self, mock_read_csv, mock_create_client, mock_getenv, mock_load_dotenv):
        mock_getenv.side_effect = lambda key: {
            'SUPABASE_URL': 'https://test.supabase.co',
            'SUPABASE_KEY': 'test-key',
            'ROOT_DIR': '/test/path'
        }.get(key)
        
        mock_read_csv.side_effect = FileNotFoundError('File not found')
        
        with patch('builtins.print') as mock_print:
            create_icd10_table()
            
        mock_print.assert_called_with('An error occurred: File not found')

    @patch('app.utils.create_icd10_db.load_dotenv')
    @patch('app.utils.create_icd10_db.os.getenv')
    @patch('app.utils.create_icd10_db.create_client')
    @patch('app.utils.create_icd10_db.pd.read_csv')
    def test_supabase_connection_error(self, mock_read_csv, mock_create_client, mock_getenv, mock_load_dotenv):
        mock_getenv.side_effect = lambda key: {
            'SUPABASE_URL': 'https://test.supabase.co',
            'SUPABASE_KEY': 'test-key',
            'ROOT_DIR': '/test/path'
        }.get(key)
        
        mock_df = MagicMock()
        mock_df.columns.tolist.return_value = ['embedded_description', 'icd_10_code', 'description']
        mock_read_csv.return_value = mock_df
        
        mock_create_client.side_effect = Exception('Connection failed')
        
        with patch('builtins.print') as mock_print:
            create_icd10_table()
            
        mock_print.assert_called_with('An error occurred: Connection failed')

    @patch('app.utils.create_icd10_db.load_dotenv')
    @patch('app.utils.create_icd10_db.os.getenv')
    @patch('app.utils.create_icd10_db.create_client')
    @patch('app.utils.create_icd10_db.pd.read_csv')
    def test_empty_csv(self, mock_read_csv, mock_create_client, mock_getenv, mock_load_dotenv):
        mock_getenv.side_effect = lambda key: {
            'SUPABASE_URL': 'https://test.supabase.co',
            'SUPABASE_KEY': 'test-key',
            'ROOT_DIR': '/test/path'
        }.get(key)
        
        mock_df = MagicMock()
        mock_df.columns.tolist.return_value = ['embedded_description', 'icd_10_code', 'description']
        mock_df.to_dict.return_value = []
        mock_read_csv.return_value = mock_df
        
        mock_supabase = MagicMock()
        mock_table = MagicMock()
        mock_supabase.table.return_value = mock_table
        mock_table.upsert.return_value = mock_table
        mock_table.execute.return_value = MagicMock()
        mock_create_client.return_value = mock_supabase
        
        with patch('builtins.print') as mock_print:
            create_icd10_table()
            
        mock_print.assert_called_with('Successfully inserted all 0 records into icd_10_codes table')

    @patch('app.utils.create_icd10_db.os.path.exists')
    @patch('app.utils.create_icd10_db.tqdm')
    @patch('app.utils.create_icd10_db.load_dotenv')
    @patch('app.utils.create_icd10_db.os.getenv')
    @patch('app.utils.create_icd10_db.create_client')
    @patch('app.utils.create_icd10_db.pd.read_csv')
    @patch('builtins.open', new_callable=mock_open)
    def test_resume_from_progress_file(self, mock_file, mock_read_csv, mock_create_client, 
                                       mock_getenv, mock_load_dotenv, mock_tqdm, mock_exists):
        mock_getenv.side_effect = lambda key: {
            'SUPABASE_URL': 'https://test.supabase.co',
            'SUPABASE_KEY': 'test-key',
            'ROOT_DIR': '/test/path'
        }.get(key)
        
        # Mock existing progress file
        mock_exists.return_value = True
        mock_file.return_value.read.return_value = json.dumps({'last_processed': 1000, 'total': 2000})
        
        mock_df = MagicMock()
        mock_df.columns.tolist.return_value = ['embedded_description', 'icd_10_code', 'description']
        mock_df.__len__.return_value = 2000
        mock_df.iloc.__getitem__.return_value = mock_df
        mock_df.to_dict.return_value = self.expected_records
        mock_read_csv.return_value = mock_df
        
        mock_supabase = MagicMock()
        mock_table = MagicMock()
        mock_supabase.table.return_value = mock_table
        mock_table.upsert.return_value = mock_table
        mock_table.execute.return_value = MagicMock()
        mock_create_client.return_value = mock_supabase
        
        mock_pbar = MagicMock()
        mock_tqdm.return_value.__enter__.return_value = mock_pbar
        
        with patch('builtins.print') as mock_print:
            create_icd10_table()
            
        mock_print.assert_any_call('Resuming from record 1000')

    @patch('app.utils.create_icd10_db.os.path.exists')
    @patch('app.utils.create_icd10_db.tqdm')
    @patch('app.utils.create_icd10_db.time.sleep')
    @patch('app.utils.create_icd10_db.load_dotenv')
    @patch('app.utils.create_icd10_db.os.getenv')
    @patch('app.utils.create_icd10_db.create_client')
    @patch('app.utils.create_icd10_db.pd.read_csv')
    @patch('builtins.open', new_callable=mock_open)
    def test_retry_mechanism(self, mock_file, mock_read_csv, mock_create_client, 
                            mock_getenv, mock_load_dotenv, mock_sleep, mock_tqdm, mock_exists):
        mock_getenv.side_effect = lambda key: {
            'SUPABASE_URL': 'https://test.supabase.co',
            'SUPABASE_KEY': 'test-key',
            'ROOT_DIR': '/test/path'
        }.get(key)
        
        mock_exists.return_value = False
        
        mock_df = MagicMock()
        mock_df.columns.tolist.return_value = ['embedded_description', 'icd_10_code', 'description']
        mock_df.__len__.return_value = 1000
        mock_df.iloc.__getitem__.return_value = mock_df
        mock_df.to_dict.return_value = self.expected_records
        mock_read_csv.return_value = mock_df
        
        mock_supabase = MagicMock()
        mock_table = MagicMock()
        mock_supabase.table.return_value = mock_table
        
        # Simulate failure then success
        mock_table.upsert.side_effect = [Exception('Network error'), MagicMock()]
        mock_table.execute.return_value = MagicMock()
        mock_create_client.return_value = mock_supabase
        
        mock_pbar = MagicMock()
        mock_tqdm.return_value.__enter__.return_value = mock_pbar
        
        with patch('builtins.print') as mock_print:
            create_icd10_table()
            
        # Should have retried
        assert mock_table.upsert.call_count == 2
        mock_sleep.assert_called()
        mock_print.assert_any_call('Batch failed (attempt 1/3): Network error')

if __name__ == '__main__':
    unittest.main()