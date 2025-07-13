import unittest
from unittest.mock import patch, MagicMock, mock_open, call
import pandas as pd
import os
import sys
import json

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from app.utils.create_patient_note_db import create_patient_note_table

class TestCreatePatientNoteDB(unittest.TestCase):
    
    def setUp(self):
        self.sample_csv_data = """patient_id,patient_uid,patient_note,age,gender
1,uid-001,Patient has mild symptoms,45,M
2,uid-002,Follow-up required,32,F"""
        
        self.expected_records = [
            {
                'patient_id': 1,
                'patient_uid': 'uid-001',
                'patient_note': 'Patient has mild symptoms',
                'age': 45,
                'gender': 'M'
            },
            {
                'patient_id': 2,
                'patient_uid': 'uid-002',
                'patient_note': 'Follow-up required',
                'age': 32,
                'gender': 'F'
            }
        ]

    @patch('app.utils.create_patient_note_db.os.path.exists')
    @patch('app.utils.create_patient_note_db.os.remove')
    @patch('app.utils.create_patient_note_db.tqdm')
    @patch('app.utils.create_patient_note_db.load_dotenv')
    @patch('app.utils.create_patient_note_db.os.getenv')
    @patch('app.utils.create_patient_note_db.create_client')
    @patch('app.utils.create_patient_note_db.pd.read_csv')
    @patch('builtins.open', new_callable=mock_open)
    def test_successful_execution_with_batching(self, mock_file, mock_read_csv, mock_create_client, 
                                                mock_getenv, mock_load_dotenv, mock_tqdm, mock_remove, mock_exists):
        mock_getenv.side_effect = lambda key: {
            'SUPABASE_URL': 'https://test.supabase.co',
            'SUPABASE_KEY': 'test-key',
            'ROOT_DIR': '/test/path'
        }.get(key)
        
        mock_exists.return_value = False  # No existing progress file
        
        # Create mock DataFrame with 1000 records to test batching
        mock_df = MagicMock()
        mock_df.columns.tolist.return_value = ['patient_id', 'patient_uid', 'patient_note', 'age', 'gender']
        mock_df.__len__.return_value = 1000
        mock_df.iloc.__getitem__.side_effect = lambda slice_obj: mock_df  # Return self for slicing
        mock_df.to_dict.return_value = self.expected_records * 500  # Simulate large batch
        mock_read_csv.return_value = mock_df
        
        mock_supabase = MagicMock()
        mock_table = MagicMock()
        mock_supabase.table.return_value = mock_table
        mock_table.insert.return_value = mock_table
        mock_table.execute.return_value = MagicMock()
        mock_create_client.return_value = mock_supabase
        
        # Mock tqdm progress bar
        mock_pbar = MagicMock()
        mock_tqdm.return_value.__enter__.return_value = mock_pbar
        
        with patch('builtins.print') as mock_print:
            create_patient_note_table()
            
        mock_load_dotenv.assert_called_once()
        mock_create_client.assert_called_once_with('https://test.supabase.co', 'test-key')
        mock_read_csv.assert_called_once_with('/test/path/PMC_Patients_clean.csv')
        
        # Should process in batches - expect multiple insert calls
        assert mock_table.insert.call_count >= 2  # At least 2 batches for 1000 records
        mock_print.assert_any_call('Total records to process: 1000')
        mock_print.assert_any_call('Successfully inserted all 1000 records into patient_notes table')

    @patch('app.utils.create_patient_note_db.load_dotenv')
    @patch('app.utils.create_patient_note_db.os.getenv')
    def test_missing_supabase_url(self, mock_getenv, mock_load_dotenv):
        mock_getenv.side_effect = lambda key: {
            'SUPABASE_URL': None,
            'SUPABASE_KEY': 'test-key'
        }.get(key)
        
        with self.assertRaises(ValueError) as context:
            create_patient_note_table()
        
        self.assertEqual(str(context.exception), 'SUPABASE_URL and SUPABASE_KEY must be set in .env file')

    @patch('app.utils.create_patient_note_db.load_dotenv')
    @patch('app.utils.create_patient_note_db.os.getenv')
    def test_missing_supabase_key(self, mock_getenv, mock_load_dotenv):
        mock_getenv.side_effect = lambda key: {
            'SUPABASE_URL': 'https://test.supabase.co',
            'SUPABASE_KEY': None
        }.get(key)
        
        with self.assertRaises(ValueError) as context:
            create_patient_note_table()
        
        self.assertEqual(str(context.exception), 'SUPABASE_URL and SUPABASE_KEY must be set in .env file')

    @patch('app.utils.create_patient_note_db.load_dotenv')
    @patch('app.utils.create_patient_note_db.os.getenv')
    def test_missing_root_dir(self, mock_getenv, mock_load_dotenv):
        mock_getenv.side_effect = lambda key: {
            'SUPABASE_URL': 'https://test.supabase.co',
            'SUPABASE_KEY': 'test-key',
            'ROOT_DIR': None
        }.get(key)
        
        with patch('builtins.print') as mock_print:
            create_patient_note_table()
            
        mock_print.assert_called_with('Please set the csv_file_path variable to the path of your patient notes CSV file')

    @patch('app.utils.create_patient_note_db.load_dotenv')
    @patch('app.utils.create_patient_note_db.os.getenv')
    @patch('app.utils.create_patient_note_db.create_client')
    @patch('app.utils.create_patient_note_db.pd.read_csv')
    def test_wrong_csv_columns(self, mock_read_csv, mock_create_client, mock_getenv, mock_load_dotenv):
        mock_getenv.side_effect = lambda key: {
            'SUPABASE_URL': 'https://test.supabase.co',
            'SUPABASE_KEY': 'test-key',
            'ROOT_DIR': '/test/path'
        }.get(key)
        
        mock_df = MagicMock()
        mock_df.columns.tolist.return_value = ['wrong', 'column', 'names', 'here', 'too']
        mock_df.iterrows.return_value = []
        mock_df.__len__.return_value = 0
        mock_read_csv.return_value = mock_df
        
        mock_supabase = MagicMock()
        mock_create_client.return_value = mock_supabase
        
        with patch('builtins.print') as mock_print:
            create_patient_note_table()
            
        expected_columns = ['patient_id', 'patient_uid', 'patient_note', 'age', 'gender']
        mock_print.assert_any_call(f'Warning: CSV columns should be: {expected_columns}')
        mock_print.assert_any_call('Found columns: [\'wrong\', \'column\', \'names\', \'here\', \'too\']')

    @patch('app.utils.create_patient_note_db.load_dotenv')
    @patch('app.utils.create_patient_note_db.os.getenv')
    @patch('app.utils.create_patient_note_db.create_client')
    @patch('app.utils.create_patient_note_db.pd.read_csv')
    def test_csv_read_error(self, mock_read_csv, mock_create_client, mock_getenv, mock_load_dotenv):
        mock_getenv.side_effect = lambda key: {
            'SUPABASE_URL': 'https://test.supabase.co',
            'SUPABASE_KEY': 'test-key',
            'ROOT_DIR': '/test/path'
        }.get(key)
        
        mock_read_csv.side_effect = FileNotFoundError('File not found')
        
        with patch('builtins.print') as mock_print:
            create_patient_note_table()
            
        mock_print.assert_called_with('An error occurred: File not found')

    @patch('app.utils.create_patient_note_db.load_dotenv')
    @patch('app.utils.create_patient_note_db.os.getenv')
    @patch('app.utils.create_patient_note_db.create_client')
    @patch('app.utils.create_patient_note_db.pd.read_csv')
    def test_supabase_connection_error(self, mock_read_csv, mock_create_client, mock_getenv, mock_load_dotenv):
        mock_getenv.side_effect = lambda key: {
            'SUPABASE_URL': 'https://test.supabase.co',
            'SUPABASE_KEY': 'test-key',
            'ROOT_DIR': '/test/path'
        }.get(key)
        
        mock_df = MagicMock()
        mock_df.columns.tolist.return_value = ['patient_id', 'patient_uid', 'patient_note', 'age', 'gender']
        mock_read_csv.return_value = mock_df
        
        mock_create_client.side_effect = Exception('Connection failed')
        
        with patch('builtins.print') as mock_print:
            create_patient_note_table()
            
        mock_print.assert_called_with('An error occurred: Connection failed')

    @patch('app.utils.create_patient_note_db.load_dotenv')
    @patch('app.utils.create_patient_note_db.os.getenv')
    @patch('app.utils.create_patient_note_db.create_client')
    @patch('app.utils.create_patient_note_db.pd.read_csv')
    def test_supabase_insert_error(self, mock_read_csv, mock_create_client, mock_getenv, mock_load_dotenv):
        mock_getenv.side_effect = lambda key: {
            'SUPABASE_URL': 'https://test.supabase.co',
            'SUPABASE_KEY': 'test-key',
            'ROOT_DIR': '/test/path'
        }.get(key)
        
        mock_df = MagicMock()
        mock_df.columns.tolist.return_value = ['patient_id', 'patient_uid', 'patient_note', 'age', 'gender']
        mock_df.iterrows.return_value = [(0, pd.Series({'patient_id': 1, 'patient_uid': 'uid-001'}))]
        mock_read_csv.return_value = mock_df
        
        mock_supabase = MagicMock()
        mock_table = MagicMock()
        mock_supabase.table.return_value = mock_table
        mock_table.insert.side_effect = Exception('Insert failed')
        mock_create_client.return_value = mock_supabase
        
        with patch('builtins.print') as mock_print:
            create_patient_note_table()
            
        mock_print.assert_called_with('An error occurred: Insert failed')

    @patch('app.utils.create_patient_note_db.load_dotenv')
    @patch('app.utils.create_patient_note_db.os.getenv')
    @patch('app.utils.create_patient_note_db.create_client')
    @patch('app.utils.create_patient_note_db.pd.read_csv')
    def test_empty_csv(self, mock_read_csv, mock_create_client, mock_getenv, mock_load_dotenv):
        mock_getenv.side_effect = lambda key: {
            'SUPABASE_URL': 'https://test.supabase.co',
            'SUPABASE_KEY': 'test-key',
            'ROOT_DIR': '/test/path'
        }.get(key)
        
        mock_df = MagicMock()
        mock_df.columns.tolist.return_value = ['patient_id', 'patient_uid', 'patient_note', 'age', 'gender']
        mock_df.iterrows.return_value = []
        mock_df.__len__.return_value = 0
        mock_read_csv.return_value = mock_df
        
        mock_supabase = MagicMock()
        mock_create_client.return_value = mock_supabase
        
        with patch('builtins.print') as mock_print:
            create_patient_note_table()
            
        mock_print.assert_called_with('Successfully inserted all 0 records into patient_note table')

    @patch('app.utils.create_patient_note_db.load_dotenv')
    @patch('app.utils.create_patient_note_db.os.getenv')
    @patch('app.utils.create_patient_note_db.create_client')
    @patch('app.utils.create_patient_note_db.pd.read_csv')
    def test_single_record(self, mock_read_csv, mock_create_client, mock_getenv, mock_load_dotenv):
        mock_getenv.side_effect = lambda key: {
            'SUPABASE_URL': 'https://test.supabase.co',
            'SUPABASE_KEY': 'test-key',
            'ROOT_DIR': '/test/path'
        }.get(key)
        
        single_record = {
            'patient_id': 999,
            'patient_uid': 'uid-999',
            'patient_note': 'Test note',
            'age': 25,
            'gender': 'F'
        }
        
        mock_df = MagicMock()
        mock_df.columns.tolist.return_value = ['patient_id', 'patient_uid', 'patient_note', 'age', 'gender']
        mock_df.iterrows.return_value = [(0, pd.Series(single_record))]
        mock_df.__len__.return_value = 1
        mock_read_csv.return_value = mock_df
        
        mock_supabase = MagicMock()
        mock_table = MagicMock()
        mock_supabase.table.return_value = mock_table
        mock_table.insert.return_value = mock_table
        mock_table.execute.return_value = MagicMock()
        mock_create_client.return_value = mock_supabase
        
        with patch('builtins.print') as mock_print:
            create_patient_note_table()
            
        mock_print.assert_any_call('Successfully added entry 1: Patient ID 999')
        mock_print.assert_called_with('Successfully inserted all 1 records into patient_notes table')

    @patch('app.utils.create_patient_note_db.os.path.exists')
    @patch('app.utils.create_patient_note_db.tqdm')
    @patch('app.utils.create_patient_note_db.load_dotenv')
    @patch('app.utils.create_patient_note_db.os.getenv')
    @patch('app.utils.create_patient_note_db.create_client')
    @patch('app.utils.create_patient_note_db.pd.read_csv')
    @patch('builtins.open', new_callable=mock_open)
    def test_connection_error_handling(self, mock_file, mock_read_csv, mock_create_client, 
                                       mock_getenv, mock_load_dotenv, mock_tqdm, mock_exists):
        mock_getenv.side_effect = lambda key: {
            'SUPABASE_URL': 'https://test.supabase.co',
            'SUPABASE_KEY': 'test-key',
            'ROOT_DIR': '/test/path'
        }.get(key)
        
        mock_exists.return_value = False
        
        mock_df = MagicMock()
        mock_df.columns.tolist.return_value = ['patient_id', 'patient_uid', 'patient_note', 'age', 'gender']
        mock_df.__len__.return_value = 500
        mock_df.iloc.__getitem__.return_value = mock_df
        mock_df.to_dict.return_value = self.expected_records
        mock_read_csv.return_value = mock_df
        
        mock_supabase = MagicMock()
        mock_table = MagicMock()
        mock_supabase.table.return_value = mock_table
        
        # Simulate connection error
        mock_table.insert.side_effect = Exception('ConnectionTerminated error_code:0')
        mock_create_client.return_value = mock_supabase
        
        mock_pbar = MagicMock()
        mock_tqdm.return_value.__enter__.return_value = mock_pbar
        
        with patch('builtins.print') as mock_print:
            create_patient_note_table()
            
        mock_print.assert_any_call('CONNECTION ERROR - Batch 1 failed (attempt 1/3)')
        mock_print.assert_any_call('CAUSE: Network connection dropped or server overloaded')

    @patch('app.utils.create_patient_note_db.os.path.exists')
    @patch('app.utils.create_patient_note_db.tqdm')
    @patch('app.utils.create_patient_note_db.load_dotenv')
    @patch('app.utils.create_patient_note_db.os.getenv')
    @patch('app.utils.create_patient_note_db.create_client')
    @patch('app.utils.create_patient_note_db.pd.read_csv')
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
        mock_file.return_value.read.return_value = json.dumps({'last_processed': 500, 'total': 1000})
        
        mock_df = MagicMock()
        mock_df.columns.tolist.return_value = ['patient_id', 'patient_uid', 'patient_note', 'age', 'gender']
        mock_df.__len__.return_value = 1000
        mock_df.iloc.__getitem__.return_value = mock_df
        mock_df.to_dict.return_value = self.expected_records
        mock_read_csv.return_value = mock_df
        
        mock_supabase = MagicMock()
        mock_table = MagicMock()
        mock_supabase.table.return_value = mock_table
        mock_table.insert.return_value = mock_table
        mock_table.execute.return_value = MagicMock()
        mock_create_client.return_value = mock_supabase
        
        mock_pbar = MagicMock()
        mock_tqdm.return_value.__enter__.return_value = mock_pbar
        
        with patch('builtins.print') as mock_print:
            create_patient_note_table()
            
        mock_print.assert_any_call('Resuming from record 500')

if __name__ == '__main__':
    unittest.main()