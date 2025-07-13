#!/usr/bin/env python3
"""
Test Intelligence Layer Database Creation
Tests the database schema setup and table creation functionality
"""

import unittest
import sys
import os
from unittest.mock import Mock, patch, MagicMock

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from app.utils.create_intelligence_db import create_intelligence_layer_tables, verify_tables


class TestCreateIntelligenceDB(unittest.TestCase):
    """Test cases for Intelligence Layer database creation"""
    
    @patch('app.utils.create_intelligence_db.load_dotenv')
    @patch('app.utils.create_intelligence_db.os.getenv')
    @patch('app.utils.create_intelligence_db.create_client')
    def test_create_intelligence_layer_tables_success(self, mock_create_client, mock_getenv, mock_load_dotenv):
        """Test successful creation of intelligence layer tables"""
        
        # Mock environment variables
        mock_getenv.side_effect = lambda key: {
            'SUPABASE_URL': 'https://test.supabase.co',
            'SUPABASE_KEY': 'test-key'
        }.get(key)
        
        # Mock Supabase client
        mock_supabase = Mock()
        mock_create_client.return_value = mock_supabase
        
        # Mock successful RPC calls for table creation
        mock_result = Mock()
        mock_result.execute.return_value = Mock()
        mock_supabase.rpc.return_value = mock_result
        
        # Run table creation
        success = create_intelligence_layer_tables()
        
        # Verify success
        self.assertTrue(success)
        
        # Verify environment was loaded
        mock_load_dotenv.assert_called_once()
        
        # Verify Supabase client was created
        mock_create_client.assert_called_once_with('https://test.supabase.co', 'test-key')
        
        # Verify RPC calls were made for table creation
        self.assertGreater(mock_supabase.rpc.call_count, 0)
        
        # Check that the main table creation SQL was called
        rpc_calls = [call[0][1]['sql'] for call in mock_supabase.rpc.call_args_list if 'sql' in call[0][1]]
        
        # Verify analysis_sessions table creation
        analysis_sessions_found = any('analysis_sessions' in sql and 'CREATE TABLE' in sql for sql in rpc_calls)
        self.assertTrue(analysis_sessions_found, "analysis_sessions table should be created")
        
        # Verify clinical_entities table creation
        clinical_entities_found = any('clinical_entities' in sql and 'CREATE TABLE' in sql for sql in rpc_calls)
        self.assertTrue(clinical_entities_found, "clinical_entities table should be created")
        
        # Verify entity_icd_mappings table creation
        entity_mappings_found = any('entity_icd_mappings' in sql and 'CREATE TABLE' in sql for sql in rpc_calls)
        self.assertTrue(entity_mappings_found, "entity_icd_mappings table should be created")
        
        # Verify analysis_cache table creation
        analysis_cache_found = any('analysis_cache' in sql and 'CREATE TABLE' in sql for sql in rpc_calls)
        self.assertTrue(analysis_cache_found, "analysis_cache table should be created")
    
    @patch('app.utils.create_intelligence_db.load_dotenv')
    @patch('app.utils.create_intelligence_db.os.getenv')
    def test_create_intelligence_layer_tables_missing_env(self, mock_getenv, mock_load_dotenv):
        """Test failure when environment variables are missing"""
        
        # Mock missing environment variables
        mock_getenv.return_value = None
        
        # Should raise ValueError for missing environment variables
        with self.assertRaises(ValueError) as context:
            create_intelligence_layer_tables()
        
        self.assertIn("SUPABASE_URL and SUPABASE_KEY must be set", str(context.exception))
    
    @patch('app.utils.create_intelligence_db.load_dotenv')
    @patch('app.utils.create_intelligence_db.os.getenv')
    @patch('app.utils.create_intelligence_db.create_client')
    @patch('builtins.print')  # Mock print to capture output
    def test_create_intelligence_layer_tables_database_error(self, mock_print, mock_create_client, mock_getenv, mock_load_dotenv):
        """Test handling of database errors during table creation"""
        
        # Mock environment variables
        mock_getenv.side_effect = lambda key: {
            'SUPABASE_URL': 'https://test.supabase.co',
            'SUPABASE_KEY': 'test-key'
        }.get(key)
        
        # Mock Supabase client that raises an exception
        mock_supabase = Mock()
        mock_create_client.return_value = mock_supabase
        mock_supabase.rpc.side_effect = Exception("Database connection failed")
        
        # Run table creation
        success = create_intelligence_layer_tables()
        
        # Verify failure
        self.assertFalse(success)
        
        # Verify error message was printed
        error_prints = [call for call in mock_print.call_args_list if 'error occurred' in str(call).lower()]
        self.assertGreater(len(error_prints), 0, "Error message should be printed")
    
    @patch('app.utils.create_intelligence_db.load_dotenv')
    @patch('app.utils.create_intelligence_db.os.getenv')
    @patch('app.utils.create_intelligence_db.create_client')
    def test_verify_tables_success(self, mock_create_client, mock_getenv, mock_load_dotenv):
        """Test successful table verification"""
        
        # Mock environment variables
        mock_getenv.side_effect = lambda key: {
            'SUPABASE_URL': 'https://test.supabase.co',
            'SUPABASE_KEY': 'test-key'
        }.get(key)
        
        # Mock Supabase client
        mock_supabase = Mock()
        mock_create_client.return_value = mock_supabase
        
        # Mock successful table queries
        mock_result = Mock()
        mock_result.execute.return_value = Mock()
        mock_supabase.table.return_value.select.return_value.limit.return_value = mock_result
        
        # Run verification
        success = verify_tables()
        
        # Verify success
        self.assertTrue(success)
        
        # Verify all required tables were checked
        expected_tables = ['analysis_sessions', 'clinical_entities', 'entity_icd_mappings', 'analysis_cache']
        table_calls = [call[0][0] for call in mock_supabase.table.call_args_list]
        
        for table in expected_tables:
            self.assertIn(table, table_calls, f"Table {table} should be verified")
    
    @patch('app.utils.create_intelligence_db.load_dotenv')
    @patch('app.utils.create_intelligence_db.os.getenv')
    @patch('app.utils.create_intelligence_db.create_client')
    def test_verify_tables_failure(self, mock_create_client, mock_getenv, mock_load_dotenv):
        """Test table verification failure"""
        
        # Mock environment variables
        mock_getenv.side_effect = lambda key: {
            'SUPABASE_URL': 'https://test.supabase.co',
            'SUPABASE_KEY': 'test-key'
        }.get(key)
        
        # Mock Supabase client
        mock_supabase = Mock()
        mock_create_client.return_value = mock_supabase
        
        # Mock table query failure
        mock_supabase.table.return_value.select.return_value.limit.return_value.execute.side_effect = Exception("Table does not exist")
        
        # Run verification
        success = verify_tables()
        
        # Verify failure
        self.assertFalse(success)
    
    def test_table_schema_completeness(self):
        """Test that all required table schemas are included"""
        
        # Read the create_intelligence_db.py file to check schema definitions
        db_file_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'app', 'utils', 'create_intelligence_db.py')
        
        with open(db_file_path, 'r') as f:
            db_content = f.read()
        
        # Check that all required tables are defined
        required_tables = [
            'analysis_sessions',
            'clinical_entities', 
            'entity_icd_mappings',
            'analysis_cache'
        ]
        
        for table in required_tables:
            self.assertIn(f'CREATE TABLE IF NOT EXISTS {table}', db_content, 
                         f"Table {table} should be defined in the schema")
        
        # Check for important columns in each table
        important_columns = {
            'analysis_sessions': ['session_id', 'status', 'risk_level', 'requires_immediate_attention'],
            'clinical_entities': ['entity_type', 'entity_text', 'confidence', 'session_id'],
            'entity_icd_mappings': ['icd_10_code', 'similarity_score', 'mapping_confidence', 'is_primary_mapping'],
            'analysis_cache': ['cache_key', 'cached_result', 'expires_at']
        }
        
        for table, columns in important_columns.items():
            for column in columns:
                self.assertIn(column, db_content, 
                             f"Column {column} should be defined in table {table}")
        
        # Check for foreign key relationships
        foreign_keys = [
            'FOREIGN KEY (session_id) REFERENCES analysis_sessions(session_id)',
            'FOREIGN KEY (entity_id) REFERENCES clinical_entities(id)',
            'FOREIGN KEY (icd_10_code) REFERENCES icd_10_codes(icd_10_code)'
        ]
        
        for fk in foreign_keys:
            self.assertIn(fk, db_content, f"Foreign key constraint should be defined: {fk}")
        
        # Check for indexes
        index_patterns = [
            'CREATE INDEX',
            'idx_analysis_sessions',
            'idx_clinical_entities', 
            'idx_entity_icd_mappings',
            'idx_analysis_cache'
        ]
        
        for pattern in index_patterns:
            self.assertIn(pattern, db_content, f"Index pattern should be present: {pattern}")
        
        # Check for functions and triggers
        function_patterns = [
            'cleanup_expired_cache',
            'update_updated_at_column',
            'CREATE TRIGGER'
        ]
        
        for pattern in function_patterns:
            self.assertIn(pattern, db_content, f"Function/trigger pattern should be present: {pattern}")


def run_database_creation_demo():
    """
    Demo function showing the database creation process
    Note: This would create real tables if .env is configured
    """
    print("🏗️  Intelligence Layer Database Creation Demo")
    print("=" * 60)
    
    try:
        print("\n📊 Testing Database Schema Creation...")
        
        # Show what tables would be created
        required_tables = [
            "analysis_sessions - Track analysis requests and metadata",
            "clinical_entities - Store extracted clinical entities", 
            "entity_icd_mappings - Store ICD-10 mappings for entities",
            "analysis_cache - Cache analysis results for performance"
        ]
        
        print("\n📋 Tables to be created:")
        for table in required_tables:
            print(f"   • {table}")
        
        print("\n🔧 Additional features:")
        print("   • Database indexes for optimal query performance")
        print("   • Cache cleanup function for maintenance")
        print("   • Timestamp update triggers")
        print("   • Foreign key constraints for data integrity")
        
        # Test the verification function (will fail without actual DB)
        print("\n🔍 Testing Table Verification...")
        try:
            success = verify_tables()
            if success:
                print("✅ All tables verified successfully!")
            else:
                print("⚠️  Table verification failed (expected without database)")
        except Exception as e:
            print(f"⚠️  Database not available: {str(e)}")
            print("   This is expected if Supabase is not configured")
        
        print("\n✨ Database Creation Demo Complete!")
        print("\nTo actually create the tables, run:")
        print("   python app/utils/create_intelligence_db.py")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")


if __name__ == "__main__":
    print("🧪 Running Intelligence Layer Database Tests")
    print("=" * 60)
    
    # Run unit tests
    unittest.main(argv=[''], exit=False, verbosity=2)
    
    # Run demo
    print("\n" + "=" * 60)
    run_database_creation_demo()