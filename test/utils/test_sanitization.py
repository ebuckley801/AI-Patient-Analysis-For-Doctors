import unittest
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from app.utils.sanitization import Sanitizer, SQLInjectionPrevention, sanitize_request_data

class TestSanitizer(unittest.TestCase):
    
    def test_sanitize_text_basic(self):
        text = "  Hello World  "
        result = Sanitizer.sanitize_text(text)
        self.assertEqual(result, "Hello World")
    
    def test_sanitize_text_html_escape(self):
        text = "<script>alert('xss')</script>"
        result = Sanitizer.sanitize_text(text)
        self.assertNotIn("<script>", result)
        self.assertIn("&lt;script&gt;", result)
    
    def test_sanitize_text_null_bytes(self):
        text = "Hello\x00World"
        result = Sanitizer.sanitize_text(text)
        self.assertEqual(result, "HelloWorld")
    
    def test_sanitize_text_whitespace_normalization(self):
        text = "Hello    \n\n   World"
        result = Sanitizer.sanitize_text(text)
        self.assertEqual(result, "Hello World")
    
    def test_sanitize_html_allowed_tags(self):
        text = "<p>Hello <strong>World</strong></p>"
        result = Sanitizer.sanitize_html(text)
        self.assertIn("<p>", result)
        self.assertIn("<strong>", result)
    
    def test_sanitize_html_disallowed_tags(self):
        text = "<script>alert('xss')</script><p>Safe content</p>"
        result = Sanitizer.sanitize_html(text)
        self.assertNotIn("<script>", result)
        self.assertIn("<p>", result)
    
    def test_sanitize_patient_uid_valid(self):
        uid = "patient-123_test"
        result = Sanitizer.sanitize_patient_uid(uid)
        self.assertEqual(result, "patient-123_test")
    
    def test_sanitize_patient_uid_invalid_chars(self):
        uid = "patient@123#test"
        result = Sanitizer.sanitize_patient_uid(uid)
        self.assertEqual(result, "patient123test")
    
    def test_sanitize_patient_uid_length_limit(self):
        uid = "a" * 150
        result = Sanitizer.sanitize_patient_uid(uid)
        self.assertEqual(len(result), 100)
    
    def test_sanitize_search_query_safe(self):
        query = "symptoms fever"
        result = Sanitizer.sanitize_search_query(query)
        self.assertEqual(result, "symptoms fever")
    
    def test_sanitize_search_query_dangerous_chars(self):
        query = "symptoms'; DROP TABLE--"
        result = Sanitizer.sanitize_search_query(query)
        self.assertNotIn("';", result)
        self.assertNotIn("--", result)
    
    def test_sanitize_patient_note_basic(self):
        note = "<p>Patient has <strong>mild</strong> symptoms</p>"
        result = Sanitizer.sanitize_patient_note(note)
        self.assertIn("<p>", result)
        self.assertIn("<strong>", result)
    
    def test_sanitize_patient_note_excessive_newlines(self):
        note = "Line 1\n\n\n\n\nLine 2"
        result = Sanitizer.sanitize_patient_note(note)
        self.assertEqual(result.count('\n'), 2)  # Should be reduced to 2
    
    def test_sanitize_integer_valid(self):
        result = Sanitizer.sanitize_integer(25, min_val=0, max_val=100)
        self.assertEqual(result, 25)
    
    def test_sanitize_integer_below_min(self):
        result = Sanitizer.sanitize_integer(-5, min_val=0, max_val=100)
        self.assertEqual(result, 0)
    
    def test_sanitize_integer_above_max(self):
        result = Sanitizer.sanitize_integer(150, min_val=0, max_val=100)
        self.assertEqual(result, 100)
    
    def test_sanitize_integer_invalid(self):
        result = Sanitizer.sanitize_integer("invalid", min_val=0, max_val=100)
        self.assertIsNone(result)
    
    def test_sanitize_gender_normalization(self):
        self.assertEqual(Sanitizer.sanitize_gender("male"), "M")
        self.assertEqual(Sanitizer.sanitize_gender("female"), "F")
        self.assertEqual(Sanitizer.sanitize_gender("MALE"), "M")
        self.assertEqual(Sanitizer.sanitize_gender("FEMALE"), "F")
    
    def test_sanitize_gender_invalid_chars(self):
        result = Sanitizer.sanitize_gender("m@le")
        self.assertEqual(result, "MLE")
    
    def test_sanitize_patient_data_complete(self):
        data = {
            'patient_id': 123,
            'patient_uid': 'uid-123',
            'patient_note': '<p>Test note</p>',
            'age': 30,
            'gender': 'male'
        }
        result = Sanitizer.sanitize_patient_data(data)
        
        self.assertEqual(result['patient_id'], 123)
        self.assertEqual(result['patient_uid'], 'uid-123')
        self.assertIn('<p>', result['patient_note'])
        self.assertEqual(result['age'], 30)
        self.assertEqual(result['gender'], 'M')
    
    def test_sanitize_patient_data_removes_none(self):
        data = {
            'patient_id': 'invalid',
            'patient_uid': 'uid-123',
            'age': 30
        }
        result = Sanitizer.sanitize_patient_data(data)
        
        self.assertNotIn('patient_id', result)  # Should be removed as it's None
        self.assertIn('patient_uid', result)
        self.assertIn('age', result)

class TestSQLInjectionPrevention(unittest.TestCase):
    
    def test_contains_sql_injection_basic_keywords(self):
        self.assertTrue(SQLInjectionPrevention.contains_sql_injection("SELECT * FROM users"))
        self.assertTrue(SQLInjectionPrevention.contains_sql_injection("DROP TABLE patients"))
        self.assertTrue(SQLInjectionPrevention.contains_sql_injection("INSERT INTO"))
    
    def test_contains_sql_injection_comments(self):
        self.assertTrue(SQLInjectionPrevention.contains_sql_injection("test -- comment"))
        self.assertTrue(SQLInjectionPrevention.contains_sql_injection("test /* comment */"))
    
    def test_contains_sql_injection_union_attacks(self):
        self.assertTrue(SQLInjectionPrevention.contains_sql_injection("' UNION SELECT"))
        self.assertTrue(SQLInjectionPrevention.contains_sql_injection("1=1 OR 2=2"))
    
    def test_contains_sql_injection_safe_text(self):
        self.assertFalse(SQLInjectionPrevention.contains_sql_injection("Patient has symptoms"))
        self.assertFalse(SQLInjectionPrevention.contains_sql_injection("Normal text content"))
    
    def test_sanitize_sql_input_removes_dangerous_patterns(self):
        dangerous_text = "test' OR 1=1 --"
        result = SQLInjectionPrevention.sanitize_sql_input(dangerous_text)
        self.assertNotIn("OR 1=1", result)
        self.assertNotIn("--", result)

class TestSanitizeRequestData(unittest.TestCase):
    
    def test_sanitize_request_data_valid(self):
        data = {
            'patient_id': 123,
            'patient_uid': 'uid-123',
            'patient_note': 'Normal patient note',
            'age': 30,
            'gender': 'M'
        }
        result = sanitize_request_data(data)
        self.assertEqual(result['patient_id'], 123)
        self.assertEqual(result['patient_uid'], 'uid-123')
    
    def test_sanitize_request_data_sql_injection(self):
        data = {
            'patient_note': "'; DROP TABLE patients; --"
        }
        with self.assertRaises(ValueError):
            sanitize_request_data(data)
    
    def test_sanitize_request_data_invalid_input(self):
        result = sanitize_request_data("not a dict")
        self.assertEqual(result, {})
    
    def test_sanitize_request_data_nested_sql_injection(self):
        data = {
            'patient_id': 123,
            'patient_note': "Normal note",
            'metadata': {
                'source': "' OR 1=1 --"
            }
        }
        # Should pass through nested data without checking (for now)
        result = sanitize_request_data(data)
        self.assertIn('patient_id', result)

if __name__ == '__main__':
    unittest.main()