import unittest
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from app.utils.validation import Validator, ValidationError, PatientNoteSchema

class TestValidator(unittest.TestCase):
    
    def test_validate_patient_id_valid(self):
        self.assertEqual(Validator.validate_patient_id(123), 123)
        self.assertEqual(Validator.validate_patient_id("456"), 456)
    
    def test_validate_patient_id_invalid(self):
        with self.assertRaises(ValidationError):
            Validator.validate_patient_id(-1)
        
        with self.assertRaises(ValidationError):
            Validator.validate_patient_id(0)
        
        with self.assertRaises(ValidationError):
            Validator.validate_patient_id("invalid")
    
    def test_validate_patient_uid_valid(self):
        self.assertEqual(Validator.validate_patient_uid("uid-123"), "uid-123")
        self.assertEqual(Validator.validate_patient_uid("user_123"), "user_123")
    
    def test_validate_patient_uid_invalid(self):
        with self.assertRaises(ValidationError):
            Validator.validate_patient_uid("")
        
        with self.assertRaises(ValidationError):
            Validator.validate_patient_uid("uid with spaces")
        
        with self.assertRaises(ValidationError):
            Validator.validate_patient_uid("a" * 101)  # Too long
    
    def test_validate_patient_note_valid(self):
        note = "Patient has mild symptoms"
        self.assertEqual(Validator.validate_patient_note(note), note)
    
    def test_validate_patient_note_invalid(self):
        with self.assertRaises(ValidationError):
            Validator.validate_patient_note("")
        
        with self.assertRaises(ValidationError):
            Validator.validate_patient_note("a" * 10001)  # Too long
    
    def test_validate_age_valid(self):
        self.assertEqual(Validator.validate_age(25), 25)
        self.assertEqual(Validator.validate_age("30"), 30)
    
    def test_validate_age_invalid(self):
        with self.assertRaises(ValidationError):
            Validator.validate_age(-1)
        
        with self.assertRaises(ValidationError):
            Validator.validate_age(151)
        
        with self.assertRaises(ValidationError):
            Validator.validate_age("invalid")
    
    def test_validate_gender_valid(self):
        self.assertEqual(Validator.validate_gender("M"), "M")
        self.assertEqual(Validator.validate_gender("F"), "F")
        self.assertEqual(Validator.validate_gender("male"), "M")
        self.assertEqual(Validator.validate_gender("female"), "F")
    
    def test_validate_gender_invalid(self):
        with self.assertRaises(ValidationError):
            Validator.validate_gender("invalid")
    
    def test_validate_search_query_valid(self):
        query = "symptoms"
        self.assertEqual(Validator.validate_search_query(query), query)
    
    def test_validate_search_query_invalid(self):
        with self.assertRaises(ValidationError):
            Validator.validate_search_query("")
        
        with self.assertRaises(ValidationError):
            Validator.validate_search_query("a")  # Too short
        
        with self.assertRaises(ValidationError):
            Validator.validate_search_query("a" * 501)  # Too long
    
    def test_validate_pagination_valid(self):
        limit, offset = Validator.validate_pagination(50, 10)
        self.assertEqual(limit, 50)
        self.assertEqual(offset, 10)
    
    def test_validate_pagination_invalid(self):
        with self.assertRaises(ValidationError):
            Validator.validate_pagination(0, 0)  # Limit too small
        
        with self.assertRaises(ValidationError):
            Validator.validate_pagination(1001, 0)  # Limit too large
        
        with self.assertRaises(ValidationError):
            Validator.validate_pagination(100, -1)  # Negative offset

class TestPatientNoteSchema(unittest.TestCase):
    
    def setUp(self):
        self.valid_data = {
            'patient_id': 123,
            'patient_uid': 'uid-123',
            'patient_note': 'Patient has mild symptoms',
            'age': 30,
            'gender': 'M'
        }
    
    def test_validate_create_request_valid(self):
        result = PatientNoteSchema.validate_create_request(self.valid_data)
        self.assertEqual(result['patient_id'], 123)
        self.assertEqual(result['patient_uid'], 'uid-123')
    
    def test_validate_create_request_missing_field(self):
        incomplete_data = self.valid_data.copy()
        del incomplete_data['patient_id']
        
        with self.assertRaises(ValidationError):
            PatientNoteSchema.validate_create_request(incomplete_data)
    
    def test_validate_create_request_invalid_data(self):
        invalid_data = self.valid_data.copy()
        invalid_data['age'] = -1
        
        with self.assertRaises(ValidationError):
            PatientNoteSchema.validate_create_request(invalid_data)
    
    def test_validate_update_request_valid(self):
        update_data = {'patient_note': 'Updated note'}
        result = PatientNoteSchema.validate_update_request(update_data)
        self.assertEqual(result['patient_note'], 'Updated note')
    
    def test_validate_update_request_empty(self):
        with self.assertRaises(ValidationError):
            PatientNoteSchema.validate_update_request({})
    
    def test_validate_update_request_partial(self):
        update_data = {'age': 35, 'gender': 'F'}
        result = PatientNoteSchema.validate_update_request(update_data)
        self.assertEqual(result['age'], 35)
        self.assertEqual(result['gender'], 'F')
        self.assertNotIn('patient_note', result)

if __name__ == '__main__':
    unittest.main()