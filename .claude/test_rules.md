# Test Development Rules

## Mandatory Test Creation
For every new file created in the following directories, a corresponding test file MUST be created:
- `app/config/` → `test/config/`
- `app/models/` → `test/models/`  
- `app/routes/` → `test/routes/`
- `app/services/` → `test/services/`
- `app/utils/` → `test/utils/`

## Test File Naming Convention
- Test files must be named `test_<original_filename>.py`
- Example: `app/utils/database.py` → `test/utils/test_database.py`

## Test Coverage Requirements
Each test file must include comprehensive tests for:

### 1. Happy Path Testing
- Normal execution with valid inputs
- Expected return values and side effects
- Proper function calls and integrations

### 2. Edge Cases
- Empty inputs (empty strings, lists, None values)
- Boundary conditions (min/max values)
- Single item vs multiple items
- Missing optional parameters

### 3. Error Handling
- Invalid input types
- Missing required parameters
- File not found errors
- Network/connection failures
- Database errors
- Permission errors

### 4. Environment & Configuration
- Missing environment variables
- Invalid configuration values
- Different environment setups

### 5. Mocking Requirements
- Mock external dependencies (databases, APIs, file systems)
- Mock environment variables
- Mock third-party library calls
- Use `unittest.mock.patch` for isolation

## Test Structure Template
```python
import unittest
from unittest.mock import patch, MagicMock
import sys
import os

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from app.module.target_file import target_function

class TestTargetFunction(unittest.TestCase):
    
    def setUp(self):
        # Setup test data and mocks
        pass
    
    def test_successful_execution(self):
        # Test happy path
        pass
    
    def test_edge_case_empty_input(self):
        # Test empty inputs
        pass
    
    def test_error_handling(self):
        # Test error conditions
        pass
    
    def tearDown(self):
        # Cleanup if needed
        pass

if __name__ == '__main__':
    unittest.main()
```

## Update Requirements
When base code is modified:
1. Review corresponding test file
2. Update tests to match new functionality
3. Add tests for new edge cases
4. Ensure all existing tests still pass
5. Add new tests for any new functions/methods

## Test Execution
- Tests must be runnable with `python -m pytest test/`
- Individual test files should be runnable with `python -m unittest test.module.test_file`
- All tests must pass before code is considered complete

## Forbidden Practices
- Do NOT create tests that require actual database connections
- Do NOT create tests that make real API calls
- Do NOT skip test creation for "simple" functions
- Do NOT leave TODO comments in test files

## These rules are MANDATORY and must be followed for every code change.