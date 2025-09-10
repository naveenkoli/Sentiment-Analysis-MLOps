import pytest
import sys
import os
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def test_imports():
    """Test basic imports work"""
    try:
        import pandas as pd
        import numpy as np
        import sklearn
        assert True
    except ImportError as e:
        pytest.fail(f"Basic imports failed: {e}")

def test_project_structure():
    """Test project structure exists"""
    project_root = Path(__file__).parent.parent
    
    # Check essential directories
    essential_dirs = ['app', 'models', 'data']
    for dir_name in essential_dirs:
        dir_path = project_root / dir_name
        assert dir_path.exists(), f"Directory {dir_name} not found"

def test_requirements_file():
    """Test requirements file exists"""
    project_root = Path(__file__).parent.parent
    requirements_file = project_root / "requirements.txt"
    assert requirements_file.exists(), "requirements.txt not found"

def test_app_files():
    """Test app files exist"""
    project_root = Path(__file__).parent.parent
    app_files = ['app/main.py', 'app/utils.py']
    
    for file_path in app_files:
        full_path = project_root / file_path
        assert full_path.exists(), f"App file {file_path} not found"

# Simple test that can run without model files
def test_text_preprocessing():
    """Test text preprocessing without loading models"""
    try:
        # Import the preprocessing function
        from app.utils import preprocess_text
        
        # Test basic preprocessing
        test_text = "This is a TEST message with Numbers123 and Special!@#"
        result = preprocess_text(test_text)
        
        # Basic assertions
        assert isinstance(result, str), "Result should be string"
        assert len(result) > 0, "Result should not be empty"
        
        # Check that preprocessing worked (lowercased, cleaned)
        assert result.islower() or len(result) == 0, "Text should be lowercased"
        
    except ImportError:
        pytest.skip("Cannot import preprocessing function - utils not available")
    except Exception as e:
        pytest.fail(f"Text preprocessing failed: {e}")

if __name__ == "__main__":
    pytest.main([__file__, "-v"])