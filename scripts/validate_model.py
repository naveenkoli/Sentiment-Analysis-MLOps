import pickle
import logging
from pathlib import Path
import json
import sys

sys.path.append(str(Path(__file__).parent.parent))
from app.utils import preprocess_text

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelValidator:
    def validate_model(self, model_path):
        """Validate trained model"""
        model_path = Path(model_path)
        
        # Check files exist
        required_files = ["tfidf_vectorizer.pkl", "label_encoder.pkl", "logistic_regression_model.pkl", "metadata.json"]
        for filename in required_files:
            if not (model_path / filename).exists():
                logger.error(f"Missing file: {filename}")
                return False
        
        # Load metadata
        with open(model_path / "metadata.json", "r") as f:
            metadata = json.load(f)
        
        accuracy = metadata.get("accuracy", 0)
        if accuracy < 0.6:
            logger.error(f"Accuracy too low: {accuracy}")
            return False
        
        # Load and test models
        try:
            with open(model_path / "tfidf_vectorizer.pkl", "rb") as f:
                tfidf = pickle.load(f)
            with open(model_path / "label_encoder.pkl", "rb") as f:
                le = pickle.load(f)
            with open(model_path / "logistic_regression_model.pkl", "rb") as f:
                model = pickle.load(f)
            
            # Test prediction
            test_text = "This is a great project with positive outcomes"
            cleaned_text = preprocess_text(test_text)
            X = tfidf.transform([cleaned_text])
            pred = model.predict(X)[0]
            sentiment = le.inverse_transform([pred])[0]
            
            logger.info(f"Test prediction: '{test_text}' -> {sentiment}")
            logger.info(f"✅ Model validation passed - Accuracy: {accuracy:.4f}")
            return True
            
        except Exception as e:
            logger.error(f"Model validation failed: {str(e)}")
            return False

def main():
    if len(sys.argv) != 2:
        print("Usage: python validate_model.py <model_path>")
        return 1
        
    validator = ModelValidator()
    if validator.validate_model(sys.argv[1]):
        print("✅ Model validation passed")
        return 0
    else:
        print("❌ Model validation failed")
        return 1

if __name__ == "__main__":
    exit(main())
