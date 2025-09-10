import pandas as pd
import pickle
import logging
import sys
from datetime import datetime
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
import json

sys.path.append(str(Path(__file__).parent.parent))
from app.utils import preprocess_text

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelTrainer:
    def __init__(self):
        self.data_dir = Path("data")
        self.model_dir = Path("models")
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create directories
        (self.data_dir / "raw").mkdir(parents=True, exist_ok=True)
        (self.data_dir / "processed").mkdir(parents=True, exist_ok=True)
        (self.model_dir / "training").mkdir(parents=True, exist_ok=True)
        (self.model_dir / "current").mkdir(parents=True, exist_ok=True)
        
    def check_for_new_data(self):
        """Check if there are new data files to process"""
        raw_data_dir = self.data_dir / "raw"
        data_files = list(raw_data_dir.glob("*.xlsx")) + list(raw_data_dir.glob("*.csv"))
        
        if not data_files:
            logger.warning("No data files found in data/raw/")
            return False
        
        # Check if current models exist
        current_metadata = self.model_dir / "current" / "metadata.json"
        if not current_metadata.exists():
            logger.info("No current models found, training new model")
            return True
            
        # Check file modification times
        with open(current_metadata, "r") as f:
            metadata = json.load(f)
            last_training = datetime.strptime(metadata.get("timestamp", "19700101_000000"), "%Y%m%d_%H%M%S")
        
        for file in data_files:
            if datetime.fromtimestamp(file.stat().st_mtime) > last_training:
                logger.info(f"New data found: {file.name}")
                return True
                
        logger.info("No new data found")
        return False
        
    def load_and_combine_data(self):
        """Load and combine all data files"""
        raw_data_dir = self.data_dir / "raw"
        data_files = list(raw_data_dir.glob("*.xlsx")) + list(raw_data_dir.glob("*.csv"))
        
        all_data = []
        for file in data_files:
            try:
                if file.suffix == '.xlsx':
                    df = pd.read_excel(file)
                else:
                    df = pd.read_csv(file)
                all_data.append(df)
                logger.info(f"Loaded {len(df)} records from {file.name}")
            except Exception as e:
                logger.error(f"Error loading {file.name}: {str(e)}")
        
        if not all_data:
            raise ValueError("No valid data files found")
            
        combined_df = pd.concat(all_data, ignore_index=True)
        
        # Save processed data
        processed_file = self.data_dir / "processed" / f"combined_data_{self.timestamp}.csv"
        combined_df.to_csv(processed_file, index=False)
        
        logger.info(f"Combined dataset: {len(combined_df)} records")
        return combined_df
        
    def preprocess_data(self, df):
        """Clean and preprocess the data"""
        # Check required columns
        required_cols = ['Remarks', 'Sentiment']
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"Required columns missing: {required_cols}")
        
        # Remove duplicates and null values
        df = df.drop_duplicates(subset=['Remarks'])
        df = df.dropna(subset=['Remarks', 'Sentiment'])
        
        # Clean sentiment labels
        df['Sentiment'] = df['Sentiment'].str.lower()
        valid_sentiments = ['positive', 'negative', 'neutral']
        df = df[df['Sentiment'].isin(valid_sentiments)]
        
        # Preprocess text
        df['cleaned_text'] = df['Remarks'].apply(preprocess_text)
        df = df[df['cleaned_text'].str.len() > 0]
        
        logger.info(f"Preprocessed data: {len(df)} records")
        return df
        
    def train_models(self, df):
        """Train the ML models"""
        X = df['cleaned_text']
        y = df['Sentiment']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train TF-IDF vectorizer
        tfidf = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
        X_train_tfidf = tfidf.fit_transform(X_train)
        X_test_tfidf = tfidf.transform(X_test)
        
        # Train label encoder
        le = LabelEncoder()
        y_train_encoded = le.fit_transform(y_train)
        y_test_encoded = le.transform(y_test)
        
        # Train model
        model = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
        model.fit(X_train_tfidf, y_train_encoded)
        
        # Evaluate
        y_pred = model.predict(X_test_tfidf)
        accuracy = accuracy_score(y_test_encoded, y_pred)
        
        logger.info(f"Model accuracy: {accuracy:.4f}")
        logger.info(f"Classification report:\n{classification_report(y_test_encoded, y_pred, target_names=le.classes_)}")
        
        return tfidf, le, model, accuracy
        
    def save_models(self, tfidf, le, model, accuracy, data_info):
        """Save trained models"""
        training_dir = self.model_dir / "training" / self.timestamp
        training_dir.mkdir(parents=True, exist_ok=True)
        
        # Save models
        with open(training_dir / "tfidf_vectorizer.pkl", "wb") as f:
            pickle.dump(tfidf, f)
        with open(training_dir / "label_encoder.pkl", "wb") as f:
            pickle.dump(le, f)
        with open(training_dir / "logistic_regression_model.pkl", "wb") as f:
            pickle.dump(model, f)
        
        # Save metadata
        metadata = {
            "timestamp": self.timestamp,
            "training_date": datetime.now().isoformat(),
            "accuracy": float(accuracy),
            "model_type": "logistic_regression",
            "data_info": data_info
        }
        
        with open(training_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
            
        logger.info(f"Models saved to {training_dir}")
        return training_dir
        
    def run_training(self):
        """Main training pipeline"""
        try:
            if not self.check_for_new_data():
                logger.info("No new data found, skipping training")
                return None, None
                
            # Load and process data
            df = self.load_and_combine_data()
            df = self.preprocess_data(df)
            
            if len(df) < 50:
                raise ValueError(f"Insufficient data: {len(df)} records")
            
            data_info = {
                "total_records": len(df),
                "sentiment_distribution": df['Sentiment'].value_counts().to_dict()
            }
            
            # Train models
            tfidf, le, model, accuracy = self.train_models(df)
            
            if accuracy < 0.6:
                raise ValueError(f"Model accuracy too low: {accuracy:.4f}")
            
            # Save models
            model_path = self.save_models(tfidf, le, model, accuracy, data_info)
            
            return model_path, accuracy
            
        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
            raise

def main():
    trainer = ModelTrainer()
    try:
        model_path, accuracy = trainer.run_training()
        if model_path:
            print(f"✅ Training completed! Models: {model_path}, Accuracy: {accuracy:.4f}")
        else:
            print("ℹ️ No training needed")
        return 0
    except Exception as e:
        print(f"❌ Training failed: {str(e)}")
        return 1

if __name__ == "__main__":
    exit(main())
