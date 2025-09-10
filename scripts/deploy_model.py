import shutil
import logging
from pathlib import Path
from datetime import datetime
import json
import sys

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelDeployer:
    def __init__(self):
        self.model_dir = Path("models")
        
    def deploy_model(self, source_path):
        """Deploy trained model to production"""
        source_path = Path(source_path)
        current_dir = self.model_dir / "current"
        backup_dir = self.model_dir / "backup"
        
        # Backup current models if they exist
        if current_dir.exists() and any(current_dir.iterdir()):
            backup_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = backup_dir / backup_timestamp
            backup_path.mkdir(parents=True, exist_ok=True)
            
            for file in current_dir.iterdir():
                if file.is_file():
                    shutil.copy2(file, backup_path / file.name)
            logger.info(f"Backed up current models to {backup_path}")
        
        # Deploy new models
        current_dir.mkdir(exist_ok=True)
        for filename in ["tfidf_vectorizer.pkl", "label_encoder.pkl", "logistic_regression_model.pkl", "metadata.json"]:
            src_file = source_path / filename
            if src_file.exists():
                shutil.copy2(src_file, current_dir / filename)
        
        # Update deployment metadata
        metadata_file = current_dir / "metadata.json"
        if metadata_file.exists():
            with open(metadata_file, "r") as f:
                metadata = json.load(f)
            metadata["deployment_timestamp"] = datetime.now().isoformat()
            with open(metadata_file, "w") as f:
                json.dump(metadata, f, indent=2)
        
        logger.info(f"✅ Model deployed from {source_path}")
        return current_dir

def main():
    if len(sys.argv) != 2:
        print("Usage: python deploy_model.py <source_model_path>")
        return 1
        
    deployer = ModelDeployer()
    try:
        deployed_path = deployer.deploy_model(sys.argv[1])
        print(f"✅ Model deployed successfully to {deployed_path}")
        return 0
    except Exception as e:
        print(f"❌ Deployment failed: {str(e)}")
        return 1

if __name__ == "__main__":
    exit(main())
