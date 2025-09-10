from fastapi import FastAPI, Request, Form, File, UploadFile, BackgroundTasks
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
import pandas as pd
import pickle
from typing import Dict, Optional
from pydantic import BaseModel
import numpy as np
from pathlib import Path
import logging
from datetime import datetime
import json
import os
from contextlib import asynccontextmanager

from app.utils import preprocess_text

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Global model variables
tfidf = None
le = None
model = None
model_metadata = {}

class ModelManager:
    def __init__(self):
        self.model_dir = Path("models/current")
        self.model_dir.mkdir(parents=True, exist_ok=True)
    
    def load_models(self):
        global tfidf, le, model, model_metadata
        
        try:
            # Try to load from current directory first
            if (self.model_dir / "tfidf_vectorizer.pkl").exists():
                with open(self.model_dir / "tfidf_vectorizer.pkl", "rb") as f:
                    tfidf = pickle.load(f)
                with open(self.model_dir / "label_encoder.pkl", "rb") as f:
                    le = pickle.load(f)
                with open(self.model_dir / "logistic_regression_model.pkl", "rb") as f:
                    model = pickle.load(f)
                    
                # Load metadata
                metadata_file = self.model_dir / "metadata.json"
                if metadata_file.exists():
                    with open(metadata_file, "r") as f:
                        model_metadata = json.load(f)
                        
                logger.info(f"Models loaded from {self.model_dir}")
            else:
                # Fallback to legacy models directory
                self._load_legacy_models()
                
        except Exception as e:
            logger.error(f"Failed to load models: {str(e)}")
            self._load_legacy_models()
    
    def _load_legacy_models(self):
        global tfidf, le, model, model_metadata
        
        try:
            legacy_dir = Path("models")
            with open(legacy_dir / "tfidf_vectorizer.pkl", "rb") as f:
                tfidf = pickle.load(f)
            with open(legacy_dir / "label_encoder.pkl", "rb") as f:
                le = pickle.load(f)
            with open(legacy_dir / "logistic_regression_model.pkl", "rb") as f:
                model = pickle.load(f)
                
            # Create initial metadata
            model_metadata = {
                "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
                "accuracy": 0.85,  # placeholder
                "model_type": "logistic_regression",
                "source": "legacy_models"
            }
            
            # Save to current directory for future use
            self.model_dir.mkdir(parents=True, exist_ok=True)
            for filename in ["tfidf_vectorizer.pkl", "label_encoder.pkl", "logistic_regression_model.pkl"]:
                src = legacy_dir / filename
                dst = self.model_dir / filename
                if src.exists():
                    import shutil
                    shutil.copy2(src, dst)
            
            with open(self.model_dir / "metadata.json", "w") as f:
                json.dump(model_metadata, f, indent=2)
                
            logger.info("✅ Legacy models migrated to current directory")
            
        except Exception as e:
            logger.error(f"❌ Failed to load legacy models: {str(e)}")
            raise
    
    def reload_models(self):
        logger.info("🔄 Reloading models...")
        self.load_models()
        return True

model_manager = ModelManager()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting Sentiment Analysis MLOps Service...")
    model_manager.load_models()
    yield
    # Shutdown
    logger.info("🛑 Shutting down...")

app = FastAPI(
    title="Sentiment Analysis MLOps",
    description="Automated sentiment analysis with model retraining",
    version="2.0.0",
    lifespan=lifespan
)

# Setup static files and templates
BASE_DIR = Path(__file__).resolve().parent.parent
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
templates = Jinja2Templates(directory=BASE_DIR / "templates")

# Pydantic models
class SentimentRequest(BaseModel):
    text: str

class SentimentResponse(BaseModel):
    sentiment: str
    confidence: float

class HealthResponse(BaseModel):
    status: str
    model_info: Dict
    timestamp: str

def predict_sentiment(text: str) -> SentimentResponse:
    if not tfidf or not le or not model:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    try:
        cleaned_text = preprocess_text(text)
        if not cleaned_text.strip():
            return SentimentResponse(sentiment="neutral", confidence=0.5)
        
        X = tfidf.transform([cleaned_text])
        proba = model.predict_proba(X)[0]
        pred = model.predict(X)[0]
        
        sentiment = le.inverse_transform([pred])[0]
        confidence = float(np.max(proba))
        
        return SentimentResponse(sentiment=sentiment, confidence=confidence)
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

# API Routes
@app.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status="healthy" if (tfidf and le and model) else "unhealthy",
        model_info=model_metadata,
        timestamp=datetime.now().isoformat()
    )

@app.post("/api/predict", response_model=SentimentResponse)
async def api_predict(request: SentimentRequest):
    return predict_sentiment(request.text)

@app.post("/api/reload-models")
async def reload_models():
    try:
        model_manager.reload_models()
        return {"status": "success", "message": "Models reloaded"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/trigger-retrain")
async def trigger_retrain(background_tasks: BackgroundTasks):
    def run_training():
        try:
            import subprocess
            result = subprocess.run(["python", "scripts/train_model.py"], capture_output=True, text=True)
            if result.returncode == 0:
                model_manager.reload_models()
                logger.info("✅ Manual retraining completed")
            else:
                logger.error(f"❌ Training failed: {result.stderr}")
        except Exception as e:
            logger.error(f"Training error: {str(e)}")
    
    background_tasks.add_task(run_training)
    return {"status": "Training started in background"}

# Web UI Routes
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {
        "request": request,
        "company_stats": {},
        "companies": [],
        "batch_result": False,
        "model_info": model_metadata
    })

@app.post("/analyze")
async def analyze_text(request: Request, text: str = Form(...)):
    result = predict_sentiment(text)
    return templates.TemplateResponse("index.html", {
        "request": request,
        "single_result": result,
        "input_text": text,
        "batch_result": False,
        "company_stats": {},
        "companies": [],
        "total_pos": 0,
        "total_neg": 0,
        "total_neu": 0,
        "sample_data": [],
        "model_info": model_metadata
    })

@app.post("/batch_analyze")
async def batch_analyze(request: Request, file: UploadFile = File(...)):
    try:
        df = pd.read_excel(file.file) if file.filename.endswith('.xlsx') else pd.read_csv(file.file)
        
        required_cols = ['Company Name', 'Opportunity Name', 'Remarks']
        if not all(col in df.columns for col in required_cols):
            return JSONResponse(
                status_code=400,
                content={"message": f"File must contain: {', '.join(required_cols)}"}
            )
        
        df['cleaned_text'] = df['Remarks'].apply(preprocess_text)
        X = tfidf.transform(df['cleaned_text'])
        proba = model.predict_proba(X)
        pred = model.predict(X)
        df['Sentiment'] = le.inverse_transform(pred)
        df['Confidence'] = np.max(proba, axis=1)
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path("data/processed")
        output_dir.mkdir(exist_ok=True)
        df.to_csv(output_dir / f"batch_results_{timestamp}.csv", index=False)
        
        company_stats = df.groupby('Company Name')['Sentiment'].value_counts().unstack().fillna(0)
        company_stats['Total'] = company_stats.sum(axis=1)
        
        total_pos = df[df['Sentiment'] == 'positive'].shape[0]
        total_neg = df[df['Sentiment'] == 'negative'].shape[0]
        total_neu = df[df['Sentiment'] == 'neutral'].shape[0]
        
        return templates.TemplateResponse("index.html", {
            "request": request,
            "batch_result": True,
            "total_pos": total_pos,
            "total_neg": total_neg,
            "total_neu": total_neu,
            "companies": company_stats.index.tolist(),
            "company_stats": company_stats.to_dict('index'),
            "sample_data": df.head(5).to_dict('records'),
            "model_info": model_metadata
        })
        
    except Exception as e:
        logger.error(f"Batch analysis error: {str(e)}")
        return JSONResponse(status_code=500, content={"message": f"Error: {str(e)}"})

if __name__ == "__main__":
    import uvicorn
    Path("logs").mkdir(exist_ok=True)
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=False)
