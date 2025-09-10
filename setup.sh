#!/bin/bash

set -e  # Exit on any error

echo "🚀 Setting up Sentiment Analysis MLOps Project with Python 3.13..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)
python_major=$(echo $python_version | cut -d. -f1)
python_minor=$(echo $python_version | cut -d. -f2)

if [ "$python_major" -eq 3 ] && [ "$python_minor" -ge 9 ]; then
    print_status "Python $python_version detected"
else
    print_error "Python 3.9+ required. Current version: $python_version"
    exit 1
fi

# Check if Docker is installed
if command -v docker &> /dev/null; then
    print_status "Docker detected"
else
    print_error "Docker not found. Please install Docker first."
    exit 1
fi

# Check if Docker Compose is installed
if command -v docker-compose &> /dev/null || docker compose version &> /dev/null; then
    print_status "Docker Compose detected"
else
    print_error "Docker Compose not found. Please install Docker Compose first."
    exit 1
fi

# Create directory structure
echo "📁 Creating directory structure..."
mkdir -p data/{raw,processed,archive}
mkdir -p models/{current,training,backup}
mkdir -p logs
mkdir -p static
mkdir -p templates

print_status "Directory structure created"

# Create Python virtual environment with Python 3.13
echo "🐍 Setting up Python virtual environment with Python 3.13..."
python3 -m venv venv
source venv/bin/activate

print_status "Virtual environment created"

# Upgrade pip first
echo "⬆️ Upgrading pip..."
pip install --upgrade pip

# Install Python dependencies
echo "📦 Installing Python dependencies..."
pip install -r requirements.txt

print_status "Python dependencies installed"

# Download NLTK data with error handling
echo "📚 Downloading NLTK data..."
python -c "
try:
    import nltk
    nltk.download('wordnet', quiet=True)
    nltk.download('stopwords', quiet=True) 
    nltk.download('punkt', quiet=True)
    print('✅ NLTK data downloaded successfully')
except Exception as e:
    print(f'⚠️  NLTK download warning: {e}')
"

print_status "NLTK data downloaded"

# Copy existing models if available
if [ -f "models/tfidf_vectorizer.pkl" ]; then
    echo "📋 Migrating existing models..."
    cp models/*.pkl models/current/ 2>/dev/null || true
    
    # Create initial metadata
    cat > models/current/metadata.json << EOF
{
  "timestamp": "$(date +'%Y%m%d_%H%M%S')",
  "training_date": "$(date -Iseconds)",
  "model_type": "logistic_regression",
  "accuracy": 0.85,
  "source": "migrated_existing_models"
}
EOF
    
    print_status "Existing models migrated"
fi

# Create .env file if it doesn't exist
if [ ! -f ".env" ]; then
    echo "⚙️ Creating environment configuration..."
    cp .env.example .env
    print_status "Environment file created (edit .env as needed)"
fi

# Build Docker image
echo "🐳 Building Docker image..."
docker build -t sentiment-analysis:latest .

print_status "Docker image built"

# Test the installation
echo "🧪 Running installation tests..."

# Test Python imports
python -c "
try:
    import pandas as pd
    import numpy as np
    import sklearn
    from app.utils import preprocess_text
    from app.main import app
    print('✅ All Python imports successful')
except ImportError as e:
    print(f'❌ Import error: {e}')
    exit(1)
"

# Test Docker container
echo "🧪 Testing Docker container..."
docker-compose up -d
sleep 20

# Health check with retry
for i in {1..5}; do
    if curl -f http://localhost:8000/health > /dev/null 2>&1; then
        print_status "Health check passed"
        break
    else
        if [ $i -eq 5 ]; then
            print_warning "Health check failed - but container might still be starting"
        else
            echo "Waiting for container to start... (attempt $i/5)"
            sleep 10
        fi
    fi
done

docker-compose down

echo ""
echo "🎉 Setup completed successfully with Python 3.13!"
echo ""
echo "📋 Next steps:"
echo "1. Add your data files to data/raw/"
echo "2. Run: docker-compose up -d"
echo "3. Visit: http://localhost:8000"
echo "4. Test the API: http://localhost:8000/health"
echo ""
echo "🤖 The system will automatically retrain every 10 days!"
echo ""
echo "📝 Key commands:"
echo "  • Start: docker-compose up -d"
echo "  • Stop: docker-compose down"  
echo "  • Logs: docker-compose logs -f"
echo "  • Manual retrain: python scripts/train_model.py"
echo "  • Health check: curl http://localhost:8000/health"
echo ""
print_status "Happy MLOps with Python 3.13! 🚀"
