#!/bin/bash
# H'uhul Multi Hive OS - Startup Script

set -e

echo "🛸 =========================================="
echo "   H'UHUL MULTI HIVE OS - STARTUP"
echo "========================================== 🛸"
echo ""

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if Ollama is installed
echo -e "${BLUE}[1/5]${NC} Checking Ollama installation..."
if ! command -v ollama &> /dev/null; then
    echo -e "${YELLOW}⚠️  Ollama not found. Please install it first:${NC}"
    echo "   curl -fsSL https://ollama.ai/install.sh | sh"
    exit 1
fi
echo -e "${GREEN}✅ Ollama found${NC}"
echo ""

# Check if Ollama is running
echo -e "${BLUE}[2/5]${NC} Checking Ollama service..."
if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo -e "${YELLOW}⚠️  Ollama not running. Starting it...${NC}"
    ollama serve > /dev/null 2>&1 &
    sleep 3
fi
echo -e "${GREEN}✅ Ollama is running${NC}"
echo ""

# Check for required models
echo -e "${BLUE}[3/5]${NC} Checking required models..."
MODELS=("qwen2.5:latest" "llama3.2:latest" "mistral:latest")

for model in "${MODELS[@]}"; do
    if ollama list | grep -q "$model"; then
        echo -e "${GREEN}   ✅ $model${NC}"
    else
        echo -e "${YELLOW}   ⚠️  $model not found. Pulling...${NC}"
        ollama pull "$model"
    fi
done
echo ""

# Check Python dependencies
echo -e "${BLUE}[4/5]${NC} Checking Python dependencies..."
cd backend
if [ -f "requirements.txt" ]; then
    pip install -q -r requirements.txt
    echo -e "${GREEN}✅ Python dependencies installed${NC}"
fi
echo ""

# Start the H'uhul Hive Server
echo -e "${BLUE}[5/5]${NC} Starting H'uhul Hive Server..."
echo ""
echo -e "${GREEN}🐝 =========================================="
echo "   H'UHUL HIVE IS NOW ONLINE"
echo "========================================== 🐝${NC}"
echo ""
echo -e "📡 Backend API: ${BLUE}http://localhost:8000${NC}"
echo -e "🌐 Frontend:    ${BLUE}Open frontend/index.html in your browser${NC}"
echo -e "📊 API Docs:    ${BLUE}http://localhost:8000/docs${NC}"
echo ""
echo -e "${YELLOW}Press Ctrl+C to stop the hive${NC}"
echo ""

python huhul_server.py
