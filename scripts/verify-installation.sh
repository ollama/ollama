#!/bin/bash
# Ollama 安裝驗證腳本
# Ollama Installation Verification Script

echo "=================================="
echo "Ollama 安裝驗證"
echo "Ollama Installation Verification"
echo "=================================="
echo ""

# Color codes for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if Ollama is installed
echo "正在檢查 Ollama 是否已安裝..."
echo "Checking if Ollama is installed..."
if command -v ollama &> /dev/null; then
    echo -e "${GREEN}✓${NC} Ollama 已安裝 (Ollama is installed)"
    OLLAMA_VERSION=$(ollama --version 2>&1)
    echo "  版本 (Version): $OLLAMA_VERSION"
else
    echo -e "${RED}✗${NC} Ollama 未安裝 (Ollama is not installed)"
    echo ""
    echo "請訪問以下鏈接下載安裝："
    echo "Please visit the following links to download and install:"
    echo "  macOS: https://ollama.com/download/Ollama.dmg"
    echo "  Windows: https://ollama.com/download/OllamaSetup.exe"
    echo "  Linux: curl -fsSL https://ollama.com/install.sh | sh"
    exit 1
fi

echo ""

# Check if Ollama service is running
echo "正在檢查 Ollama 服務狀態..."
echo "Checking Ollama service status..."
if curl -s http://localhost:11434/api/version > /dev/null 2>&1; then
    echo -e "${GREEN}✓${NC} Ollama 服務正在運行 (Ollama service is running)"
else
    echo -e "${YELLOW}⚠${NC} Ollama 服務未運行 (Ollama service is not running)"
    echo "  請在另一個終端運行 (Please run in another terminal): ollama serve"
fi

echo ""

# Check system resources
echo "正在檢查系統資源..."
echo "Checking system resources..."

# Check available RAM
if command -v free &> /dev/null; then
    TOTAL_RAM=$(free -g | awk '/^Mem:/{print $2}')
    echo "  可用記憶體 (Available RAM): ${TOTAL_RAM}GB"
    if [ "$TOTAL_RAM" -lt 8 ]; then
        echo -e "${YELLOW}  ⚠ 建議至少 8GB RAM 來運行 7B 模型${NC}"
        echo "    Recommended at least 8GB RAM for 7B models"
    else
        echo -e "${GREEN}  ✓ 記憶體充足${NC} (RAM is sufficient)"
    fi
elif command -v vm_stat &> /dev/null; then
    # macOS
    TOTAL_RAM=$(sysctl -n hw.memsize | awk '{print int($1/1024/1024/1024)}')
    echo "  可用記憶體 (Available RAM): ${TOTAL_RAM}GB"
    if [ "$TOTAL_RAM" -lt 8 ]; then
        echo -e "${YELLOW}  ⚠ 建議至少 8GB RAM 來運行 7B 模型${NC}"
        echo "    Recommended at least 8GB RAM for 7B models"
    else
        echo -e "${GREEN}  ✓ 記憶體充足${NC} (RAM is sufficient)"
    fi
fi

echo ""

# Check available disk space
echo "正在檢查磁碟空間..."
echo "Checking disk space..."
if command -v df &> /dev/null; then
    if [ -d "$HOME/.ollama" ]; then
        DISK_SPACE=$(df -h "$HOME/.ollama" | awk 'NR==2 {print $4}')
    else
        DISK_SPACE=$(df -h "$HOME" | awk 'NR==2 {print $4}')
    fi
    echo "  可用磁碟空間 (Available disk space): $DISK_SPACE"
fi

echo ""

# Check for GPU
echo "正在檢查 GPU 支援..."
echo "Checking GPU support..."

# Check for NVIDIA GPU
if command -v nvidia-smi &> /dev/null; then
    echo -e "${GREEN}✓${NC} 檢測到 NVIDIA GPU (NVIDIA GPU detected)"
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
    echo "  GPU: $GPU_NAME"
elif command -v rocm-smi &> /dev/null; then
    echo -e "${GREEN}✓${NC} 檢測到 AMD GPU (AMD GPU detected)"
else
    echo "  未檢測到專用 GPU，將使用 CPU 運行"
    echo "  No dedicated GPU detected, will use CPU"
fi

echo ""
echo "=================================="
echo "驗證完成 (Verification completed)"
echo "=================================="
echo ""

# Suggest next steps
if command -v ollama &> /dev/null && curl -s http://localhost:11434/api/version > /dev/null 2>&1; then
    echo "🎉 您的系統已準備好使用 Ollama！"
    echo "🎉 Your system is ready to use Ollama!"
    echo ""
    echo "下一步 (Next steps):"
    echo "  1. 運行您的第一個模型 (Run your first model):"
    echo "     ollama run gemma3:1b"
    echo ""
    echo "  2. 查看可用模型 (View available models):"
    echo "     https://ollama.com/library"
    echo ""
    echo "  3. 閱讀快速入門指南 (Read quickstart guide):"
    echo "     docs/zh-CN/quickstart.md"
else
    echo "請確保 Ollama 服務正在運行："
    echo "Please make sure Ollama service is running:"
    echo "  ollama serve"
fi

echo ""
