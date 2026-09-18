# Ollama 中文文檔

歡迎使用 Ollama！這裡是 Ollama 的中文文檔資源。

## 📚 文檔導航

### 🚀 快速開始
- **[快速入門指南](./quickstart.md)** - 5 分鐘快速上手 Ollama
- **[安裝指南](./install.md)** - 詳細的安裝步驟和故障排除
- **[測試指南](./testing.md)** - 完整的安裝測試流程

### 📖 其他文檔資源
- [故障排除指南](./troubleshooting.md) - 常見問題解決方案
- [SSH 金鑰設置指南](./ssh-setup.md) - 配置 SSH 金鑰以訪問 Git 存儲庫
- [API 文檔](../api.md)（英文）
- [Modelfile 文檔](../modelfile.md)（英文）
- [開發指南](../development.md)（英文）

## 🎯 推薦閱讀路徑

### 新手用戶
1. 先閱讀 [安裝指南](./install.md) 完成安裝
2. 然後按照 [快速入門指南](./quickstart.md) 運行第一個模型
3. 探索 [模型庫](https://ollama.com/library) 尋找適合的模型

### 開發者
1. 閱讀 [快速入門指南](./quickstart.md) 了解基本用法
2. 查看 [API 文檔](../api.md) 學習如何集成
3. 參考 [開發指南](../development.md) 了解如何從源碼構建

## 💡 常見使用場景

### 1. 本地 AI 助手
```bash
ollama run gemma3
>>> 你好！我需要幫助...
```

### 2. 代碼開發助手
```bash
ollama run codellama
>>> 請幫我寫一個 Python 函數來計算斐波那契數列
```

### 3. 文檔處理
```bash
ollama run gemma3 "總結這個文件：$(cat document.txt)"
```

### 4. 圖像理解
```bash
ollama run llava "描述這張圖片：/path/to/image.jpg"
```

## 🌟 熱門模型推薦

| 模型 | 大小 | 適用場景 | 命令 |
|-----|------|---------|------|
| Gemma 3 (1B) | 815MB | 輕量級對話 | `ollama run gemma3:1b` |
| Gemma 3 (4B) | 3.3GB | 通用對話 | `ollama run gemma3` |
| Llama 3.2 | 2.0GB | 平衡性能 | `ollama run llama3.2` |
| CodeLlama | 3.8GB | 編程助手 | `ollama run codellama` |
| QwQ | 20GB | 複雜推理 | `ollama run qwq` |
| DeepSeek-R1 | 4.7GB | 代碼生成 | `ollama run deepseek-r1` |

## 🔧 快速安裝

### macOS
```bash
# 下載 DMG 檔案並安裝
open https://ollama.com/download/Ollama.dmg
```

### Windows
```powershell
# 下載並運行安裝程式
start https://ollama.com/download/OllamaSetup.exe
```

### Linux
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

### Docker
```bash
docker run -d -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama
```

## 🆘 需要幫助？

如果您遇到任何問題：

1. 查看 [安裝指南](./install.md) 中的故障排除章節
2. 搜索 [GitHub Issues](https://github.com/ollama/ollama/issues)
3. 加入 [Discord 社群](https://discord.gg/ollama) 尋求幫助
4. 訪問 [Reddit 社群](https://reddit.com/r/ollama)

## 📝 貢獻

歡迎為中文文檔做出貢獻！如果您發現任何問題或有改進建議：

1. 訪問 [GitHub 儲存庫](https://github.com/ollama/ollama)
2. 提交 Issue 或 Pull Request
3. 幫助改進文檔翻譯和內容

## 🔗 相關連結

- **官方網站**：https://ollama.com
- **GitHub**：https://github.com/ollama/ollama
- **模型庫**：https://ollama.com/library
- **Discord**：https://discord.gg/ollama
- **Reddit**：https://reddit.com/r/ollama

## 📄 授權

Ollama 採用 MIT 授權。詳見 [LICENSE](../../LICENSE) 檔案。

---

最後更新：2025-12-30
