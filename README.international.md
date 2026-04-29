<p align="center">
  <a href="https://ollama.com">
    <img src="https://github.com/ollama/ollama/assets/3325447/0d0b44e2-8f4a-4e99-9b52-a5c1c741c8f7" alt="ollama" width="200"/>
  </a>
</p>

# Ollama — International

Selecione seu idioma. Consulte o `README.md` original (inglês) para a lista completa de integrações da comunidade.

Sumário rápido (todas as línguas abaixo têm esta mesma estrutura):
- Instalação (macOS, Windows, Linux, Docker)
- Comandos básicos (`ollama`, `ollama launch`, `ollama run`)
- Exemplo de chamada à API REST
- Links úteis (CLI, Modelfile, import)

---

## Português (Brasil)

Comece a construir com modelos abertos.

### Download
- macOS: `curl -fsSL https://ollama.com/install.sh | sh` ou [download manual](https://ollama.com/download/Ollama.dmg)
- Windows: `irm https://ollama.com/install.ps1 | iex` ou [download manual](https://ollama.com/download/OllamaSetup.exe)
- Linux: `curl -fsSL https://ollama.com/install.sh | sh` ([instalação manual](https://docs.ollama.com/linux#manual-install))
- Docker: imagem oficial `ollama/ollama` no Docker Hub
- Bibliotecas: [ollama-python](https://github.com/ollama/ollama-python), [ollama-js](https://github.com/ollama/ollama-js)
- Comunidade: [Discord](https://discord.gg/ollama), [𝕏](https://x.com/ollama), [Reddit](https://reddit.com/r/ollama)

### Primeiros passos
```
ollama
```
- Lançar integração específica: `ollama launch claude`
- Assistente de IA (OpenClaw): `ollama launch openclaw`
- Conversar com modelo: `ollama run gemma3`
- Mais detalhes: [guia rápido](https://docs.ollama.com/quickstart)

### API REST (exemplo)
```shell
curl http://localhost:11434/api/chat -d '{
  "model": "gemma3",
  "messages": [{"role": "user", "content": "Why is the sky blue?"}],
  "stream": false
}'
```
Documentação completa: [API](https://docs.ollama.com/api)

### Documentação útil
- [Referência da CLI](https://docs.ollama.com/cli)
- [Importando modelos](https://docs.ollama.com/import)
- [Referência do Modelfile](https://docs.ollama.com/modelfile)

---

## 한국어 (Korean)

오픈 모델로 시작하세요.

### 다운로드
- macOS: `curl -fsSL https://ollama.com/install.sh | sh` 또는 [수동 다운로드](https://ollama.com/download/Ollama.dmg)
- Windows: `irm https://ollama.com/install.ps1 | iex` 또는 [수동 다운로드](https://ollama.com/download/OllamaSetup.exe)
- Linux: `curl -fsSL https://ollama.com/install.sh | sh` ([수동 설치](https://docs.ollama.com/linux#manual-install))
- Docker: 공식 이미지 `ollama/ollama` (Docker Hub)
- 라이브러리: [ollama-python](https://github.com/ollama/ollama-python), [ollama-js](https://github.com/ollama/ollama-js)
- 커뮤니티: [Discord](https://discord.gg/ollama), [𝕏](https://x.com/ollama), [Reddit](https://reddit.com/r/ollama)

### 시작하기
```
ollama
```
- 통합 실행: `ollama launch claude`
- AI 비서(OpenClaw): `ollama launch openclaw`
- 모델과 대화: `ollama run gemma3`
- 자세히: [퀵스타트](https://docs.ollama.com/quickstart)

### REST API (예시)
```shell
curl http://localhost:11434/api/chat -d '{
  "model": "gemma3",
  "messages": [{"role": "user", "content": "Why is the sky blue?"}],
  "stream": false
}'
```
전체 문서: [API](https://docs.ollama.com/api)

### 문서
- [CLI 레퍼런스](https://docs.ollama.com/cli)
- [모델 불러오기](https://docs.ollama.com/import)
- [Modelfile 레퍼런스](https://docs.ollama.com/modelfile)

---

## 简体中文 (Simplified Chinese)

开始使用开源模型。

### 下载
- macOS：`curl -fsSL https://ollama.com/install.sh | sh` 或 [手动下载](https://ollama.com/download/Ollama.dmg)
- Windows：`irm https://ollama.com/install.ps1 | iex` 或 [手动下载](https://ollama.com/download/OllamaSetup.exe)
- Linux：`curl -fsSL https://ollama.com/install.sh | sh`（[手动安装](https://docs.ollama.com/linux#manual-install)）
- Docker：官方镜像 `ollama/ollama`（Docker Hub）
- 库：[ollama-python](https://github.com/ollama/ollama-python)、[ollama-js](https://github.com/ollama/ollama-js)
- 社区： [Discord](https://discord.gg/ollama)、[𝕏](https://x.com/ollama)、[Reddit](https://reddit.com/r/ollama)

### 快速开始
```
ollama
```
- 启动集成：`ollama launch claude`
- AI 助手（OpenClaw）：`ollama launch openclaw`
- 与模型聊天：`ollama run gemma3`
- 更多细节： [快速开始](https://docs.ollama.com/quickstart)

### REST API（示例）
```shell
curl http://localhost:11434/api/chat -d '{
  "model": "gemma3",
  "messages": [{"role": "user", "content": "Why is the sky blue?"}],
  "stream": false
}'
```
完整文档： [API](https://docs.ollama.com/api)

### 文档
- [CLI 参考](https://docs.ollama.com/cli)
- [导入模型](https://docs.ollama.com/import)
- [Modelfile 参考](https://docs.ollama.com/modelfile)

---

## Italiano

Inizia a costruire con modelli open source.

### Download
- macOS: `curl -fsSL https://ollama.com/install.sh | sh` oppure [download manuale](https://ollama.com/download/Ollama.dmg)
- Windows: `irm https://ollama.com/install.ps1 | iex` oppure [download manuale](https://ollama.com/download/OllamaSetup.exe)
- Linux: `curl -fsSL https://ollama.com/install.sh | sh` ([installazione manuale](https://docs.ollama.com/linux#manual-install))
- Docker: immagine ufficiale `ollama/ollama` su Docker Hub
- Librerie: [ollama-python](https://github.com/ollama/ollama-python), [ollama-js](https://github.com/ollama/ollama-js)
- Community: [Discord](https://discord.gg/ollama), [𝕏](https://x.com/ollama), [Reddit](https://reddit.com/r/ollama)

### Primi passi
```
ollama
```
- Avviare un’integrazione: `ollama launch claude`
- Assistente IA (OpenClaw): `ollama launch openclaw`
- Chattare con un modello: `ollama run gemma3`
- Dettagli: [guida rapida](https://docs.ollama.com/quickstart)

### API REST (esempio)
```shell
curl http://localhost:11434/api/chat -d '{
  "model": "gemma3",
  "messages": [{"role": "user", "content": "Why is the sky blue?"}],
  "stream": false
}'
```
Documentazione completa: [API](https://docs.ollama.com/api)

### Documentazione utile
- [Riferimento CLI](https://docs.ollama.com/cli)
- [Importazione modelli](https://docs.ollama.com/import)
- [Riferimento Modelfile](https://docs.ollama.com/modelfile)

---

## 日本語 (Japanese)

オープンモデルで始めましょう。

### ダウンロード
- macOS: `curl -fsSL https://ollama.com/install.sh | sh` または [手動ダウンロード](https://ollama.com/download/Ollama.dmg)
- Windows: `irm https://ollama.com/install.ps1 | iex` または [手動ダウンロード](https://ollama.com/download/OllamaSetup.exe)
- Linux: `curl -fsSL https://ollama.com/install.sh | sh`（[手動インストール](https://docs.ollama.com/linux#manual-install)）
- Docker: 公式イメージ `ollama/ollama`（Docker Hub）
- ライブラリ: [ollama-python](https://github.com/ollama/ollama-python), [ollama-js](https://github.com/ollama/ollama-js)
- コミュニティ: [Discord](https://discord.gg/ollama), [𝕏](https://x.com/ollama), [Reddit](https://reddit.com/r/ollama)

### はじめに
```
ollama
```
- 特定の統合を起動: `ollama launch claude`
- AIアシスタント (OpenClaw): `ollama launch openclaw`
- モデルと対話: `ollama run gemma3`
- 詳細: [クイックスタート](https://docs.ollama.com/quickstart)

### REST API（例）
```shell
curl http://localhost:11434/api/chat -d '{
  "model": "gemma3",
  "messages": [{"role": "user", "content": "Why is the sky blue?"}],
  "stream": false
}'
```
詳細ドキュメント: [API](https://docs.ollama.com/api)

### ドキュメント
- [CLI リファレンス](https://docs.ollama.com/cli)
- [モデルのインポート](https://docs.ollama.com/import)
- [Modelfile リファレンス](https://docs.ollama.com/modelfile)

---

## ไทย (Thai)

เริ่มต้นสร้างด้วยโมเดลแบบเปิด

### ดาวน์โหลด
- macOS: `curl -fsSL https://ollama.com/install.sh | sh` หรือ [ดาวน์โหลดด้วยตนเอง](https://ollama.com/download/Ollama.dmg)
- Windows: `irm https://ollama.com/install.ps1 | iex` หรือ [ดาวน์โหลดด้วยตนเอง](https://ollama.com/download/OllamaSetup.exe)
- Linux: `curl -fsSL https://ollama.com/install.sh | sh` ([ติดตั้งด้วยตนเอง](https://docs.ollama.com/linux#manual-install))
- Docker: อิมเมจทางการ `ollama/ollama` บน Docker Hub
- ไลบรารี: [ollama-python](https://github.com/ollama/ollama-python), [ollama-js](https://github.com/ollama/ollama-js)
- ชุมชน: [Discord](https://discord.gg/ollama), [𝕏](https://x.com/ollama), [Reddit](https://reddit.com/r/ollama)

### เริ่มต้นใช้งาน
```
ollama
```
- เรียกใช้การผสานงาน: `ollama launch claude`
- ผู้ช่วย AI (OpenClaw): `ollama launch openclaw`
- แชทกับโมเดล: `ollama run gemma3`
- รายละเอียดเพิ่มเติม: [คู่มือเริ่มต้นด่วน](https://docs.ollama.com/quickstart)

### REST API (ตัวอย่าง)
```shell
curl http://localhost:11434/api/chat -d '{
  "model": "gemma3",
  "messages": [{"role": "user", "content": "Why is the sky blue?"}],
  "stream": false
}'
```
ดูเอกสารเต็ม: [API](https://docs.ollama.com/api)

### เอกสารประกอบ
- [CLI อ้างอิง](https://docs.ollama.com/cli)
- [การนำเข้ารุ่นโมเดล](https://docs.ollama.com/import)
- [เอกสาร Modelfile](https://docs.ollama.com/modelfile)

---

## Français (French)

Commencez à créer avec des modèles ouverts.

### Téléchargement
- macOS : `curl -fsSL https://ollama.com/install.sh | sh` ou [téléchargement manuel](https://ollama.com/download/Ollama.dmg)
- Windows : `irm https://ollama.com/install.ps1 | iex` ou [téléchargement manuel](https://ollama.com/download/OllamaSetup.exe)
- Linux : `curl -fsSL https://ollama.com/install.sh | sh` ([installation manuelle](https://docs.ollama.com/linux#manual-install))
- Docker : image officielle `ollama/ollama` sur Docker Hub
- Bibliothèques : [ollama-python](https://github.com/ollama/ollama-python), [ollama-js](https://github.com/ollama/ollama-js)
- Communauté : [Discord](https://discord.gg/ollama), [𝕏](https://x.com/ollama), [Reddit](https://reddit.com/r/ollama)

### Premiers pas
```
ollama
```
- Lancer une intégration : `ollama launch claude`
- Assistant IA (OpenClaw) : `ollama launch openclaw`
- Discuter avec un modèle : `ollama run gemma3`
- Plus de détails : [guide de démarrage](https://docs.ollama.com/quickstart)

### API REST (exemple)
```shell
curl http://localhost:11434/api/chat -d '{
  "model": "gemma3",
  "messages": [{"role": "user", "content": "Why is the sky blue?"}],
  "stream": false
}'
```
Documentation complète : [API](https://docs.ollama.com/api)

### Documentation utile
- [Référence CLI](https://docs.ollama.com/cli)
- [Import de modèles](https://docs.ollama.com/import)
- [Référence Modelfile](https://docs.ollama.com/modelfile)

---

## Español (Spanish)

Comienza a construir con modelos abiertos.

### Descarga
- macOS: `curl -fsSL https://ollama.com/install.sh | sh` o [descarga manual](https://ollama.com/download/Ollama.dmg)
- Windows: `irm https://ollama.com/install.ps1 | iex` o [descarga manual](https://ollama.com/download/OllamaSetup.exe)
- Linux: `curl -fsSL https://ollama.com/install.sh | sh` ([instalación manual](https://docs.ollama.com/linux#manual-install))
- Docker: imagen oficial `ollama/ollama` en Docker Hub
- Bibliotecas: [ollama-python](https://github.com/ollama/ollama-python), [ollama-js](https://github.com/ollama/ollama-js)
- Comunidad: [Discord](https://discord.gg/ollama), [𝕏](https://x.com/ollama), [Reddit](https://reddit.com/r/ollama)

### Primeros pasos
```
ollama
```
- Lanzar una integración: `ollama launch claude`
- Asistente de IA (OpenClaw): `ollama launch openclaw`
- Chatear con un modelo: `ollama run gemma3`
- Más detalles: [guía rápida](https://docs.ollama.com/quickstart)

### API REST (ejemplo)
```shell
curl http://localhost:11434/api/chat -d '{
  "model": "gemma3",
  "messages": [{"role": "user", "content": "Why is the sky blue?"}],
  "stream": false
}'
```
Documentación completa: [API](https://docs.ollama.com/api)

### Documentación útil
- [Referencia de CLI](https://docs.ollama.com/cli)
- [Importar modelos](https://docs.ollama.com/import)
- [Referencia de Modelfile](https://docs.ollama.com/modelfile)
