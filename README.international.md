# Ollama (Português)

Comece a construir com modelos abertos.

## Download
- macOS: `curl -fsSL https://ollama.com/install.sh | sh` ou [download manual](https://ollama.com/download/Ollama.dmg)
- Windows: `irm https://ollama.com/install.ps1 | iex` ou [download manual](https://ollama.com/download/OllamaSetup.exe)
- Linux: `curl -fsSL https://ollama.com/install.sh | sh` ([instalação manual](https://docs.ollama.com/linux#manual-install))
- Docker: imagem oficial `ollama/ollama` no Docker Hub
- Bibliotecas: [ollama-python](https://github.com/ollama/ollama-python), [ollama-js](https://github.com/ollama/ollama-js)
- Comunidade: [Discord](https://discord.gg/ollama), [𝕏](https://x.com/ollama), [Reddit](https://reddit.com/r/ollama)

## Primeiros passos
```
ollama
```
Para lançar integrações: `ollama launch claude`

Assistente de IA com OpenClaw: `ollama launch openclaw`

Conversar com um modelo: `ollama run gemma3`

Mais detalhes: [guia rápido](https://docs.ollama.com/quickstart)

## API REST
```shell
curl http://localhost:11434/api/chat -d '{
  "model": "gemma3",
  "messages": [{"role": "user", "content": "Why is the sky blue?"}],
  "stream": false
}'
```
Documentação completa: [API](https://docs.ollama.com/api)

## Documentação
- [Referência da CLI](https://docs.ollama.com/cli)
- [Importando modelos](https://docs.ollama.com/import)
- [Referência do Modelfile](https://docs.ollama.com/modelfile)

Para a lista completa de integrações da comunidade, consulte `README.md` (inglês).

---

# Ollama (한국어)

오픈 모델로 시작하세요.

## 다운로드
- macOS: `curl -fsSL https://ollama.com/install.sh | sh` 또는 [수동 다운로드](https://ollama.com/download/Ollama.dmg)
- Windows: `irm https://ollama.com/install.ps1 | iex` 또는 [수동 다운로드](https://ollama.com/download/OllamaSetup.exe)
- Linux: `curl -fsSL https://ollama.com/install.sh | sh` ([수동 설치](https://docs.ollama.com/linux#manual-install))
- Docker: 공식 이미지 `ollama/ollama` (Docker Hub)
- 라이브러리: [ollama-python](https://github.com/ollama/ollama-python), [ollama-js](https://github.com/ollama/ollama-js)
- 커뮤니티: [Discord](https://discord.gg/ollama), [𝕏](https://x.com/ollama), [Reddit](https://reddit.com/r/ollama)

## 시작하기
```
ollama
```
통합 실행: `ollama launch claude`

AI 비서(OpenClaw): `ollama launch openclaw`

모델과 대화: `ollama run gemma3`

자세히: [퀵스타트](https://docs.ollama.com/quickstart)

## REST API
```shell
curl http://localhost:11434/api/chat -d '{
  "model": "gemma3",
  "messages": [{"role": "user", "content": "Why is the sky blue?"}],
  "stream": false
}'
```
전체 문서: [API](https://docs.ollama.com/api)

## 문서
- [CLI 레퍼런스](https://docs.ollama.com/cli)
- [모델 불러오기](https://docs.ollama.com/import)
- [Modelfile 레퍼런스](https://docs.ollama.com/modelfile)

전체 커뮤니티 통합 목록은 영어 `README.md`를 참고하세요.

---

# Ollama (简体中文)

开始使用开源模型。

## 下载
- macOS：`curl -fsSL https://ollama.com/install.sh | sh` 或 [手动下载](https://ollama.com/download/Ollama.dmg)
- Windows：`irm https://ollama.com/install.ps1 | iex` 或 [手动下载](https://ollama.com/download/OllamaSetup.exe)
- Linux：`curl -fsSL https://ollama.com/install.sh | sh`（[手动安装](https://docs.ollama.com/linux#manual-install)）
- Docker：官方镜像 `ollama/ollama`（Docker Hub）
- 库：[ollama-python](https://github.com/ollama/ollama-python)、[ollama-js](https://github.com/ollama/ollama-js)
- 社区： [Discord](https://discord.gg/ollama)、[𝕏](https://x.com/ollama)、[Reddit](https://reddit.com/r/ollama)

## 快速开始
```
ollama
```
启动集成：`ollama launch claude`

AI 助手（OpenClaw）：`ollama launch openclaw`

与模型聊天：`ollama run gemma3`

更多细节： [快速开始](https://docs.ollama.com/quickstart)

## REST API
```shell
curl http://localhost:11434/api/chat -d '{
  "model": "gemma3",
  "messages": [{"role": "user", "content": "Why is the sky blue?"}],
  "stream": false
}'
```
完整文档： [API](https://docs.ollama.com/api)

## 文档
- [CLI 参考](https://docs.ollama.com/cli)
- [导入模型](https://docs.ollama.com/import)
- [Modelfile 参考](https://docs.ollama.com/modelfile)

完整的社区集成列表请查看英文版 `README.md`。
