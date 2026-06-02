---
name: project-scope
description: Ollama is a popular open-source platform for running Large Language Models (LLMs) locally. The goal of this project is to use Ollama to enable open source models on IBM's frameworks. The benifit of using open-source LLMS locally is privacy and control. This skill binds the context of the code base to describe and summarize the repo to a developer so that the developer can better understand the project and its scope. Use when user asks about "project scope", "project achetecture", "project overview", or "project structure".
---
# Project Scope
## Instructions
### Step 1: Analyze Repository Structure
- Use [`list_files`](.) with recursive option to map the complete directory structure
- Identify key directories: [`cmd/`](cmd/), [`llm/`](llm/), [`server/`](server/), [`convert/`](convert/), [`api/`](api/)
- Understand the organization pattern (Go modules, build systems, documentation)

### Step 2: Read Core Documentation
- Read [`CONTRIBUTING.md`](CONTRIBUTING.md) for development guidelines
- Read [`docs/development.md`](docs/development.md) for setup instructions
- Review [`README.md`](README.md) if present for project overview
- Check [`docs/`](docs/) directory for architecture documentation

### Step 3: Identify Entry Points and Main Components
- Examine [`cmd/`](cmd/) directory for CLI entry points
- Use [`list_code_definition_names`](cmd/) to identify main functions and commands
- Review [`server/`](server/) for API server implementation
- Check [`api/`](api/) for API definitions

### Step 4: Understand Model Conversion Pipeline
- Analyze [`convert/`](convert/) directory structure
- Review conversion implementations for different model formats (Llama, Gemma, Mistral, etc.)
- Understand how models are imported and processed

### Step 5: Map LLM Runtime Components
- Explore [`llm/`](llm/) directory for inference engine
- Check [`llama/`](llama/) for LLaMA-specific implementations
- Review [`runner/`](runner/) for model execution logic
- Examine [`kvcache/`](kvcache/) for caching mechanisms

### Step 6: Identify Platform-Specific Code
- Search for IBM Z / s390x specific implementations using [`search_files`](.)
- Look for architecture-specific build configurations in [`cmake/`](cmake/)
- Check [`CMakePresets.json`](CMakePresets.json) for build presets
- Review [`scripts/`](scripts/) for platform-specific build scripts

### Step 7: Understand API Compatibility Layers
- Review [`middleware/`](middleware/) for API compatibility (OpenAI, Anthropic)
- Check [`openai/`](openai/) and [`anthropic/`](anthropic/) directories
- Read [`docs/api/`](docs/api/) for API documentation

### Step 8: Generate Comprehensive Summary
- Create a Mermaid architecture diagram showing:
  - Main components and their relationships
  - Data flow from API → Server → LLM Runtime → Model
  - Build system and dependencies
- Summarize:
  - Project purpose and goals
  - Key technologies (Go, CMake, LLaMA.cpp)
  - Supported model formats
  - API compatibility layers
  - Platform-specific considerations for IBM Z/LinuxOne
  - Development workflow and contribution process
