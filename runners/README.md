# `runners`

Ollama uses a subprocess model to run one or more child processes to load the LLM.  On some platforms (Linux non-containerized, MacOS) these executables are carried as payloads inside the main executable via the ../build package.  Extraction and discovery of these runners at runtime is implemented in this package.  This package also provides the abstraction to communicate with these subprocesses. 
