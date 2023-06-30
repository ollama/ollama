FROM python:3-slim-bullseye
RUN apt-get update && apt-get install -y build-essential gcc ninja-build libopenblas-dev
RUN python -m pip install --upgrade pip setuptools
RUN CMAKE_ARGS="-DLLAMA_OPENBLAS=on" FORCE_CMAKE=1 pip install ollama
ENTRYPOINT ["ollama"]
