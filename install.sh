#!/bin/bash

python3 -m venv venv
source venv/bin/activate
brew install cmake go
brew install portaudio
pip install pyaudio
pip install SpeechRecognition
git clone https://github.com/jmorganca/ollama.git
cd ollama
go generate ./...
go build .
