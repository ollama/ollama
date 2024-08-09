# Speech to Text Prototype

### To run
`make {/path/to/whisper.cpp/server}`
- replace `whisperServer` in `routes.go` with path to server

## CLI
`./ollama run llama3 [PROMPT] --speech`
- processes voice audio with the provided prompt

`./ollama run llama3 --speech`
- enters interactive mode for continuous voice chat
- TODO: fix exiting interactive mode

Notes: uses default model


## api/generate
### Request fields
- `speech` (required):
    - `audio` (required): path to audio file
    - `model` (optional): path to whisper model, uses default if null
    - `transcribe` (optional): if true, will transcribe and return the audio file
    - `keep_alive`: (optional): sets how long the model is stored in memory (default: `5m`)
- `prompt` (optional): if not null, passed in with the transcribed audio

#### Transcription
```
curl http://localhost:11434/api/generate -d '{
    "speech": {
        "model": "/Users/royhan-ollama/.ollama/whisper/ggml-base.en.bin",
        "audio": "/Users/royhan-ollama/ollama/llm/whisper.cpp/samples/jfk.wav",
        "transcribe": true,
        "keep_alive": "1m"
    },
    "stream": false
}' | jq
```

#### Response Generation
```
curl http://localhost:11434/api/generate -d '{
    "model": "llama3",
    "prompt": "What do you think about this quote?",
    "speech": {
        "model": "/Users/royhan-ollama/.ollama/whisper/ggml-base.en.bin",
        "audio": "/Users/royhan-ollama/ollama/llm/whisper.cpp/samples/jfk.wav",
        "keep_alive": "1m"
    },
    "stream": false
}' | jq
```

## api/chat
### Request fields
- `model` (required): language model to chat with
- `speech` (optional):
    - `model` (optional): path to whisper model, uses default if null
    - `keep_alive`: (optional): sets how long the model is stored in memory (default: `5m`)
-  `run_speech` (optional): either this flag must be true or `speech` must be passed in for speech mode to run
- `messages`/`message`/`audio` (required): path to audio file

```
curl http://localhost:11434/api/chat -d '{
    "model": "llama3",
    "speech": {
        "model": "/Users/royhan-ollama/.ollama/whisper/ggml-base.en.bin",
        "keep_alive": "10m"
    },
    "messages": [
        {
            "role": "system",
            "content": "You are a Canadian Nationalist"
        },
        {
            "role": "user",
            "content": "What do you think about this quote?",
            "audio": "/Users/royhan-ollama/ollama/llm/whisper.cpp/samples/jfk.wav"
        }
    ],
    "stream": false
}' | jq
```