# Whisper Prototype

### To run
`make {/path/to/whisper.cpp/server}`

### Update routes.go
- replace `whisperServer` with path to server

## api/generate
### Request fields
    - "audio" (required): path to audio file
    - "whisper_model" (required): path to whisper model
    - "transcribe" (optional): if true, will transcribe and return the audio file
    - "prompt" (optional): if not null, passed in with the transcribed audio
