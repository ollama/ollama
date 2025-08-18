# Running ollama CLI command examples

- go run . --help
- go run . serve
- go run . pull llama3.2

# Start Electron app (dev)

- go build .
- mkdir -p dist/darwin-amd64/lib/ollama && ([ -f go.mod ] && go build -o dist/darwin-amd64/lib/ollama/ollama ./cmd/ollama || true)
- cd macapp && npm install && npm start

# Testing API

- curl http://localhost:11434/v1/models

# Building CLI release

- go build -o dist/darwin/ollama .
- go build -o dist/darwin-amd64/lib/ollama .

# Building Mac Application

- cd macapp/ && npx electron-forge make --arch=arm64 (works)
- cd macapp/ && npx electron-forge package --arch=arm64 (?)

# TBD

osascript -e 'tell application "Ollama" to quit' || true
rm -rf /Applications/Ollama.app
cp -R out/Ollama-darwin-arm64/Ollama.app /Applications/
xattr -dr com.apple.quarantine /Applications/Ollama.app
open /Applications/Ollama.app
