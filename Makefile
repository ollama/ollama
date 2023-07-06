default: ollama

.PHONY: llama
llama:
	cmake -S llama -B llama/build -DLLAMA_METAL=on
	cmake --build llama/build

.PHONY: ollama
ollama: llama
	go build .

.PHONY: app
app: ollama
	npm run --prefix app make

clean:
	go clean
	rm -rf llama/build
