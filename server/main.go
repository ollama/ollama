package main

import (
	"bytes"
	"context"
	"fmt"
	"io"
	"net/http"
	"os"
	"runtime"

	"github.com/sashabaranov/go-openai"

	llama "github.com/go-skynet/go-llama.cpp"
)


type Model interface {
	Name() string
	Handler(w http.ResponseWriter, r *http.Request)
}

type LLama7B struct {
	llama *llama.LLama
}

func NewLLama7B() *LLama7B {
	llama, err := llama.New("./models/7B/ggml-model-q4_0.bin", llama.EnableF16Memory, llama.SetContext(128), llama.EnableEmbeddings, llama.SetGPULayers(128))
	if err != nil {
		fmt.Println("Loading the model failed:", err.Error())
		os.Exit(1)
	}

	return &LLama7B{
		llama: llama,
	}
}

func (l *LLama7B) Name() string {
	return "LLaMA 7B"
}

func (m *LLama7B) Handler(w http.ResponseWriter, r *http.Request) {
	var text bytes.Buffer
	io.Copy(&text, r.Body)

	_, err := m.llama.Predict(text.String(), llama.Debug, llama.SetTokenCallback(func(token string) bool {
		w.Write([]byte(token))
		return true
	}), llama.SetTokens(512), llama.SetThreads(runtime.NumCPU()), llama.SetTopK(90), llama.SetTopP(0.86), llama.SetStopWords("llama"))

	if err != nil {
		fmt.Println("Predict failed:", err.Error())
		os.Exit(1)
	}

	embeds, err := m.llama.Embeddings(text.String())
	if err != nil {
		fmt.Printf("Embeddings: error %s \n", err.Error())
	}
	fmt.Printf("Embeddings: %v", embeds)

	w.Header().Set("Content-Type", "text/event-stream")
    w.Header().Set("Cache-Control", "no-cache")
    w.Header().Set("Connection", "keep-alive")
}

type GPT4 struct {
	apiKey string
}

func (g *GPT4) Name() string {
	return "OpenAI GPT-4"
}

func (g *GPT4) Handler(w http.ResponseWriter, r *http.Request) {
	w.WriteHeader(http.StatusOK)
	client := openai.NewClient("your token")
	resp, err := client.CreateChatCompletion(
		context.Background(),
		openai.ChatCompletionRequest{
			Model: openai.GPT3Dot5Turbo,
			Messages: []openai.ChatCompletionMessage{
				{
					Role:    openai.ChatMessageRoleUser,
					Content: "Hello!",
				},
			},
		},
	)
	if err != nil {
		fmt.Printf("chat completion error: %v\n", err)
		return
	}

	fmt.Println(resp.Choices[0].Message.Content)

	w.Header().Set("Content-Type", "text/plain; charset=utf-8")
	w.WriteHeader(http.StatusOK)
}

// TODO: add subcommands to spawn different models
func main() {
	model := &LLama7B{}
	
	http.HandleFunc("/generate", model.Handler)

	fmt.Println("Starting server on :8080")
	if err := http.ListenAndServe(":8080", nil); err != nil {
		fmt.Printf("Error starting server: %s\n", err)
		return
	}
}
