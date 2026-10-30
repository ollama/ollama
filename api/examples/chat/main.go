package main

import (
	"context"
	"fmt"
	"log"

	"github.com/ollama/ollama/api"
)

func main() {
	client, err := api.ClientFromEnvironment()
	if err != nil {
		log.Fatal(err)
	}

	messages := []api.Message{
		{
			Role:    "system",
			Content: "Provide very brief, concise responses",
		},
		{
			Role:    "user",
			Content: "Name some unusual animals",
		},
		{
			Role:    "assistant",
			Content: "Monotreme, platypus, echidna",
		},
		{
			Role:    "user",
			Content: "which of these is the most dangerous?",
		},
	}

	ctx := context.Background()
	req := &api.ChatRequest{
		Model:    "llama3.2",
		Messages: messages,
	}

	respFunc := func(resp api.ChatResponse) error {
		fmt.Print(resp.Message.Content)
		return nil
	}

	err = client.Chat(ctx, req, respFunc)
	if err != nil {
		log.Fatal(err)
	}
}
