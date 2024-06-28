package main

import (
	"bufio"
	"context"
	"fmt"
	"log"
	"os"

	"github.com/ollama/ollama/api"
)

func main() {
	client, err := api.ClientFromEnvironment()
	if err != nil {
		log.Fatal(err)
	}

	messages := []api.Message{
		api.Message{
			Role:    "system",
			Content: "Provide very brief, concise responses",
		},
	}

	ctx := context.Background()
	req := &api.ChatRequest{
		Model:    "llama3",
		Messages: messages,
	}

	respString := ""
	respFunc := func(resp api.ChatResponse) error {
		respString += resp.Message.Content
		if resp.Done {
			fmt.Println("\t", respString)
			messages = append(messages, api.Message{
				Role:    "assistant",
				Content: respString,
			})
			respString = ""
		}
		return nil
	}

	scanner := bufio.NewScanner(os.Stdin)
	for scanner.Scan() {
		req.Messages = append(req.Messages, api.Message{
			Role:    "user",
			Content: scanner.Text(),
		})
		err = client.Chat(ctx, req, respFunc)
		if err != nil {
			log.Fatal(err)
		}
	}
}
