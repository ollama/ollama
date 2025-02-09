package main

import (
    "context"
    "fmt"
    "log"

    "github.com/ollama/ollama/api"
)

func runChat(ctx context.Context, client *api.Client) error {
    if client == nil {
        return fmt.Errorf("client cannot be nil")
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

    req := &api.ChatRequest{
        Model:    "llama3.2",
        Messages: messages,
    }

    respFunc := func(resp api.ChatResponse) error {
        fmt.Print(resp.Message.Content)
        return nil
    }

    return client.Chat(ctx, req, respFunc)
}

func main() {
    client, err := api.ClientFromEnvironment()
    if err != nil {
        log.Fatal(err)
    }

    ctx := context.Background()
    err = runChat(ctx, client)
    if err != nil {
        log.Fatal(err)
    }
}
