package main

import (
	"context"
	"encoding/json"
	"fmt"
	"github.com/ollama/ollama/api"
	"log"
	"time"
)

func main() {
	client, err := api.ClientFromEnvironment()
	if err != nil {
		log.Fatal(err)
	}

	messages := []api.Message{
		api.Message{
			Role:    "system",
			Content: "You give short answers with no extra detail.",
		},
		api.Message{
			Role:    "user",
			Content: "What time is it? I want to know the time in american format.",
		},
		api.Message{
			Role: "assistant",
			ToolCalls: []api.ToolCall{
				api.ToolCall{
					Function: api.ToolCallFunction{
						Name: "get_current_time",
						Arguments: map[string]any{
							"format": "12-hour",
						},
					},
				},
			},
		},
		api.Message{
			Role:    "tool",
			Content: "11:53AM",
		},
		api.Message{
			Role:    "assistant",
			Content: "The current time is 11:53 AM.",
		},
		api.Message{
			Role:    "user",
			Content: "What time is it in european format?",
		},
	}
	ctx := context.Background()
	stream := false
	req := &api.ChatRequest{
		Model:    "mistral",
		Messages: messages,
		Options: map[string]interface{}{
			"temperature": 0,
		},
		Stream: &stream,
		Tools: api.Tools{
			api.Tool{
				Type: "function",
				Function: api.ToolFunction{
					Name:        "get_current_time",
					Description: "Get the current time.",
					Parameters: struct {
						Type       string   `json:"type"`
						Required   []string `json:"required"`
						Properties map[string]struct {
							Type        string   `json:"type"`
							Description string   `json:"description"`
							Enum        []string `json:"enum,omitempty"`
						} `json:"properties"`
					}{
						Type:     "object",
						Required: []string{"format"},
						Properties: map[string]struct {
							Type        string   `json:"type"`
							Description string   `json:"description"`
							Enum        []string `json:"enum,omitempty"`
						}{
							"format": {
								Type: "string",
								Enum: []string{
									"12-hour",
									"24-hour",
								},
								Description: "The clock format to use. Either 12-hour or a 24-hour format.",
							},
						}},
				},
			},
		},
	}
	respFunc := func(resp api.ChatResponse) error {
		if len(resp.Message.ToolCalls) >= 0 && resp.Done {
			answer := ""
			for _, call := range resp.Message.ToolCalls {
				if call.Function.Name == "get_current_time" {
					jsonArguments := call.Function.Arguments.String()
					var args map[string]interface{}
					json.Unmarshal([]byte(jsonArguments), &args)
					clockFormat := args["format"]
					if clockFormat == "24-hour" {
						answer = time.Now().Format("15:04")
					} else {
						answer = time.Now().Format("03:04PM")
					}
				}
			}
			req.Messages = append(req.Messages, api.Message{
				Role:    "tool",
				Content: answer,
			})
			if err := client.Chat(ctx, req, func(response api.ChatResponse) error {
				fmt.Println(response.Message.Content)
				return nil
			}); err != nil {
				log.Fatal(err)
			}
		}
		return nil
	}

	err = client.Chat(ctx, req, respFunc)
	if err != nil {
		log.Fatal(err)
	}
}
