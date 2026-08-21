package main

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/ollama/ollama/api"
)

func TestCloudChatClientUsesBearerAuthAndNDJSON(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if got := r.Header.Get("Authorization"); got != "Bearer test-key" {
			t.Errorf("authorization = %q", got)
		}
		if got := r.Header.Get("Accept"); got != "application/x-ndjson" {
			t.Errorf("accept = %q", got)
		}
		var request api.ChatRequest
		if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
			t.Fatal(err)
		}
		if request.Model != "glm-5.2" {
			t.Errorf("model = %q", request.Model)
		}
		_, _ = w.Write([]byte("{\"message\":{\"role\":\"assistant\",\"content\":\"ok\"},\"done\":true}\n"))
	}))
	defer server.Close()

	client := cloudChatClient{apiKey: "test-key", endpoint: server.URL}
	var got string
	if err := client.Chat(context.Background(), &api.ChatRequest{Model: reviewModelRemote}, func(part api.ChatResponse) error {
		got += part.Message.Content
		return nil
	}); err != nil {
		t.Fatal(err)
	}
	if got != "ok" {
		t.Fatalf("content = %q", got)
	}
}

func TestReadChatResponse(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/x-ndjson")
		_, _ = w.Write([]byte("{\"message\":{\"role\":\"assistant\",\"content\":\"ok\"},\"done\":true}\n"))
	}))
	defer server.Close()
	response, err := http.Get(server.URL)
	if err != nil {
		t.Fatal(err)
	}
	defer response.Body.Close()
	var got string
	if err := readChatResponse(response, func(part api.ChatResponse) error { got += part.Message.Content; return nil }); err != nil {
		t.Fatal(err)
	}
	if got != "ok" {
		t.Fatalf("content = %q", got)
	}
}
