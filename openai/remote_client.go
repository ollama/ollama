package openai

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strings"

	"github.com/ollama/ollama/api"
)

// RemoteClient calls OpenAI-compatible chat/completions endpoints.
type RemoteClient struct {
	base    *url.URL
	apiKey  string
	headers map[string]string
	http    *http.Client
}

func NewRemoteClient(base *url.URL, apiKey string, headers map[string]string) *RemoteClient {
	var headersCopy map[string]string
	if headers != nil {
		headersCopy = make(map[string]string, len(headers))
		for k, v := range headers {
			headersCopy[k] = v
		}
	}

	return &RemoteClient{
		base:    base,
		apiKey:  apiKey,
		headers: headersCopy,
		http:    http.DefaultClient,
	}
}

func (c *RemoteClient) CreateChatCompletion(ctx context.Context, req ChatCompletionRequest) (ChatCompletion, error) {
	req.Stream = false
	resp, err := c.do(ctx, req, false)
	if err != nil {
		return ChatCompletion{}, err
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return ChatCompletion{}, err
	}
	if resp.StatusCode >= http.StatusBadRequest {
		return ChatCompletion{}, parseOpenAIError(resp.StatusCode, body)
	}

	var out ChatCompletion
	if err := json.Unmarshal(body, &out); err != nil {
		return ChatCompletion{}, err
	}
	return out, nil
}

// StreamChatCompletion streams chat completion chunks. fn is called for each chunk.
func (c *RemoteClient) StreamChatCompletion(ctx context.Context, req ChatCompletionRequest, fn func(ChatCompletionChunk) error) error {
	req.Stream = true
	resp, err := c.do(ctx, req, true)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	if resp.StatusCode >= http.StatusBadRequest {
		body, _ := io.ReadAll(resp.Body)
		return parseOpenAIError(resp.StatusCode, body)
	}

	scanner := bufio.NewScanner(resp.Body)
	buf := make([]byte, 0, 64*1024)
	scanner.Buffer(buf, 10*1024*1024)

	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" || strings.HasPrefix(line, "event:") {
			continue
		}
		if !strings.HasPrefix(line, "data:") {
			continue
		}

		data := strings.TrimSpace(strings.TrimPrefix(line, "data:"))
		if data == "" {
			continue
		}
		if data == "[DONE]" {
			return nil
		}

		var chunk ChatCompletionChunk
		if err := json.Unmarshal([]byte(data), &chunk); err != nil {
			return err
		}

		if err := fn(chunk); err != nil {
			return err
		}
	}

	if err := scanner.Err(); err != nil {
		return err
	}

	return nil
}

func (c *RemoteClient) do(ctx context.Context, req ChatCompletionRequest, stream bool) (*http.Response, error) {
	if c.base == nil {
		return nil, errors.New("base url is nil")
	}
	if c.apiKey == "" && (c.headers == nil || c.headers["Authorization"] == "") {
		return nil, errors.New("api key is required")
	}

	b, err := json.Marshal(req)
	if err != nil {
		return nil, err
	}

	endpoint := c.base.JoinPath("chat", "completions")
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint.String(), bytes.NewReader(b))
	if err != nil {
		return nil, err
	}

	httpReq.Header.Set("Content-Type", "application/json")
	if stream {
		httpReq.Header.Set("Accept", "text/event-stream")
	} else {
		httpReq.Header.Set("Accept", "application/json")
	}

	for k, v := range c.headers {
		if v != "" {
			httpReq.Header.Set(k, v)
		}
	}

	if httpReq.Header.Get("Authorization") == "" && c.apiKey != "" {
		httpReq.Header.Set("Authorization", "Bearer "+c.apiKey)
	}

	return c.http.Do(httpReq)
}

func parseOpenAIError(status int, body []byte) error {
	var resp ErrorResponse
	if err := json.Unmarshal(body, &resp); err == nil && resp.Error.Message != "" {
		return api.StatusError{StatusCode: status, ErrorMessage: resp.Error.Message}
	}

	msg := strings.TrimSpace(string(body))
	if msg == "" {
		msg = fmt.Sprintf("upstream error (%d)", status)
	}
	return api.StatusError{StatusCode: status, ErrorMessage: msg}
}
