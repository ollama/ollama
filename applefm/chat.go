package applefm

import (
	"bufio"
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"net/http"
	"strings"
	"time"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/llm"
)

// fm serve only knows this model name.
const fmModel = "system"

type chatMessage struct {
	Role    string `json:"role"`
	Content any    `json:"content"` // string, or []contentPart when there are images
}

type contentPart struct {
	Type     string    `json:"type"`
	Text     string    `json:"text,omitempty"`
	ImageURL *imageURL `json:"image_url,omitempty"`
}

type imageURL struct {
	URL string `json:"url"`
}

type responseFormat struct {
	Type       string     `json:"type"`
	JSONSchema jsonSchema `json:"json_schema"`
}

type jsonSchema struct {
	Name   string          `json:"name"`
	Schema json.RawMessage `json:"schema"`
}

type chatCompletionRequest struct {
	Model               string          `json:"model"`
	Messages            []chatMessage   `json:"messages"`
	Stream              bool            `json:"stream"`
	StreamOptions       map[string]bool `json:"stream_options,omitempty"`
	Temperature         *float32        `json:"temperature,omitempty"`
	TopP                *float32        `json:"top_p,omitempty"`
	TopK                *int            `json:"top_k,omitempty"`
	Seed                *int            `json:"seed,omitempty"`
	MaxCompletionTokens *int            `json:"max_completion_tokens,omitempty"`
	ResponseFormat      *responseFormat `json:"response_format,omitempty"`
}

type chatCompletionChunk struct {
	Choices []struct {
		Delta struct {
			Content string `json:"content"`
		} `json:"delta"`
		FinishReason string `json:"finish_reason"`
	} `json:"choices"`
	Usage *struct {
		PromptTokens     int `json:"prompt_tokens"`
		CompletionTokens int `json:"completion_tokens"`
	} `json:"usage"`
	Error *errorBody `json:"error"`
}

type errorBody struct {
	Message string `json:"message"`
}

func toChatMessages(msgs []api.Message) []chatMessage {
	out := make([]chatMessage, 0, len(msgs))
	for _, m := range msgs {
		if len(m.Images) == 0 {
			out = append(out, chatMessage{Role: m.Role, Content: m.Content})
			continue
		}
		parts := []contentPart{{Type: "text", Text: m.Content}}
		for _, img := range m.Images {
			url := "data:" + http.DetectContentType(img) + ";base64," + base64.StdEncoding.EncodeToString(img)
			parts = append(parts, contentPart{Type: "image_url", ImageURL: &imageURL{URL: url}})
		}
		out = append(out, chatMessage{Role: m.Role, Content: parts})
	}
	return out
}

func toResponseFormat(format json.RawMessage) (*responseFormat, error) {
	switch string(bytes.TrimSpace(format)) {
	case "", "null", `""`:
		return nil, nil
	case `"json"`:
		// Guided generation needs concrete properties, so an open-ended
		// object can't be expressed.
		return nil, api.StatusError{
			StatusCode:   http.StatusBadRequest,
			ErrorMessage: `format "json" is not supported by Apple Foundation Models, provide a JSON schema instead`,
		}
	}
	return &responseFormat{Type: "json_schema", JSONSchema: jsonSchema{Name: "response", Schema: format}}, nil
}

func newChatCompletionRequest(msgs []api.Message, format json.RawMessage, opts *api.Options) (*chatCompletionRequest, error) {
	rf, err := toResponseFormat(format)
	if err != nil {
		return nil, err
	}

	req := &chatCompletionRequest{
		Model:          fmModel,
		Messages:       toChatMessages(msgs),
		Stream:         true,
		StreamOptions:  map[string]bool{"include_usage": true},
		ResponseFormat: rf,
	}
	if opts != nil {
		req.Temperature = &opts.Temperature
		if opts.TopP > 0 {
			req.TopP = &opts.TopP
		}
		if opts.TopK > 0 {
			req.TopK = &opts.TopK
		}
		if opts.Seed >= 0 {
			req.Seed = &opts.Seed
		}
		if opts.NumPredict > 0 {
			req.MaxCompletionTokens = &opts.NumPredict
		}
		if len(opts.Stop) > 0 {
			slog.Warn("apple foundation models do not support stop sequences, ignoring", "stop", opts.Stop)
		}
	}
	return req, nil
}

// Chat implements llm.LlamaServer.
func (c *Client) Chat(ctx context.Context, req llm.ChatRequest, fn func(llm.ChatResponse)) error {
	creq, err := newChatCompletionRequest(req.Messages, req.Format, req.Options)
	if err != nil {
		return err
	}
	return c.stream(ctx, creq, fn)
}

// Completion implements llm.LlamaServer by sending the prompt as a single
// user message.
func (c *Client) Completion(ctx context.Context, req llm.CompletionRequest, fn func(llm.CompletionResponse)) error {
	msg := api.Message{Role: "user", Content: req.Prompt}
	for _, m := range req.Media {
		msg.Images = append(msg.Images, m.Data)
	}
	return llm.CompleteWithChat(c, []api.Message{msg})(ctx, req, fn)
}

func (c *Client) stream(ctx context.Context, creq *chatCompletionRequest, fn func(llm.ChatResponse)) error {
	if c.baseURL == "" {
		return errors.New("fm serve is not running")
	}

	body, err := json.Marshal(creq)
	if err != nil {
		return err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.baseURL+"/v1/chat/completions", bytes.NewReader(body))
	if err != nil {
		return err
	}
	req.Header.Set("Content-Type", "application/json")

	start := time.Now()
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return fmt.Errorf("fm serve request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		var e struct {
			Error errorBody `json:"error"`
		}
		if err := json.NewDecoder(resp.Body).Decode(&e); err != nil || e.Error.Message == "" {
			e.Error.Message = resp.Status
		}
		return api.StatusError{StatusCode: resp.StatusCode, ErrorMessage: e.Error.Message}
	}

	final := llm.ChatResponse{Done: true, DoneReason: llm.DoneReasonStop}
	var firstToken time.Time
	var finished bool

	scanner := bufio.NewScanner(resp.Body)
	scanner.Buffer(make([]byte, 0, 64*1024), 16*1024*1024)
	for scanner.Scan() {
		// Errors arrive as an `event: error` whose data carries an error
		// object, so only data lines need to be read.
		data, ok := strings.CutPrefix(scanner.Text(), "data:")
		if !ok {
			continue
		}
		data = strings.TrimSpace(data)
		if data == "[DONE]" {
			finished = true
			break
		}

		var chunk chatCompletionChunk
		if err := json.Unmarshal([]byte(data), &chunk); err != nil {
			return fmt.Errorf("fm serve: invalid stream chunk: %w", err)
		}
		if chunk.Error != nil {
			return errors.New(chunk.Error.Message)
		}

		if chunk.Usage != nil {
			final.PromptEvalCount = chunk.Usage.PromptTokens
			final.EvalCount = chunk.Usage.CompletionTokens
		}
		for _, choice := range chunk.Choices {
			if choice.FinishReason == "length" {
				final.DoneReason = llm.DoneReasonLength
			}
			if choice.Delta.Content == "" {
				continue
			}
			if firstToken.IsZero() {
				firstToken = time.Now()
			}
			fn(llm.ChatResponse{Message: api.Message{Role: "assistant", Content: choice.Delta.Content}})
		}
	}
	if err := scanner.Err(); err != nil {
		return err
	}
	if !finished {
		return errors.New("fm serve ended the response early")
	}

	if firstToken.IsZero() {
		firstToken = time.Now()
	}
	// fm reports "stop" when it reaches max_completion_tokens.
	if limit := creq.MaxCompletionTokens; limit != nil && final.EvalCount >= *limit {
		final.DoneReason = llm.DoneReasonLength
	}
	final.PromptEvalDuration = firstToken.Sub(start)
	final.EvalDuration = time.Since(firstToken)
	fn(final)
	return nil
}
