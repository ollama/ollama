package llm

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"net/http"

	"github.com/ollama/ollama/api"
)

// SystemOne answers a /v1/systemone request with llama-server, which builds the
// prompt for the decision models llama.cpp supports. The state and questions
// pass through as given; images are sent as data URLs.
func (s *llamaServerRunner) SystemOne(ctx context.Context, state, questions json.RawMessage, images []api.ImageData) (answers, usage json.RawMessage, err error) {
	request := struct {
		State     json.RawMessage `json:"state"`
		Images    []string        `json:"images,omitempty"`
		Questions json.RawMessage `json:"questions"`
	}{State: state, Questions: questions}
	for _, image := range images {
		data, err := llamaServerMediaBytes(image)
		if err != nil {
			return nil, nil, api.StatusError{StatusCode: http.StatusBadRequest, ErrorMessage: err.Error()}
		}
		request.Images = append(request.Images, "data:"+http.DetectContentType(data)+";base64,"+base64.StdEncoding.EncodeToString(data))
	}
	data, err := json.Marshal(request)
	if err != nil {
		return nil, nil, err
	}

	if err := s.sem.Acquire(ctx, 1); err != nil {
		return nil, nil, err
	}
	defer s.sem.Release(1)

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, fmt.Sprintf("http://127.0.0.1:%d/v1/systemone", s.port), bytes.NewReader(data))
	if err != nil {
		return nil, nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	res, err := s.httpClient().Do(req)
	if err != nil {
		if ctx.Err() != nil {
			return nil, nil, ctx.Err()
		}
		if msg := s.lastErrMsg(); msg != "" {
			return nil, nil, fmt.Errorf("decision failed: %s: %w", msg, err)
		}
		return nil, nil, err
	}
	defer res.Body.Close()
	body, err := io.ReadAll(res.Body)
	if err != nil {
		return nil, nil, err
	}
	if res.StatusCode != http.StatusOK {
		return nil, nil, api.StatusError{StatusCode: res.StatusCode, ErrorMessage: s.statusErrorMessage(body)}
	}
	var response struct {
		Answers json.RawMessage `json:"answers"`
		Usage   json.RawMessage `json:"usage"`
	}
	if err := json.Unmarshal(body, &response); err != nil {
		return nil, nil, fmt.Errorf("invalid decision response: %w", err)
	}
	return response.Answers, response.Usage, nil
}
