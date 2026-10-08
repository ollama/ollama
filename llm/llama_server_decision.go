package llm

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/decision"
)

// SystemOne delegates template rendering and candidate readout to llama.cpp.
func (s *llamaServerRunner) SystemOne(ctx context.Context, input decision.Request) (decision.Response, error) {
	var result decision.Response
	if err := s.sem.Acquire(ctx, 1); err != nil {
		return result, err
	}
	defer s.sem.Release(1)
	status, err := s.getServerStatusRetry(ctx)
	if err != nil {
		return result, err
	}
	if status != ServerStatusReady {
		return result, fmt.Errorf("unexpected server status: %s", status)
	}
	request := struct {
		State     json.RawMessage     `json:"state"`
		Questions *decision.Questions `json:"questions"`
		Images    []string            `json:"images,omitempty"`
	}{State: input.State, Questions: input.Questions}
	for _, image := range input.Images {
		data, err := llamaServerMediaBytes(image)
		if err != nil {
			return result, api.StatusError{StatusCode: http.StatusBadRequest, ErrorMessage: err.Error()}
		}
		mime := http.DetectContentType(data)
		if !strings.HasPrefix(mime, "image/") {
			return result, api.StatusError{StatusCode: http.StatusBadRequest, ErrorMessage: "invalid decision image"}
		}
		request.Images = append(request.Images, "data:"+mime+";base64,"+base64.StdEncoding.EncodeToString(data))
	}
	if err := s.scoreRequest(ctx, "/v1/systemone", request, &result); err != nil {
		return result, err
	}
	result.Model = input.Model
	return result, nil
}
