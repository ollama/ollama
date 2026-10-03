package mlxrunner

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/mlx"
	"github.com/ollama/ollama/mlx/mlxthread"
)

func (c *Client) Extract(ctx context.Context, req api.ExtractRequest) (*api.ExtractResponse, error) {
	body, err := json.Marshal(req)
	if err != nil {
		return nil, err
	}
	r, err := http.NewRequestWithContext(ctx, http.MethodPost, fmt.Sprintf("http://127.0.0.1:%d/v1/extract", c.port), strings.NewReader(string(body)))
	if err != nil {
		return nil, err
	}
	r.Header.Set("Content-Type", "application/json")
	resp, err := c.client.Do(r)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
		return nil, api.StatusError{StatusCode: resp.StatusCode, ErrorMessage: strings.TrimSpace(string(body))}
	}
	var out api.ExtractResponse
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		return nil, err
	}
	return &out, nil
}

func (r *Runner) extractHandler(w http.ResponseWriter, req *http.Request) {
	if r.Extractor == nil {
		http.Error(w, "model does not support extraction", http.StatusBadRequest)
		return
	}
	var input api.ExtractRequest
	if err := json.NewDecoder(http.MaxBytesReader(w, req.Body, 2<<20)).Decode(&input); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	if err := input.Validate(); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	out, err := mlxthread.Call(req.Context(), r.mlxThread, func() (*api.ExtractResponse, error) {
		var out *api.ExtractResponse
		var err error
		mlx.Scoped(func() {
			out, err = r.Extractor.Extract(req.Context(), input)
		})
		return out, err
	})
	if err != nil {
		status := http.StatusInternalServerError
		var se api.StatusError
		if errors.As(err, &se) {
			status = se.StatusCode
		}
		http.Error(w, err.Error(), status)
		return
	}
	if err := json.NewEncoder(w).Encode(out); err != nil {
		return
	}
}
