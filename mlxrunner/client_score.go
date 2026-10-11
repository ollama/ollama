package mlxrunner

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/llm"
)

var _ llm.Scorer = (*Client)(nil)

func (c *Client) Score(ctx context.Context, input llm.ScoreRequest) (llm.ScoreResponse, error) {
	var result llm.ScoreResponse
	data, err := json.Marshal(input)
	if err != nil {
		return result, err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, fmt.Sprintf("http://127.0.0.1:%d/v1/score", c.port), bytes.NewReader(data))
	if err != nil {
		return result, err
	}
	req.Header.Set("Content-Type", "application/json")
	res, err := c.client.Do(req)
	if err != nil {
		if errMsg := c.status.LastError(); errMsg != "" {
			return result, fmt.Errorf("mlx runner failed: %s", errMsg)
		}
		return result, err
	}
	defer res.Body.Close()
	if res.StatusCode != http.StatusOK {
		body, err := io.ReadAll(res.Body)
		if err != nil {
			return result, err
		}
		return result, api.StatusError{StatusCode: res.StatusCode, ErrorMessage: strings.TrimSpace(string(body))}
	}
	err = json.NewDecoder(res.Body).Decode(&result)
	return result, err
}
