package mlxrunner

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/llm"
)

var _ llm.Speaker = (*Client)(nil)

func (c *Client) Speech(ctx context.Context, input llm.SpeechRequest, fn func([]byte) error) error {
	data, err := json.Marshal(input)
	if err != nil {
		return err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, fmt.Sprintf("http://127.0.0.1:%d/v1/audio/speech", c.port), bytes.NewReader(data))
	if err != nil {
		return err
	}
	req.Header.Set("Content-Type", "application/json")
	res, err := c.client.Do(req)
	if err != nil {
		if c.status != nil {
			if errMsg := c.status.LastError(); errMsg != "" {
				return fmt.Errorf("mlx runner failed: %s", errMsg)
			}
		}
		return err
	}
	defer res.Body.Close()
	if res.StatusCode != http.StatusOK {
		body, err := io.ReadAll(res.Body)
		if err != nil {
			return err
		}
		return api.StatusError{StatusCode: res.StatusCode, ErrorMessage: strings.TrimSpace(string(body))}
	}

	buf := make([]byte, 32<<10)
	for {
		n, err := res.Body.Read(buf)
		if n > 0 {
			if err := fn(buf[:n]); err != nil {
				return err
			}
		}
		if errors.Is(err, io.EOF) {
			return nil
		} else if err != nil {
			return err
		}
	}
}
