package registry

import (
	"context"
	"io"
	"net/http"

	"bllamo.com/client/ollama"
)

type Client struct {
	BaseURL    string
	HTTPClient *http.Client
}

func (c *Client) oclient() *ollama.Client {
	return (*ollama.Client)(c)
}

// Push pushes a manifest to the server.
func (c *Client) Push(ctx context.Context, ref string, manifest []byte) ([]Requirement, error) {
	// TODO(bmizerany): backoff
	v, err := ollama.Do[PushResponse](ctx, c.oclient(), "POST", "/v1/push", &PushRequest{
		Ref:      ref,
		Manifest: manifest,
	})
	if err != nil {
		return nil, err
	}
	return v.Requirements, nil
}

func PushLayer(ctx context.Context, dstURL string, size int64, file io.Reader) error {
	req, err := http.NewRequest("PUT", dstURL, file)
	if err != nil {
		return err
	}
	req.ContentLength = size

	res, err := http.DefaultClient.Do(req)
	if err != nil {
		return err
	}
	defer res.Body.Close()
	if res.StatusCode != 200 {
		e := &ollama.Error{Status: res.StatusCode}
		msg, err := io.ReadAll(res.Body)
		if err != nil {
			return err
		}
		// TODO(bmizerany): format error message
		e.Message = string(msg)
	}
	return nil
}
