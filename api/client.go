package api

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"

	"github.com/jmorganca/ollama/signature"
)

type Client struct {
	Name       string
	Version    string
	URL        string
	HTTP       http.Client
	Headers    http.Header
	PrivateKey []byte
}

func checkError(resp *http.Response, body []byte) error {
	if resp.StatusCode >= 200 && resp.StatusCode < 400 {
		return nil
	}

	apiError := Error{Code: int32(resp.StatusCode)}

	err := json.Unmarshal(body, &apiError)
	if err != nil {
		// Use the full body as the message if we fail to decode a response.
		apiError.Message = string(body)
	}

	return apiError
}

func (c *Client) do(ctx context.Context, method string, path string, stream bool, reqData any, respData any) error {
	var reqBody io.Reader
	var data []byte
	var err error
	if reqData != nil {
		data, err = json.Marshal(reqData)
		if err != nil {
			return err
		}
		reqBody = bytes.NewReader(data)
	}

	url := fmt.Sprintf("%s%s", c.URL, path)

	req, err := http.NewRequestWithContext(ctx, method, url, reqBody)
	if err != nil {
		return err
	}

	if c.PrivateKey != nil {
		s := signature.SignatureData{
			Method: method,
			Path:   url,
			Data:   data,
		}
		authHeader, err := signature.SignAuthData(s, c.PrivateKey)
		if err != nil {
			return err
		}
		req.Header.Set("Authorization", authHeader)
	}

	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "application/json")

	for k, v := range c.Headers {
		req.Header[k] = v
	}

	respObj, err := c.HTTP.Do(req)
	if err != nil {
		return err
	}
	defer respObj.Body.Close()

	respBody, err := io.ReadAll(respObj.Body)
	if err != nil {
		return err
	}

	if err := checkError(respObj, respBody); err != nil {
		return err
	}

	if len(respBody) > 0 && respData != nil {
		if err := json.Unmarshal(respBody, respData); err != nil {
			return err
		}
	}
	return nil
}
