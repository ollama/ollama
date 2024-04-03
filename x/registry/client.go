package registry

import (
	"cmp"
	"context"
	"encoding/xml"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"

	"bllamo.com/client/ollama"
	"bllamo.com/registry/apitype"
)

type Client struct {
	BaseURL    string
	HTTPClient *http.Client
}

func (c *Client) oclient() *ollama.Client {
	return (*ollama.Client)(c)
}

type PushParams struct {
	Uploaded []apitype.CompletePart
}

// Push pushes a manifest to the server.
func (c *Client) Push(ctx context.Context, ref string, manifest []byte, p *PushParams) ([]apitype.Requirement, error) {
	p = cmp.Or(p, &PushParams{})
	// TODO(bmizerany): backoff
	v, err := ollama.Do[apitype.PushResponse](ctx, c.oclient(), "POST", "/v1/push", &apitype.PushRequest{
		Ref:      ref,
		Manifest: manifest,
		Uploaded: p.Uploaded,
	})
	if err != nil {
		return nil, err
	}
	return v.Requirements, nil
}

func PushLayer(ctx context.Context, body io.ReaderAt, url string, off, n int64) (apitype.CompletePart, error) {
	var zero apitype.CompletePart
	if off < 0 {
		return zero, errors.New("off must be >0")
	}

	file := io.NewSectionReader(body, off, n)
	req, err := http.NewRequest("PUT", url, file)
	if err != nil {
		return zero, err
	}
	req.ContentLength = n

	// TODO(bmizerany): take content type param
	req.Header.Set("Content-Type", "text/plain")

	if n >= 0 {
		req.Header.Set("x-amz-copy-source-range", fmt.Sprintf("bytes=%d-%d", off, off+n-1))
	}

	res, err := http.DefaultClient.Do(req)
	if err != nil {
		return zero, err
	}
	defer res.Body.Close()
	if res.StatusCode != 200 {
		e := parseS3Error(res)
		return zero, fmt.Errorf("unexpected status code: %d; %w", res.StatusCode, e)
	}
	etag := strings.Trim(res.Header.Get("ETag"), `"`)
	cp := apitype.CompletePart{
		URL:  url,
		ETag: etag,
		// TODO(bmizerany): checksum
	}
	return cp, nil
}

type s3Error struct {
	XMLName   xml.Name `xml:"Error"`
	Code      string   `xml:"Code"`
	Message   string   `xml:"Message"`
	Resource  string   `xml:"Resource"`
	RequestId string   `xml:"RequestId"`
}

func (e *s3Error) Error() string {
	return fmt.Sprintf("S3 (%s): %s: %s: %s", e.RequestId, e.Resource, e.Code, e.Message)
}

// parseS3Error parses an XML error response from S3.
func parseS3Error(res *http.Response) error {
	var se *s3Error
	if err := xml.NewDecoder(res.Body).Decode(&se); err != nil {
		return err
	}
	return se
}
