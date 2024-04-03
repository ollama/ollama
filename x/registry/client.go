package registry

import (
	"cmp"
	"context"
	"encoding/xml"
	"fmt"
	"io"
	"net/http"

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

func PushLayer(ctx context.Context, dstURL string, off, size int64, file io.ReaderAt) (etag string, err error) {
	sr := io.NewSectionReader(file, off, size)
	req, err := http.NewRequestWithContext(ctx, "PUT", dstURL, sr)
	if err != nil {
		return "", err
	}
	req.ContentLength = size

	res, err := http.DefaultClient.Do(req)
	if err != nil {
		return "", err
	}
	defer res.Body.Close()
	if res.StatusCode != 200 {
		return "", parseS3Error(res)
	}
	return res.Header.Get("ETag"), nil
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
