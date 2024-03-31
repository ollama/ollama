package ollama

import (
	"bytes"
	"cmp"
	"context"
	"encoding/json"
	"io"
	"io/fs"
	"iter"
	"net/http"
	"os"
	"strings"

	"bllamo.com/client/ollama/apitype"
	"bllamo.com/types/empty"
)

// TODO(bmizerany): PROGRESS INDICATORS!!!!

const DefaultBaseURL = "http://localhost:11434"

var envBaseURL = cmp.Or(os.Getenv("OLLAMA_BASE_URL"), DefaultBaseURL)

// Default returns a new client with the default base URL.
func Default() *Client {
	return &Client{BaseURL: envBaseURL}
}

// I_Acknowledge_This_API_Is_Under_Development is a flag that must be set to
// true for any instance of Client to work.
var I_Acknowledge_This_API_Is_Under_Development bool

// Client is a client for the Ollama API.
type Client struct {
	// BaseURL is the base URL of the Ollama API.
	BaseURL string
}

// Build requests the remote Ollama service to build a model. It uploads any
// source files the server needs.
func (c *Client) Build(ctx context.Context, ref string, modelfile []byte, source fs.FS) error {
	panic("not implemented")
}

// Push requests the remote Ollama service to push a model to the server.
func (c *Client) Push(ctx context.Context, ref string) error {
	_, err := Do[empty.Message](ctx, "POST", c.BaseURL+"/v1/push", apitype.PushRequest{Name: ref})
	return err
}

func (c *Client) Pull(ctx context.Context, ref string) error {
	panic("not implemented")
}

func (c *Client) List(ctx context.Context) iter.Seq2[apitype.Model, error] {
	panic("not implemented")
}

func (c *Client) Show(ctx context.Context, ref string) (*apitype.Model, error) {
	panic("not implemented")
}

func (c *Client) Remove(ctx context.Context, ref string) error {
	panic("not implemented")
}

func (c *Client) Copy(ctx context.Context, dstRef, srcRef string) error {
	panic("not implemented")
}

func (c *Client) Run(ctx context.Context, ref string, messages []apitype.Message) error {
	panic("not implemented")
}

type Error struct {
	Status  int    `json:"-"`
	Code    string `json:"code"`
	Message string `json:"message"`
	Field   string `json:"field,omitempty"`
	RawBody []byte `json:"-"`
}

func (e *Error) Error() string {
	var b strings.Builder
	b.WriteString("ollama: ")
	b.WriteString(e.Code)
	if e.Message != "" {
		b.WriteString(": ")
		b.WriteString(e.Message)
	}
	return b.String()
}

func Do[Res any](ctx context.Context, method, urlStr string, in any) (*Res, error) {
	var body bytes.Buffer
	// TODO(bmizerany): pool and reuse this buffer AND the encoder
	if err := encodeJSON(&body, in); err != nil {
		return nil, err
	}
	req, err := http.NewRequestWithContext(ctx, method, urlStr, &body)
	if err != nil {
		return nil, err
	}

	res, err := http.DefaultClient.Do(req)
	if err != nil {
		return nil, err
	}
	defer res.Body.Close()

	if res.StatusCode/100 != 2 {
		var b bytes.Buffer
		body := io.TeeReader(res.Body, &b)
		e, err := decodeJSON[Error](body)
		if err != nil {
			return nil, err
		}
		e.RawBody = b.Bytes()
		return nil, e
	}

	return decodeJSON[Res](res.Body)
}

// decodeJSON decodes JSON from r into a new value of type T.
//
// NOTE: This is (and encodeJSON) are copies and paste from oweb.go, please
// do not try and consolidate so we can keep ollama/client free from
// dependencies which are moving targets and not pulling enough weight to
// justify their inclusion.
func decodeJSON[T any](r io.Reader) (*T, error) {
	var v T
	if err := json.NewDecoder(r).Decode(&v); err != nil {
		return nil, err
	}
	return &v, nil
}

// NOTE: see NOT above decodeJSON
func encodeJSON(w io.Writer, v any) error {
	// TODO(bmizerany): pool and reuse encoder
	return json.NewEncoder(w).Encode(v)
}
