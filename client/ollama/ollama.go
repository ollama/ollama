package ollama

import (
	"cmp"
	"context"
	"io/fs"
	"iter"
	"os"

	"bllamo.com/client/ollama/apitype"
	"bllamo.com/oweb"
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
	_, err := oweb.Do[empty.Message](ctx, "POST", c.BaseURL+"/v1/push", apitype.PushRequest{Name: ref})
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
