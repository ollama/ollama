package main

import (
	"context"
	"fmt"
	"os"

	"github.com/ollama/ollama/api"
)

var readFile = os.ReadFile
var clientFromEnvironment = func() (APIClient, error) {
	return api.ClientFromEnvironment()
}

type APIClient interface {
	Generate(ctx context.Context, req *api.GenerateRequest, respFunc api.GenerateResponseFunc) error
}

func run(args []string) error {
	if len(args) <= 1 {
		return fmt.Errorf("usage: <image name>")
	}

	imgData, err := readFile(args[1])
	if err != nil {
		return err
	}

	client, err := clientFromEnvironment()
	if err != nil {
		return err
	}

	req := &api.GenerateRequest{
		Model:  "llava",
		Prompt: "describe this image",
		Images: []api.ImageData{imgData},
	}

	ctx := context.Background()
	respFunc := func(resp api.GenerateResponse) error {
		fmt.Print(resp.Response)
		return nil
	}

	err = client.Generate(ctx, req, respFunc)
	if err != nil {
		return err
	}
	fmt.Println()
	return nil
}

func main() {
	if err := run(os.Args); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}
