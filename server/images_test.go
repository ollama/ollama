package server

import (
	"context"
	"testing"

	"github.com/ollama/ollama/api"
	"github.com/stretchr/testify/assert"
)

func TestPushModel(t *testing.T) {
	noOpProgress := func(resp api.ProgressResponse) {}

	tests := []struct {
		modelStr string
		regOpts  *registryOptions
		wantErr  string
	}{
		{
			modelStr: "http://example.com/namespace/repo:tag",
			regOpts:  &registryOptions{Insecure: false},
			wantErr:  "insecure protocol http",
		},
		{
			modelStr: "docker://Example/repo:tag",
			regOpts:  &registryOptions{},
			wantErr:  "namespace must be lowercase, but is Example",
		},
		{
			modelStr: "docker://example/Repo:tag",
			regOpts:  &registryOptions{},
			wantErr:  "model name must be lowercase, but is Repo",
		},
	}

	for _, tt := range tests {
		t.Run(tt.modelStr, func(t *testing.T) {
			err := PushModel(context.Background(), tt.modelStr, tt.regOpts, noOpProgress)

			assert.Error(t, err)
			assert.EqualError(t, err, tt.wantErr)
		})
	}
}
