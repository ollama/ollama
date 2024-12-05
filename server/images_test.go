package server

import (
	"context"
	"strings"
	"testing"

	"github.com/ollama/ollama/api"
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

			if tt.wantErr != "" {
				if err == nil {
					t.Errorf("PushModel() error = %v, wantErr %v", err, tt.wantErr)
				} else if !strings.Contains(err.Error(), tt.wantErr) {
					t.Errorf("PushModel() error = %v, wantErr %v", err, tt.wantErr)
				}
				return
			}
		})
	}
}
