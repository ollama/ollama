package openai

import (
	"encoding/json"
	"testing"

	"github.com/ollama/ollama/api"
)

func TestFromCompleteRequest_Temperature(t *testing.T) {
	tests := []struct {
		name    string
		body    string
		wantSet bool
		want    float32
	}{
		{name: "omitted", body: `{"model":"test-model","prompt":"Hello"}`, want: 0.25},
		{name: "null", body: `{"model":"test-model","prompt":"Hello","temperature":null}`, want: 0.25},
		{name: "zero", body: `{"model":"test-model","prompt":"Hello","temperature":0}`, wantSet: true, want: 0},
		{name: "nonzero", body: `{"model":"test-model","prompt":"Hello","temperature":0.5}`, wantSet: true, want: 0.5},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var req CompletionRequest
			if err := json.Unmarshal([]byte(tt.body), &req); err != nil {
				t.Fatal(err)
			}
			result, err := FromCompleteRequest(req)
			if err != nil {
				t.Fatal(err)
			}
			if value, set := result.Options["temperature"]; set != tt.wantSet {
				t.Errorf("temperature option = %v (present: %v), want present: %v", value, set, tt.wantSet)
			}

			// The middleware serializes the converted request before the server reads it.
			body, err := json.Marshal(result)
			if err != nil {
				t.Fatal(err)
			}
			var native api.GenerateRequest
			if err := json.Unmarshal(body, &native); err != nil {
				t.Fatal(err)
			}

			// Request options are applied after the model's options by the server.
			opts := api.DefaultOptions()
			if err := opts.FromMap(map[string]any{"temperature": 0.25}); err != nil {
				t.Fatal(err)
			}
			if err := opts.FromMap(native.Options); err != nil {
				t.Fatal(err)
			}
			if opts.Temperature != tt.want {
				t.Errorf("effective temperature = %v, want %v", opts.Temperature, tt.want)
			}
		})
	}
}
