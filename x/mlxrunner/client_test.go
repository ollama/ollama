package mlxrunner

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"strings"
	"testing"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/llm"
)

func TestCompletionForwardsThink(t *testing.T) {
	boolPtr := func(v bool) *bool { return &v }

	testCases := []struct {
		name  string
		think *api.ThinkValue
		want  *bool
	}{
		{name: "unset", think: nil, want: nil},
		{name: "enabled", think: &api.ThinkValue{Value: true}, want: boolPtr(true)},
		{name: "disabled", think: &api.ThinkValue{Value: false}, want: boolPtr(false)},
		{name: "level maps to enabled", think: &api.ThinkValue{Value: "high"}, want: boolPtr(true)},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			var got completionRequest

			rt := roundTripFunc(func(r *http.Request) (*http.Response, error) {
				if r.URL.Path != "/completion" {
					t.Fatalf("request path = %q, want %q", r.URL.Path, "/completion")
				}

				if err := json.NewDecoder(r.Body).Decode(&got); err != nil {
					return nil, err
				}

				return &http.Response{
					StatusCode: http.StatusOK,
					Header:     make(http.Header),
					Body:       io.NopCloser(strings.NewReader("{\"done\":true}\n")),
					Request:    r,
				}, nil
			})

			c := &Client{
				port: 11434,
				client: &http.Client{
					Transport: rt,
				},
			}

			err := c.Completion(context.Background(), llm.CompletionRequest{
				Prompt: "hello",
				Think:  tc.think,
			}, func(llm.CompletionResponse) {})
			if err != nil {
				t.Fatalf("completion request failed: %v", err)
			}

			if got.Prompt != "hello" {
				t.Fatalf("prompt = %q, want %q", got.Prompt, "hello")
			}

			switch {
			case tc.want == nil && got.Think != nil:
				t.Fatalf("think = %v, want nil", *got.Think)
			case tc.want != nil && got.Think == nil:
				t.Fatalf("think = nil, want %v", *tc.want)
			case tc.want != nil && got.Think != nil && *tc.want != *got.Think:
				t.Fatalf("think = %v, want %v", *got.Think, *tc.want)
			}
		})
	}
}

func TestCompletionForwardsOnlySpecifiedSamplingOptions(t *testing.T) {
	var got completionRequest

	rt := roundTripFunc(func(r *http.Request) (*http.Response, error) {
		if err := json.NewDecoder(r.Body).Decode(&got); err != nil {
			return nil, err
		}

		return &http.Response{
			StatusCode: http.StatusOK,
			Header:     make(http.Header),
			Body:       io.NopCloser(strings.NewReader("{\"done\":true}\n")),
			Request:    r,
		}, nil
	})

	c := &Client{
		port: 11434,
		client: &http.Client{
			Transport: rt,
		},
	}

	opts := &api.Options{
		Temperature:      1.0,
		TopP:             0.95,
		MinP:             0.1,
		TopK:             20,
		RepeatLastN:      128,
		RepeatPenalty:    1.2,
		PresencePenalty:  1.5,
		FrequencyPenalty: 0.25,
		NumPredict:       64,
	}

	err := c.Completion(context.Background(), llm.CompletionRequest{
		Prompt:  "hello",
		Options: opts,
		ExplicitOptions: map[string]struct{}{
			"temperature":      {},
			"top_k":            {},
			"repeat_penalty":   {},
			"presence_penalty": {},
		},
	}, func(llm.CompletionResponse) {})
	if err != nil {
		t.Fatalf("completion request failed: %v", err)
	}

	if got.Options == nil {
		t.Fatal("options = nil, want serialized options")
	}

	if got.Options.Temperature == nil || *got.Options.Temperature != opts.Temperature {
		t.Fatalf("temperature = %v, want %v", got.Options.Temperature, opts.Temperature)
	}
	if got.Options.TopK == nil || *got.Options.TopK != opts.TopK {
		t.Fatalf("top_k = %v, want %v", got.Options.TopK, opts.TopK)
	}
	if got.Options.RepeatPenalty == nil || *got.Options.RepeatPenalty != opts.RepeatPenalty {
		t.Fatalf("repeat_penalty = %v, want %v", got.Options.RepeatPenalty, opts.RepeatPenalty)
	}
	if got.Options.PresencePenalty == nil || *got.Options.PresencePenalty != opts.PresencePenalty {
		t.Fatalf("presence_penalty = %v, want %v", got.Options.PresencePenalty, opts.PresencePenalty)
	}
	if got.Options.TopP != nil {
		t.Fatalf("top_p = %v, want nil", *got.Options.TopP)
	}
	if got.Options.MinP != nil {
		t.Fatalf("min_p = %v, want nil", *got.Options.MinP)
	}
	if got.Options.RepeatLastN != nil {
		t.Fatalf("repeat_last_n = %v, want nil", *got.Options.RepeatLastN)
	}
	if got.Options.FrequencyPenalty != nil {
		t.Fatalf("frequency_penalty = %v, want nil", *got.Options.FrequencyPenalty)
	}
	if got.Options.NumPredict != opts.NumPredict {
		t.Fatalf("num_predict = %d, want %d", got.Options.NumPredict, opts.NumPredict)
	}
}

type roundTripFunc func(*http.Request) (*http.Response, error)

func (f roundTripFunc) RoundTrip(r *http.Request) (*http.Response, error) {
	return f(r)
}
