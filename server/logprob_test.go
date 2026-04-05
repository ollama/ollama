package server

import (
	"reflect"
	"testing"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/llm"
)

func TestLogprobBytes(t *testing.T) {
	tests := []struct {
		name string
		tlp  llm.TokenLogprob
		want []int
	}{
		{
			name: "prefers Bytes over Token",
			tlp:  llm.TokenLogprob{Token: "a", Bytes: []byte{0xF0}},
			want: []int{0xF0},
		},
		{
			name: "falls back to Token when Bytes is nil",
			tlp:  llm.TokenLogprob{Token: "hi"},
			want: []int{104, 105},
		},
		{
			name: "falls back to Token when Bytes is empty",
			tlp:  llm.TokenLogprob{Token: "a", Bytes: []byte{}},
			want: []int{97},
		},
		{
			name: "empty Token and nil Bytes",
			tlp:  llm.TokenLogprob{Token: ""},
			want: nil,
		},
		{
			name: "multi-byte raw bytes",
			tlp:  llm.TokenLogprob{Token: "\ufffd\ufffd", Bytes: []byte{0xF0, 0x9F}},
			want: []int{0xF0, 0x9F},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := logprobBytes(tt.tlp)
			if !reflect.DeepEqual(got, tt.want) {
				t.Errorf("logprobBytes() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestToAPILogprobsPartialUTF8(t *testing.T) {
	input := []llm.Logprob{{
		TokenLogprob: llm.TokenLogprob{
			Token:   string([]byte{0xF0}),
			Logprob: -0.5,
			Bytes:   []byte{0xF0, 0x9F},
		},
		TopLogprobs: []llm.TokenLogprob{{
			Token:   string([]byte{0x98}),
			Logprob: -1.0,
			Bytes:   []byte{0x98},
		}},
	}}

	got := toAPILogprobs(input)

	want := []api.Logprob{{
		TokenLogprob: api.TokenLogprob{
			Token:   string([]byte{0xF0}),
			Logprob: -0.5,
			Bytes:   []int{0xF0, 0x9F},
		},
		TopLogprobs: []api.TokenLogprob{{
			Token:   string([]byte{0x98}),
			Logprob: -1.0,
			Bytes:   []int{0x98},
		}},
	}}

	if !reflect.DeepEqual(got, want) {
		t.Errorf("toAPILogprobs() = %+v, want %+v", got, want)
	}
}
