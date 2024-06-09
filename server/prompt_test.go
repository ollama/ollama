package server

import (
	"strings"
	"testing"

	"github.com/ollama/ollama/api"
)

func TestPrompt(t *testing.T) {
	tests := []struct {
		name     string
		template string
		tools    string
		system   string
		prompt   string
		results  string
		response string
		generate bool
		want     string
	}{
		{
			name:     "simple prompt",
			template: "[INST] {{ .System }} {{ .Prompt }} [/INST]",
			system:   "You are a Wizard.",
			prompt:   "What are the potion ingredients?",
			want:     "[INST] You are a Wizard. What are the potion ingredients? [/INST]",
		},
		{
			name:     "implicit response",
			template: "[INST] {{ .System }} {{ .Prompt }} [/INST]",
			system:   "You are a Wizard.",
			prompt:   "What are the potion ingredients?",
			response: "I don't know.",
			want:     "[INST] You are a Wizard. What are the potion ingredients? [/INST]I don't know.",
		},
		{
			name:     "response",
			template: "[INST] {{ .System }} {{ .Prompt }} [/INST] {{ .Response }}",
			system:   "You are a Wizard.",
			prompt:   "What are the potion ingredients?",
			response: "I don't know.",
			want:     "[INST] You are a Wizard. What are the potion ingredients? [/INST] I don't know.",
		},
		{
			name:     "cut",
			template: "<system>{{ .System }}</system><user>{{ .Prompt }}</user><assistant>{{ .Response }}</assistant>",
			system:   "You are a Wizard.",
			prompt:   "What are the potion ingredients?",
			response: "I don't know.",
			generate: true,
			want:     "<system>You are a Wizard.</system><user>What are the potion ingredients?</user><assistant>I don't know.",
		},
		{
			name:     "nocut",
			template: "<system>{{ .System }}</system><user>{{ .Prompt }}</user><assistant>{{ .Response }}</assistant>",
			system:   "You are a Wizard.",
			prompt:   "What are the potion ingredients?",
			response: "I don't know.",
			want:     "<system>You are a Wizard.</system><user>What are the potion ingredients?</user><assistant>I don't know.</assistant>",
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got, err := Prompt(tc.template, tc.tools, tc.system, tc.prompt, tc.results, tc.response, tc.generate)
			if err != nil {
				t.Errorf("error = %v", err)
			}

			if got != tc.want {
				t.Errorf("got = %v, want %v", got, tc.want)
			}
		})
	}
}

func TestChatPrompt(t *testing.T) {
	tests := []struct {
		name     string
		template string
		messages []api.Message
		tools    string
		window   int
		want     string
	}{
		{
			name:     "simple prompt",
			template: "[INST] {{ .Prompt }} [/INST]",
			messages: []api.Message{
				{Role: "user", Content: "Hello"},
			},
			window: 1024,
			want:   "[INST] Hello [/INST]",
		},
		{
			name:     "with system message",
			template: "[INST] {{ if .System }}<<SYS>>{{ .System }}<</SYS>> {{ end }}{{ .Prompt }} [/INST]",
			messages: []api.Message{
				{Role: "system", Content: "You are a Wizard."},
				{Role: "user", Content: "Hello"},
			},
			window: 1024,
			want:   "[INST] <<SYS>>You are a Wizard.<</SYS>> Hello [/INST]",
		},
		{
			name:     "with response",
			template: "[INST] {{ if .System }}<<SYS>>{{ .System }}<</SYS>> {{ end }}{{ .Prompt }} [/INST] {{ .Response }}",
			messages: []api.Message{
				{Role: "system", Content: "You are a Wizard."},
				{Role: "user", Content: "Hello"},
				{Role: "assistant", Content: "I am?"},
			},
			window: 1024,
			want:   "[INST] <<SYS>>You are a Wizard.<</SYS>> Hello [/INST] I am?",
		},
		{
			name:     "with implicit response",
			template: "[INST] {{ if .System }}<<SYS>>{{ .System }}<</SYS>> {{ end }}{{ .Prompt }} [/INST]",
			messages: []api.Message{
				{Role: "system", Content: "You are a Wizard."},
				{Role: "user", Content: "Hello"},
				{Role: "assistant", Content: "I am?"},
			},
			window: 1024,
			want:   "[INST] <<SYS>>You are a Wizard.<</SYS>> Hello [/INST]I am?",
		},
		{
			name:     "with conversation",
			template: "[INST] {{ if .System }}<<SYS>>{{ .System }}<</SYS>> {{ end }}{{ .Prompt }} [/INST] {{ .Response }} ",
			messages: []api.Message{
				{Role: "system", Content: "You are a Wizard."},
				{Role: "user", Content: "What are the potion ingredients?"},
				{Role: "assistant", Content: "sugar"},
				{Role: "user", Content: "Anything else?"},
			},
			window: 1024,
			want:   "[INST] <<SYS>>You are a Wizard.<</SYS>> What are the potion ingredients? [/INST] sugar [INST] Anything else? [/INST] ",
		},
		{
			name:     "with truncation",
			template: "{{ .System }} {{ .Prompt }} {{ .Response }} ",
			messages: []api.Message{
				{Role: "system", Content: "You are a Wizard."},
				{Role: "user", Content: "Hello"},
				{Role: "assistant", Content: "I am?"},
				{Role: "user", Content: "Why is the sky blue?"},
				{Role: "assistant", Content: "The sky is blue from rayleigh scattering"},
			},
			window: 10,
			want:   "You are a Wizard. Why is the sky blue? The sky is blue from rayleigh scattering",
		},
		{
			name:     "images",
			template: "{{ .System }} {{ .Prompt }}",
			messages: []api.Message{
				{Role: "system", Content: "You are a Wizard."},
				{Role: "user", Content: "Hello", Images: []api.ImageData{[]byte("base64")}},
			},
			window: 1024,
			want:   "You are a Wizard. [img-0] Hello",
		},
		{
			name:     "images truncated",
			template: "{{ .System }} {{ .Prompt }}",
			messages: []api.Message{
				{Role: "system", Content: "You are a Wizard."},
				{Role: "user", Content: "Hello", Images: []api.ImageData{[]byte("img1"), []byte("img2")}},
			},
			window: 1024,
			want:   "You are a Wizard. [img-0] [img-1] Hello",
		},
		{
			name:     "empty list",
			template: "{{ .System }} {{ .Prompt }}",
			messages: []api.Message{},
			window:   1024,
			want:     "",
		},
		{
			name:     "empty prompt",
			template: "[INST] {{ if .System }}<<SYS>>{{ .System }}<</SYS>> {{ end }}{{ .Prompt }} [/INST] {{ .Response }} ",
			messages: []api.Message{
				{Role: "user", Content: ""},
			},
			window: 1024,
			want:   "",
		},
	}

	encode := func(s string) ([]int, error) {
		words := strings.Fields(s)
		return make([]int, len(words)), nil
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got, err := ChatPrompt(tc.template, tc.messages, tc.tools, tc.window, encode)
			if err != nil {
				t.Errorf("error = %v", err)
			}

			if got != tc.want {
				t.Errorf("got: %q, want: %q", got, tc.want)
			}
		})
	}
}
