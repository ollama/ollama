package server

import (
	"bytes"
	"strings"
	"testing"

	"github.com/jmorganca/ollama/api"
)

func TestPrompt(t *testing.T) {
	tests := []struct {
		name     string
		template string
		vars     PromptVars
		want     string
		wantErr  bool
	}{
		{
			name:     "System Prompt",
			template: "[INST] {{ .System }} {{ .Prompt }} [/INST]",
			vars: PromptVars{
				System: "You are a Wizard.",
				Prompt: "What are the potion ingredients?",
			},
			want: "[INST] You are a Wizard. What are the potion ingredients? [/INST]",
		},
		{
			name:     "System Prompt with Response",
			template: "[INST] {{ .System }} {{ .Prompt }} [/INST] {{ .Response }}",
			vars: PromptVars{
				System:   "You are a Wizard.",
				Prompt:   "What are the potion ingredients?",
				Response: "I don't know.",
			},
			want: "[INST] You are a Wizard. What are the potion ingredients? [/INST] I don't know.",
		},
		{
			name:     "Conditional Logic Nodes",
			template: "[INST] {{if .First}}Hello!{{end}} {{ .System }} {{ .Prompt }} [/INST] {{ .Response }}",
			vars: PromptVars{
				First:    true,
				System:   "You are a Wizard.",
				Prompt:   "What are the potion ingredients?",
				Response: "I don't know.",
			},
			want: "[INST] Hello! You are a Wizard. What are the potion ingredients? [/INST] I don't know.",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := Prompt(tt.template, tt.vars)
			if (err != nil) != tt.wantErr {
				t.Errorf("Prompt() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if got != tt.want {
				t.Errorf("Prompt() got = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestModel_PreResponsePrompt(t *testing.T) {
	tests := []struct {
		name     string
		template string
		vars     PromptVars
		want     string
		wantErr  bool
	}{
		{
			name:     "No Response in Template",
			template: "[INST] {{ .System }} {{ .Prompt }} [/INST]",
			vars: PromptVars{
				System: "You are a Wizard.",
				Prompt: "What are the potion ingredients?",
			},
			want: "[INST] You are a Wizard. What are the potion ingredients? [/INST]",
		},
		{
			name:     "Response in Template",
			template: "[INST] {{ .System }} {{ .Prompt }} [/INST] {{ .Response }}",
			vars: PromptVars{
				System: "You are a Wizard.",
				Prompt: "What are the potion ingredients?",
			},
			want: "[INST] You are a Wizard. What are the potion ingredients? [/INST] ",
		},
		{
			name:     "Response in Template with Trailing Formatting",
			template: "<|im_start|>user\n{{ .Prompt }}<|im_end|><|im_start|>assistant\n{{ .Response }}<|im_end|>",
			vars: PromptVars{
				Prompt: "What are the potion ingredients?",
			},
			want: "<|im_start|>user\nWhat are the potion ingredients?<|im_end|><|im_start|>assistant\n",
		},
		{
			name:     "Response in Template with Alternative Formatting",
			template: "<|im_start|>user\n{{.Prompt}}<|im_end|><|im_start|>assistant\n{{.Response}}<|im_end|>",
			vars: PromptVars{
				Prompt: "What are the potion ingredients?",
			},
			want: "<|im_start|>user\nWhat are the potion ingredients?<|im_end|><|im_start|>assistant\n",
		},
	}

	for _, tt := range tests {
		m := Model{Template: tt.template}
		t.Run(tt.name, func(t *testing.T) {
			got, err := m.PreResponsePrompt(tt.vars)
			if (err != nil) != tt.wantErr {
				t.Errorf("PreResponsePrompt() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if got != tt.want {
				t.Errorf("PreResponsePrompt() got = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestModel_PostResponsePrompt(t *testing.T) {
	tests := []struct {
		name     string
		template string
		vars     PromptVars
		want     string
		wantErr  bool
	}{
		{
			name:     "No Response in Template",
			template: "[INST] {{ .System }} {{ .Prompt }} [/INST]",
			vars: PromptVars{
				Response: "I don't know.",
			},
			want: "I don't know.",
		},
		{
			name:     "Response in Template",
			template: "[INST] {{ .System }} {{ .Prompt }} [/INST] {{ .Response }}",
			vars: PromptVars{
				Response: "I don't know.",
			},
			want: "I don't know.",
		},
		{
			name:     "Response in Template with Trailing Formatting",
			template: "<|im_start|>user\n{{ .Prompt }}<|im_end|><|im_start|>assistant\n{{ .Response }}<|im_end|>",
			vars: PromptVars{
				Response: "I don't know.",
			},
			want: "I don't know.<|im_end|>",
		},
		{
			name:     "Response in Template with Alternative Formatting",
			template: "<|im_start|>user\n{{.Prompt}}<|im_end|><|im_start|>assistant\n{{.Response}}<|im_end|>",
			vars: PromptVars{
				Response: "I don't know.",
			},
			want: "I don't know.<|im_end|>",
		},
	}

	for _, tt := range tests {
		m := Model{Template: tt.template}
		t.Run(tt.name, func(t *testing.T) {
			got, err := m.PostResponseTemplate(tt.vars)
			if (err != nil) != tt.wantErr {
				t.Errorf("PostResponseTemplate() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if got != tt.want {
				t.Errorf("PostResponseTemplate() got = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestModel_PreResponsePrompt_PostResponsePrompt(t *testing.T) {
	tests := []struct {
		name     string
		template string
		preVars  PromptVars
		postVars PromptVars
		want     string
		wantErr  bool
	}{
		{
			name:     "Response in Template",
			template: "<|im_start|>user\n{{.Prompt}}<|im_end|><|im_start|>assistant\n{{.Response}}<|im_end|>",
			preVars: PromptVars{
				Prompt: "What are the potion ingredients?",
			},
			postVars: PromptVars{
				Prompt:   "What are the potion ingredients?",
				Response: "Sugar.",
			},
			want: "<|im_start|>user\nWhat are the potion ingredients?<|im_end|><|im_start|>assistant\nSugar.<|im_end|>",
		},
		{
			name:     "No Response in Template",
			template: "<|im_start|>user\n{{.Prompt}}<|im_end|><|im_start|>assistant\n",
			preVars: PromptVars{
				Prompt: "What are the potion ingredients?",
			},
			postVars: PromptVars{
				Prompt:   "What are the potion ingredients?",
				Response: "Spice.",
			},
			want: "<|im_start|>user\nWhat are the potion ingredients?<|im_end|><|im_start|>assistant\nSpice.",
		},
	}

	for _, tt := range tests {
		m := Model{Template: tt.template}
		t.Run(tt.name, func(t *testing.T) {
			pre, err := m.PreResponsePrompt(tt.preVars)
			if (err != nil) != tt.wantErr {
				t.Errorf("PreResponsePrompt() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			post, err := m.PostResponseTemplate(tt.postVars)
			if err != nil {
				t.Errorf("PostResponseTemplate() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			result := pre + post
			if result != tt.want {
				t.Errorf("Prompt() got = %v, want %v", result, tt.want)
			}
		})
	}
}

func chatHistoryEqual(a, b ChatHistory) bool {
	if len(a.Prompts) != len(b.Prompts) {
		return false
	}
	for i, v := range a.Prompts {

		if v.First != b.Prompts[i].First {
			return false
		}

		if v.Response != b.Prompts[i].Response {
			return false
		}

		if v.Prompt != b.Prompts[i].Prompt {
			return false
		}

		if v.System != b.Prompts[i].System {
			return false
		}

		if len(v.Images) != len(b.Prompts[i].Images) {
			return false
		}

		for j, img := range v.Images {
			if img.ID != b.Prompts[i].Images[j].ID {
				return false
			}

			if !bytes.Equal(img.Data, b.Prompts[i].Images[j].Data) {
				return false
			}
		}
	}
	return a.LastSystem == b.LastSystem
}

func TestChat(t *testing.T) {
	tests := []struct {
		name    string
		model   Model
		msgs    []api.Message
		want    ChatHistory
		wantErr string
	}{
		{
			name: "Single Message",
			model: Model{
				Template: "[INST] {{ .System }} {{ .Prompt }} [/INST]",
			},
			msgs: []api.Message{
				{
					Role:    "system",
					Content: "You are a Wizard.",
				},
				{
					Role:    "user",
					Content: "What are the potion ingredients?",
				},
			},
			want: ChatHistory{
				Prompts: []PromptVars{
					{
						System: "You are a Wizard.",
						Prompt: "What are the potion ingredients?",
						First:  true,
					},
				},
				LastSystem: "You are a Wizard.",
			},
		},
		{
			name: "Message History",
			model: Model{
				Template: "[INST] {{ .System }} {{ .Prompt }} [/INST]",
			},
			msgs: []api.Message{
				{
					Role:    "system",
					Content: "You are a Wizard.",
				},
				{
					Role:    "user",
					Content: "What are the potion ingredients?",
				},
				{
					Role:    "assistant",
					Content: "sugar",
				},
				{
					Role:    "user",
					Content: "Anything else?",
				},
			},
			want: ChatHistory{
				Prompts: []PromptVars{
					{
						System:   "You are a Wizard.",
						Prompt:   "What are the potion ingredients?",
						Response: "sugar",
						First:    true,
					},
					{
						Prompt: "Anything else?",
					},
				},
				LastSystem: "You are a Wizard.",
			},
		},
		{
			name: "Assistant Only",
			model: Model{
				Template: "[INST] {{ .System }} {{ .Prompt }} [/INST]",
			},
			msgs: []api.Message{
				{
					Role:    "assistant",
					Content: "everything nice",
				},
			},
			want: ChatHistory{
				Prompts: []PromptVars{
					{
						Response: "everything nice",
						First:    true,
					},
				},
			},
		},
		{
			name: "Last system message is preserved from modelfile",
			model: Model{
				Template: "[INST] {{ .System }} {{ .Prompt }} [/INST]",
				System:   "You are Mojo Jojo.",
			},
			msgs: []api.Message{
				{
					Role:    "user",
					Content: "hi",
				},
			},
			want: ChatHistory{
				Prompts: []PromptVars{
					{
						System: "You are Mojo Jojo.",
						Prompt: "hi",
						First:  true,
					},
				},
				LastSystem: "You are Mojo Jojo.",
			},
		},
		{
			name: "Last system message is preserved from messages",
			model: Model{
				Template: "[INST] {{ .System }} {{ .Prompt }} [/INST]",
				System:   "You are Mojo Jojo.",
			},
			msgs: []api.Message{
				{
					Role:    "system",
					Content: "You are Professor Utonium.",
				},
			},
			want: ChatHistory{
				Prompts: []PromptVars{
					{
						System: "You are Professor Utonium.",
						First:  true,
					},
				},
				LastSystem: "You are Professor Utonium.",
			},
		},
		{
			name: "Invalid Role",
			msgs: []api.Message{
				{
					Role:    "not-a-role",
					Content: "howdy",
				},
			},
			wantErr: "invalid role: not-a-role",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := tt.model.ChatPrompts(tt.msgs)
			if tt.wantErr != "" {
				if err == nil {
					t.Errorf("ChatPrompt() expected error, got nil")
				}
				if !strings.Contains(err.Error(), tt.wantErr) {
					t.Errorf("ChatPrompt() error = %v, wantErr %v", err, tt.wantErr)
				}
				return
			}
			if !chatHistoryEqual(*got, tt.want) {
				t.Errorf("ChatPrompt() got = %#v, want %#v", got, tt.want)
			}
		})
	}
}
