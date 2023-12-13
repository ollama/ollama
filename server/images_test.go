package server

import (
	"strings"
	"testing"

	"github.com/jmorganca/ollama/api"
)

func TestChat(t *testing.T) {
	tests := []struct {
		name     string
		template string
		msgs     []api.Message
		want     string
		wantErr  string
	}{
		{
			name:     "Single Message",
			template: "[INST] {{ .System }} {{ .Prompt }} [/INST]",
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
			want: "[INST] You are a Wizard. What are the potion ingredients? [/INST]",
		},
		{
			name:     "Message History",
			template: "[INST] {{ .System }} {{ .Prompt }} [/INST]",
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
			want: "[INST] You are a Wizard. What are the potion ingredients? [/INST]sugar[INST]  Anything else? [/INST]",
		},
		{
			name:     "Assistant Only",
			template: "[INST] {{ .System }} {{ .Prompt }} [/INST]",
			msgs: []api.Message{
				{
					Role:    "assistant",
					Content: "everything nice",
				},
			},
			want: "[INST]   [/INST]everything nice",
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
		m := Model{
			Template: tt.template,
		}
		t.Run(tt.name, func(t *testing.T) {
			got, _, err := m.ChatPrompt(tt.msgs)
			if tt.wantErr != "" {
				if err == nil {
					t.Errorf("ChatPrompt() expected error, got nil")
				}
				if !strings.Contains(err.Error(), tt.wantErr) {
					t.Errorf("ChatPrompt() error = %v, wantErr %v", err, tt.wantErr)
				}
			}
			if got != tt.want {
				t.Errorf("ChatPrompt() got = %v, want %v", got, tt.want)
			}
		})
	}
}
