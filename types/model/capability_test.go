package model

import "testing"

func TestChatTemplateCapabilities(t *testing.T) {
	tests := []struct {
		name         string
		chatTemplate string
		wantTools    bool
		wantThinking bool
	}{
		{
			name:         "tools variable",
			chatTemplate: "{% if tools %}{{ tools }}{% endif %}",
			wantTools:    true,
		},
		{
			name:         "tool call",
			chatTemplate: "{% for tool_call in message.tool_calls %}{{ tool_call }}{% endfor %}",
			wantTools:    true,
		},
		{
			name:         "thinking tags",
			chatTemplate: "<think>{{ content }}</think>",
			wantThinking: true,
		},
		{
			name:         "thinking split",
			chatTemplate: "{% set content = content.split('</think>')[-1] %}",
			wantThinking: true,
		},
		{
			name:         "double quoted thinking split",
			chatTemplate: `{% set content = content.split("</think>")[-1] %}`,
			wantThinking: true,
		},
		{
			name:         "reasoning content exclusion",
			chatTemplate: "{% set content = content.split('</think>')[-1] %}{{ reasoning_content }}",
		},
		{
			name:         "special token exclusion",
			chatTemplate: "{% set content = content.split('</think>')[-1] %}<SPECIAL_12>",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := ChatTemplateHasToolSupport(tt.chatTemplate); got != tt.wantTools {
				t.Errorf("ChatTemplateHasToolSupport() = %v, want %v", got, tt.wantTools)
			}
			if got := ChatTemplateHasThinkingSupport(tt.chatTemplate); got != tt.wantThinking {
				t.Errorf("ChatTemplateHasThinkingSupport() = %v, want %v", got, tt.wantThinking)
			}
		})
	}
}
