package renderers

import (
	"slices"

	"github.com/ollama/ollama/api"
)

// developerMessagesAsSystem is an opt-in fallback for renderers whose model
// protocol uses system turns for developer instructions. Keep the original
// history intact so other renderers and compaction can preserve its roles.
func developerMessagesAsSystem(messages []api.Message) []api.Message {
	var normalized []api.Message
	for i, message := range messages {
		if message.Role == "developer" {
			if normalized == nil {
				normalized = slices.Clone(messages)
			}
			normalized[i].Role = "system"
		}
	}
	if normalized != nil {
		return normalized
	}
	return messages
}
