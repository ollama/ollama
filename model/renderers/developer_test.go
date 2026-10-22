package renderers

import (
	"slices"
	"strings"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/ollama/ollama/api"
)

func TestDeveloperRoleFallbackPreservesInstructions(t *testing.T) {
	names := []string{
		"qwen3-coder", "qwen3-vl-instruct", "qwen3-vl-thinking", "qwen3.5", "tev1", "qwen3.8", "ornith",
		"cogito", "deepseek3.1", "olmo3", "olmo3.1", "olmo3-think", "olmo3-32b-think",
		"glm-4.7", "glm-ocr",
		"laguna", "poolside-v1", "cohere", "glimmer",
	}
	cases := map[string][]api.Message{
		"leading": {{Role: "developer", Content: "PROJECT_POLICY"}, {Role: "user", Content: "Continue"}},
		"mixed history": {
			{Role: "system", Content: "BASE_POLICY"},
			{Role: "developer", Content: "PROJECT_POLICY"},
			{Role: "user", Content: "Hello"},
			{Role: "assistant", Content: "Hello."},
			{Role: "developer", Content: "LATER_POLICY"},
			{Role: "user", Content: "Continue"},
		},
	}
	for _, name := range names {
		for placement, messages := range cases {
			t.Run(name+"/"+placement, func(t *testing.T) {
				original := slices.Clone(messages)
				system := slices.Clone(messages)
				for i := range system {
					if system[i].Role == "developer" {
						system[i].Role = "system"
					}
				}
				// Exercise each renderer directly: fallback belongs to the model,
				// rather than being imposed by API conversion or the registry.
				got, err := rendererForName(name).Render(messages, nil, &api.ThinkValue{Value: false})
				require.NoError(t, err)
				want, err := rendererForName(name).Render(system, nil, &api.ThinkValue{Value: false})
				require.NoError(t, err)
				require.Equal(t, want, got, "developer message must use the model's system fallback")
				// Equality alone can pass when both prompts drop the same policy.
				position := 0
				for _, message := range original {
					if message.Role != "system" && message.Role != "developer" {
						continue
					}
					index := strings.Index(got[position:], message.Content)
					if index < 0 || strings.Count(got, message.Content) != 1 {
						t.Fatalf("instruction %q was lost, duplicated, or reordered: %q", message.Content, got)
					}
					position += index + len(message.Content)
				}
				require.Equal(t, original, messages, "renderer mutated the source history")
			})
		}
	}
}

func TestNativeDeveloperTurnIsPreserved(t *testing.T) {
	for _, name := range []string{"functiongemma", "gemma4", "gemma4-small", "gemma4-large", "lfm2", "lfm2-thinking", "nemotron-3-nano", "nemotron-3.5-nano", "nemotron-3.5-super"} {
		t.Run(name, func(t *testing.T) {
			messages := []api.Message{{Role: "user", Content: "Hello"}, {Role: "assistant", Content: "Hello."}, {Role: "developer", Content: "POLICY"}, {Role: "user", Content: "Continue"}}
			original := slices.Clone(messages)
			got, err := RenderWithRenderer(name, messages, nil, nil)
			require.NoError(t, err)
			require.Contains(t, got, "developer\nPOLICY", "native developer turn was rewritten")
			require.Equal(t, original, messages, "renderer mutated the source history")
		})
	}
}

type developerPreservingRenderer struct{ mockRenderer }

func (*developerPreservingRenderer) Render(messages []api.Message, _ []api.Tool, _ *api.ThinkValue) (string, error) {
	return messages[0].Role, nil
}

func TestCustomRendererReceivesOriginalDeveloperRole(t *testing.T) {
	// An override for a built-in name must still own its role semantics.
	const name = "qwen3.5"
	previous, existed := registry.renderers[name]
	t.Cleanup(func() {
		if existed {
			registry.renderers[name] = previous
		} else {
			delete(registry.renderers, name)
		}
	})
	Register(name, func() Renderer { return &developerPreservingRenderer{} })
	got, err := RenderWithRenderer(name, []api.Message{{Role: "developer", Content: "Policy"}}, nil, nil)
	require.NoError(t, err)
	require.Equal(t, "developer", got)
}
