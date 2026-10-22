package renderers

import (
	"reflect"
	"slices"
	"strings"
	"testing"

	"github.com/ollama/ollama/api"
)

func TestDeveloperRoleFallbackMatchesSystem(t *testing.T) {
	names := []string{
		"qwen3-coder", "qwen3-vl-instruct", "qwen3-vl-thinking", "qwen3.5", "tev1", "qwen3.8", "ornith",
		"cogito", "deepseek3.1", "olmo3", "olmo3.1", "olmo3-think", "olmo3-32b-think",
		"gemma4", "gemma4-small", "gemma4-large", "glm-4.7", "flint", "glm-ocr",
		"laguna", "poolside-v1", "cohere", "glimmer",
	}
	const policy = "DEVELOPER_POLICY_7F3A"
	for _, name := range names {
		for _, placement := range []string{"leading", "after_system", "after_conversation"} {
			t.Run(name+"/"+placement, func(t *testing.T) {
				var messages []api.Message
				switch placement {
				case "after_system":
					messages = append(messages, api.Message{Role: "system", Content: "Base policy"})
				case "after_conversation":
					messages = append(messages, api.Message{Role: "user", Content: "Hello"}, api.Message{Role: "assistant", Content: "Hello."})
				}
				messages = append(messages, api.Message{Role: "developer", Content: policy}, api.Message{Role: "user", Content: "Continue"})
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
				if err != nil {
					t.Fatal(err)
				}
				want, err := rendererForName(name).Render(system, nil, &api.ThinkValue{Value: false})
				if err != nil {
					t.Fatal(err)
				}
				if got != want {
					t.Fatal("developer message did not use the model's system fallback")
				}
				if placement == "leading" && !strings.Contains(got, policy) {
					t.Fatal("leading developer instruction disappeared")
				}
				if !reflect.DeepEqual(messages, original) {
					t.Fatal("renderer mutated the source history")
				}
			})
		}
	}
}

func TestNativeDeveloperTurnIsPreserved(t *testing.T) {
	for _, name := range []string{"functiongemma", "lfm2", "lfm2-thinking", "nemotron-3-nano", "nemotron-3.5-nano", "nemotron-3.5-super"} {
		t.Run(name, func(t *testing.T) {
			messages := []api.Message{{Role: "user", Content: "Hello"}, {Role: "assistant", Content: "Hello."}, {Role: "developer", Content: "POLICY"}, {Role: "user", Content: "Continue"}}
			original := slices.Clone(messages)
			got, err := RenderWithRenderer(name, messages, nil, nil)
			if err != nil {
				t.Fatal(err)
			}
			if !strings.Contains(got, "developer\nPOLICY") {
				t.Fatal("native developer turn was rewritten")
			}
			if !reflect.DeepEqual(messages, original) {
				t.Fatal("renderer mutated the source history")
			}
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
	if err != nil || got != "developer" {
		t.Fatalf("role = %q, err = %v", got, err)
	}
}
