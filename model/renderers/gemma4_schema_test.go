package renderers

import (
	"testing"

	"github.com/ollama/ollama/api"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestGemma4RendererPreservesSchemaKeywordPropertyNames(t *testing.T) {
	for _, name := range []string{"description", "nullable", "properties", "required", "type"} {
		t.Run(name, func(t *testing.T) {
			tools := []api.Tool{{
				Type: "function",
				Function: api.ToolFunction{
					Name: "create_ticket",
					Parameters: api.ToolFunctionParameters{
						Type: "object",
						Properties: testPropsMap(map[string]api.ToolProperty{
							name: {Type: api.PropertyType{"string"}, Description: "Value"},
						}),
					},
				},
			}}

			got, err := (&Gemma4Renderer{}).Render(
				[]api.Message{{Role: "user", Content: "Create a ticket"}}, tools, nil)
			require.NoError(t, err)
			assert.Contains(t, got, "properties:{"+name+":{description:"+g4Q+"Value"+g4Q+",type:"+g4Q+"STRING"+g4Q+"}}")
		})
	}
}

func TestGemma4RendererPreservesSchemaKeywordPropertyNamesWhenNested(t *testing.T) {
	tools := []api.Tool{{
		Type: "function",
		Function: api.ToolFunction{
			Name: "create_ticket",
			Parameters: api.ToolFunctionParameters{
				Type: "object",
				Properties: testPropsMap(map[string]api.ToolProperty{
					"payload": {
						Type: api.PropertyType{"object"},
						Properties: testPropsMap(map[string]api.ToolProperty{
							"type": {Type: api.PropertyType{"string"}, Description: "Kind"},
						}),
					},
					"items": {
						Type: api.PropertyType{"array"},
						Items: map[string]any{
							"type": "object",
							"properties": map[string]any{
								"required": map[string]any{"type": "string", "description": "Rule"},
							},
						},
					},
				}),
			},
		},
	}}

	got, err := (&Gemma4Renderer{}).Render(
		[]api.Message{{Role: "user", Content: "Create a ticket"}}, tools, nil)
	require.NoError(t, err)
	assert.Contains(t, got, "properties:{type:{description:"+g4Q+"Kind"+g4Q+",type:"+g4Q+"STRING"+g4Q+"}}")
	assert.Contains(t, got, "properties:{required:{description:"+g4Q+"Rule"+g4Q+",type:"+g4Q+"STRING"+g4Q+"}}")
}
