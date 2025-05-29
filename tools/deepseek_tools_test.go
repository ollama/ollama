package tools

import (
	"fmt"
	"path/filepath"
	"testing"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/template"
	"github.com/stretchr/testify/assert"
)

func TestDeepSeekToolParser(t *testing.T) {
	p := filepath.Join("testdata")
	t1 := api.ToolCall{
		Function: api.ToolCallFunction{
			Name: "get_current_weather",
			Arguments: map[string]any{
				"format":   "fahrenheit",
				"location": "San Francisco, CA",
			},
			Index: 0,
		},
	}

	// t2 := api.ToolCall{
	// 	Function: api.ToolCallFunction{
	// 		Name: "get_current_weather",
	// 		Arguments: map[string]any{
	// 			"format":   "celsius",
	// 			"location": "Toronto, Canada",
	// 		},
	// 		Index: 1,
	// 	},
	// }

	tests := []struct {
		name             string
		template         string
		output           string
		expectedToolCall []api.ToolCall
		expectedTokens   string
	}{
		{
			name: "single tool call",
			output: `<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>get_current_weather
` + "```json\n" + `{"format":"fahrenheit","location":"San Francisco, CA"}` + "\n```" + `<｜tool▁call▁end｜>`,
			expectedToolCall: []api.ToolCall{t1},
			expectedTokens:   "",
		},
		// 		{
		// 			name:     "multiple tool calls",
		// 			template: `"<｜tool▁call▁begin｜>function<｜tool▁sep｜>get_current_weather\n` + "```json\n" + `{"format":"fahrenheit","location":"San Francisco, CA"}` + "\n```" + `<｜tool▁call▁end｜>"`,
		// 			output: `<｜tool▁call▁begin｜>function<｜tool▁sep｜>get_current_weather
		// ` + "```json\n" + `{"format":"fahrenheit","location":"San Francisco, CA"}` + "\n```" + `<｜tool▁call▁end｜>
		// <｜tool▁call▁begin｜>function<｜tool▁sep｜>get_current_weather
		// ` + "```json\n" + `{"format":"celsius","location":"Toronto, Canada"}` + "\n```" + `<｜tool▁call▁end｜>`,
		// 			expectedToolCall: []api.ToolCall{t1, t2},
		// 			expectedTokens:   "",
		// 		},
		// 		{
		// 			name:             "invalid tool call format",
		// 			template:         `{{"<｜tool▁call▁begin｜>function<｜tool▁sep｜>get_current_weather\n` + "```json\n" + `{"format":"fahrenheit","location":"San Francisco, CA"}` + "\n```" + `<｜tool▁call▁end｜>"}}`,
		// 			output:           "This is just some text without a tool call",
		// 			expectedToolCall: nil,
		// 			expectedTokens:   "This is just some text without a tool call",
		// 		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tmpl, err := template.Parse(readFile(t, p, "deepseek-r1.gotmpl").String())
			if err != nil {
				t.Fatal(err)
			}
			fmt.Println(tmpl.Template.Root.String())

			parser, err := NewDeepSeekToolParser(tmpl.Template)
			assert.NoError(t, err)

			tools, content := parser.Add(tt.output)
			assert.Equal(t, tt.expectedToolCall, tools)
			assert.Equal(t, tt.expectedTokens, content)
		})
	}
}
