package template

import (
	"bytes"
	"testing"
	"text/template"

	"github.com/ollama/ollama/api"
)

func TestRewritePropertiesCheck(t *testing.T) {
	makeToolWithProps := func(props *api.ToolProperties) api.Tool {
		return api.Tool{
			Type: "function",
			Function: api.ToolFunction{
				Name:        "test",
				Description: "test function",
				Parameters:  api.NewToolFunctionParametersWithProps("object", nil, props),
			},
		}
	}

	tests := []struct {
		name     string
		template string
		data     interface{}
		expected string
	}{
		{
			name:     "if statement with Properties gets rewritten to HasProperties",
			template: `{{if .Function.Parameters.Properties}}Has props{{else}}No props{{end}}`,
			data:     makeToolWithProps(nil),
			expected: "No props", // Should use HasProperties which returns false for empty
		},
		{
			name:     "if statement with Properties and non-empty properties",
			template: `{{if .Function.Parameters.Properties}}Has props{{else}}No props{{end}}`,
			data: makeToolWithProps(func() *api.ToolProperties {
				p := api.NewToolProperties()
				p.Set("test", api.ToolProperty{Type: api.PropertyType{"string"}})
				return p
			}()),
			expected: "Has props", // Should use HasProperties which returns true
		},
		{
			name:     "range over Properties should not be rewritten",
			template: `{{range $k, $v := .Function.Parameters.Properties}}{{$k}} {{end}}`,
			data: makeToolWithProps(func() *api.ToolProperties {
				p := api.NewToolProperties()
				p.Set("foo", api.ToolProperty{Type: api.PropertyType{"string"}})
				p.Set("bar", api.ToolProperty{Type: api.PropertyType{"number"}})
				return p
			}()),
			expected: "foo bar ", // Should still use Properties() for ranging
		},
		{
			name: "complex template with both if and range",
			template: `{{if .Function.Parameters.Properties}}Args:
{{range $k, $v := .Function.Parameters.Properties}}  {{$k}}
{{end}}{{else}}No args{{end}}`,
			data: makeToolWithProps(func() *api.ToolProperties {
				p := api.NewToolProperties()
				p.Set("location", api.ToolProperty{Type: api.PropertyType{"string"}})
				return p
			}()),
			expected: "Args:\n  location\n",
		},
		{
			name:     "if with and condition",
			template: `{{if and .Function.Parameters.Properties (gt (len .Function.Parameters.Properties) 0)}}yes{{else}}no{{end}}`,
			data:     makeToolWithProps(nil),
			expected: "no", // Empty, so HasProperties returns false
		},
		{
			name:     "len function on Properties gets rewritten to Len method",
			template: `{{len .Function.Parameters.Properties}}`,
			data:     makeToolWithProps(nil),
			expected: "0", // Empty properties should have length 0
		},
		{
			name:     "len function on non-empty Properties",
			template: `{{len .Function.Parameters.Properties}}`,
			data: makeToolWithProps(func() *api.ToolProperties {
				p := api.NewToolProperties()
				p.Set("foo", api.ToolProperty{Type: api.PropertyType{"string"}})
				p.Set("bar", api.ToolProperty{Type: api.PropertyType{"number"}})
				return p
			}()),
			expected: "2", // Two properties
		},
		{
			name:     "nested len in and/gt (gpt-oss pattern)",
			template: `{{if and .Function.Parameters.Properties (gt (len .Function.Parameters.Properties) 0)}}has props{{else}}no props{{end}}`,
			data:     makeToolWithProps(nil),
			expected: "no props", // Empty, so both checks should be false
		},
		{
			name:     "nested len in and/gt with properties",
			template: `{{if and .Function.Parameters.Properties (gt (len .Function.Parameters.Properties) 0)}}has props{{else}}no props{{end}}`,
			data: makeToolWithProps(func() *api.ToolProperties {
				p := api.NewToolProperties()
				p.Set("test", api.ToolProperty{Type: api.PropertyType{"string"}})
				return p
			}()),
			expected: "has props", // Has properties, both checks should be true
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Use text/template directly and call rewritePropertiesCheck
			tmpl, err := template.New("test").Parse(tt.template)
			if err != nil {
				t.Fatalf("Failed to parse template: %v", err)
			}

			// Apply the rewrite
			rewritePropertiesCheck(tmpl)

			var buf bytes.Buffer
			err = tmpl.Execute(&buf, tt.data)
			if err != nil {
				t.Fatalf("Failed to execute template: %v", err)
			}

			if buf.String() != tt.expected {
				t.Errorf("Expected %q, got %q", tt.expected, buf.String())
			}
		})
	}
}
