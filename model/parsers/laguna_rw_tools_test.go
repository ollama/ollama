package parsers

import "github.com/ollama/ollama/api"

func readWriteToolsLaguna() []api.Tool {
	pathProps := api.NewToolPropertiesMap()
	pathProps.Set("path", api.ToolProperty{Type: api.PropertyType{"string"}})
	pathProps.Set("content", api.ToolProperty{Type: api.PropertyType{"string"}})
	readTool := api.Tool{
		Function: api.ToolFunction{
			Name: "read",
			Parameters: api.ToolFunctionParameters{
				Type:       "object",
				Properties: pathProps,
				Required:   []string{"path"},
			},
		},
	}
	contentProps := api.NewToolPropertiesMap()
	contentProps.Set("path", api.ToolProperty{Type: api.PropertyType{"string"}})
	contentProps.Set("content", api.ToolProperty{Type: api.PropertyType{"string"}})
	writeTool := api.Tool{
		Function: api.ToolFunction{
			Name: "write",
			Parameters: api.ToolFunctionParameters{
				Type:       "object",
				Properties: contentProps,
				Required:   []string{"path", "content"},
			},
		},
	}
	return []api.Tool{readTool, writeTool}
}
