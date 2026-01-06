package renderers

import "github.com/ollama/ollama/api"

// testPropsMap creates a ToolPropertiesMap from a map (convenience function for tests, order not preserved)
func testPropsMap(m map[string]api.ToolProperty) *api.ToolPropertiesMap {
	props := api.NewToolPropertiesMap()
	for k, v := range m {
		props.Set(k, v)
	}
	return props
}

// testArgs creates ToolCallFunctionArguments from a map (convenience function for tests, order not preserved)
func testArgs(m map[string]any) api.ToolCallFunctionArguments {
	args := api.NewToolCallFunctionArguments()
	for k, v := range m {
		args.Set(k, v)
	}
	return args
}

// orderedArg represents a key-value pair for ordered argument creation
type orderedArg struct {
	Key   string
	Value any
}

// testArgsOrdered creates ToolCallFunctionArguments with a specific key order
func testArgsOrdered(pairs []orderedArg) api.ToolCallFunctionArguments {
	args := api.NewToolCallFunctionArguments()
	for _, p := range pairs {
		args.Set(p.Key, p.Value)
	}
	return args
}

// orderedProp represents a key-value pair for ordered property creation
type orderedProp struct {
	Key   string
	Value api.ToolProperty
}

// testPropsOrdered creates a ToolPropertiesMap with a specific key order
func testPropsOrdered(pairs []orderedProp) *api.ToolPropertiesMap {
	props := api.NewToolPropertiesMap()
	for _, p := range pairs {
		props.Set(p.Key, p.Value)
	}
	return props
}
