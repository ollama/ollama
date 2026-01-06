package parsers

import (
	"encoding/json"

	"github.com/google/go-cmp/cmp"
	"github.com/ollama/ollama/api"
)

// argsComparer provides cmp options for comparing ToolCallFunctionArguments
// It compares by logical equality (same keys with same values) not by order
var argsComparer = cmp.Comparer(func(a, b api.ToolCallFunctionArguments) bool {
	// Convert both to maps and compare
	aMap := a.ToMap()
	bMap := b.ToMap()
	if len(aMap) != len(bMap) {
		return false
	}
	for k, av := range aMap {
		bv, ok := bMap[k]
		if !ok {
			return false
		}
		// Use JSON encoding for deep comparison of values
		aJSON, _ := json.Marshal(av)
		bJSON, _ := json.Marshal(bv)
		if string(aJSON) != string(bJSON) {
			return false
		}
	}
	return true
})

// propsComparer provides cmp options for comparing ToolPropertiesMap
var propsComparer = cmp.Comparer(func(a, b *api.ToolPropertiesMap) bool {
	if a == nil && b == nil {
		return true
	}
	if a == nil || b == nil {
		return false
	}
	aJSON, _ := json.Marshal(a)
	bJSON, _ := json.Marshal(b)
	return string(aJSON) == string(bJSON)
})

// toolsComparer combines argsComparer and propsComparer for comparing tools
var toolsComparer = cmp.Options{argsComparer, propsComparer}

// toolCallEqual compares two tool calls by comparing their components
// It compares arguments by logical equality (same keys with same values) not by order
func toolCallEqual(a, b api.ToolCall) bool {
	if a.ID != b.ID {
		return false
	}
	if a.Function.Index != b.Function.Index {
		return false
	}
	if a.Function.Name != b.Function.Name {
		return false
	}
	// Compare arguments by logical equality using argsComparer logic
	aMap := a.Function.Arguments.ToMap()
	bMap := b.Function.Arguments.ToMap()
	if len(aMap) != len(bMap) {
		return false
	}
	for k, av := range aMap {
		bv, ok := bMap[k]
		if !ok {
			return false
		}
		aJSON, _ := json.Marshal(av)
		bJSON, _ := json.Marshal(bv)
		if string(aJSON) != string(bJSON) {
			return false
		}
	}
	return true
}

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
