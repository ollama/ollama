package parsers

import "github.com/ollama/ollama/api"

func toolArgumentTypes(tool *api.Tool, key string) api.PropertyType {
	if tool == nil || tool.Function.Parameters.Properties == nil {
		return nil
	}
	prop, ok := tool.Function.Parameters.Properties.Get(key)
	if !ok {
		return nil
	}
	if len(prop.AnyOf) == 0 {
		return prop.Type
	}
	var types api.PropertyType
	for _, choice := range prop.AnyOf {
		types = append(types, choice.Type...)
	}
	return types
}
