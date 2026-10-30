package renderers

import (
	"encoding/json"
	"fmt"
	"reflect"
	"strconv"
	"strings"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/internal/orderedmap"
)

// Go's JSON marshaler writes float64(1.0) as 1 and float64(-0.0) as -0.
// Preserve the float type before PythonJSON formats values for the template.
func flintPythonJSON(value any) (string, error) {
	converted, err := flintJSONValue(reflect.ValueOf(value), 0)
	if err != nil {
		return "", err
	}
	return PythonJSON(converted, 0)
}

func flintJSONValue(value reflect.Value, depth int) (any, error) {
	if depth > maxPythonJSONDepth {
		return nil, fmt.Errorf("JSON nesting exceeds %d levels", maxPythonJSONDepth)
	}
	if !value.IsValid() {
		return nil, nil
	}
	if value.CanInterface() {
		switch v := value.Interface().(type) {
		case json.Number, json.RawMessage:
			return v, nil
		case *api.ToolPropertiesMap:
			if v == nil {
				return nil, nil
			}
			if v.Len() == 0 {
				b, err := json.Marshal(v)
				if err != nil || string(b) == "null" {
					return nil, err
				}
			}
			properties := orderedmap.New[string, any]()
			for name, property := range v.All() {
				converted, err := flintJSONValue(reflect.ValueOf(property), depth+1)
				if err != nil {
					return nil, err
				}
				properties.Set(name, converted)
			}
			return properties, nil
		case api.ToolCallFunctionArguments:
			arguments := orderedmap.New[string, any]()
			for name, argument := range v.All() {
				converted, err := flintJSONValue(reflect.ValueOf(argument), depth+1)
				if err != nil {
					return nil, err
				}
				arguments.Set(name, converted)
			}
			return arguments, nil
		case json.Marshaler:
			return v, nil
		}
	}

	switch value.Kind() {
	case reflect.Interface, reflect.Pointer:
		if value.IsNil() {
			return nil, nil
		}
		return flintJSONValue(value.Elem(), depth)
	case reflect.Float32, reflect.Float64:
		s := strconv.FormatFloat(value.Float(), 'g', -1, value.Type().Bits())
		if !strings.ContainsAny(s, ".eE") {
			s += ".0"
		}
		return json.Number(s), nil
	case reflect.Array, reflect.Slice:
		if value.Kind() == reflect.Slice && value.IsNil() {
			return nil, nil
		}
		if value.Type().Elem().Kind() == reflect.Uint8 {
			return value.Interface(), nil
		}
		items := make([]any, value.Len())
		for i := range items {
			converted, err := flintJSONValue(value.Index(i), depth+1)
			if err != nil {
				return nil, err
			}
			items[i] = converted
		}
		return items, nil
	case reflect.Map:
		if value.IsNil() {
			return nil, nil
		}
		if value.Type().Key().Kind() != reflect.String {
			return value.Interface(), nil
		}
		entries := make(map[string]any, value.Len())
		for _, key := range value.MapKeys() {
			converted, err := flintJSONValue(value.MapIndex(key), depth+1)
			if err != nil {
				return nil, err
			}
			entries[key.String()] = converted
		}
		return entries, nil
	case reflect.Struct:
		fields := orderedmap.New[string, any]()
		for i := range value.NumField() {
			field := value.Type().Field(i)
			if !field.IsExported() {
				continue
			}
			name, options, _ := strings.Cut(field.Tag.Get("json"), ",")
			if name == "-" {
				continue
			}
			if name == "" {
				name = field.Name
			}
			member := value.Field(i)
			if strings.Contains(","+options+",", ",omitempty,") && flintEmptyJSONValue(member) {
				continue
			}
			converted, err := flintJSONValue(member, depth+1)
			if err != nil {
				return nil, err
			}
			fields.Set(name, converted)
		}
		return fields, nil
	default:
		return value.Interface(), nil
	}
}

func flintEmptyJSONValue(value reflect.Value) bool {
	switch value.Kind() {
	case reflect.Array, reflect.Map, reflect.Slice, reflect.String:
		return value.Len() == 0
	case reflect.Bool, reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64,
		reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64,
		reflect.Float32, reflect.Float64, reflect.Interface, reflect.Pointer:
		return value.IsZero()
	default:
		return false
	}
}
