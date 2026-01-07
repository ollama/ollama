package safetensors

import (
	"fmt"
	"reflect"
	"strings"

	"github.com/ollama/ollama/x/imagegen/mlx"
)

// LoadModule loads weights into a struct using reflection and struct tags.
//
// Struct tags use the format: `weight:"path[,optional]"`
//   - path: the weight name suffix (appended to prefix)
//   - optional: if present, missing weights don't cause errors
//   - "-": skip this field entirely
//   - no tag on struct pointer: recurse with current prefix
//   - no tag on *mlx.Array: skip (computed fields don't need loading)
//
// For slices of struct pointers, the loader iterates with .0, .1, .2... suffixes.
// The slice must be pre-allocated to the correct length.
//
// Example:
//
//	type Attention struct {
//	    QProj *nn.Linear  `weight:"self_attn.q_proj"`
//	    KProj *nn.Linear  `weight:"self_attn.k_proj"`
//	    Cache *mlx.Array  // no tag = skipped (computed field)
//	}
//
//	err := LoadModule(&attn, weights, "model.layers.0")
func LoadModule(dst any, weights *ModelWeights, prefix string) error {
	v := reflect.ValueOf(dst)
	if v.Kind() != reflect.Ptr || v.IsNil() {
		return fmt.Errorf("LoadModule: dst must be a non-nil pointer")
	}
	v = v.Elem()
	if v.Kind() != reflect.Struct {
		return fmt.Errorf("LoadModule: dst must be a pointer to struct, got %v", v.Kind())
	}

	var errs []string
	loadStruct(v, weights, prefix, &errs, false)

	if len(errs) > 0 {
		return fmt.Errorf("LoadModule: missing weights:\n  %s", strings.Join(errs, "\n  "))
	}
	return nil
}

// loadStruct recursively loads weights into a struct value.
func loadStruct(v reflect.Value, weights *ModelWeights, prefix string, errs *[]string, parentOptional bool) {
	t := v.Type()

	for i := 0; i < t.NumField(); i++ {
		field := t.Field(i)
		fieldVal := v.Field(i)

		// Skip unexported fields
		if !fieldVal.CanSet() {
			continue
		}

		// Parse tag
		tag, hasTag := field.Tag.Lookup("weight")
		if tag == "-" {
			continue
		}

		// Parse tag options
		optional := parentOptional
		weightPath := tag
		if idx := strings.Index(tag, ","); idx != -1 {
			weightPath = tag[:idx]
			if strings.Contains(tag[idx+1:], "optional") {
				optional = true
			}
		}

		// Build full path
		fullPath := joinPath(prefix, weightPath)

		// For struct pointers without a tag, recurse with current prefix
		if !hasTag && fieldVal.Kind() == reflect.Ptr {
			elemType := fieldVal.Type().Elem()
			if elemType.Kind() == reflect.Struct && elemType != reflect.TypeOf(mlx.Array{}) {
				if fieldVal.IsNil() {
					fieldVal.Set(reflect.New(elemType))
				}
				loadStruct(fieldVal.Elem(), weights, prefix, errs, optional)
				continue
			}
		}

		// Handle by kind
		switch fieldVal.Kind() {
		case reflect.Ptr:
			elemType := fieldVal.Type().Elem()

			// *mlx.Array - load directly (but skip if no tag - computed fields)
			if fieldVal.Type() == reflect.TypeOf((*mlx.Array)(nil)) {
				if !hasTag {
					continue // no tag on *mlx.Array = computed field, skip
				}
				arr, err := weights.GetTensor(fullPath)
				if err != nil {
					if !optional {
						*errs = append(*errs, fullPath)
					}
					continue
				}
				fieldVal.Set(reflect.ValueOf(arr))
				continue
			}

			// Pointer to struct - allocate and recurse
			if elemType.Kind() == reflect.Struct {
				if optional && !hasWeightsWithPrefix(weights, fullPath) {
					continue
				}
				if fieldVal.IsNil() {
					fieldVal.Set(reflect.New(elemType))
				}
				loadStruct(fieldVal.Elem(), weights, fullPath, errs, optional)
			}

		case reflect.Slice:
			elemType := fieldVal.Type().Elem()
			if elemType.Kind() == reflect.Ptr && elemType.Elem().Kind() == reflect.Struct {
				loadSlice(fieldVal, weights, fullPath, errs)
			}
		}
	}
}

// hasWeightsWithPrefix checks if any weights exist with the given prefix.
func hasWeightsWithPrefix(weights *ModelWeights, prefix string) bool {
	for _, name := range weights.ListTensors() {
		if strings.HasPrefix(name, prefix+".") || name == prefix {
			return true
		}
	}
	return false
}

// loadSlice loads weights into each element of a slice of struct pointers.
func loadSlice(v reflect.Value, weights *ModelWeights, prefix string, errs *[]string) {
	elemStructType := v.Type().Elem().Elem()

	for i := 0; i < v.Len(); i++ {
		elem := v.Index(i)
		if elem.IsNil() {
			elem.Set(reflect.New(elemStructType))
		}
		loadStruct(elem.Elem(), weights, fmt.Sprintf("%s.%d", prefix, i), errs, false)
	}
}

// joinPath joins path segments with dots, handling empty segments.
func joinPath(prefix, suffix string) string {
	if prefix == "" {
		return suffix
	}
	if suffix == "" {
		return prefix
	}
	return prefix + "." + suffix
}
