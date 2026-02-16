//go:build mlx

package safetensors

import (
	"fmt"
	"reflect"
	"strings"

	"github.com/ollama/ollama/x/imagegen/mlx"
	"github.com/ollama/ollama/x/imagegen/nn"
)

// WeightSource is an interface for loading weights.
// Both ModelWeights (directory-based) and ManifestWeights (blob-based) implement this.
type WeightSource interface {
	GetTensor(name string) (*mlx.Array, error)
	ListTensors() []string
	HasTensor(name string) bool
	Quantization() string // Returns "FP4", "FP8", or ""
}

// quantizationParams returns groupSize, bits, mode for a quantization type.
// Returns defaults (32, 8, "affine") for unknown types (backward compatibility).
func quantizationParams(quantization string) (groupSize, bits int, mode string) {
	switch strings.ToUpper(quantization) {
	case "FP4":
		return 32, 4, "affine"
	default:
		return 32, 8, "affine" // FP8 or unknown
	}
}

// Transformer allows structs to transform weight arrays before assignment.
// Implement this to apply operations like transpose during loading.
type Transformer interface {
	Transform(field string, arr *mlx.Array) *mlx.Array
}

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
func LoadModule(dst any, weights WeightSource, prefix string) error {
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
func loadStruct(v reflect.Value, weights WeightSource, prefix string, errs *[]string, parentOptional bool) {
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

		// Handle nn.LinearLayer interface fields specially
		if field.Type == reflect.TypeOf((*nn.LinearLayer)(nil)).Elem() {
			if !hasTag {
				continue // no tag = skip
			}
			layer, err := LoadLinearLayer(weights, fullPath)
			if err != nil {
				if !optional {
					*errs = append(*errs, fullPath+": "+err.Error())
				}
				continue
			}
			fieldVal.Set(reflect.ValueOf(layer))
			continue
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
				// Transform before assigning if parent implements Transformer
				if t, ok := v.Addr().Interface().(Transformer); ok {
					arr = t.Transform(field.Name, arr)
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
func hasWeightsWithPrefix(weights WeightSource, prefix string) bool {
	for _, name := range weights.ListTensors() {
		if strings.HasPrefix(name, prefix+".") || name == prefix {
			return true
		}
	}
	return false
}

// loadSlice loads weights into each element of a slice of struct pointers.
func loadSlice(v reflect.Value, weights WeightSource, prefix string, errs *[]string) {
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

// LoadLinearLayer loads a linear layer from weights, automatically detecting if it's quantized.
// If {path}.weight_scale exists, dequantizes the weights.
func LoadLinearLayer(weights WeightSource, path string) (nn.LinearLayer, error) {
	// Check if this is a quantized layer by looking for scale tensor
	scalePath := path + ".weight_scale"
	if weights.HasTensor(scalePath) {
		weight, err := weights.GetTensor(path + ".weight")
		if err != nil {
			return nil, fmt.Errorf("failed to load quantized weight %s: %w", path, err)
		}

		scales, err := weights.GetTensor(scalePath)
		if err != nil {
			return nil, fmt.Errorf("failed to load scales %s: %w", scalePath, err)
		}

		// Bias is optional
		var bias *mlx.Array
		biasPath := path + ".bias"
		if weights.HasTensor(biasPath) {
			bias, _ = weights.GetTensor(biasPath)
		}

		var qbiases *mlx.Array
		qbiasPath := path + ".weight_qbias"
		if weights.HasTensor(qbiasPath) {
			qbiases, _ = weights.GetTensor(qbiasPath)
		}

		groupSize, bits, mode := quantizationParams(weights.Quantization())

		if mlx.MetalIsAvailable() {
			return &nn.QuantizedLinear{
				Weight:    weight,
				Scales:    scales,
				QBiases:   qbiases,
				Bias:      bias,
				GroupSize: groupSize,
				Bits:      bits,
				Mode:      mode,
			}, nil
		}

		dequantized := mlx.Dequantize(weight, scales, qbiases, groupSize, bits, mode)
		return nn.NewLinear(dequantized, bias), nil
	}

	// Load as regular Linear
	weight, err := weights.GetTensor(path + ".weight")
	if err != nil {
		return nil, fmt.Errorf("failed to load weight %s: %w", path, err)
	}

	// Bias is optional
	var bias *mlx.Array
	biasPath := path + ".bias"
	if weights.HasTensor(biasPath) {
		bias, _ = weights.GetTensor(biasPath)
	}

	return nn.NewLinear(weight, bias), nil
}
