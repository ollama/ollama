package model

import (
	"log/slog"
	"reflect"
	"strconv"

	"github.com/ollama/ollama/logutil"
	"github.com/ollama/ollama/ml"
)

// PopulateVisionFromBackend populates vision model fields using tensors from a separate backend.
// This is used for split GGUF models where the vision encoder is in a separate file.
//
// Parameters:
//   - visionBackend: The backend loaded from the vision GGUF file
//   - visionModel: A pointer to the vision model struct to populate
//   - prefix: The tensor name prefix (e.g., "v" for vision tensors)
func PopulateVisionFromBackend(visionBackend ml.Backend, visionModel interface{}, prefix string) error {
	if visionBackend == nil {
		return ErrNoVisionModel
	}

	v := reflect.ValueOf(visionModel)
	if v.Kind() != reflect.Pointer {
		slog.Error("PopulateVisionFromBackend: visionModel must be a pointer")
		return ErrUnsupportedModel
	}

	v = v.Elem()
	if v.Kind() != reflect.Struct {
		slog.Error("PopulateVisionFromBackend: visionModel must point to a struct")
		return ErrUnsupportedModel
	}

	populated := populateVisionFields(visionBackend, v, []Tag{{name: prefix}})
	if !populated {
		slog.Warn("PopulateVisionFromBackend: no tensors were populated")
	}

	return nil
}

// populateVisionFields recursively populates struct fields with tensors from a backend.
// Returns true if any tensor was successfully populated.
// populateVisionFields recursively populates struct fields with tensors from a backend.
// Returns true if any tensor was successfully populated.
func populateVisionFields(backend ml.Backend, v reflect.Value, tags []Tag) bool {
	t := v.Type()
	anyPopulated := false

	if t.Kind() != reflect.Struct {
		return false
	}

	for i := range t.NumField() {
		tt := t.Field(i).Type
		vv := v.Field(i)
		if !vv.CanSet() {
			continue
		}

		// Copy tags and add field's gguf tag if present
		tagsCopy := make([]Tag, len(tags))
		copy(tagsCopy, tags)
		if tag := t.Field(i).Tag.Get("gguf"); tag != "" {
			tagsCopy = append(tagsCopy, parseTag(tag))
		}

		// Handle ml.Tensor interface fields
		if tt == reflect.TypeOf((*ml.Tensor)(nil)).Elem() {
			tensorNames := buildTensorNames(tagsCopy)
			for _, tensorName := range tensorNames {
				if tensor := backend.Get(tensorName); tensor != nil {
					logutil.Trace("PopulateVision: found tensor", "name", tensorName, "shape", tensor.Shape())
					vv.Set(reflect.ValueOf(tensor))
					anyPopulated = true
					break // Found a match, stop looking for alternatives for this field
				}
			}
		} else if tt.Kind() == reflect.Pointer {
			// Handle pointer to struct (e.g., *nn.Linear)
			elemType := tt.Elem()
			if elemType.Kind() == reflect.Struct {
				newStruct := reflect.New(elemType)
				if populateVisionFields(backend, newStruct.Elem(), tagsCopy) {
					vv.Set(newStruct)
					anyPopulated = true
				}
			}
		} else if tt.Kind() == reflect.Slice {
			// Handle slices (e.g., []VisionEncoderLayer)
			for j := 0; j < vv.Len(); j++ {
				elem := vv.Index(j)
				layerTags := append(tagsCopy, Tag{name: strconv.Itoa(j)})
				if elem.Kind() == reflect.Struct {
					if populateVisionFields(backend, elem, layerTags) {
						anyPopulated = true
					}
				} else if elem.Kind() == reflect.Pointer && !elem.IsNil() {
					if populateVisionFields(backend, elem.Elem(), layerTags) {
						anyPopulated = true
					}
				}
			}
		} else if tt.Kind() == reflect.Struct {
			// Handle embedded structs
			if populateVisionFields(backend, vv, tagsCopy) {
				anyPopulated = true
			}
		}
	}

	return anyPopulated
}

// buildTensorNames constructs all possible tensor names from a list of tags, checking alternatives.
// Returns a slice of fully qualified names.
func buildTensorNames(tags []Tag) []string {
	// Start with one empty path
	names := []string{""}

	for _, tag := range tags {
		var nextNames []string

		// Gather all variants for this segment (primary + alternatives)
		variants := []string{}
		if tag.name != "" {
			variants = append(variants, tag.name)
		}
		variants = append(variants, tag.alternatives...)

		if len(variants) == 0 {
			continue
		}

		// Cartesian product: extend each existing name with each variant
		for _, name := range names {
			for _, variant := range variants {
				if name == "" {
					nextNames = append(nextNames, variant)
				} else {
					nextNames = append(nextNames, name+"."+variant)
				}
			}
		}
		names = nextNames
	}
	return names
}
