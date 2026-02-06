package mlx

// #include "generated.h"
import "C"

import (
	"encoding/binary"
	"encoding/json"
	"errors"
	"io"
	"iter"
	"log/slog"
	"maps"
	"slices"
	"unsafe"

	"github.com/ollama/ollama/types/model"
)

func Load(path string) iter.Seq2[string, *Array] {
	return func(yield func(string, *Array) bool) {
		string2array := C.mlx_map_string_to_array_new()
		defer C.mlx_map_string_to_array_free(string2array)

		string2string := C.mlx_map_string_to_string_new()
		defer C.mlx_map_string_to_string_free(string2string)

		cPath := C.CString(path)
		defer C.free(unsafe.Pointer(cPath))

		cpu := C.mlx_default_cpu_stream_new()
		defer C.mlx_stream_free(cpu)

		C.mlx_load_safetensors(&string2array, &string2string, cPath, cpu)

		it := C.mlx_map_string_to_array_iterator_new(string2array)
		defer C.mlx_map_string_to_array_iterator_free(it)

		for {
			var key *C.char
			value := C.mlx_array_new()
			if C.mlx_map_string_to_array_iterator_next(&key, &value, it) != 0 {
				break
			}

			name := C.GoString(key)
			if !yield(name, &Array{ctx: value, desc: tensorDesc{name: name, numRefs: 1000}}) {
				break
			}
		}
	}
}

func Parse(root *model.Root, path string) (map[string]Quantization, error) {
	f, err := root.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	var n uint64
	if err := binary.Read(f, binary.LittleEndian, &n); err != nil {
		return nil, err
	}

	bts := make([]byte, n)
	if _, err := io.ReadFull(f, bts); err != nil {
		return nil, err
	}

	var m struct {
		Metadata struct {
			Quantization map[string]Quantization `json:"quantization"`
		} `json:"__metadata__"`
	}
	if err := json.Unmarshal(bts, &m); err != nil {
		return nil, err
	}

	return m.Metadata.Quantization, nil
}

func LoadWeights(root *model.Root, match string, states map[string]*Array) error {
	slog.Debug("Loading weights from", "file", match)
	for name, weight := range Load(root.JoinPath("blobs", root.Real(match))) {
		if state, ok := states[name]; ok {
			*state = *weight
		}
	}

	return nil
}

func LoadQuantizations(root *model.Root, match string, quantizations map[string]*Quantization) error {
	slog.Debug("Loading quantizations from", "file", match)
	metadata, err := Parse(root, match)
	if err != nil {
		return err
	}

	for name := range metadata {
		if q, ok := quantizations[name+".weight"]; ok {
			q.GroupSize = metadata[name].GroupSize
			q.Bits = metadata[name].Bits
			q.Mode = metadata[name].Mode
		}
	}

	return nil
}

type AfterLoadFunc func(*model.Root) ([]*Array, error)

func LoadAll(root *model.Root, states map[string]*Array, quantizations map[string]*Quantization, afterLoadFuncs []AfterLoadFunc) error {
	matches, err := root.Glob("model*.safetensors")
	if err != nil {
		return err
	}

	for match := range matches {
		if err := errors.Join(
			LoadWeights(root, match, states),
			LoadQuantizations(root, match, quantizations),
		); err != nil {
			return err
		}
	}

	for _, afterLoadFunc := range afterLoadFuncs {
		weights, err := afterLoadFunc(root)
		if err != nil {
			return err
		}

		for _, weight := range weights {
			weight.desc.numRefs = 1000
			Eval(weight)

			var freeAll func(...*Array)
			freeAll = func(inputs ...*Array) {
				for _, input := range inputs {
					input.desc.numRefs = 0
					freeAll(input.desc.inputs...)
				}
				Free(inputs...)
			}

			freeAll(weight.desc.inputs...)
		}
	}

	Eval(slices.Collect(maps.Values(states))...)
	ClearCache()
	slog.Info("Loaded weights", "count", len(states), "memory", Memory{})
	return nil
}

func UnloadAll(states map[string]*Array) {
	weights := slices.Collect(maps.Values(states))
	for _, weight := range weights {
		weight.desc.numRefs = 0
	}

	numBytes := Free(weights...)
	slog.Info("Unloaded weights", "count", len(states), "num_bytes", PrettyBytes(numBytes), "memory", Memory{})
}
