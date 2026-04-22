package model

import (
	"encoding/json"
	"strings"

	"github.com/ollama/ollama/x/imagegen/manifest"
)

// readConfigQuantOverrides parses config.json's "quantization" (or
// "quantization_config") block. Returns global defaults and per-path
// overrides keyed by <path>.weight. Overrides that omit "mode" default to
// "affine" (mx.nn.Linear.to_quantized's default). Missing or malformed
// config is a silent fallback (zero values, nil error).
func readConfigQuantOverrides(m *manifest.ModelManifest) (TensorQuantInfo, map[string]*TensorQuantInfo, error) {
	data, err := m.ReadConfig("config.json")
	if err != nil {
		return TensorQuantInfo{}, nil, nil
	}

	var cfg map[string]json.RawMessage
	if err := json.Unmarshal(data, &cfg); err != nil {
		return TensorQuantInfo{}, nil, nil
	}

	raw, ok := cfg["quantization"]
	if !ok {
		raw, ok = cfg["quantization_config"]
	}
	if !ok {
		return TensorQuantInfo{}, nil, nil
	}

	var block map[string]json.RawMessage
	if err := json.Unmarshal(raw, &block); err != nil {
		return TensorQuantInfo{}, nil, nil
	}

	var defaults TensorQuantInfo
	var overrides map[string]*TensorQuantInfo

	for key, val := range block {
		switch key {
		case "group_size":
			_ = json.Unmarshal(val, &defaults.GroupSize)
		case "bits":
			_ = json.Unmarshal(val, &defaults.Bits)
		case "mode":
			_ = json.Unmarshal(val, &defaults.Mode)
		default:
			var inner map[string]json.RawMessage
			if err := json.Unmarshal(val, &inner); err != nil {
				continue
			}
			info := &TensorQuantInfo{}
			if v, ok := inner["group_size"]; ok {
				_ = json.Unmarshal(v, &info.GroupSize)
			}
			if v, ok := inner["bits"]; ok {
				_ = json.Unmarshal(v, &info.Bits)
			}
			if v, ok := inner["mode"]; ok {
				_ = json.Unmarshal(v, &info.Mode)
			}
			if info.Mode == "" {
				info.Mode = "affine"
			}
			info.QuantType = strings.ToUpper(info.Mode)
			if overrides == nil {
				overrides = make(map[string]*TensorQuantInfo)
			}
			overrides[key+".weight"] = info
		}
	}

	if defaults.Mode != "" {
		defaults.QuantType = strings.ToUpper(defaults.Mode)
	}
	return defaults, overrides, nil
}
