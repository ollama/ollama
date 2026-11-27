package server

import (
	"fmt"

	"github.com/ollama/ollama/manifest"
	"github.com/ollama/ollama/mlxrunner/model/lora"
	"github.com/ollama/ollama/types/model"
)

func peftLayers(base []*modelLayer, cfg model.ConfigV2, files map[string]string) ([]*modelLayer, error) {
	if cfg.ModelFormat != manifest.FormatSafetensors || cfg.RemoteHost != "" {
		return nil, errAdaptersUnsupported
	}
	var m manifest.Manifest
	for _, l := range base {
		m.Layers = append(m.Layers, l.Layer)
	}
	ref, err := m.Adapter()
	if err != nil {
		return nil, err
	}
	if ref.Weights != "" {
		return nil, fmt.Errorf("FROM already has a PEFT adapter")
	}
	raw, err := m.ReadConfig("config.json")
	if err != nil {
		return nil, err
	}
	layout, err := lora.ModelLayout(raw)
	if err != nil {
		return nil, err
	}
	ref = manifest.AdapterRef{Config: files[manifest.AdapterConfigName], Weights: files[manifest.AdapterWeightsName]}
	if _, err := lora.Inspect(ref, layout); err != nil {
		return nil, err
	}
	cfgLayer, err := manifest.NewLayerFromLayer(ref.Config, manifest.MediaTypeImageJSON, "")
	if err != nil {
		return nil, err
	}
	cfgLayer.Name = manifest.AdapterConfigName
	weightLayer, err := manifest.NewLayerFromLayer(ref.Weights, manifest.MediaTypePEFTAdapter, "")
	if err != nil {
		return nil, err
	}
	weightLayer.Name = manifest.AdapterWeightsName
	return []*modelLayer{{Layer: cfgLayer}, {Layer: weightLayer}}, nil
}
