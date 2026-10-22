package create

import (
	"fmt"
	"path/filepath"
	"slices"

	"github.com/ollama/ollama/fs/safetensors"
)

const gradiumPhononArchitecture = "GradiumPhononTTSForConditionalGeneration"

func isPhononArchitecture(architecture string) bool {
	return architecture == gradiumPhononArchitecture
}

// The publisher keeps its default speaker latents beside the model weights.
// Import them as a separate tensor so the runner can project them with the
// checkpoint's own speaker projection at load time.
func prepareGradiumPhononInventory(inv Inventory) (Inventory, error) {
	const filename = "default-voice.safetensors"
	reader, err := safetensors.OpenForExtraction(filepath.Join(inv.Dir, filename))
	if err != nil {
		return Inventory{}, fmt.Errorf("open Phonon default voice: %w", err)
	}
	defer reader.Close()
	if names := reader.ListTensors(); !slices.Equal(names, []string{"speaker_wavs"}) {
		return Inventory{}, fmt.Errorf("Phonon default voice must contain only speaker_wavs, got %v", names)
	}
	voice, err := reader.GetTensor("speaker_wavs")
	if err != nil {
		return Inventory{}, fmt.Errorf("read Phonon default voice: %w", err)
	}
	if voice.Dtype != "F32" || len(voice.Shape) != 3 || voice.Shape[0] != 1 || voice.Shape[1] != 512 || voice.Shape[2] <= 0 {
		return Inventory{}, fmt.Errorf("unsupported Phonon default voice: %s %v", voice.Dtype, voice.Shape)
	}
	if !inv.Has("flow_lm.condition_provider.conditioners.speaker_wavs.output_proj.weight") {
		return Inventory{}, fmt.Errorf("Phonon checkpoint has no speaker projection")
	}
	if inv.Has("voices.default") {
		return Inventory{}, fmt.Errorf("Phonon checkpoint already defines voices.default")
	}
	inv.Tensors["voices.default"] = SourceTensor{
		Name: "speaker_wavs", Dtype: voice.Dtype, Shape: voice.Shape, File: filename,
	}
	return inv, nil
}
