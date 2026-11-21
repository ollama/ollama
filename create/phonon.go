package create

import (
	"errors"
	"fmt"
	"os"
	"path"
	"path/filepath"
	"slices"
	"strings"

	"github.com/ollama/ollama/fs/safetensors"
)

const gradiumPhononArchitecture = "GradiumPhononTTSForConditionalGeneration"

func isPhononArchitecture(architecture string) bool {
	return architecture == gradiumPhononArchitecture
}

// The publisher keeps its default speaker latents beside the model weights
// and any other voices in voices/NAME.safetensors. Import each as a separate
// tensor so the runner can project them with the checkpoint's own speaker
// projection at load time.
func prepareGradiumPhononInventory(inv Inventory) (Inventory, error) {
	if !inv.Has("flow_lm.condition_provider.conditioners.speaker_wavs.output_proj.weight") {
		return Inventory{}, fmt.Errorf("Phonon checkpoint has no speaker projection")
	}
	files, err := phononVoiceFiles(inv.Dir)
	if err != nil {
		return Inventory{}, err
	}
	if err := addPhononVoice(inv, "default", "default-voice.safetensors"); err != nil {
		return Inventory{}, err
	}
	for _, file := range files {
		if err := addPhononVoice(inv, strings.TrimSuffix(path.Base(file), ".safetensors"), file); err != nil {
			return Inventory{}, err
		}
	}
	return inv, nil
}

func addPhononVoice(inv Inventory, name, file string) error {
	reader, err := safetensors.OpenForExtraction(filepath.Join(inv.Dir, file))
	if err != nil {
		return fmt.Errorf("open Phonon %s voice: %w", name, err)
	}
	defer reader.Close()
	if names := reader.ListTensors(); !slices.Equal(names, []string{"speaker_wavs"}) {
		return fmt.Errorf("Phonon %s voice must contain only speaker_wavs, got %v", name, names)
	}
	voice, err := reader.GetTensor("speaker_wavs")
	if err != nil {
		return fmt.Errorf("read Phonon %s voice: %w", name, err)
	}
	if voice.Dtype != "F32" || len(voice.Shape) != 3 || voice.Shape[0] != 1 || voice.Shape[1] != 512 || voice.Shape[2] <= 0 {
		return fmt.Errorf("unsupported Phonon %s voice: %s %v", name, voice.Dtype, voice.Shape)
	}
	if inv.Has("voices." + name) {
		return fmt.Errorf("Phonon checkpoint already defines voices.%s", name)
	}
	inv.Tensors["voices."+name] = SourceTensor{
		Name: "speaker_wavs", Dtype: voice.Dtype, Shape: voice.Shape, File: file,
	}
	return nil
}

// phononVoiceFiles lists a checkpoint's voices/NAME.safetensors files.
func phononVoiceFiles(dir string) ([]string, error) {
	entries, err := os.ReadDir(filepath.Join(dir, "voices"))
	if errors.Is(err, os.ErrNotExist) {
		return nil, nil
	}
	if err != nil {
		return nil, err
	}
	var files []string
	for _, entry := range entries {
		if name, ok := strings.CutSuffix(entry.Name(), ".safetensors"); ok && name != "" && !entry.IsDir() {
			files = append(files, path.Join("voices", entry.Name()))
		}
	}
	return files, nil
}
