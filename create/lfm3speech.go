package create

import (
	"errors"
	"fmt"
	"os"
	"path"
	"path/filepath"

	"github.com/ollama/ollama/fs/safetensors"
)

// lfm3CodecPrefix names the imported speech codec tensors; the runner loads
// the codec from this prefix and the subfolder's config.json.
const lfm3CodecPrefix = "codec."

// lfm3CodecDirs maps the LFM3 speech-output architectures to the subfolder
// holding their LiquidCodec checkpoint.
var lfm3CodecDirs = map[string]string{
	"Lfm3OmniForConditionalGeneration":  "audio_tokenizer",
	"Lfm3AudioForConditionalGeneration": "audio_tokenizer",
}

// lfm3CodecDir returns the codec subfolder of dir's checkpoint, or "" when
// the checkpoint does not speak.
func lfm3CodecDir(dir string) (string, error) {
	if _, err := os.Stat(filepath.Join(dir, "config.json")); errors.Is(err, os.ErrNotExist) {
		return "", nil
	} else if err != nil {
		return "", err
	}
	cfg, _, err := readSourceModelConfig(dir)
	if err != nil {
		return "", err
	}
	return lfm3CodecDirs[cfg.Architecture()], nil
}

// lfm3CodecFile is the codec subfolder's file name, slash-separated as
// transfer lists and manifest config names are.
func lfm3CodecFile(dir, codecDir, name string) (string, error) {
	file := path.Join(codecDir, name)
	info, err := os.Stat(filepath.Join(dir, filepath.FromSlash(file)))
	if err != nil {
		return "", fmt.Errorf("LFM3 speech output requires %s: %w", file, err)
	}
	if !info.Mode().IsRegular() {
		return "", fmt.Errorf("LFM3 speech codec %s must be a regular file", file)
	}
	return file, nil
}

// prepareLFM3SpeechInventory adds the codec checkpoint, kept in its own
// subfolder by the publisher, under lfm3CodecPrefix.
func prepareLFM3SpeechInventory(inv Inventory) (Inventory, error) {
	codecDir := lfm3CodecDirs[inv.Config.Architecture()]
	if codecDir == "" {
		return inv, nil
	}
	file, err := lfm3CodecFile(inv.Dir, codecDir, "model.safetensors")
	if err != nil {
		return Inventory{}, err
	}
	reader, err := safetensors.OpenForExtraction(filepath.Join(inv.Dir, filepath.FromSlash(file)))
	if err != nil {
		return Inventory{}, fmt.Errorf("open LFM3 speech codec: %w", err)
	}
	defer reader.Close()
	names := reader.ListTensors()
	if len(names) == 0 {
		return Inventory{}, fmt.Errorf("LFM3 speech codec %s has no tensors", file)
	}
	for _, name := range names {
		td, err := reader.GetTensor(name)
		if err != nil {
			return Inventory{}, fmt.Errorf("read LFM3 speech codec tensor %s: %w", name, err)
		}
		key := lfm3CodecPrefix + name
		if inv.Has(key) {
			return Inventory{}, fmt.Errorf("LFM3 checkpoint already defines %s", key)
		}
		inv.Tensors[key] = SourceTensor{Name: name, Dtype: td.Dtype, Shape: td.Shape, File: file}
	}
	return inv, nil
}
