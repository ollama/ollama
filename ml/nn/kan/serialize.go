package kan

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"os"
	"path/filepath"
)

const (
	kanFileVersion = 1
	kanMagic       = 0x4B414E31 // "KAN1"
)

// layerHeader is written before each layer's coefficient data.
type layerHeader struct {
	Magic      uint32
	Version    uint32
	NumBasis   uint32
	Order      uint32
	NumWeights uint32
}

// metadata stores convergence state and config alongside the weights.
type metadata struct {
	Version   int               `json:"version"`
	Config    Config            `json:"config"`
	Converged map[string]bool   `json:"converged"`
	Steps     map[string]int    `json:"steps"`
	EMALoss   map[string]float64 `json:"ema_loss"`
}

// Save writes all KAN layer parameters to the given directory.
// Creates one binary file per layer and a metadata.json.
//
// Directory structure:
//
//	{dir}/
//	  metadata.json
//	  layer_0.bin
//	  layer_1.bin
//	  ...
func (s *ShadowTrainer) Save(dir string) error {
	s.mu.RLock()
	defer s.mu.RUnlock()

	if err := os.MkdirAll(dir, 0o755); err != nil {
		return fmt.Errorf("kan: create directory: %w", err)
	}

	meta := metadata{
		Version:   kanFileVersion,
		Config:    s.cfg,
		Converged: make(map[string]bool),
		Steps:     make(map[string]int),
		EMALoss:   make(map[string]float64),
	}

	for key, state := range s.layers {
		meta.Converged[key] = state.converged
		meta.Steps[key] = state.stepCount
		meta.EMALoss[key] = state.emaLoss

		if err := saveLayer(filepath.Join(dir, key+".bin"), state.kan, s.cfg); err != nil {
			return fmt.Errorf("kan: save layer %s: %w", key, err)
		}
	}

	metaBytes, err := json.MarshalIndent(meta, "", "  ")
	if err != nil {
		return fmt.Errorf("kan: marshal metadata: %w", err)
	}

	if err := os.WriteFile(filepath.Join(dir, "metadata.json"), metaBytes, 0o644); err != nil {
		return fmt.Errorf("kan: write metadata: %w", err)
	}

	slog.Info("KAN parameters saved", "dir", dir, "layers", len(s.layers))
	return nil
}

// Load reads KAN layer parameters from the given directory.
// Layers that were converged will be marked as such, skipping further training.
func (s *ShadowTrainer) Load(dir string) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	metaBytes, err := os.ReadFile(filepath.Join(dir, "metadata.json"))
	if err != nil {
		if os.IsNotExist(err) {
			return nil // No saved state, start fresh
		}
		return fmt.Errorf("kan: read metadata: %w", err)
	}

	var meta metadata
	if err := json.Unmarshal(metaBytes, &meta); err != nil {
		return fmt.Errorf("kan: unmarshal metadata: %w", err)
	}

	for key, converged := range meta.Converged {
		weights, err := loadLayer(filepath.Join(dir, key+".bin"))
		if err != nil {
			slog.Warn("KAN: failed to load layer, will retrain", "layer", key, "error", err)
			continue
		}

		state := &layerState{
			kan:       NewLayerFromWeights(s.cfg, weights),
			converged: converged,
			stepCount: meta.Steps[key],
			emaLoss:   meta.EMALoss[key],
		}
		if converged {
			state.convergenceCount = s.cfg.ConvergenceWindow
		}
		s.layers[key] = state
	}

	slog.Info("KAN parameters loaded", "dir", dir, "layers", len(s.layers))
	return nil
}

func saveLayer(path string, kan *Layer, cfg Config) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()

	weights := kan.GetCoefficients()

	hdr := layerHeader{
		Magic:      kanMagic,
		Version:    kanFileVersion,
		NumBasis:   uint32(cfg.NumBasis),
		Order:      uint32(cfg.Order),
		NumWeights: uint32(len(weights)),
	}

	if err := binary.Write(f, binary.LittleEndian, &hdr); err != nil {
		return err
	}

	return binary.Write(f, binary.LittleEndian, weights)
}

func loadLayer(path string) ([]float32, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	var hdr layerHeader
	if err := binary.Read(f, binary.LittleEndian, &hdr); err != nil {
		return nil, err
	}

	if hdr.Magic != kanMagic {
		return nil, fmt.Errorf("invalid KAN file magic: %x", hdr.Magic)
	}
	if hdr.Version != kanFileVersion {
		return nil, fmt.Errorf("unsupported KAN file version: %d", hdr.Version)
	}

	weights := make([]float32, hdr.NumWeights)
	if err := binary.Read(f, binary.LittleEndian, weights); err != nil {
		if err == io.EOF {
			return nil, fmt.Errorf("truncated KAN file")
		}
		return nil, err
	}

	return weights, nil
}
