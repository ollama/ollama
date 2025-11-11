package models

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"sync"
	"time"
)

// ModelInfo represents a downloadable model
type ModelInfo struct {
	ID              string    `json:"id"`
	Name            string    `json:"name"`
	Size            int64     `json:"size"`
	Description     string    `json:"description"`
	Provider        string    `json:"provider"`
	Version         string    `json:"version"`
	Downloaded      bool      `json:"downloaded"`
	DownloadedAt    time.Time `json:"downloaded_at,omitempty"`
	LastUsed        time.Time `json:"last_used,omitempty"`
	UsageCount      int       `json:"usage_count"`
	LocalPath       string    `json:"local_path,omitempty"`
	RequiredRAM     int64     `json:"required_ram"`
	ContextWindow   int       `json:"context_window"`
	Capabilities    []string  `json:"capabilities"`
}

// DownloadProgress tracks download status
type DownloadProgress struct {
	ModelID       string    `json:"model_id"`
	Status        string    `json:"status"` // downloading, completed, failed, paused
	BytesDownloaded int64   `json:"bytes_downloaded"`
	TotalBytes    int64     `json:"total_bytes"`
	Speed         float64   `json:"speed"` // bytes per second
	ETA           int       `json:"eta"` // seconds
	StartedAt     time.Time `json:"started_at"`
	Error         string    `json:"error,omitempty"`
}

// ModelManager manages local model storage and downloads
type ModelManager struct {
	modelsDir    string
	downloads    map[string]*DownloadProgress
	downloadsMu  sync.RWMutex
	models       map[string]*ModelInfo
	modelsMu     sync.RWMutex
}

// NewModelManager creates a new model manager
func NewModelManager(modelsDir string) *ModelManager {
	return &ModelManager{
		modelsDir: modelsDir,
		downloads: make(map[string]*DownloadProgress),
		models:    make(map[string]*ModelInfo),
	}
}

// ListAvailableModels returns all models available for download
func (mm *ModelManager) ListAvailableModels() []*ModelInfo {
	// In production, this would fetch from a model registry
	return []*ModelInfo{
		{
			ID:            "llama3.3-70b",
			Name:          "Llama 3.3 70B",
			Size:          40 * 1024 * 1024 * 1024, // 40 GB
			Description:   "Meta's Llama 3.3 with 70B parameters",
			Provider:      "meta",
			Version:       "3.3",
			RequiredRAM:   80 * 1024 * 1024 * 1024,
			ContextWindow: 8192,
			Capabilities:  []string{"chat", "completion"},
		},
		{
			ID:            "llama3.1-8b",
			Name:          "Llama 3.1 8B",
			Size:          4.7 * 1024 * 1024 * 1024, // 4.7 GB
			Description:   "Meta's Llama 3.1 with 8B parameters",
			Provider:      "meta",
			Version:       "3.1",
			RequiredRAM:   8 * 1024 * 1024 * 1024,
			ContextWindow: 8192,
			Capabilities:  []string{"chat", "completion"},
		},
		{
			ID:            "mistral-7b",
			Name:          "Mistral 7B",
			Size:          4.1 * 1024 * 1024 * 1024, // 4.1 GB
			Description:   "Mistral AI's 7B parameter model",
			Provider:      "mistral",
			Version:       "0.3",
			RequiredRAM:   8 * 1024 * 1024 * 1024,
			ContextWindow: 32768,
			Capabilities:  []string{"chat", "completion"},
		},
		{
			ID:            "qwen2.5-72b",
			Name:          "Qwen 2.5 72B",
			Size:          41 * 1024 * 1024 * 1024, // 41 GB
			Description:   "Alibaba's Qwen 2.5 with 72B parameters",
			Provider:      "alibaba",
			Version:       "2.5",
			RequiredRAM:   80 * 1024 * 1024 * 1024,
			ContextWindow: 32768,
			Capabilities:  []string{"chat", "completion", "multilingual"},
		},
	}
}

// ListLocalModels returns all downloaded models
func (mm *ModelManager) ListLocalModels() ([]*ModelInfo, error) {
	mm.modelsMu.RLock()
	defer mm.modelsMu.RUnlock()

	models := make([]*ModelInfo, 0, len(mm.models))
	for _, model := range mm.models {
		if model.Downloaded {
			models = append(models, model)
		}
	}

	return models, nil
}

// GetModelInfo returns info about a specific model
func (mm *ModelManager) GetModelInfo(modelID string) (*ModelInfo, error) {
	mm.modelsMu.RLock()
	defer mm.modelsMu.RUnlock()

	if model, ok := mm.models[modelID]; ok {
		return model, nil
	}

	// Check available models
	for _, model := range mm.ListAvailableModels() {
		if model.ID == modelID {
			return model, nil
		}
	}

	return nil, fmt.Errorf("model not found: %s", modelID)
}

// DownloadModel starts downloading a model
func (mm *ModelManager) DownloadModel(ctx context.Context, modelID string, downloadURL string) error {
	// Check if already downloading
	mm.downloadsMu.RLock()
	if _, exists := mm.downloads[modelID]; exists {
		mm.downloadsMu.RUnlock()
		return fmt.Errorf("model already downloading: %s", modelID)
	}
	mm.downloadsMu.RUnlock()

	// Get model info
	modelInfo, err := mm.GetModelInfo(modelID)
	if err != nil {
		return err
	}

	// Create download progress tracker
	progress := &DownloadProgress{
		ModelID:    modelID,
		Status:     "downloading",
		TotalBytes: modelInfo.Size,
		StartedAt:  time.Now(),
	}

	mm.downloadsMu.Lock()
	mm.downloads[modelID] = progress
	mm.downloadsMu.Unlock()

	// Start download in goroutine
	go mm.performDownload(ctx, modelID, downloadURL, progress)

	return nil
}

// performDownload performs the actual download
func (mm *ModelManager) performDownload(ctx context.Context, modelID, downloadURL string, progress *DownloadProgress) {
	// Create model directory
	modelPath := filepath.Join(mm.modelsDir, modelID)
	if err := os.MkdirAll(filepath.Dir(modelPath), 0755); err != nil {
		progress.Status = "failed"
		progress.Error = err.Error()
		return
	}

	// In production, this would actually download from the URL
	// For now, simulate download
	totalBytes := progress.TotalBytes
	chunkSize := int64(1024 * 1024) // 1 MB chunks
	downloaded := int64(0)

	for downloaded < totalBytes {
		select {
		case <-ctx.Done():
			progress.Status = "paused"
			return
		default:
		}

		// Simulate download chunk
		time.Sleep(100 * time.Millisecond)
		downloaded += chunkSize
		if downloaded > totalBytes {
			downloaded = totalBytes
		}

		// Update progress
		progress.BytesDownloaded = downloaded
		progress.Speed = float64(downloaded) / time.Since(progress.StartedAt).Seconds()
		if progress.Speed > 0 {
			remaining := totalBytes - downloaded
			progress.ETA = int(float64(remaining) / progress.Speed)
		}
	}

	// Mark as completed
	progress.Status = "completed"
	progress.BytesDownloaded = totalBytes

	// Update model info
	mm.modelsMu.Lock()
	if model, exists := mm.models[modelID]; exists {
		model.Downloaded = true
		model.DownloadedAt = time.Now()
		model.LocalPath = modelPath
	} else {
		// Add new model
		modelInfo, _ := mm.GetModelInfo(modelID)
		if modelInfo != nil {
			modelInfo.Downloaded = true
			modelInfo.DownloadedAt = time.Now()
			modelInfo.LocalPath = modelPath
			mm.models[modelID] = modelInfo
		}
	}
	mm.modelsMu.Unlock()
}

// GetDownloadProgress returns download progress for a model
func (mm *ModelManager) GetDownloadProgress(modelID string) (*DownloadProgress, error) {
	mm.downloadsMu.RLock()
	defer mm.downloadsMu.RUnlock()

	if progress, ok := mm.downloads[modelID]; ok {
		return progress, nil
	}

	return nil, fmt.Errorf("no download in progress for model: %s", modelID)
}

// CancelDownload cancels an ongoing download
func (mm *ModelManager) CancelDownload(modelID string) error {
	mm.downloadsMu.Lock()
	defer mm.downloadsMu.Unlock()

	if progress, ok := mm.downloads[modelID]; ok {
		progress.Status = "cancelled"
		delete(mm.downloads, modelID)
		return nil
	}

	return fmt.Errorf("no download in progress for model: %s", modelID)
}

// DeleteModel removes a downloaded model
func (mm *ModelManager) DeleteModel(modelID string) error {
	mm.modelsMu.Lock()
	defer mm.modelsMu.Unlock()

	model, ok := mm.models[modelID]
	if !ok {
		return fmt.Errorf("model not found: %s", modelID)
	}

	// Remove model file
	if model.LocalPath != "" {
		if err := os.RemoveAll(model.LocalPath); err != nil {
			return fmt.Errorf("failed to delete model file: %w", err)
		}
	}

	// Remove from models map
	delete(mm.models, modelID)

	return nil
}

// CompareModels returns comparison between two models
func (mm *ModelManager) CompareModels(modelID1, modelID2 string) (map[string]interface{}, error) {
	model1, err := mm.GetModelInfo(modelID1)
	if err != nil {
		return nil, err
	}

	model2, err := mm.GetModelInfo(modelID2)
	if err != nil {
		return nil, err
	}

	comparison := map[string]interface{}{
		"models": map[string]interface{}{
			"model1": model1,
			"model2": model2,
		},
		"differences": map[string]interface{}{
			"size_diff":     model1.Size - model2.Size,
			"ram_diff":      model1.RequiredRAM - model2.RequiredRAM,
			"context_diff":  model1.ContextWindow - model2.ContextWindow,
			"provider_same": model1.Provider == model2.Provider,
		},
	}

	return comparison, nil
}

// GetStorageStats returns storage statistics
func (mm *ModelManager) GetStorageStats() (map[string]interface{}, error) {
	mm.modelsMu.RLock()
	defer mm.modelsMu.RUnlock()

	totalSize := int64(0)
	modelCount := 0

	for _, model := range mm.models {
		if model.Downloaded {
			totalSize += model.Size
			modelCount++
		}
	}

	// Get disk space
	var stat struct {
		Total int64
		Free  int64
		Used  int64
	}

	// In production, use syscall to get actual disk stats
	stat.Total = 1024 * 1024 * 1024 * 1024 // 1 TB
	stat.Used = totalSize
	stat.Free = stat.Total - stat.Used

	return map[string]interface{}{
		"total_models":     modelCount,
		"total_size":       totalSize,
		"disk_total":       stat.Total,
		"disk_used":        stat.Used,
		"disk_free":        stat.Free,
		"disk_usage_pct":   float64(stat.Used) / float64(stat.Total) * 100,
	}, nil
}

// UpdateModelUsage updates usage statistics for a model
func (mm *ModelManager) UpdateModelUsage(modelID string) error {
	mm.modelsMu.Lock()
	defer mm.modelsMu.Unlock()

	if model, ok := mm.models[modelID]; ok {
		model.UsageCount++
		model.LastUsed = time.Now()
		return nil
	}

	return fmt.Errorf("model not found: %s", modelID)
}
