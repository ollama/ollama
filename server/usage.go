package server

import (
	"sync"
	"time"

	"github.com/ollama/ollama/api"
)

type ModelUsage struct {
	Requests         int64
	PromptTokens     int64
	CompletionTokens int64
}

type UsageTracker struct {
	mu     sync.Mutex
	start  time.Time
	models map[string]*ModelUsage
}

func NewUsageTracker() *UsageTracker {
	return &UsageTracker{
		start:  time.Now().UTC(),
		models: make(map[string]*ModelUsage),
	}
}

func (u *UsageTracker) Record(model string, promptTokens, completionTokens int) {
	u.mu.Lock()
	defer u.mu.Unlock()

	m, ok := u.models[model]
	if !ok {
		m = &ModelUsage{}
		u.models[model] = m
	}

	m.Requests++
	m.PromptTokens += int64(promptTokens)
	m.CompletionTokens += int64(completionTokens)
}

func (u *UsageTracker) Stats() api.UsageResponse {
	u.mu.Lock()
	defer u.mu.Unlock()

	byModel := make([]api.ModelUsageData, 0, len(u.models))
	for model, usage := range u.models {
		byModel = append(byModel, api.ModelUsageData{
			Model:            model,
			Requests:         usage.Requests,
			PromptTokens:     usage.PromptTokens,
			CompletionTokens: usage.CompletionTokens,
		})
	}

	return api.UsageResponse{
		Start: u.start,
		Usage: byModel,
	}
}
