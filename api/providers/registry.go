package providers

import (
	"fmt"
	"sync"
)

// Registry manages all providers
type Registry struct {
	providers map[string]Provider
	mu        sync.RWMutex
}

var globalRegistry = &Registry{
	providers: make(map[string]Provider),
}

// GetRegistry returns the global provider registry
func GetRegistry() *Registry {
	return globalRegistry
}

// Register registers a provider
func (r *Registry) Register(id string, provider Provider) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.providers[id] = provider
}

// Get returns a provider by ID
func (r *Registry) Get(id string) (Provider, error) {
	r.mu.RLock()
	defer r.mu.RUnlock()

	provider, ok := r.providers[id]
	if !ok {
		return nil, fmt.Errorf("provider not found: %s", id)
	}

	return provider, nil
}

// List returns all registered providers
func (r *Registry) List() map[string]Provider {
	r.mu.RLock()
	defer r.mu.RUnlock()

	providers := make(map[string]Provider)
	for k, v := range r.providers {
		providers[k] = v
	}

	return providers
}

// Remove removes a provider
func (r *Registry) Remove(id string) {
	r.mu.Lock()
	defer r.mu.Unlock()
	delete(r.providers, id)
}

// CreateProvider creates a new provider from config
func CreateProvider(providerType, apiKey, baseURL string) (Provider, error) {
	switch providerType {
	case "openai":
		return NewOpenAIProvider(apiKey, baseURL), nil
	case "anthropic":
		return NewAnthropicProvider(apiKey), nil
	case "google":
		return NewGoogleProvider(apiKey), nil
	case "groq":
		return NewGroqProvider(apiKey), nil
	default:
		return nil, fmt.Errorf("unknown provider type: %s", providerType)
	}
}
