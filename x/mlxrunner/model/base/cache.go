package base

import "github.com/ollama/ollama/x/mlxrunner/cache"

// Cacher is implemented by models that support custom caching mechanisms.
type Cacher interface {
	Cache() []cache.Cache
}
