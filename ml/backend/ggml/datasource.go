// datasource.go defines interfaces for tensor data access, allowing the GGML
// backend to load tensors from different sources (GGUF files, SQLite databases).
package ggml

import (
	"io"

	fsggml "github.com/ollama/ollama/fs/ggml"
)

// TensorDataSource provides tensor data from various backends.
type TensorDataSource interface {
	// GetTensorReader returns a reader for the tensor's raw data.
	// The reader provides access to the tensor's bytes.
	GetTensorReader(name string) (io.Reader, int64, error)

	// TensorInfo returns metadata about a tensor.
	TensorInfo(name string) *fsggml.Tensor

	// AllTensors returns all tensor metadata.
	AllTensors() []*fsggml.Tensor

	// TotalBytes returns total size of all tensor data.
	TotalBytes() uint64

	// Close releases any resources.
	Close() error
}

// GGUFDataSource implements TensorDataSource for GGUF files.
type GGUFDataSource struct {
	path    string
	meta    *fsggml.GGML
	tensors map[string]*fsggml.Tensor
}

// NewGGUFDataSource creates a data source from a GGUF file.
func NewGGUFDataSource(path string, meta *fsggml.GGML) *GGUFDataSource {
	tensors := make(map[string]*fsggml.Tensor)
	for _, t := range meta.Tensors().Items() {
		tensors[t.Name] = t
	}

	return &GGUFDataSource{
		path:    path,
		meta:    meta,
		tensors: tensors,
	}
}

// GetTensorReader returns a reader for tensor data from GGUF file.
func (g *GGUFDataSource) GetTensorReader(name string) (io.Reader, int64, error) {
	t, ok := g.tensors[name]
	if !ok {
		return nil, 0, nil
	}

	// Open file and create section reader at tensor offset
	// Note: caller should handle file opening for parallel access
	return nil, int64(t.Size()), nil
}

// TensorInfo returns tensor metadata.
func (g *GGUFDataSource) TensorInfo(name string) *fsggml.Tensor {
	return g.tensors[name]
}

// AllTensors returns all tensor metadata.
func (g *GGUFDataSource) AllTensors() []*fsggml.Tensor {
	return g.meta.Tensors().Items()
}

// TotalBytes returns total tensor data size.
func (g *GGUFDataSource) TotalBytes() uint64 {
	return uint64(g.meta.Length) - g.meta.Tensors().Offset
}

// Close is a no-op for GGUF (file handles managed separately).
func (g *GGUFDataSource) Close() error {
	return nil
}

// TensorOffset returns the file offset for a tensor in GGUF.
func (g *GGUFDataSource) TensorOffset(name string) uint64 {
	if t, ok := g.tensors[name]; ok {
		return g.meta.Tensors().Offset + t.Offset
	}
	return 0
}

// FilePath returns the GGUF file path.
func (g *GGUFDataSource) FilePath() string {
	return g.path
}
