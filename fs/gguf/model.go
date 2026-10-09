package gguf

import (
	"fmt"
	"maps"
	"math"
	"os"
	"slices"
	"strings"
)

// Model is the metadata view of a logical GGUF model. A model may span
// multiple files, but callers see one set of key-values and tensors.
type Model struct {
	files    []string
	fileSize uint64
	kv       *Metadata
	tensors  Tensors
}

// ReadModel reads a GGUF model and any additional shard files. The first file
// supplies the model key-values; tensor metadata and derived counts are
// aggregated across all files.
func ReadModel(path string, maxArraySize int, shards ...string) (*Model, error) {
	paths := make([]string, 1, len(shards)+1)
	paths[0] = path
	paths = append(paths, shards...)

	var kv *Metadata
	var tensors []TensorInfo
	var fileSize, parameterCount, tensorBytes uint64
	tensorNames := make(map[string]struct{})
	var architecture string
	var fileType FileType

	for i, path := range paths {
		metadata, err := ReadFileMetadata(path, maxArraySize)
		if err != nil {
			return nil, fmt.Errorf("read GGUF metadata %q: %w", path, err)
		}
		info, err := os.Stat(path)
		if err != nil {
			return nil, fmt.Errorf("stat GGUF file %q: %w", path, err)
		}
		if info.Size() < 0 || uint64(info.Size()) > math.MaxUint64-fileSize {
			return nil, fmt.Errorf("%w GGUF file sizes overflow", ErrUnsupported)
		}
		fileSize += uint64(info.Size())
		if i == 0 {
			kvCopy := *metadata
			kvCopy.values = maps.Clone(metadata.values)
			kv = &kvCopy
			architecture = metadata.Architecture()
			fileType = metadata.FileType()
		} else {
			if arch := metadata.Architecture(); architecture != "unknown" && arch != "unknown" && arch != architecture {
				return nil, fmt.Errorf("%w GGUF shard %q architecture %q does not match %q", ErrUnsupported, path, arch, architecture)
			}
			if shardFileType := metadata.FileType(); fileType != FileTypeUnknown && shardFileType != FileTypeUnknown && shardFileType != fileType {
				return nil, fmt.Errorf("%w GGUF shard %q file type %s does not match %s", ErrUnsupported, path, shardFileType, fileType)
			}
		}

		if metadata.parameterCount > math.MaxUint64-parameterCount {
			return nil, fmt.Errorf("%w GGUF parameter count overflows", ErrUnsupported)
		}
		parameterCount += metadata.parameterCount

		for _, tensor := range metadata.tensors {
			if _, ok := tensorNames[tensor.Name]; ok {
				return nil, fmt.Errorf("%w GGUF contains duplicate tensor %q", ErrUnsupported, tensor.Name)
			}
			tensorNames[tensor.Name] = struct{}{}

			size := tensor.NumBytes()
			if size < 0 || uint64(size) > math.MaxUint64-tensorBytes {
				return nil, fmt.Errorf("%w GGUF tensor sizes overflow", ErrUnsupported)
			}
			tensorBytes += uint64(size)
			tensors = append(tensors, cloneTensorInfo(tensor))
		}
	}

	kv.parameterCount = parameterCount
	kv.tensors = tensors
	kv.values["general.parameter_count"] = Value{value: parameterCount}
	return &Model{
		files:    paths,
		fileSize: fileSize,
		kv:       kv,
		tensors:  Tensors{items: tensors, size: tensorBytes},
	}, nil
}

// Files returns the GGUF files that make up the model, with the primary file first.
func (m *Model) Files() []string {
	if m == nil {
		return nil
	}
	return slices.Clone(m.files)
}

// FileSize returns the aggregate size of the model's GGUF files.
func (m *Model) FileSize() uint64 {
	if m == nil {
		return 0
	}
	return m.fileSize
}

// KV returns the model key-values.
func (m *Model) KV() *Metadata {
	if m == nil {
		return nil
	}
	return m.kv
}

// Tensors returns the model tensor metadata.
func (m *Model) Tensors() Tensors {
	if m == nil {
		return Tensors{}
	}
	return m.tensors
}

// Tensors is a read-only view of tensor metadata.
type Tensors struct {
	items []TensorInfo
	size  uint64
}

// Items returns all tensors, or tensors whose names start with prefix.
func (t Tensors) Items(prefix ...string) []TensorInfo {
	if len(prefix) == 0 {
		return cloneTensorInfos(t.items)
	}

	var items []TensorInfo
	for _, tensor := range t.items {
		if strings.HasPrefix(tensor.Name, prefix[0]) {
			items = append(items, cloneTensorInfo(tensor))
		}
	}
	return items
}

// Size returns the total tensor data size. When prefixes are supplied, each
// tensor whose name starts with any prefix is counted once.
func (t Tensors) Size(prefixes ...string) uint64 {
	if len(prefixes) == 0 {
		return t.size
	}

	var size uint64
	for _, tensor := range t.items {
		for _, prefix := range prefixes {
			if strings.HasPrefix(tensor.Name, prefix) {
				size += uint64(tensor.NumBytes())
				break
			}
		}
	}
	return size
}

func cloneTensorInfos(tensors []TensorInfo) []TensorInfo {
	cloned := make([]TensorInfo, len(tensors))
	for i, tensor := range tensors {
		cloned[i] = cloneTensorInfo(tensor)
	}
	return cloned
}

func cloneTensorInfo(tensor TensorInfo) TensorInfo {
	tensor.Shape = slices.Clone(tensor.Shape)
	return tensor
}
