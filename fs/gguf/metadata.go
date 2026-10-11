package gguf

import (
	"fmt"
	"math"
	"slices"
	"strings"
)

const defaultMetadataArraySize = 1024

// Metadata contains the GGUF key-values and tensor descriptors needed after
// the underlying file has been closed.
type Metadata struct {
	values         map[string]Value
	tensors        []TensorInfo
	omitted        []string
	parameterCount uint64
}

// ReadFileMetadata reads one GGUF file without retaining array values larger
// than maxArraySize. Zero uses the runtime default and a negative value retains
// all arrays.
func ReadFileMetadata(path string, maxArraySize int) (_ *Metadata, err error) {
	if maxArraySize == 0 {
		maxArraySize = defaultMetadataArraySize
	}
	f, err := open(path, maxArraySize)
	if err != nil {
		return nil, err
	}
	defer func() {
		if closeErr := f.Close(); err == nil {
			err = closeErr
		}
	}()

	m := &Metadata{values: make(map[string]Value)}
	for _, kv := range f.KeyValues() {
		m.values[kv.Key] = kv.Value
		if _, ok := kv.Value.value.(skippedArray); ok {
			m.omitted = append(m.omitted, kv.Key)
		}
	}
	for _, tensor := range f.TensorInfos() {
		m.tensors = append(m.tensors, tensor)
		n := tensor.NumValues()
		if n < 0 || uint64(n) > math.MaxUint64-m.parameterCount {
			return nil, fmt.Errorf("%w parameter count overflows", ErrUnsupported)
		}
		m.parameterCount += uint64(n)
	}
	if err := f.validateTensorData(); err != nil {
		return nil, err
	}
	return m, nil
}

// ExactKeyValue returns a key without architecture qualification.
func (m *Metadata) ExactKeyValue(key string) KeyValue {
	if m == nil {
		return KeyValue{}
	}
	value, ok := m.values[key]
	if !ok {
		return KeyValue{}
	}
	return KeyValue{Key: key, Value: value}
}

func (m *Metadata) KeyValue(key string) KeyValue {
	if m == nil {
		return KeyValue{}
	}
	if strings.HasPrefix(key, "split.") {
		if value := m.ExactKeyValue(key); value.Valid() {
			return value
		}
	}
	if !strings.HasPrefix(key, "general.") && !strings.HasPrefix(key, "tokenizer.") {
		key = m.Architecture() + "." + key
	}
	value, ok := m.values[key]
	if !ok {
		return KeyValue{}
	}
	return KeyValue{Key: key, Value: value}
}

func (m *Metadata) Has(key string) bool {
	return m.KeyValue(key).Valid()
}

func (m *Metadata) NumTensors() int {
	if m == nil {
		return 0
	}
	return len(m.tensors)
}

func (m *Metadata) String(key string, defaultValue ...string) string {
	value := m.KeyValue(key)
	if !value.Valid() && len(defaultValue) > 0 {
		return defaultValue[0]
	}
	return value.String()
}

func (m *Metadata) Uint(key string, defaultValue ...uint64) uint64 {
	if n, ok := m.UintOK(key); ok {
		return n
	}
	if len(defaultValue) > 0 {
		return defaultValue[0]
	}
	return 0
}

// UintOK returns a non-negative integer value as uint64.
func (m *Metadata) UintOK(key string) (uint64, bool) {
	value := m.KeyValue(key)
	if n, ok := value.UintOK(); ok {
		return n, true
	}
	if n, ok := value.IntOK(); ok && n >= 0 {
		return uint64(n), true
	}
	return 0, false
}

func (m *Metadata) Bool(key string, defaultValue ...bool) bool {
	value := m.KeyValue(key)
	if _, ok := value.Any().(bool); ok {
		return value.Bool()
	}
	if len(defaultValue) > 0 {
		return defaultValue[0]
	}
	return false
}

func (m *Metadata) Architecture() string {
	return m.String("general.architecture", "unknown")
}

func (m *Metadata) Kind() string {
	return m.String("general.type", "unknown")
}

func (m *Metadata) FileType() FileType {
	n := m.Uint("general.file_type", uint64(FileTypeUnknown))
	if n > math.MaxUint32 {
		return FileTypeUnknown
	}
	return FileType(n)
}

func (m *Metadata) BlockCount() uint64 {
	return m.Uint("block_count")
}

func (m *Metadata) EmbeddingLength() uint64 {
	return m.Uint("embedding_length")
}

func (m *Metadata) ContextLength() uint64 {
	return m.Uint("context_length")
}

func (m *Metadata) ChatTemplate() string {
	return m.String("tokenizer.chat_template")
}

func (m *Metadata) HeadCountMax() uint64 {
	_, maximum := m.uintRange("attention.head_count", 1)
	return maximum
}

func (m *Metadata) HeadCountKVMin() uint64 {
	minimum, _ := m.uintRange("attention.head_count_kv", 1)
	return minimum
}

func (m *Metadata) uintRange(key string, defaultValue uint64) (uint64, uint64) {
	value := m.KeyValue(key)
	values := value.Uints()
	if len(values) == 0 {
		for _, n := range value.Ints() {
			if n < 0 {
				return defaultValue, defaultValue
			}
			values = append(values, uint64(n))
		}
	}
	if len(values) == 0 {
		n := m.Uint(key, defaultValue)
		return n, n
	}
	return slices.Min(values), slices.Max(values)
}

func (m *Metadata) ParameterCount() uint64 {
	if m == nil {
		return 0
	}
	return m.parameterCount
}

// OmittedKeys returns array keys whose values exceeded the requested limit.
func (m *Metadata) OmittedKeys() []string {
	if m == nil {
		return nil
	}
	return slices.Clone(m.omitted)
}

func (m *Metadata) TensorInfos(prefix ...string) []TensorInfo {
	if m == nil {
		return nil
	}
	if len(prefix) == 0 {
		return cloneTensorInfos(m.tensors)
	}
	var tensors []TensorInfo
	for _, tensor := range m.tensors {
		if strings.HasPrefix(tensor.Name, prefix[0]) {
			tensors = append(tensors, cloneTensorInfo(tensor))
		}
	}
	return tensors
}

func (m *Metadata) Values() map[string]any {
	if m == nil {
		return nil
	}
	values := make(map[string]any, len(m.values))
	for key, value := range m.values {
		if _, ok := value.value.(skippedArray); ok {
			values[key] = []any{}
		} else {
			values[key] = value.Any()
		}
	}
	return values
}
