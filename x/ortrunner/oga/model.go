package oga

// #include "oga.h"
// #include <stdlib.h>
import "C"

import (
	"fmt"
	"unsafe"
)

// ogaError converts an OgaResult to a Go error and frees the result.
func ogaError(result *C.OgaResult) error {
	if result == nil {
		return nil
	}
	msg := C.GoString(C.OgaResultGetError(result))
	C.OgaDestroyResult(result)
	return fmt.Errorf("ort genai: %s", msg)
}

// Config wraps an OgaConfig handle.
type Config struct {
	ctx *C.OgaConfig
}

// NewConfig creates a new ORT GenAI config from a model directory path.
func NewConfig(modelDir string) (*Config, error) {
	cPath := C.CString(modelDir)
	defer C.free(unsafe.Pointer(cPath))

	var cfg *C.OgaConfig
	if err := ogaError(C.OgaCreateConfig(cPath, &cfg)); err != nil {
		return nil, err
	}
	return &Config{ctx: cfg}, nil
}

// ClearProviders removes all execution providers from the config.
func (c *Config) ClearProviders() error {
	return ogaError(C.OgaConfigClearProviders(c.ctx))
}

// AppendProvider adds an execution provider to the config.
func (c *Config) AppendProvider(name string) error {
	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))
	return ogaError(C.OgaConfigAppendProvider(c.ctx, cName))
}

// SetProviderOption sets a key/value option on an execution provider.
func (c *Config) SetProviderOption(provider, key, value string) error {
	cProvider := C.CString(provider)
	defer C.free(unsafe.Pointer(cProvider))
	cKey := C.CString(key)
	defer C.free(unsafe.Pointer(cKey))
	cValue := C.CString(value)
	defer C.free(unsafe.Pointer(cValue))
	return ogaError(C.OgaConfigSetProviderOption(c.ctx, cProvider, cKey, cValue))
}

// Close frees the config.
func (c *Config) Close() {
	if c.ctx != nil {
		C.OgaDestroyConfig(c.ctx)
		c.ctx = nil
	}
}

// Model wraps an OgaModel handle.
type Model struct {
	ctx *C.OgaModel
}

// NewModel creates a model from a config.
func NewModel(config *Config) (*Model, error) {
	var m *C.OgaModel
	if err := ogaError(C.OgaCreateModelFromConfig(config.ctx, &m)); err != nil {
		return nil, err
	}
	return &Model{ctx: m}, nil
}

// Close frees the model.
func (m *Model) Close() {
	if m.ctx != nil {
		C.OgaDestroyModel(m.ctx)
		m.ctx = nil
	}
}
