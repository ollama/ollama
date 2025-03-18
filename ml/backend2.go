package ml

import (
	"context"

	"github.com/ollama/ollama/fs"
)

type Device int

const (
	CPU Device = iota
	GPU
)

type Backend2 interface {
	Close()

	NewContext() Context

	Scheduler() Scheduler

	Get(fs.TensorReader, Device) Tensor
	LoadAll(context.Context) error
}
