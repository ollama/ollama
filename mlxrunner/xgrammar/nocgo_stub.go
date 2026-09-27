//go:build !cgo

// Package xgrammar wraps the xgrammar C API. The real implementation lives
// in cgo files; this stub keeps dependents compiling when built without cgo
// (e.g. `go build` on Windows with no C toolchain). Every entry point panics
// at run time: without cgo there is no xgrammar library to call into.

package xgrammar

import "errors"

type Compiler struct{}

func New(dir string, pieces []string, vocabSize int, stopIDs []int32, threads int, cacheBytes int64) (*Compiler, error) {
	return nil, errors.New("github.com/ollama/ollama/mlxrunner/xgrammar requires cgo")
}

func (c *Compiler) Path() string {
	panic("github.com/ollama/ollama/mlxrunner/xgrammar requires cgo")
}

func (c *Compiler) Version() string {
	panic("github.com/ollama/ollama/mlxrunner/xgrammar requires cgo")
}

func (c *Compiler) Compile(source string) (*Matcher, error) {
	return nil, errors.New("github.com/ollama/ollama/mlxrunner/xgrammar requires cgo")
}

func (c *Compiler) Close() {}

type Matcher struct{}

func (m *Matcher) Terminated() bool {
	panic("github.com/ollama/ollama/mlxrunner/xgrammar requires cgo")
}

func (m *Matcher) Fill(row []int32) (bool, error) {
	panic("github.com/ollama/ollama/mlxrunner/xgrammar requires cgo")
}

func (m *Matcher) Accept(tokenID int32) error {
	panic("github.com/ollama/ollama/mlxrunner/xgrammar requires cgo")
}

func (m *Matcher) Rollback(n int) error {
	panic("github.com/ollama/ollama/mlxrunner/xgrammar requires cgo")
}

func (m *Matcher) Close() {}
