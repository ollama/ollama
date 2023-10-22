package main

import (
	"embed"
	"io"
	"os"
	"path/filepath"
)

//go:embed ggml-metal.metal
var fs embed.FS

func init() {
	exec, err := os.Executable()
	if err != nil {
		return
	}

	exec, err = filepath.EvalSymlinks(exec)
	if err != nil {
		return
	}

	dst, err := os.Create(filepath.Join(filepath.Dir(exec), "ggml-metal.metal"))
	if err != nil {
		return
	}
	defer dst.Close()

	src, err := fs.Open("ggml-metal.metal")
	if err != nil {
		return
	}
	defer src.Close()

	if _, err := io.Copy(dst, src); err != nil {
		return
	}
}
