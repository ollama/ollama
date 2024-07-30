package llm

import (
	"embed"
	"syscall"
)

//go:embed build/darwin/arm64/*/bin/*
var libEmbed embed.FS

var LlamaServerSysProcAttr = &syscall.SysProcAttr{}
