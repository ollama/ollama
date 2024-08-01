package llm

import (
	"embed"
	"syscall"
)

//go:embed build/darwin/x86_64/*/bin/*
var libEmbed embed.FS

var LlamaServerSysProcAttr = &syscall.SysProcAttr{}
