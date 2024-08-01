package llm

import (
	"embed"
	"syscall"
)

//go:embed build/linux/*/*/bin/*
var libEmbed embed.FS

var LlamaServerSysProcAttr = &syscall.SysProcAttr{}
