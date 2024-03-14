package lib

import (
	"fmt"
	"strings"
	"time"
)

// //go:wasm-module env
// //export _sleep
// func Sleep(dur int64)

//go:wasm-module ollama
//export generate
func Generate(
	modelNamePtr *byte,
	modelNameSize uint32,

	promptPtr *byte,
	promptSize uint32,

	systemPtr *byte,
	systemSize uint32,

	templatePtr *byte,
	templateSize uint32,

	contextPtr *byte, // encoded []int64
	contextSize uint32,

	rawVal uint32, // 0=false or 1=true

	keepaliveDur uint64,

	imagesPtr *byte, // encoded [][]byte
	imagesSize uint32,

	optionsPtr *byte, // cbor
	optionsSize uint32,

	errBufferPtr *byte,
	errBufferSize uint32,
	errBufferWrittenPtr *uint32,

	idPtr *uint64,
) uint32

//go:wasm-module ollama
//export next
func Next(
	jobId uint64,

	wait uint64, // 0 for default

	tokenBufferPtr *byte,
	tokenBufferSize uint32,
	tokenBufferWrittenPtr *uint32,

	errBufferPtr *byte,
	errBufferSize uint32,
	errBufferWrittenPtr *uint32,

) uint32

//export generate
func generate() {
	model := []byte("gemma:2b-instruct")
	prompt := []byte("What is your name?")
	system := []byte("You are a chatbot assistant named Odo. You are to answer all user's questions in one sentence.")
	template := []byte("[INST] {{ if .System }}{{ .System }} {{ end }}{{ .Prompt }} [/INST]")
	errBuf := make([]byte, 512)
	var errW uint32
	var id uint64
	id = 0
	err := Generate(&model[0], uint32(len(model)), &prompt[0], uint32(len(prompt)), &system[0], uint32(len(system)), &template[0], uint32(len(template)), nil, 0, 0, uint64(5*time.Minute), nil, 0, nil, 0, &errBuf[0], uint32(len(errBuf)), &errW, &id)

	if err != 0 {
		panic(string(errBuf[:errW]))
	}

	if id == 0 {
		panic("id == 0!")
	}

	tokenBuffer := make([]byte, 1024)
	var tokenBufferW uint32

	var sb strings.Builder

	for {
		err = Next(id, 0, &tokenBuffer[0], uint32(len(tokenBuffer)), &tokenBufferW, &errBuf[0], uint32(len(errBuf)), &errW)
		if err != 0 {
			break
		}

		sb.Write(tokenBuffer[:tokenBufferW])
	}

	if !strings.Contains(sb.String(), "Odo") {
		panic("LLM did not answer the question")
	}

	fmt.Print(sb.String())
}
