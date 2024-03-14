package main

import "github.com/taubyte/vm-orbit/satellite"

type Error uint32

const (
	ErrorNone Error = iota
	ErrorReadMemory
	ErrorWriteMemory
	ErrorBufferTooSmall
	ErrorErrorBufferTooSmall
	ErrorRetuned
	ErrorTimeout
	ErrorEOF
	ErrorPull
	ErrorPullNotFound
	ErrorPullStatus
	ErrorModelNotFound
	ErrorEmbeddingNotSupportedInGenerate
	ErrorJobNotFound
)

func returnError(
	module satellite.Module,
	errBufferPtr uint32,
	errBufferSize uint32,
	errBufferWrittenPtr uint32,
	err error,
) Error {
	if err == nil {
		return ErrorNone
	}

	errString := err.Error()

	if uint32(len(errString)) > errBufferSize {
		return ErrorErrorBufferTooSmall
	}

	n, _ := module.WriteString(errBufferPtr, errString)
	module.WriteUint32(errBufferWrittenPtr, n)

	return ErrorRetuned
}

func returnMemoryRead(
	module satellite.Module,
	errBufferPtr uint32,
	errBufferSize uint32,
	errBufferWrittenPtr uint32,
	errString string,
) Error {
	if uint32(len(errString)) > errBufferSize {
		return ErrorErrorBufferTooSmall
	}

	n, _ := module.WriteString(errBufferPtr, errString)
	module.WriteUint32(errBufferWrittenPtr, n)

	return ErrorReadMemory
}
