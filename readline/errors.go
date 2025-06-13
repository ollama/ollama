package readline

import (
	"errors"
)

var (
	ErrInterrupt       = errors.New("Interrupt")
	ErrNewLineDetected = errors.New("new line detected")
)

type InterruptError struct {
	Line []rune
}

func (*InterruptError) Error() string {
	return "Interrupted"
}
