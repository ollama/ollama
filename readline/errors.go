package readline

import (
	"errors"
)

var ErrInterrupt = errors.New("Interrupt")

// ErrExpandOutput is returned when user presses Ctrl+O to expand tool output
var ErrExpandOutput = errors.New("ExpandOutput")

type InterruptError struct {
	Line []rune
}

func (*InterruptError) Error() string {
	return "Interrupted"
}
