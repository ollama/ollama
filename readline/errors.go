package readline

import (
	"errors"
)

var (
	ErrInterrupt  = errors.New("Interrupt")
	ErrEditPrompt = errors.New("EditPrompt")
	ErrVimHelp    = errors.New("VimHelp")
)

type InterruptError struct {
	Line []rune
}

func (*InterruptError) Error() string {
	return "Interrupted"
}
