package readline

import (
	"errors"
)

var (
	ErrInterrupt  = errors.New("Interrupt")
	ErrEditPrompt = errors.New("EditPrompt")
)
