//go:build windows

package cmd

import (
	"errors"
	"os"
)

func openPTY() (*os.File, *os.File, error) {
	return nil, nil, errors.New("pty is not available on windows")
}
