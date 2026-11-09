//go:build !windows

package cmd

import (
	"os"

	"github.com/creack/pty"
)

func openPTY() (*os.File, *os.File, error) {
	return pty.Open()
}
