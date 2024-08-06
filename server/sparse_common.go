//go:build !windows

package server

import "os"

func setSparse(file *os.File) error {
	return nil
}
