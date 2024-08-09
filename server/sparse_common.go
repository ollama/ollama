//go:build !windows

package server

import "os"

func setSparse(*os.File) {
}
