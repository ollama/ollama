//go:build !windows

package discover

import "syscall"

func discoverSysProcAttr() *syscall.SysProcAttr {
	return nil
}
