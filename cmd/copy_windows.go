//go:build windows
// +build windows

package cmd

import (
	"os"
	"path/filepath"
	"syscall"
	"unsafe"
)

func localCopy(src, target string) error {
	// Create target directory if it doesn't exist
	dirPath := filepath.Dir(target)
	if err := os.MkdirAll(dirPath, 0o755); err != nil {
		return err
	}

	// Open source file
	sourceFile, err := os.Open(src)
	if err != nil {
		return err
	}
	defer sourceFile.Close()

	// Create target file
	targetFile, err := os.Create(target)
	if err != nil {
		return err
	}
	defer targetFile.Close()

	// Use CopyFileExW to copy the file
	err = copyFileEx(src, target)
	if err != nil {
		return err
	}

	return nil
}

func copyFileEx(src, dst string) error {
	kernel32 := syscall.NewLazyDLL("kernel32.dll")
	copyFileEx := kernel32.NewProc("CopyFileExW")

	srcPtr, err := syscall.UTF16PtrFromString(src)
	if err != nil {
		return err
	}

	dstPtr, err := syscall.UTF16PtrFromString(dst)
	if err != nil {
		return err
	}

	r1, _, err := copyFileEx.Call(
		uintptr(unsafe.Pointer(srcPtr)),
		uintptr(unsafe.Pointer(dstPtr)),
		0, 0, 0, 0)

	if r1 == 0 {
		return err
	}

	return nil
}
