package cmd

import "errors"

func localCopy(src, target string) error {
	return errors.New("no local copy implementation for windows")
}

/* func localCopy(src, target string) error {
	dirPath := filepath.Dir(target)

	if err := os.MkdirAll(dirPath, 0o755); err != nil {
		return err
	}

	sourceFile, err := os.Open(src)
	if err != nil {
		return err
	}
	defer sourceFile.Close()

	targetFile, err := os.Create(target)
	if err != nil {
		return err
	}
	defer targetFile.Close()

	sourceHandle := syscall.Handle(sourceFile.Fd())
	targetHandle := syscall.Handle(targetFile.Fd())

	err = copyFileEx(sourceHandle, targetHandle)
	if err != nil {
		return err
	}

	return nil
}

func copyFileEx(srcHandle, dstHandle syscall.Handle) error {
	kernel32 := syscall.NewLazyDLL("kernel32.dll")
	copyFileEx := kernel32.NewProc("CopyFileExW")

	r1, _, err := copyFileEx.Call(
		uintptr(srcHandle),
		uintptr(dstHandle),
		0, 0, 0, 0)

	if r1 == 0 {
		return err
	}

	return nil
}
*/
