package cmd

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"os"
	"os/exec"
	"path"
	"path/filepath"
	"strings"
	"syscall"
	"unsafe"

	"github.com/ollama/ollama/api"
	"golang.org/x/sys/windows"
)

const (
	Installer = "OllamaSetup.exe"
)

func startApp(ctx context.Context, client *api.Client) error {
	if len(isProcRunning(Installer)) > 0 {
		return fmt.Errorf("upgrade in progress...")
	}
	AppName := "ollama app.exe"
	exe, err := os.Executable()
	if err != nil {
		return err
	}
	appExe := filepath.Join(filepath.Dir(exe), AppName)
	_, err = os.Stat(appExe)
	if errors.Is(err, os.ErrNotExist) {
		// Try the standard install location
		localAppData := os.Getenv("LOCALAPPDATA")
		appExe = filepath.Join(localAppData, "Ollama", AppName)
		_, err := os.Stat(appExe)
		if errors.Is(err, os.ErrNotExist) {
			// Finally look in the path
			appExe, err = exec.LookPath(AppName)
			if err != nil {
				return errors.New("could not locate ollama app")
			}
		}
	}

	cmd_path := "c:\\Windows\\system32\\cmd.exe"
	cmd := exec.Command(cmd_path, "/c", appExe, "--hide", "--fast-startup")
	cmd.SysProcAttr = &syscall.SysProcAttr{CreationFlags: 0x08000000, HideWindow: true}

	cmd.Stdin = strings.NewReader("")
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr

	if err := cmd.Start(); err != nil {
		return fmt.Errorf("unable to start ollama app %w", err)
	}

	if cmd.Process != nil {
		defer cmd.Process.Release() //nolint:errcheck
	}
	return waitForServer(ctx, client)
}

// enumPIDs converts the byte count reported by EnumProcesses into the number
// of process IDs written. It returns the used portion of pids and whether the
// buffer was filled completely, which means the process list may have been
// truncated and the caller should grow the buffer and query again.
func enumPIDs(pids []uint32, bytesReturned uint32) (used []uint32, full bool) {
	if len(pids) == 0 {
		return pids, true
	}

	const pidSize = uint32(unsafe.Sizeof(uint32(0)))
	count := bytesReturned / pidSize
	if count > uint32(len(pids)) {
		count = uint32(len(pids))
	}

	return pids[:count], count == uint32(len(pids))
}

// enumerateProcessPIDs returns the running process IDs. enum is injectable so
// the grow-and-retry path can be tested without depending on how many
// processes happen to be running on the host.
func enumerateProcessPIDs(enum func([]uint32, *uint32) error) ([]uint32, error) {
	pids := make([]uint32, 2048)
	for {
		var bytesReturned uint32
		if err := enum(pids, &bytesReturned); err != nil {
			return nil, err
		}
		if bytesReturned == 0 {
			return nil, nil
		}

		used, full := enumPIDs(pids, bytesReturned)
		if !full {
			return used, nil
		}

		pids = make([]uint32, len(pids)*2)
	}
}

// processExecutableName returns the base name of the executable for pid.
// Scoping the handle to this function closes it as soon as the lookup
// finishes, rather than holding every handle until isProcRunning returns.
func processExecutableName(pid uint32) (string, bool) {
	hProcess, err := windows.OpenProcess(windows.PROCESS_QUERY_INFORMATION|windows.PROCESS_VM_READ, false, pid)
	if err != nil {
		return "", false
	}
	defer windows.CloseHandle(hProcess)

	var module windows.Handle
	var cbNeeded uint32
	cb := uint32(unsafe.Sizeof(module))
	if err := windows.EnumProcessModules(hProcess, &module, cb, &cbNeeded); err != nil {
		return "", false
	}

	var sz uint32 = 1024 * 8
	moduleName := make([]uint16, sz)
	cb = uint32(len(moduleName)) * uint32(unsafe.Sizeof(uint16(0)))
	if err := windows.GetModuleBaseName(hProcess, module, &moduleName[0], cb); err != nil && err != syscall.ERROR_INSUFFICIENT_BUFFER {
		return "", false
	}

	return path.Base(strings.ToLower(syscall.UTF16ToString(moduleName))), true
}

func isProcRunning(procName string) []uint32 {
	pids, err := enumerateProcessPIDs(windows.EnumProcesses)
	if err != nil {
		slog.Debug("failed to check for running installers", "error", err)
		return nil
	}

	var matches []uint32
	for _, pid := range pids {
		if pid == 0 {
			continue
		}

		exeFile, ok := processExecutableName(pid)
		if ok && strings.EqualFold(exeFile, procName) {
			matches = append(matches, pid)
		}
	}
	return matches
}
