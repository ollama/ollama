package cmd

import (
	"context"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"syscall"

	"golang.org/x/sys/windows"

	"github.com/jmorganca/ollama/api"
)

func init() {
	var inMode uint32
	var outMode uint32
	var errMode uint32

	in := windows.Handle(os.Stdin.Fd())
	if err := windows.GetConsoleMode(in, &inMode); err == nil {
		windows.SetConsoleMode(in, inMode|windows.ENABLE_VIRTUAL_TERMINAL_INPUT) //nolint:errcheck
	}

	out := windows.Handle(os.Stdout.Fd())
	if err := windows.GetConsoleMode(out, &outMode); err == nil {
		windows.SetConsoleMode(out, outMode|windows.ENABLE_VIRTUAL_TERMINAL_PROCESSING) //nolint:errcheck
	}

	errf := windows.Handle(os.Stderr.Fd())
	if err := windows.GetConsoleMode(errf, &errMode); err == nil {
		windows.SetConsoleMode(errf, errMode|windows.ENABLE_VIRTUAL_TERMINAL_PROCESSING) //nolint:errcheck
	}
}

func startApp(ctx context.Context, client *api.Client) error {
	// log.Printf("XXX Attempting to find and start ollama app")
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
				return fmt.Errorf("could not locate ollama app")
			}
		}
	}
	// log.Printf("XXX attempting to start app %s", appExe)

	cmd_path := "c:\\Windows\\system32\\cmd.exe"
	cmd := exec.Command(cmd_path, "/c", appExe)
	// TODO - these hide flags aren't working - still pops up a command window for some reason
	cmd.SysProcAttr = &syscall.SysProcAttr{CreationFlags: 0x08000000, HideWindow: true}

	// TODO this didn't help either...
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
