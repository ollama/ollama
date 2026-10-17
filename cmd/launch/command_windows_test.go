package launch

import (
	"context"
	"fmt"
	"os"
	"strings"
	"testing"

	"golang.org/x/sys/windows"
)

func TestBackgroundCommandHasNoConsole(t *testing.T) {
	if os.Getenv("OLLAMA_TEST_BACKGROUND_COMMAND") == "1" {
		console, _, _ := windows.NewLazySystemDLL("kernel32.dll").NewProc("GetConsoleWindow").Call()
		fmt.Print(console)
		os.Exit(0)
	}

	cmd := backgroundCommandContext(context.Background(), os.Args[0], "-test.run=^TestBackgroundCommandHasNoConsole$")
	cmd.Env = append(os.Environ(), "OLLAMA_TEST_BACKGROUND_COMMAND=1")
	if cmd.SysProcAttr == nil || cmd.SysProcAttr.CreationFlags&windows.CREATE_NO_WINDOW == 0 {
		t.Fatal("background command must prevent console allocation")
	}
	output, err := cmd.Output()
	if err != nil {
		t.Fatal(err)
	}
	if got := strings.TrimSpace(string(output)); got != "0" {
		t.Fatalf("background command has console window %s", got)
	}
}
