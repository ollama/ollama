package chat

import (
	"context"
	"errors"
	"fmt"
	"os/exec"
	"runtime"
	"strings"
)

func writeClipboard(ctx context.Context, text string) error {
	if ctx == nil {
		ctx = context.Background()
	}
	switch runtime.GOOS {
	case "darwin":
		return runClipboardCommand(ctx, text, "pbcopy")
	case "windows":
		return runClipboardCommand(ctx, text, "clip")
	default:
		for _, candidate := range []struct {
			name string
			args []string
		}{
			{name: "wl-copy"},
			{name: "xclip", args: []string{"-selection", "clipboard"}},
			{name: "xsel", args: []string{"--clipboard", "--input"}},
		} {
			if _, err := exec.LookPath(candidate.name); err != nil {
				continue
			}
			return runClipboardCommand(ctx, text, candidate.name, candidate.args...)
		}
		return errors.New("no clipboard command found")
	}
}

func runClipboardCommand(ctx context.Context, text, name string, args ...string) error {
	cmd := exec.CommandContext(ctx, name, args...)
	cmd.Stdin = strings.NewReader(text)
	if output, err := cmd.CombinedOutput(); err != nil {
		if len(output) > 0 {
			return fmt.Errorf("%s: %w: %s", name, err, strings.TrimSpace(string(output)))
		}
		return fmt.Errorf("%s: %w", name, err)
	}
	return nil
}
