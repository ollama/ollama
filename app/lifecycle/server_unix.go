//go:build !windows

package lifecycle

import (
	"context"
	"os/exec"
)

func getCmd(ctx context.Context, cmd string) *exec.Cmd {
	return exec.CommandContext(ctx, cmd, "serve")
}
