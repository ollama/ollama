//go:build !windows

package launch

import (
	"context"
	"os/exec"
)

func backgroundCommandContext(ctx context.Context, name string, args ...string) *exec.Cmd {
	return exec.CommandContext(ctx, name, args...)
}
