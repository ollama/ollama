//go:build !windows

package tools

import (
	"os/exec"
	"testing"
)

func TestConfigureBashCommandSetsProcessGroup(t *testing.T) {
	cmd := exec.Command("bash", "-c", "true")
	configureBashCommand(cmd)
	if cmd.SysProcAttr == nil || !cmd.SysProcAttr.Setpgid {
		t.Fatalf("configureBashCommand should start bash in a new process group")
	}
}
