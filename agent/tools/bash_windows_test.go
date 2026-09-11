//go:build windows

package tools

import (
	"strings"
	"testing"
)

func TestPowerShellCommandScriptUsesWideOutString(t *testing.T) {
	script := powerShellCommandScript("Get-ChildItem", `C:\cwd.txt`)
	if !strings.Contains(script, "Out-String -Stream -Width 4096") {
		t.Fatalf("script = %q, want explicit Out-String width", script)
	}
}
