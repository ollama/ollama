//go:build windows

package discover

import "testing"

func TestDiscoverSysProcAttrHasNoWindow(t *testing.T) {
	attr := discoverSysProcAttr()
	if attr == nil {
		t.Fatal("expected non-nil SysProcAttr on Windows")
	}
	if attr.CreationFlags&createNoWindow == 0 {
		t.Fatalf("expected CREATE_NO_WINDOW flag, got %#x", attr.CreationFlags)
	}
}
