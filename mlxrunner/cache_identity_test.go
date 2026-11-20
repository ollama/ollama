package mlxrunner

import (
	"debug/macho"
	"encoding/binary"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
)

func TestUniversalExecutableBuildID(t *testing.T) {
	data := make([]byte, 16<<10)
	be, le := binary.BigEndian, binary.LittleEndian
	be.PutUint32(data, macho.MagicFat)
	be.PutUint32(data[4:], 2)
	for i, cpu := range []macho.Cpu{macho.CpuAmd64, macho.CpuArm64} {
		offset := uint32((i + 1) * 4096)
		arch := data[8+i*20:]
		be.PutUint32(arch, uint32(cpu))
		be.PutUint32(arch[8:], offset)
		be.PutUint32(arch[12:], 4096)
		be.PutUint32(arch[16:], 12)
		header := data[offset:]
		le.PutUint32(header, macho.Magic64)
		le.PutUint32(header[4:], uint32(cpu))
		copy(header[256:], "\xff Go build ID: \""+[]string{"amd64-build", "arm64-build"}[i]+"\"\n \xff")
	}
	path := filepath.Join(t.TempDir(), "universal")
	if err := os.WriteFile(path, data, 0o600); err != nil {
		t.Fatal(err)
	}
	id, err := executableBuildID(path)
	if err != nil || id != runtime.GOARCH+"-build" {
		t.Fatalf("universal build ID selected %q: %v", id, err)
	}
}

func TestExecutableBuildID(t *testing.T) {
	exe, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	id, err := executableBuildID(exe)
	if err != nil || id == "" {
		t.Fatalf("test executable build ID: %q, %v", id, err)
	}
	path := filepath.Join(t.TempDir(), "runner")
	for _, contents := range []string{
		"", "not a Go binary", `prefix Go build ID: "invented"`,
		"\xff Go build ID: \"\"\n \xff", "\xff Go build ID: \"truncated",
		"\xff Go build ID: \"first-build\"\n \xff",
		"\xff Go build ID: \"second-build\"\n \xff",
	} {
		if err := os.WriteFile(path, []byte(contents), 0o600); err != nil {
			t.Fatal(err)
		}
		got, err := executableBuildID(path)
		if strings.Contains(contents, "-build") {
			if err != nil || !strings.Contains(contents, got) || got == "" {
				t.Fatalf("valid embedded stamp: %q, %v", got, err)
			}
		} else if err == nil || got != "" {
			t.Fatalf("accepted missing or damaged build ID: %q, %v", got, err)
		}
	}
}
