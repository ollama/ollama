//go:build windows || darwin

package logrotate

import (
	"os"
	"path/filepath"
	"strconv"
	"testing"
)

func TestRotate(t *testing.T) {
	logDir := t.TempDir()
	logFile := filepath.Join(logDir, "testlog.log")

	// No log exists
	Rotate(logFile)

	if err := os.WriteFile(logFile, []byte("1"), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(logFile); os.IsNotExist(err) {
		t.Fatal("expected log file to exist")
	}

	// First rotation
	Rotate(logFile)
	if _, err := os.Stat(filepath.Join(logDir, "testlog-1.log")); os.IsNotExist(err) {
		t.Fatal("expected rotated log file to exist")
	}
	if _, err := os.Stat(filepath.Join(logDir, "testlog-2.log")); !os.IsNotExist(err) {
		t.Fatal("expected no second rotated log file")
	}
	if _, err := os.Stat(logFile); !os.IsNotExist(err) {
		t.Fatal("expected original log file to be moved")
	}

	// Should be a no-op without a new log
	Rotate(logFile)
	if _, err := os.Stat(filepath.Join(logDir, "testlog-1.log")); os.IsNotExist(err) {
		t.Fatal("expected rotated log file to still exist")
	}
	if _, err := os.Stat(filepath.Join(logDir, "testlog-2.log")); !os.IsNotExist(err) {
		t.Fatal("expected no second rotated log file")
	}
	if _, err := os.Stat(logFile); !os.IsNotExist(err) {
		t.Fatal("expected no original log file")
	}

	for i := 2; i <= MaxLogFiles+1; i++ {
		if err := os.WriteFile(logFile, []byte(strconv.Itoa(i)), 0o644); err != nil {
			t.Fatal(err)
		}
		if _, err := os.Stat(logFile); os.IsNotExist(err) {
			t.Fatal("expected log file to exist")
		}
		Rotate(logFile)
		if _, err := os.Stat(logFile); !os.IsNotExist(err) {
			t.Fatal("expected log file to be moved")
		}
		for j := 1; j < i; j++ {
			if _, err := os.Stat(filepath.Join(logDir, "testlog-"+strconv.Itoa(j)+".log")); os.IsNotExist(err) {
				t.Fatalf("expected rotated log file %d to exist", j)
			}
		}
		if _, err := os.Stat(filepath.Join(logDir, "testlog-"+strconv.Itoa(i+1)+".log")); !os.IsNotExist(err) {
			t.Fatalf("expected no rotated log file %d", i+1)
		}
	}
}

func TestRotateFilename(t *testing.T) {
	for _, tt := range []struct {
		name     string
		filename string
		rotated  string
	}{
		{"extensionless", "server", "server-1"},
		{"extensionless in dotted directory", "server", "server-1"},
		{"multiple extensions", "server.debug.log", "server.debug-1.log"},
		{"unicode basename", "günlük.log", "günlük-1.log"},
	} {
		t.Run(tt.name, func(t *testing.T) {
			dir := t.TempDir()
			if tt.name == "extensionless in dotted directory" {
				dir = filepath.Join(dir, "logs.d")
				if err := os.Mkdir(dir, 0o755); err != nil {
					t.Fatal(err)
				}
			}
			filename := filepath.Join(dir, tt.filename)
			if err := os.WriteFile(filename, []byte("first"), 0o600); err != nil {
				t.Fatal(err)
			}

			Rotate(filename)

			data, err := os.ReadFile(filepath.Join(dir, tt.rotated))
			if err != nil {
				t.Fatal(err)
			}
			if string(data) != "first" {
				t.Fatalf("rotated content = %q, want first", data)
			}
			if _, err := os.Stat(filename); !os.IsNotExist(err) {
				t.Fatalf("original file still exists or cannot be inspected: %v", err)
			}
		})
	}
}
