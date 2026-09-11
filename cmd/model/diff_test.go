package main

import (
	"bytes"
	"context"
	"encoding/binary"
	"os"
	"path/filepath"
	"testing"
)

type cancelAfterHeader struct {
	bytes.Buffer
	cancel context.CancelFunc
}

func (w *cancelAfterHeader) Write(p []byte) (int, error) {
	n, err := w.Buffer.Write(p)
	if bytes.Contains(w.Bytes(), []byte("\n+++ ")) {
		w.cancel()
	}
	return n, err
}

func writeSingleByteSafetensors(t *testing.T, value byte) string {
	t.Helper()
	header := `{"weight":{"dtype":"U8","shape":[1],"data_offsets":[0,1]}}`
	var data bytes.Buffer
	binary.Write(&data, binary.LittleEndian, uint64(len(header)))
	data.WriteString(header)
	data.WriteByte(value)
	path := filepath.Join(t.TempDir(), "model.safetensors")
	if err := os.WriteFile(path, data.Bytes(), 0o644); err != nil {
		t.Fatal(err)
	}
	return path
}

func TestExitCodes(t *testing.T) {
	var paths []string
	for _, value := range []byte{1, 2} {
		paths = append(paths, writeSingleByteSafetensors(t, value))
	}
	for _, tc := range []struct {
		name string
		args []string
		code int
	}{
		{"equal", []string{paths[0], paths[0]}, 0},
		{"different", []string{paths[0], paths[1]}, 1},
		{"missing", []string{paths[0], paths[1] + ".missing"}, 2},
		{"json removed", []string{"--json", paths[0], paths[1]}, 2},
		{"stats metadata conflict", []string{"--stats", "--metadata-only", paths[0], paths[1]}, 2},
		{"no args", nil, 2},
		{"help", []string{"--help"}, 0},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var out, stderr bytes.Buffer
			if code := run(t.Context(), tc.args, &out, &stderr); code != tc.code {
				t.Fatalf("code=%d, want %d: %s", code, tc.code, stderr.String())
			}
			if tc.code < 2 && tc.name != "help" && stderr.Len() != 0 {
				t.Fatalf("comparison emitted stderr noise: %s", stderr.String())
			}
			if tc.code < 2 && tc.name != "help" && out.Len() == 0 {
				t.Fatal("successful comparison emitted no report")
			}
		})
	}
}

func TestInputHeaderPrecedesComparison(t *testing.T) {
	path := writeSingleByteSafetensors(t, 1)
	ctx, cancel := context.WithCancel(t.Context())
	out := &cancelAfterHeader{cancel: cancel}
	var stderr bytes.Buffer
	if code := run(ctx, []string{path, path}, out, &stderr); code != 2 {
		t.Fatalf("code=%d, want interrupted comparison error: %s", code, stderr.String())
	}
	if got := out.String(); !bytes.HasPrefix([]byte(got), []byte("--- ")) || !bytes.Contains([]byte(got), []byte("\n+++ ")) {
		t.Fatalf("resolved input header was not written before comparison: %q", got)
	}
	if bytes.Contains(out.Bytes(), []byte("Summary\n")) {
		t.Fatalf("report body was written after comparison cancellation: %q", out.String())
	}
}
