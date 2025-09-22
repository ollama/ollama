package cmd

import (
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"io"
	"os"
	"os/user"
	"path/filepath"
	"runtime"
	"testing"

	"github.com/google/go-cmp/cmp"
	"github.com/google/go-cmp/cmp/cmpopts"
	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/fs/ggml"
	"github.com/ollama/ollama/parser"
)

func TestCreateRequest(t *testing.T) {
	cases := []struct {
		modelfile parser.Modelfile
		expected  *api.CreateRequest
	}{
		{
			parser.Modelfile{
				Commands: []parser.Command{
					{Name: "model", Args: "test"},
				},
			},
			&api.CreateRequest{
				From:    "test",
				License: []string(nil),
			},
		},
		{
			parser.Modelfile{
				Commands: []parser.Command{
					{Name: "model", Args: "test"},
					{Name: "template", Args: "some template"},
				},
			},
			&api.CreateRequest{
				From:     "test",
				Template: "some template",
				License:  []string(nil),
			},
		},
		{
			parser.Modelfile{
				Commands: []parser.Command{
					{Name: "model", Args: "test"},
					{Name: "license", Args: "single license"},
					{Name: "temperature", Args: "0.5"},
					{Name: "message", Args: "user: Hello"},
				},
			},
			&api.CreateRequest{
				From:       "test",
				License:    []string{"single license"},
				Parameters: map[string]any{"temperature": float32(0.5)},
				Messages: []api.Message{
					{Role: "user", Content: "Hello"},
				},
			},
		},
		{
			parser.Modelfile{
				Commands: []parser.Command{
					{Name: "model", Args: "test"},
					{Name: "temperature", Args: "0.5"},
					{Name: "top_k", Args: "1"},
					{Name: "system", Args: "You are a bot."},
					{Name: "license", Args: "license1"},
					{Name: "license", Args: "license2"},
					{Name: "message", Args: "user: Hello there!"},
					{Name: "message", Args: "assistant: Hi! How are you?"},
				},
			},
			&api.CreateRequest{
				From:       "test",
				License:    []string{"license1", "license2"},
				System:     "You are a bot.",
				Parameters: map[string]any{"temperature": float32(0.5), "top_k": int64(1)},
				Messages: []api.Message{
					{Role: "user", Content: "Hello there!"},
					{Role: "assistant", Content: "Hi! How are you?"},
				},
			},
		},
	}

	for _, c := range cases {
		actual, err := createRequest(&c.modelfile, "")
		if err != nil {
			t.Fatal(err)
		}

		if diff := cmp.Diff(actual, c.expected,
			cmpopts.EquateEmpty(),
			cmpopts.SortSlices(func(a, b api.File) bool { return a.Path < b.Path }),
		); diff != "" {
			t.Errorf("mismatch (-got +want):\n%s", diff)
		}
	}
}

func createBinFile(t *testing.T, d string, kv map[string]any, ti []*ggml.Tensor) (string, string) {
	t.Helper()

	f, err := os.CreateTemp(d, "testbin.*.gguf")
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()

	if err := ggml.WriteGGUF(f, kv, ti); err != nil {
		t.Fatal(err)
	}

	// Calculate sha256 of file
	if _, err := f.Seek(0, 0); err != nil {
		t.Fatal(err)
	}

	sha256sum := sha256.New()
	if _, err := io.Copy(sha256sum, f); err != nil {
		t.Fatal(err)
	}

	return f.Name(), "sha256:" + hex.EncodeToString(sha256sum.Sum(nil))
}

func TestCreateRequestFiles(t *testing.T) {
	d := t.TempDir()
	n1, d1 := createBinFile(t, d, nil, nil)
	n2, d2 := createBinFile(t, d, map[string]any{"foo": "bar"}, nil)

	cases := []struct {
		modelfile parser.Modelfile
		expected  *api.CreateRequest
	}{
		{
			parser.Modelfile{
				Commands: []parser.Command{
					{Name: "model", Args: n1},
				},
			},
			&api.CreateRequest{
				Files: []api.File{
					{
						Name:   filepath.Base(n1),
						Path:   n1,
						Digest: d1,
					},
				},
				License: []string(nil),
			},
		},
		{
			parser.Modelfile{
				Commands: []parser.Command{
					{Name: "model", Args: n1},
					{Name: "model", Args: n2},
				},
			},
			&api.CreateRequest{
				Files: []api.File{
					{
						Name:   filepath.Base(n1),
						Path:   n1,
						Digest: d1,
					},
					{
						Name:   filepath.Base(n2),
						Path:   n2,
						Digest: d2,
					},
				},
				License: []string(nil),
			},
		},
	}

	for _, c := range cases {
		actual, err := createRequest(&c.modelfile, d)
		if err != nil {
			t.Error(err)
		}

		if diff := cmp.Diff(actual, c.expected,
			cmpopts.EquateEmpty(),
			cmpopts.SortSlices(func(a, b api.File) bool { return a.Path < b.Path }),
		); diff != "" {
			t.Errorf("mismatch (-got +want):\n%s", diff)
		}
	}
}

func TestExpandPath(t *testing.T) {
	home := t.TempDir()
	t.Setenv("HOME", home)
	t.Setenv("USERPROFILE", home)

	cwd, err := os.Getwd()
	if err != nil {
		t.Fatal(err)
	}

	u, err := user.Current()
	if err != nil {
		t.Fatal(err)
	}

	volume := ""
	if runtime.GOOS == "windows" {
		volume = "D:"
	}

	cases := []struct {
		input,
		dir,
		want string
		err error
	}{
		{"~", "", home, nil},
		{"~/path/to/file", "", filepath.Join(home, filepath.ToSlash("path/to/file")), nil},
		{"~" + u.Username + "/path/to/file", "", filepath.Join(u.HomeDir, filepath.ToSlash("path/to/file")), nil},
		{"~nonexistentuser/path/to/file", "", "", user.UnknownUserError("nonexistentuser")},
		{"relative/path/to/file", "", filepath.Join(cwd, filepath.ToSlash("relative/path/to/file")), nil},
		{volume + "/absolute/path/to/file", "", filepath.ToSlash(volume + "/absolute/path/to/file"), nil},
		{volume + "/absolute/path/to/file", filepath.ToSlash("another/path"), filepath.ToSlash(volume + "/absolute/path/to/file"), nil},
		{".", cwd, cwd, nil},
		{".", "", cwd, nil},
		{"", cwd, cwd, nil},
		{"", "", cwd, nil},
		{"file", "path/to", filepath.Join(cwd, filepath.ToSlash("path/to/file")), nil},
	}

	for _, tt := range cases {
		t.Run(tt.input, func(t *testing.T) {
			got, err := expandPath(tt.input, tt.dir)
			// On Windows, user.Lookup does not map syscall errors to user.UnknownUserError
			// so we special case the test to just check for an error.
			// See https://cs.opensource.google/go/go/+/refs/tags/go1.25.1:src/os/user/lookup_windows.go;l=455
			if runtime.GOOS != "windows" && !errors.Is(err, tt.err) {
				t.Fatalf("expandPath(%q) error = %v, wantErr %v", tt.input, err, tt.err)
			} else if tt.err != nil && err == nil {
				t.Fatal("test case expected to fail on windows")
			}

			if got != tt.want {
				t.Errorf("expandPath(%q) = %v, want %v", tt.input, got, tt.want)
			}
		})
	}
}
