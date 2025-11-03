//go:build windows || darwin

package server

import (
	"context"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/ollama/ollama/app/store"
)

func TestNew(t *testing.T) {
	tmpDir := t.TempDir()
	st := &store.Store{DBPath: filepath.Join(tmpDir, "db.sqlite")}
	defer st.Close() // Ensure database is closed before cleanup
	s := New(st, false)

	if s == nil {
		t.Fatal("expected non-nil server")
	}

	if s.bin == "" {
		t.Error("expected non-empty bin path")
	}
}

func TestServerCmd(t *testing.T) {
	os.Unsetenv("OLLAMA_HOST")
	os.Unsetenv("OLLAMA_ORIGINS")
	os.Unsetenv("OLLAMA_MODELS")
	var defaultModels string
	home, err := os.UserHomeDir()
	if err == nil {
		defaultModels = filepath.Join(home, ".ollama", "models")
		os.MkdirAll(defaultModels, 0o755)
	}

	tmpModels := t.TempDir()
	tests := []struct {
		name     string
		settings store.Settings
		want     []string
		dont     []string
	}{
		{
			name:     "default",
			settings: store.Settings{},
			want:     []string{"OLLAMA_MODELS=" + defaultModels},
			dont:     []string{"OLLAMA_HOST=", "OLLAMA_ORIGINS="},
		},
		{
			name:     "expose",
			settings: store.Settings{Expose: true},
			want:     []string{"OLLAMA_HOST=0.0.0.0", "OLLAMA_MODELS=" + defaultModels},
			dont:     []string{"OLLAMA_ORIGINS="},
		},
		{
			name:     "browser",
			settings: store.Settings{Browser: true},
			want:     []string{"OLLAMA_ORIGINS=*", "OLLAMA_MODELS=" + defaultModels},
			dont:     []string{"OLLAMA_HOST="},
		},
		{
			name:     "models",
			settings: store.Settings{Models: tmpModels},
			want:     []string{"OLLAMA_MODELS=" + tmpModels},
			dont:     []string{"OLLAMA_HOST=", "OLLAMA_ORIGINS="},
		},
		{
			name:     "inaccessible_models",
			settings: store.Settings{Models: "/nonexistent/external/drive/models"},
			want:     []string{},
			dont:     []string{"OLLAMA_MODELS="},
		},
		{
			name: "all",
			settings: store.Settings{
				Expose:  true,
				Browser: true,
				Models:  tmpModels,
			},
			want: []string{
				"OLLAMA_HOST=0.0.0.0",
				"OLLAMA_ORIGINS=*",
				"OLLAMA_MODELS=" + tmpModels,
			},
			dont: []string{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tmpDir := t.TempDir()
			st := &store.Store{DBPath: filepath.Join(tmpDir, "db.sqlite")}
			defer st.Close() // Ensure database is closed before cleanup
			st.SetSettings(tt.settings)
			s := &Server{
				store: st,
			}

			cmd, err := s.cmd(t.Context())
			if err != nil {
				t.Fatalf("s.cmd() error = %v", err)
			}

			for _, want := range tt.want {
				found := false
				for _, env := range cmd.Env {
					if strings.Contains(env, want) {
						found = true
						break
					}
				}
				if !found {
					t.Errorf("expected environment variable containing %s", want)
				}
			}

			for _, dont := range tt.dont {
				for _, env := range cmd.Env {
					if strings.Contains(env, dont) {
						t.Errorf("unexpected environment variable: %s", env)
					}
				}
			}

			if cmd.Cancel == nil {
				t.Error("expected non-nil cancel function")
			}
		})
	}
}

func TestGetInferenceComputer(t *testing.T) {
	tests := []struct {
		name string
		log  string
		exp  []InferenceCompute
	}{
		{
			name: "metal",
			log: `time=2025-06-30T09:23:07.374-07:00 level=DEBUG source=sched.go:108 msg="starting llm scheduler"
time=2025-06-30T09:23:07.416-07:00 level=INFO source=types.go:130 msg="inference compute" id=0 library=metal variant="" compute="" driver=0.0 name="" total="96.0 GiB" available="96.0 GiB"
time=2025-06-30T09:25:56.197-07:00 level=DEBUG source=ggml.go:155 msg="key not found" key=general.alignment default=32
`,
			exp: []InferenceCompute{{
				Library: "metal",
				Driver:  "0.0",
				VRAM:    "96.0 GiB",
			}},
		},
		{
			name: "cpu",
			log: `time=2025-07-01T17:59:51.470Z level=INFO source=gpu.go:377 msg="no compatible GPUs were discovered"
time=2025-07-01T17:59:51.470Z level=INFO source=types.go:130 msg="inference compute" id=0 library=cpu variant="" compute="" driver=0.0 name="" total="31.3 GiB" available="30.4 GiB"
[GIN] 2025/07/01 - 18:00:09 | 200 |      50.263µs | 100.126.204.152 | HEAD     "/"
`,
			exp: []InferenceCompute{{
				Library: "cpu",
				Driver:  "0.0",
				VRAM:    "31.3 GiB",
			}},
		},
		{
			name: "cuda1",
			log: `time=2025-07-01T19:33:43.162Z level=DEBUG source=amd_linux.go:419 msg="amdgpu driver not detected /sys/module/amdgpu"
releasing cuda driver library
time=2025-07-01T19:33:43.162Z level=INFO source=types.go:130 msg="inference compute" id=GPU-452cac9f-6960-839c-4fb3-0cec83699196 library=cuda variant=v12 compute=6.1 driver=12.7 name="NVIDIA GeForce GT 1030" total="3.9 GiB" available="3.9 GiB"
[GIN] 2025/07/01 - 18:00:09 | 200 |      50.263µs | 100.126.204.152 | HEAD     "/"
`,
			exp: []InferenceCompute{{
				Library: "cuda",
				Variant: "v12",
				Compute: "6.1",
				Driver:  "12.7",
				Name:    "NVIDIA GeForce GT 1030",
				VRAM:    "3.9 GiB",
			}},
		},
		{
			name: "frank",
			log: `time=2025-07-01T19:36:13.315Z level=INFO source=amd_linux.go:386 msg="amdgpu is supported" gpu=GPU-9abb57639fa80c50 gpu_type=gfx1030
		releasing cuda driver library
		time=2025-07-01T19:36:13.315Z level=INFO source=types.go:130 msg="inference compute" id=GPU-d6de3398-9932-6902-11ec-fee8e424c8a2 library=cuda variant=v12 compute=7.5 driver=12.8 name="NVIDIA GeForce RTX 2080 Ti" total="10.6 GiB" available="10.4 GiB"
		time=2025-07-01T19:36:13.315Z level=INFO source=types.go:130 msg="inference compute" id=GPU-9abb57639fa80c50 library=rocm variant="" compute=gfx1030 driver=6.3 name=1002:73bf total="16.0 GiB" available="1.3 GiB"
		[GIN] 2025/07/01 - 18:00:09 | 200 |      50.263µs | 100.126.204.152 | HEAD     "/"
		`,
			exp: []InferenceCompute{
				{
					Library: "cuda",
					Variant: "v12",
					Compute: "7.5",
					Driver:  "12.8",
					Name:    "NVIDIA GeForce RTX 2080 Ti",
					VRAM:    "10.6 GiB",
				},
				{
					Library: "rocm",
					Compute: "gfx1030",
					Driver:  "6.3",
					Name:    "1002:73bf",
					VRAM:    "16.0 GiB",
				},
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tmpDir := t.TempDir()
			serverLogPath = filepath.Join(tmpDir, "server.log")
			err := os.WriteFile(serverLogPath, []byte(tt.log), 0o644)
			if err != nil {
				t.Fatalf("failed to write log file %s: %s", serverLogPath, err)
			}
			ctx, cancel := context.WithTimeout(t.Context(), 10*time.Millisecond)
			defer cancel()
			ics, err := GetInferenceComputer(ctx)
			if err != nil {
				t.Fatalf(" failed to get inference compute: %v", err)
			}
			if !reflect.DeepEqual(ics, tt.exp) {
				t.Fatalf("got:\n%#v\nwant:\n%#v", ics, tt.exp)
			}
		})
	}
}

func TestGetInferenceComputerTimeout(t *testing.T) {
	ctx, cancel := context.WithTimeout(t.Context(), 10*time.Millisecond)
	defer cancel()
	tmpDir := t.TempDir()
	serverLogPath = filepath.Join(tmpDir, "server.log")
	err := os.WriteFile(serverLogPath, []byte("foo\nbar\nbaz\n"), 0o644)
	if err != nil {
		t.Fatalf("failed to write log file %s: %s", serverLogPath, err)
	}
	_, err = GetInferenceComputer(ctx)
	if err == nil {
		t.Fatal("expected timeout")
	}
	if !strings.Contains(err.Error(), "timeout") {
		t.Fatalf("unexpected error: %s", err)
	}
}
