package runners

import (
	"fmt"
	"log/slog"
	"os"
	"path"
	"runtime"
	"strings"
	"sync"
	"testing"
	"testing/fstest"

	"github.com/ollama/ollama/gpu"
)

var runnerTmpDir = sync.OnceValues(func() (string, error) {
	tmpDir, err := os.MkdirTemp("", "testing")
	if err != nil {
		return "", fmt.Errorf("failed to make tmp dir %w", err)
	}
	return tmpDir, nil
})

func TestRefreshRunners(t *testing.T) {
	slog.SetLogLoggerLevel(slog.LevelDebug)

	payloadFS := fstest.MapFS{
		path.Join(runtime.GOOS, runtime.GOARCH, "cpu", "ollama_llama_server"):     {Data: []byte("hello, world\n")},
		path.Join(runtime.GOOS, runtime.GOARCH, "cpu_avx", "ollama_llama_server"): {Data: []byte("hello, world\n")},
		path.Join(runtime.GOOS, runtime.GOARCH, "foo_v12", "ollama_llama_server"): {Data: []byte("hello, world\n")},
		path.Join(runtime.GOOS, runtime.GOARCH, "foo_v11", "ollama_llama_server"): {Data: []byte("hello, world\n")},
		path.Join(runtime.GOOS, runtime.GOARCH, "bar", "ollama_llama_server"):     {Data: []byte("hello, world\n")},
	}
	tmpDir, err := runnerTmpDir()
	if err != nil {
		t.Fatalf("failed to make tmp dir %s", err)
	}
	t.Setenv("OLLAMA_TMPDIR", tmpDir)
	rDir, err := Refresh(payloadFS)
	if err != nil {
		t.Fatalf("failed to extract to %s %s", tmpDir, err)
	}
	if !strings.Contains(rDir, tmpDir) {
		t.Fatalf("runner dir %s was not in tmp dir %s", rDir, tmpDir)
	}

	// spot check results
	servers := GetAvailableServers(rDir)
	slog.Debug("GetAvailableServers", "response", servers)
	if len(servers) < 1 {
		t.Fatalf("expected at least 1 server")
	}

	// Refresh contents
	rDir, err = extractRunners(payloadFS)
	if err != nil {
		t.Fatalf("failed to extract to %s %s", tmpDir, err)
	}
	if !strings.Contains(rDir, tmpDir) {
		t.Fatalf("runner dir %s was not in tmp dir %s", rDir, tmpDir)
	}

	// Verify correct server response list
	type testCase struct {
		input  []gpu.GpuInfo
		expect []string
	}
	testCases := map[string]*testCase{
		"cpu":     {input: []gpu.GpuInfo{{Library: "cpu"}}, expect: []string{"cpu"}},
		"cpu_avx": {input: []gpu.GpuInfo{{Library: "cpu", Variant: "avx"}}, expect: []string{"cpu_avx"}},
	}
	if runtime.GOOS == "darwin" && runtime.GOARCH == "arm64" {
		// macos arm doesn't append "cpu"
		testCases["foo no variant"] = &testCase{input: []gpu.GpuInfo{{Library: "foo"}}, expect: []string{"foo_v11", "foo_v12"}}
		testCases["foo v11"] = &testCase{input: []gpu.GpuInfo{{Library: "foo", Variant: "v11"}}, expect: []string{"foo_v11", "foo_v12"}}
		testCases["foo v12"] = &testCase{input: []gpu.GpuInfo{{Library: "foo", Variant: "v12"}}, expect: []string{"foo_v12", "foo_v11"}}
		testCases["foo v11 and v12"] = &testCase{input: []gpu.GpuInfo{{Library: "foo", Variant: "v11"}, {Library: "foo", Variant: "v12"}}, expect: []string{"foo_v11", "foo_v12"}}
	} else {
		// All other platforms append "cpu" as a last ditch runner
		testCases["foo no variant"] = &testCase{input: []gpu.GpuInfo{{Library: "foo"}}, expect: []string{"foo_v11", "foo_v12", "cpu"}}
		testCases["foo v11"] = &testCase{input: []gpu.GpuInfo{{Library: "foo", Variant: "v11"}}, expect: []string{"foo_v11", "foo_v12", "cpu"}}
		testCases["foo v12"] = &testCase{input: []gpu.GpuInfo{{Library: "foo", Variant: "v12"}}, expect: []string{"foo_v12", "foo_v11", "cpu"}}
		testCases["foo v11 and v12"] = &testCase{input: []gpu.GpuInfo{{Library: "foo", Variant: "v11"}, {Library: "foo", Variant: "v12"}}, expect: []string{"foo_v11", "foo_v12", "cpu"}}
	}
	for k, v := range testCases {
		t.Run(k, func(t *testing.T) {
			resp := ServersForGpu(v.input)
			if len(resp) != len(v.expect) {
				t.Fatalf("expected length %d, got %d => %+v", len(v.expect), len(resp), resp)
			}
			for i := range resp {
				if resp[i] != v.expect[i] {
					t.Fatalf("expected offset %d, got %s wanted %s", i, resp[i], v.expect[i])
				}
			}
		})
	}

	cleanupTmpDirs()

	Cleanup(payloadFS)
}
