package launch

import (
	"encoding/json"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"github.com/ollama/ollama/envconfig"
)

func TestMuxConfigPreservesEndpointsAndClearsStaleCredentials(t *testing.T) {
	dir := t.TempDir()
	t.Setenv("MUX_CONFIG_DIR", dir)
	t.Setenv("OLLAMA_HOST", "http://127.0.0.1:11435")
	path := filepath.Join(dir, "endpoints.json")
	original := `{"custom":{"keep":true},"endpoints":[{"name":"personal","adapterType":"openai","apiKey":"personal-secret","isDefault":true,"futureField":42},{"name":"ollama-launch","apiKey":"stale-secret","headers":{"Authorization":"Bearer stale"}}]}`
	if err := os.WriteFile(path, []byte(original), 0o600); err != nil {
		t.Fatal(err)
	}
	runner := &Mux{}
	if err := runner.Edit([]LaunchModel{{Name: "deepseek-v4.1-flash:cloud", ContextLength: 1048576}}); err != nil {
		t.Fatal(err)
	}
	root, endpoints, err := readMuxEndpoints(path)
	if err != nil {
		t.Fatal(err)
	}
	if len(endpoints) != 2 {
		t.Fatalf("got %d endpoints", len(endpoints))
	}
	var before map[string]json.RawMessage
	var saved struct {
		Endpoints []map[string]json.RawMessage `json:"endpoints"`
	}
	json.Unmarshal([]byte(original), &before)
	json.Unmarshal([]byte(original), &saved)
	if !reflect.DeepEqual(endpoints[0], saved.Endpoints[0]) {
		t.Fatal("existing endpoint changed")
	}
	var customBefore, customAfter any
	json.Unmarshal(before["custom"], &customBefore)
	json.Unmarshal(root["custom"], &customAfter)
	if !reflect.DeepEqual(customBefore, customAfter) {
		t.Fatal("unknown root field changed")
	}
	var endpoint struct {
		AdapterType, BaseURL, Model string
		ContextWindow               int
		APIKey                      *string
		Headers                     map[string]string
	}
	data, _ := json.Marshal(endpoints[1])
	json.Unmarshal(data, &endpoint)
	if endpoint.AdapterType != "ollama" || endpoint.BaseURL != envconfig.ConnectableHost().String() || endpoint.Model != "deepseek-v4.1-flash:cloud" || endpoint.ContextWindow != 1048576 {
		t.Fatalf("bad launch endpoint: %+v", endpoint)
	}
	if endpoint.APIKey != nil || len(endpoint.Headers) > 0 {
		t.Fatal("stale credentials retained")
	}
	if !reflect.DeepEqual(runner.Models(), []string{"deepseek-v4.1-flash:cloud"}) {
		t.Fatal("model not restored")
	}
	if err := runner.Edit([]LaunchModel{{Name: "qwen3:8b"}}); err != nil {
		t.Fatal(err)
	}
	_, endpoints, _ = readMuxEndpoints(path)
	if len(endpoints) != 2 {
		t.Fatal("repeated config duplicated endpoints")
	}
	if !reflect.DeepEqual(runner.Paths(), []string{path}) {
		t.Fatal("config path not reported")
	}
}

func TestMuxInvalidConfigIsNotOverwritten(t *testing.T) {
	for _, content := range []string{"{", "null", `{"endpoints":{}}`} {
		t.Run(content, func(t *testing.T) {
			dir := t.TempDir()
			t.Setenv("MUX_CONFIG_DIR", dir)
			path := filepath.Join(dir, "endpoints.json")
			os.WriteFile(path, []byte(content), 0o600)
			if err := (&Mux{}).Edit([]LaunchModel{{Name: "qwen3:8b"}}); err == nil {
				t.Fatal("expected error")
			}
			got, _ := os.ReadFile(path)
			if string(got) != content {
				t.Fatal("invalid config overwritten")
			}
		})
	}
}

func TestMuxFreshConfigAndEmptySelection(t *testing.T) {
	dir := filepath.Join(t.TempDir(), "nested")
	t.Setenv("MUX_CONFIG_DIR", dir)
	runner := &Mux{}
	if err := runner.Edit(nil); err != nil {
		t.Fatal(err)
	}
	if len(runner.Paths()) != 0 || len(runner.Models()) != 0 {
		t.Fatal("unexpected existing config")
	}
	if err := runner.Edit([]LaunchModel{{Name: "qwen3:8b"}}); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(runner.Models(), []string{"qwen3:8b"}) {
		t.Fatal("fresh model not saved")
	}
}

func TestMuxLaunchArgs(t *testing.T) {
	t.Setenv("OLLAMA_HOST", "http://127.0.0.1:11435")
	want := []string{"--endpoint", muxEndpointName, "--adapter-type", "ollama", "--base-url", envconfig.ConnectableHost().String(), "--model", "deepseek-v4.1-flash:cloud", "-p", "hello"}
	if got := muxLaunchArgs("deepseek-v4.1-flash:cloud", []string{"-p", "hello"}); !reflect.DeepEqual(got, want) {
		t.Fatalf("got %q, want %q", got, want)
	}
	spec, err := LookupIntegrationSpec("mux")
	if err != nil || spec.Runner.String() != "Mux" {
		t.Fatal("Mux is not registered")
	}
}

func TestMuxRunUsesConfiguredBinaryAndForwardsExit(t *testing.T) {
	if os.PathSeparator == '\\' {
		t.Skip("shell fixture is Unix-only")
	}
	dir := t.TempDir()
	t.Setenv("PATH", dir)
	path := filepath.Join(dir, "mux")
	if err := os.WriteFile(path, []byte("#!/bin/sh\nexit 17\n"), 0o700); err != nil {
		t.Fatal(err)
	}
	if err := (&Mux{}).Run("qwen3:8b", nil, nil); err == nil {
		t.Fatal("expected child failure")
	}
	os.Remove(path)
	if err := (&Mux{}).Run("qwen3:8b", nil, nil); err == nil {
		t.Fatal("expected missing binary error")
	}
}

func TestMuxLaunchHelp(t *testing.T) {
	cmd := LaunchCmd(nil, nil)
	if !strings.Contains(cmd.Long, "mux             Mux") {
		t.Fatal("Mux missing from launch help")
	}
}
