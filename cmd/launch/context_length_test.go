package launch

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"strings"
	"testing"

	"github.com/ollama/ollama/api"
)

func testLauncherClientWithStatus(t *testing.T, contextLength int) (*launcherClient, *int) {
	t.Helper()

	statusCalls := 0
	client := &http.Client{Transport: roundTripFunc(func(r *http.Request) (*http.Response, error) {
		if r.URL.Path != "/api/status" {
			return &http.Response{
				StatusCode: http.StatusNotFound,
				Body:       io.NopCloser(strings.NewReader(`{"error":"not found"}`)),
				Header:     make(http.Header),
			}, nil
		}
		statusCalls++
		return &http.Response{
			StatusCode: http.StatusOK,
			Body:       io.NopCloser(strings.NewReader(fmt.Sprintf(`{"cloud":{"disabled":false,"source":"none"},"context_length":%d}`, contextLength))),
			Header:     make(http.Header),
		}, nil
	})}

	u, err := url.Parse("http://ollama.test")
	if err != nil {
		t.Fatal(err)
	}
	return &launcherClient{apiClient: api.NewClient(u, client)}, &statusCalls
}

func TestPrepareClaudeLaunchModelsWarnsForLowLocalContext(t *testing.T) {
	client, _ := testLauncherClientWithStatus(t, 32*1024)

	oldConfirm := DefaultConfirmPrompt
	defer func() { DefaultConfirmPrompt = oldConfirm }()

	var gotPrompt string
	DefaultConfirmPrompt = func(prompt string, options ConfirmOptions) (bool, error) {
		gotPrompt = prompt
		if !options.DefaultNo {
			t.Fatal("expected warning prompt to default to no")
		}
		if options.YesLabel != "Continue" || options.NoLabel != "Cancel" {
			t.Fatalf("labels = %q/%q, want Continue/Cancel", options.YesLabel, options.NoLabel)
		}
		return false, nil
	}

	_, err := client.prepareLaunchModelsForRun(context.Background(), &Claude{}, "llama3.2", []LaunchModel{{Name: "llama3.2"}})
	if !errors.Is(err, ErrCancelled) {
		t.Fatalf("error = %v, want ErrCancelled", err)
	}
	for _, want := range []string{
		"Claude Code works best with at least 100k context.",
		"Current local context: 32k.",
		"Continue launching Claude Code?",
	} {
		if !strings.Contains(gotPrompt, want) {
			t.Fatalf("prompt missing %q:\n%s", want, gotPrompt)
		}
	}
}

func TestPrepareClaudeLaunchModelsSetsHighLocalContextWithoutWarning(t *testing.T) {
	client, _ := testLauncherClientWithStatus(t, 128*1024)

	oldConfirm := DefaultConfirmPrompt
	defer func() { DefaultConfirmPrompt = oldConfirm }()
	DefaultConfirmPrompt = func(prompt string, options ConfirmOptions) (bool, error) {
		t.Fatalf("did not expect prompt, got %q", prompt)
		return false, nil
	}

	models, err := client.prepareLaunchModelsForRun(context.Background(), &Claude{}, "llama3.2", []LaunchModel{{Name: "llama3.2"}})
	if err != nil {
		t.Fatalf("prepareLaunchModelsForRun error = %v", err)
	}
	if len(models) != 1 || models[0].ContextLength != 128*1024 {
		t.Fatalf("models = %+v, want local context length set", models)
	}
}

func TestPrepareClaudeLaunchModelsMatchesLatestSuffix(t *testing.T) {
	client, _ := testLauncherClientWithStatus(t, 128*1024)

	models, err := client.prepareLaunchModelsForRun(context.Background(), &Claude{}, "gemma4", []LaunchModel{{Name: "gemma4:latest"}})
	if err != nil {
		t.Fatalf("prepareLaunchModelsForRun error = %v", err)
	}
	if len(models) != 1 {
		t.Fatalf("models = %+v, want existing model updated without fallback", models)
	}
	if models[0].Name != "gemma4:latest" || models[0].ContextLength != 128*1024 {
		t.Fatalf("model = %+v, want latest-suffixed model updated with context length", models[0])
	}
}

func TestPrepareClaudeLaunchModelsYesPrintsShortWarning(t *testing.T) {
	client, _ := testLauncherClientWithStatus(t, 32*1024)

	oldConfirm := DefaultConfirmPrompt
	defer func() { DefaultConfirmPrompt = oldConfirm }()
	DefaultConfirmPrompt = func(prompt string, options ConfirmOptions) (bool, error) {
		t.Fatalf("did not expect prompt with --yes, got %q", prompt)
		return false, nil
	}

	restoreConfirm := withLaunchConfirmPolicy(launchConfirmPolicy{yes: true})
	defer restoreConfirm()

	output := captureContextWarningStderr(t, func() {
		models, err := client.prepareLaunchModelsForRun(context.Background(), &Claude{}, "llama3.2", []LaunchModel{{Name: "llama3.2"}})
		if err != nil {
			t.Fatalf("prepareLaunchModelsForRun error = %v", err)
		}
		if len(models) != 1 || models[0].ContextLength != 32*1024 {
			t.Fatalf("models = %+v, want local context length set", models)
		}
	})

	for _, want := range []string{
		"Warning: Claude Code works best with at least 100k context; current local context is 32k.",
		"Continuing because --yes was provided.",
	} {
		if !strings.Contains(output, want) {
			t.Fatalf("stderr missing %q:\n%s", want, output)
		}
	}
}

func TestPrepareClaudeLaunchModelsSkipsCloudModels(t *testing.T) {
	client, statusCalls := testLauncherClientWithStatus(t, 32*1024)

	oldConfirm := DefaultConfirmPrompt
	defer func() { DefaultConfirmPrompt = oldConfirm }()
	DefaultConfirmPrompt = func(prompt string, options ConfirmOptions) (bool, error) {
		t.Fatalf("did not expect prompt, got %q", prompt)
		return false, nil
	}

	models, err := client.prepareLaunchModelsForRun(context.Background(), &Claude{}, "glm-5:cloud", []LaunchModel{{Name: "glm-5:cloud", Remote: true}})
	if err != nil {
		t.Fatalf("prepareLaunchModelsForRun error = %v", err)
	}
	if *statusCalls != 0 {
		t.Fatalf("status calls = %d, want 0", *statusCalls)
	}
	if len(models) != 1 || models[0].ContextLength != 0 {
		t.Fatalf("models = %+v, want unchanged cloud model", models)
	}
}

func TestPrepareOpenCodeLaunchModelsSetsLocalLimits(t *testing.T) {
	client, _ := testLauncherClientWithStatus(t, 32*1024)

	models := client.prepareLaunchModelsForConfig(context.Background(), &OpenCode{}, "llama3.2", []LaunchModel{
		{Name: "llama3.2", ContextLength: 131072},
		{Name: "glm-5:cloud", ContextLength: 202752, MaxOutputTokens: 131072},
	})

	if len(models) != 2 {
		t.Fatalf("models = %+v, want 2", models)
	}
	if models[0].ContextLength != 32*1024 || models[0].MaxOutputTokens != 8192 {
		t.Fatalf("local model = %+v, want server context and derived output", models[0])
	}
	if models[1].ContextLength != 202752 || models[1].MaxOutputTokens != 131072 {
		t.Fatalf("cloud model = %+v, want unchanged cloud limits", models[1])
	}

	entries := buildModelEntries(models)
	local, _ := entries["llama3.2"].(map[string]any)
	limit, _ := local["limit"].(map[string]any)
	if limit["context"] != 32*1024 || limit["output"] != 8192 {
		t.Fatalf("local limit = %v, want context/output from server context", limit)
	}
}

func TestOpenCodeLocalMaxOutputTokens(t *testing.T) {
	tests := map[int]int{
		4096:       2048,
		16 * 1024:  4096,
		32 * 1024:  8192,
		128 * 1024: 8192,
	}
	for contextLength, want := range tests {
		if got := openCodeLocalMaxOutputTokens(contextLength); got != want {
			t.Fatalf("openCodeLocalMaxOutputTokens(%d) = %d, want %d", contextLength, got, want)
		}
	}
}

func TestPrepareOpenCodeLaunchModelsWarnsForLowLocalContext(t *testing.T) {
	client, _ := testLauncherClientWithStatus(t, 32*1024)

	oldConfirm := DefaultConfirmPrompt
	defer func() { DefaultConfirmPrompt = oldConfirm }()

	var gotPrompt string
	DefaultConfirmPrompt = func(prompt string, options ConfirmOptions) (bool, error) {
		gotPrompt = prompt
		if !options.DefaultNo {
			t.Fatal("expected warning prompt to default to no")
		}
		return false, nil
	}

	_, err := client.prepareLaunchModelsForRun(context.Background(), &OpenCode{}, "llama3.2", []LaunchModel{{Name: "llama3.2"}})
	if !errors.Is(err, ErrCancelled) {
		t.Fatalf("error = %v, want ErrCancelled", err)
	}
	for _, want := range []string{
		"OpenCode works best with at least 64k context.",
		"Current local context: 32k.",
		"Continue launching OpenCode?",
	} {
		if !strings.Contains(gotPrompt, want) {
			t.Fatalf("prompt missing %q:\n%s", want, gotPrompt)
		}
	}
}

func TestPrepareOpenCodeLaunchModelsAppliesLocalLimitsWithCloudPrimary(t *testing.T) {
	client, _ := testLauncherClientWithStatus(t, 32*1024)

	oldConfirm := DefaultConfirmPrompt
	defer func() { DefaultConfirmPrompt = oldConfirm }()
	DefaultConfirmPrompt = func(prompt string, options ConfirmOptions) (bool, error) {
		t.Fatalf("did not expect prompt for cloud primary, got %q", prompt)
		return false, nil
	}

	models, err := client.prepareLaunchModelsForRun(context.Background(), &OpenCode{}, "glm-5:cloud", []LaunchModel{
		{Name: "glm-5:cloud", Remote: true, ContextLength: 202752, MaxOutputTokens: 131072},
		{Name: "llama3.2", ContextLength: 131072},
	})
	if err != nil {
		t.Fatalf("prepareLaunchModelsForRun error = %v", err)
	}
	if len(models) != 2 {
		t.Fatalf("models = %+v, want 2", models)
	}
	if models[0].ContextLength != 202752 || models[0].MaxOutputTokens != 131072 {
		t.Fatalf("cloud model = %+v, want unchanged cloud limits", models[0])
	}
	if models[1].ContextLength != 32*1024 || models[1].MaxOutputTokens != 8192 {
		t.Fatalf("local model = %+v, want server context and derived output", models[1])
	}
}

func TestPrepareOpenCodeLaunchModelsSkipsAllCloudModels(t *testing.T) {
	client, statusCalls := testLauncherClientWithStatus(t, 32*1024)

	models, err := client.prepareLaunchModelsForRun(context.Background(), &OpenCode{}, "glm-5:cloud", []LaunchModel{{Name: "glm-5:cloud", Remote: true}})
	if err != nil {
		t.Fatalf("prepareLaunchModelsForRun error = %v", err)
	}
	if *statusCalls != 0 {
		t.Fatalf("status calls = %d, want 0", *statusCalls)
	}
	if len(models) != 1 || models[0].ContextLength != 0 {
		t.Fatalf("models = %+v, want unchanged cloud model", models)
	}
}

func TestLaunchModelsWithOpenCodeLocalLimitsDoesNotAppendMissingCloudPrimary(t *testing.T) {
	models := launchModelsWithOpenCodeLocalLimits("glm-5:cloud", []LaunchModel{{Name: "llama3.2"}}, 32*1024)
	if len(models) != 1 {
		t.Fatalf("models = %+v, want no fallback cloud primary appended", models)
	}
	if models[0].Name != "llama3.2" || models[0].ContextLength != 32*1024 || models[0].MaxOutputTokens != 8192 {
		t.Fatalf("local model = %+v, want local limits applied", models[0])
	}
}

func captureContextWarningStderr(t *testing.T, fn func()) string {
	t.Helper()

	old := os.Stderr
	r, w, err := os.Pipe()
	if err != nil {
		t.Fatal(err)
	}
	os.Stderr = w
	defer func() {
		os.Stderr = old
		r.Close()
	}()

	fn()

	if err := w.Close(); err != nil {
		t.Fatal(err)
	}
	data, err := io.ReadAll(r)
	if err != nil {
		t.Fatal(err)
	}
	return string(data)
}

func TestFormatContextLength(t *testing.T) {
	tests := map[int]string{
		32 * 1024: "32k",
		64 * 1024: "64k",
		100_000:   "100k",
		100_001:   "100001",
	}
	for tokens, want := range tests {
		if got := formatContextLength(tokens); got != want {
			t.Fatalf("formatContextLength(%d) = %q, want %q", tokens, got, want)
		}
	}
}
