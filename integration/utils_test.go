//go:build integration

package integration

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"math"
	"math/rand"
	"net"
	"net/http"
	"net/url"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"slices"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/format"
	"github.com/ollama/ollama/types/model"
)

var (
	// testModel is set via OLLAMA_TEST_MODEL env var. When set, all tests
	// that loop over model lists will test only this model, and smol is
	// also overridden to use it.
	testModel = os.Getenv("OLLAMA_TEST_MODEL")

	smol   = defaultTestModel("llama3.2:1b")
	stream = false
)

var (
	started = time.Now()

	blueSkyPrompt   = "why is the sky blue? Be brief but factual in your reply"
	blueSkyExpected = []string{"rayleigh", "scatter", "atmosphere", "nitrogen", "oxygen", "wavelength", "interact"}

	rainbowPrompt    = "how do rainbows form? Be brief but factual in your reply"
	rainbowFollowups = []string{
		"Explain the physics involved in them.  Be brief in your reply",
		"Explain the chemistry involved in them.  Be brief in your reply",
		"What are common myths related to them? Be brief in your reply",
		"Can they form if there is no rain?  Be brief in your reply",
		"Can they form if there are no clouds?  Be brief in your reply",
		"Do they happen on other planets? Be brief in your reply",
	}
	rainbowExpected = []string{"water", "droplet", "mist", "glow", "refract", "reflect", "scatter", "particles", "wave", "color", "spectrum", "raindrop", "atmosphere", "frequency", "shower", "sky", "shimmer", "light", "storm", "sunny", "sunburst", "phenomenon", "mars", "venus", "jupiter", "rain", "sun", "rainbow", "optical", "gold", "cloud", "planet", "prism", "fog", "ice"}
)

func init() {
	logger := slog.New(slog.NewTextHandler(os.Stdout, &slog.HandlerOptions{Level: slog.LevelDebug}))
	slog.SetDefault(logger)

	if testModel != "" {
		slog.Info("test model override", "model", testModel)
	}
}

func defaultTestModel(model string) string {
	if testModel != "" {
		return testModel
	}
	return model
}

// testModels returns the override model as a single-element slice when
// OLLAMA_TEST_MODEL is set, otherwise returns the provided default list.
func testModels(defaults []string) []string {
	if testModel != "" {
		return []string{testModel}
	}
	return defaults
}

// requireCapability skips the test if the model does not advertise the
// given capability. If the model is missing locally, it first goes through
// the normal pull-if-missing path so tests still behave correctly on cold
// hosts. For local-only models where Show may not return capabilities
// (e.g. models created via ollama create), this is a best-effort check.
func requireCapability(ctx context.Context, t *testing.T, client *api.Client, modelName string, cap model.Capability) {
	t.Helper()

	resp, err := client.Show(ctx, &api.ShowRequest{Name: modelName})
	var statusError api.StatusError
	if errors.As(err, &statusError) && statusError.StatusCode == http.StatusNotFound {
		if err := PullIfMissing(ctx, client, modelName); err != nil {
			t.Skipf("model %s not available: %v", modelName, err)
		}

		resp, err = client.Show(ctx, &api.ShowRequest{Name: modelName})
	}

	if err != nil {
		t.Fatalf("failed to show model %s: %v", modelName, err)
	}
	if len(resp.Capabilities) > 0 && !slices.Contains(resp.Capabilities, cap) {
		t.Skipf("model %s does not have capability %q (has %v)", modelName, cap, resp.Capabilities)
	}
}

// pullOrSkip pulls a model if it isn't already present locally. If the
// pull fails (e.g. model not in registry), the test is skipped instead
// of failed. PullIfMissing already checks Show first, so local-only
// models that exist will return immediately without hitting the registry.
func pullOrSkip(ctx context.Context, t *testing.T, client *api.Client, modelName string) {
	t.Helper()
	if err := PullIfMissing(ctx, client, modelName); err != nil {
		t.Skipf("model %s not available: %v", modelName, err)
	}
}

func FindPort() string {
	port := 0
	if a, err := net.ResolveTCPAddr("tcp", "localhost:0"); err == nil {
		var l *net.TCPListener
		if l, err = net.ListenTCP("tcp", a); err == nil {
			port = l.Addr().(*net.TCPAddr).Port
			l.Close()
		}
	}
	if port == 0 {
		port = rand.Intn(65535-49152) + 49152 // get a random port in the ephemeral range
	}
	return strconv.Itoa(port)
}

func GetTestEndpoint() (*api.Client, string) {
	defaultPort := "11434"
	ollamaHost := os.Getenv("OLLAMA_HOST")

	scheme, hostport, ok := strings.Cut(ollamaHost, "://")
	if !ok {
		scheme, hostport = "http", ollamaHost
	}

	// trim trailing slashes
	hostport = strings.TrimRight(hostport, "/")

	host, port, err := net.SplitHostPort(hostport)
	if err != nil {
		host, port = "127.0.0.1", defaultPort
		if ip := net.ParseIP(strings.Trim(hostport, "[]")); ip != nil {
			host = ip.String()
		} else if hostport != "" {
			host = hostport
		}
	}

	if os.Getenv("OLLAMA_TEST_EXISTING") == "" && runtime.GOOS != "windows" && port == defaultPort {
		port = FindPort()
	}

	slog.Info("server connection", "host", host, "port", port)

	return api.NewClient(
		&url.URL{
			Scheme: scheme,
			Host:   net.JoinHostPort(host, port),
		},
		http.DefaultClient), fmt.Sprintf("%s:%s", host, port)
}

// Server lifecycle management
var (
	serverMutex sync.Mutex
	serverReady bool
	serverLog   bytes.Buffer
	serverDone  chan int
	serverCmd   *exec.Cmd
)

func startServer(t *testing.T, ctx context.Context, ollamaHost string) error {
	// Make sure the server has been built
	CLIName, err := filepath.Abs("../ollama")
	if err != nil {
		return fmt.Errorf("failed to get absolute path: %w", err)
	}

	if runtime.GOOS == "windows" {
		CLIName += ".exe"
	}
	_, err = os.Stat(CLIName)
	if err != nil {
		return fmt.Errorf("CLI missing, did you forget to 'go build .' first?  %w", err)
	}
	serverMutex.Lock()
	defer serverMutex.Unlock()
	if serverReady {
		return nil
	}
	serverDone = make(chan int)
	serverLog.Reset()

	if tmp := os.Getenv("OLLAMA_HOST"); tmp != ollamaHost {
		slog.Info("setting env", "OLLAMA_HOST", ollamaHost)
		t.Setenv("OLLAMA_HOST", ollamaHost)
	}

	serverCmd = exec.Command(CLIName, "serve")
	serverCmd.Stderr = &serverLog
	serverCmd.Stdout = &serverLog
	go func() {
		slog.Info("starting server", "url", ollamaHost)
		if err := serverCmd.Run(); err != nil {
			// "signal: killed" expected during normal shutdown
			if !strings.Contains(err.Error(), "signal") {
				slog.Info("failed to run server", "error", err)
			}
		}
		var code int
		if serverCmd.ProcessState != nil {
			code = serverCmd.ProcessState.ExitCode()
		}
		slog.Info("server exited")
		serverDone <- code
	}()

	serverReady = true
	return nil
}

func PullIfMissing(ctx context.Context, client *api.Client, modelName string) error {
	slog.Info("checking status of model", "model", modelName)
	showReq := &api.ShowRequest{Name: modelName}

	showCtx, cancel := context.WithDeadlineCause(
		ctx,
		time.Now().Add(20*time.Second),
		fmt.Errorf("show for existing model %s took too long", modelName),
	)
	defer cancel()
	_, err := client.Show(showCtx, showReq)
	var statusError api.StatusError
	switch {
	case errors.As(err, &statusError) && statusError.StatusCode == http.StatusNotFound:
		break
	case err != nil:
		return err
	default:
		slog.Info("model already present", "model", modelName)
		return nil
	}
	slog.Info("model missing", "model", modelName)

	stallDuration := 2 * time.Minute // Includes checksum verification, which can take a while on larger models and slower systems.
	stallTimer := time.NewTimer(stallDuration)
	fn := func(resp api.ProgressResponse) error {
		// fmt.Print(".")
		if !stallTimer.Reset(stallDuration) {
			return errors.New("stall was detected, aborting status reporting")
		}
		return nil
	}

	stream := true
	pullReq := &api.PullRequest{Name: modelName, Stream: &stream}

	var pullError error

	done := make(chan int)
	go func() {
		pullError = client.Pull(ctx, pullReq, fn)
		done <- 0
	}()

	select {
	case <-stallTimer.C:
		return errors.New("download stalled")
	case <-done:
		return pullError
	}
}

// Returns an Client, the testEndpoint, and a cleanup function, fails the test on errors
// Starts the server if needed
func InitServerConnection(ctx context.Context, t *testing.T) (*api.Client, string, func()) {
	client, testEndpoint := GetTestEndpoint()
	cleanup := func() {}
	if os.Getenv("OLLAMA_TEST_EXISTING") == "" && runtime.GOOS != "windows" {
		err := startServer(t, ctx, testEndpoint)
		if err != nil {
			t.Fatal(err)
		}
		cleanup = func() {
			serverMutex.Lock()
			defer serverMutex.Unlock()
			serverReady = false

			slog.Info("shutting down server")
			serverCmd.Process.Signal(os.Interrupt)
			slog.Info("waiting for server to exit")
			<-serverDone
			slog.Info("terminate complete")

			if t.Failed() || os.Getenv("OLLAMA_TEST_LOG_SERVER") != "" {
				slog.Warn("SERVER LOG FOLLOWS")
				io.Copy(os.Stderr, bytes.NewReader(serverLog.Bytes()))
				slog.Warn("END OF SERVER")
			}
			slog.Info("cleanup complete", "failed", t.Failed())
		}
	}
	// Make sure server is online and healthy before returning
	for {
		select {
		case <-ctx.Done():
			t.Fatalf("context done before server ready: %v", ctx.Err())
		default:
		}
		listCtx, cancel := context.WithDeadlineCause(
			ctx,
			time.Now().Add(10*time.Second),
			fmt.Errorf("list models took too long"),
		)
		defer cancel()
		models, err := client.ListRunning(listCtx)
		if err != nil {
			if runtime.GOOS == "windows" {
				t.Fatalf("did you forget to start the server: %v", err)
			}
			time.Sleep(10 * time.Millisecond)
			continue
		}
		if len(models.Models) > 0 {
			names := make([]string, len(models.Models))
			for i, m := range models.Models {
				names[i] = m.Name
			}
			slog.Info("currently loaded", "models", names)
		}
		break
	}

	return client, testEndpoint, cleanup
}

func ChatTestHelper(ctx context.Context, t *testing.T, req api.ChatRequest, anyResp []string) {
	client, _, cleanup := InitServerConnection(ctx, t)
	defer cleanup()
	pullOrSkip(ctx, t, client, req.Model)
	DoChat(ctx, t, client, req, anyResp, 30*time.Second, 10*time.Second)
}

func DoGenerate(ctx context.Context, t *testing.T, client *api.Client, genReq api.GenerateRequest, anyResp []string, initialTimeout, streamTimeout time.Duration) []int {
	stallTimer := time.NewTimer(initialTimeout)
	// mu guards the fields written by the streaming callback, which runs on
	// the client goroutine while the stall and ctx-done paths read them.
	var mu sync.Mutex
	var buf bytes.Buffer
	var thinkBuf bytes.Buffer
	var doneReason string
	var context []int
	fn := func(response api.GenerateResponse) error {
		// fmt.Print(".")
		mu.Lock()
		buf.Write([]byte(response.Response))
		thinkBuf.WriteString(response.Thinking)
		if response.Done {
			doneReason = response.DoneReason
		}
		if len(response.Context) > 0 {
			context = response.Context
		}
		mu.Unlock()
		if !stallTimer.Reset(streamTimeout) {
			return errors.New("stall was detected while streaming response, aborting")
		}
		return nil
	}

	stream := true
	genReq.Stream = &stream
	done := make(chan int)
	var genErr error
	go func() {
		genErr = client.Generate(ctx, &genReq, fn)
		done <- 0
	}()

	verify := func() bool {
		// Verify the response contains the expected data
		mu.Lock()
		response := buf.String()
		thinking := thinkBuf.String()
		reason := doneReason
		mu.Unlock()
		if containsExpectedResponse(response, anyResp) {
			return true
		}
		if strings.TrimSpace(response) == "" {
			if containsExpectedResponse(thinking, anyResp) {
				slog.Warn("keywords found only in thinking; budget likely exhausted", "model", genReq.Model, "done_reason", reason)
			}
			t.Errorf("%s: model returned empty content (done_reason=%q)", genReq.Model, reason)
			return false
		}
		t.Errorf("%s: none of %v found in %s", genReq.Model, anyResp, response)
		return false
	}
	partial := func() string {
		mu.Lock()
		defer mu.Unlock()
		return buf.String()
	}

	select {
	case <-stallTimer.C:
		if response := partial(); response == "" {
			t.Errorf("generate never started.  Timed out after :%s", initialTimeout.String())
		} else {
			t.Errorf("generate stalled.  Response so far:%s", response)
		}
	case <-done:
		if genErr != nil && strings.Contains(genErr.Error(), "model requires more system memory") {
			slog.Warn("model is too large for the target test system", "model", genReq.Model, "error", genErr)
			return context
		}
		if genErr != nil {
			t.Errorf("%s failed with %s request prompt %s", genErr, genReq.Model, genReq.Prompt)
			return context
		}
		if verify() {
			slog.Info("test pass", "model", genReq.Model, "prompt", genReq.Prompt, "contains", anyResp, "response", partial())
		}
	case <-ctx.Done():
		// On slow systems, we might timeout before some models finish rambling, so check what we have so far to see
		// if it's considered a pass - the stallTimer will detect hangs, but we want to consider slow systems a pass
		// if they are still generating valid responses
		slog.Warn("outer test context done while waiting for generate")
		verify()
	}
	mu.Lock()
	defer mu.Unlock()
	return context
}

// Generate a set of requests
// By default each request uses llama3.2 as the model
func GenerateRequests() ([]api.GenerateRequest, [][]string) {
	return []api.GenerateRequest{
			{
				Model:     smol,
				Prompt:    "why is the ocean blue? Be brief but factual in your reply",
				Stream:    &stream,
				KeepAlive: &api.Duration{Duration: 10 * time.Second},
			}, {
				Model:     smol,
				Prompt:    "why is the color of dirt brown? Be brief but factual in your reply",
				Stream:    &stream,
				KeepAlive: &api.Duration{Duration: 10 * time.Second},
			}, {
				Model:     smol,
				Prompt:    rainbowPrompt,
				Stream:    &stream,
				KeepAlive: &api.Duration{Duration: 10 * time.Second},
			}, {
				Model:     smol,
				Prompt:    "what is the origin of independence day? Be brief but factual in your reply",
				Stream:    &stream,
				KeepAlive: &api.Duration{Duration: 10 * time.Second},
			}, {
				Model:     smol,
				Prompt:    "what is the composition of air? Be brief but factual in your reply",
				Stream:    &stream,
				KeepAlive: &api.Duration{Duration: 10 * time.Second},
			},
		},
		[][]string{
			{"sunlight", "scatter", "interact", "color", "surface", "depth", "red", "orange", "yellow", "absorb", "wavelength", "water", "molecule"},
			{"soil", "organic", "earth", "black", "tan", "chemical", "processes", "pigment", "particle", "iron oxide", "rust", "air", "water", "wet", "mixture", "mixing", "mineral", "element", "decomposed", "matter", "wavelength"},
			rainbowExpected,
			{"fourth", "july", "declaration", "independence"},
			{"nitrogen", "oxygen", "carbon", "dioxide", "water", "vapor", "fluid", "particles", "gas"},
		}
}

// summarizeMessages returns a compact string form of the messages suitable
// for logs and error output. Image byte payloads are replaced with a
// "<image: N bytes>" marker so vision tests don't dump huge integer arrays.
func summarizeMessages(msgs []api.Message) string {
	var b strings.Builder
	b.WriteByte('[')
	for i, m := range msgs {
		if i > 0 {
			b.WriteString(", ")
		}
		fmt.Fprintf(&b, "{Role:%s Content:%q", m.Role, m.Content)
		if m.Thinking != "" {
			fmt.Fprintf(&b, " Thinking:%q", m.Thinking)
		}
		if len(m.Images) > 0 {
			b.WriteString(" Images:[")
			for j, img := range m.Images {
				if j > 0 {
					b.WriteString(", ")
				}
				fmt.Fprintf(&b, "<image: %d bytes>", len(img))
			}
			b.WriteByte(']')
		}
		if len(m.ToolCalls) > 0 {
			fmt.Fprintf(&b, " ToolCalls:%+v", m.ToolCalls)
		}
		if m.ToolName != "" {
			fmt.Fprintf(&b, " ToolName:%s", m.ToolName)
		}
		if m.ToolCallID != "" {
			fmt.Fprintf(&b, " ToolCallID:%s", m.ToolCallID)
		}
		b.WriteByte('}')
	}
	b.WriteByte(']')
	return b.String()
}

func DoChat(ctx context.Context, t *testing.T, client *api.Client, req api.ChatRequest, anyResp []string, initialTimeout, streamTimeout time.Duration) *api.Message {
	stallTimer := time.NewTimer(initialTimeout)
	// mu guards the fields written by the streaming callback, which runs on
	// the client goroutine while the stall and ctx-done paths read them.
	var mu sync.Mutex
	var buf bytes.Buffer
	var thinkBuf bytes.Buffer
	var doneReason string
	role := "assistant"
	fn := func(response api.ChatResponse) error {
		// fmt.Print(".")
		mu.Lock()
		role = response.Message.Role
		buf.Write([]byte(response.Message.Content))
		thinkBuf.WriteString(response.Message.Thinking)
		if response.Done {
			doneReason = response.DoneReason
		}
		mu.Unlock()
		if !stallTimer.Reset(streamTimeout) {
			return errors.New("stall was detected while streaming response, aborting")
		}
		return nil
	}

	stream := true
	req.Stream = &stream
	done := make(chan int)
	var genErr error
	go func() {
		genErr = client.Chat(ctx, &req, fn)
		done <- 0
	}()

	verify := func() bool {
		// Verify the response contains the expected data
		mu.Lock()
		response := buf.String()
		thinking := thinkBuf.String()
		reason := doneReason
		mu.Unlock()
		if containsExpectedResponse(response, anyResp) {
			return true
		}
		if strings.TrimSpace(response) == "" {
			if containsExpectedResponse(thinking, anyResp) {
				slog.Warn("keywords found only in thinking; budget likely exhausted", "model", req.Model, "done_reason", reason)
			}
			t.Errorf("%s: model returned empty content (done_reason=%q) -- request was:%s", req.Model, reason, summarizeMessages(req.Messages))
			return false
		}
		t.Errorf("%s: none of %v found in \"%s\" -- request was:%s", req.Model, anyResp, response, summarizeMessages(req.Messages))
		return false
	}
	msg := func() *api.Message {
		mu.Lock()
		defer mu.Unlock()
		return &api.Message{Role: role, Content: buf.String()}
	}

	select {
	case <-stallTimer.C:
		if response := msg().Content; response == "" {
			t.Errorf("generate never started.  Timed out after :%s", initialTimeout.String())
		} else {
			t.Errorf("generate stalled.  Response so far:%s", response)
		}
	case <-done:
		if genErr != nil && strings.Contains(genErr.Error(), "model requires more system memory") {
			slog.Warn("model is too large for the target test system", "model", req.Model, "error", genErr)
			return nil
		}
		if genErr != nil {
			t.Errorf("%s failed with %s request prompt %s", genErr, req.Model, summarizeMessages(req.Messages))
			return msg()
		}
		if verify() {
			slog.Info("test pass", "model", req.Model, "messages", summarizeMessages(req.Messages), "contains", anyResp, "response", msg().Content)
		}
	case <-ctx.Done():
		// On slow systems, we might timeout before some models finish rambling, so check what we have so far to see
		// if it's considered a pass - the stallTimer will detect hangs, but we want to consider slow systems a pass
		// if they are still generating valid responses
		slog.Warn("outer test context done while waiting for chat")
		verify()
	}
	return msg()
}

func containsExpectedResponse(response string, anyResp []string) bool {
	lowerResponse := strings.ToLower(response)
	normalizedResponse := normalizeResponseText(response)
	for _, resp := range anyResp {
		if strings.Contains(lowerResponse, strings.ToLower(resp)) {
			return true
		}
		if strings.Contains(normalizedResponse, normalizeResponseText(resp)) {
			return true
		}
	}
	return false
}

func normalizeResponseText(s string) string {
	return strings.Join(strings.Fields(strings.ToLower(s)), " ")
}

func ChatRequests() ([]api.ChatRequest, [][]string) {
	genReqs, results := GenerateRequests()
	reqs := make([]api.ChatRequest, len(genReqs))
	// think := api.ThinkValue{Value: "low"}
	for i := range reqs {
		reqs[i].Model = genReqs[i].Model
		reqs[i].Stream = genReqs[i].Stream
		reqs[i].KeepAlive = genReqs[i].KeepAlive
		// reqs[i].Think = &think
		reqs[i].Messages = []api.Message{
			{
				Role:    "user",
				Content: genReqs[i].Prompt,
			},
		}
	}
	return reqs, results
}

func preloadGenerateModel(ctx context.Context, t *testing.T, client *api.Client, req api.GenerateRequest) {
	t.Helper()
	slog.Info("loading", "model", req.Model)
	err := client.Generate(ctx, &req, func(response api.GenerateResponse) error { return nil })
	if err != nil {
		skipIfMLXUnsupported(t, err)
		t.Fatalf("failed to load model %s: %s", req.Model, err)
	}
}

// skipIfMLXUnsupported converts an MLX runner startup error into a test skip
// when the fingerprint matches "the MLX stack is not wired up on this host",
// and only on platforms where MLX is not yet expected to work. On Apple
// Silicon (darwin/arm64) MLX must work, so the same errors there fall
// through and fail the test — we never want to mask a real Mac regression.
//
// The fingerprints are the exact wrapper strings produced by the MLX code
// paths (see x/mlxrunner/server.go, x/mlxrunner/mlx/dynamic.go). Model-level errors
// (unsupported architecture, tensor mismatches, runtime failures) do not
// contain these strings, so this helper will not mask them.
func skipIfMLXUnsupported(t *testing.T, err error) {
	t.Helper()
	if err == nil {
		return
	}
	targetGOOS, targetGOARCH := targetPlatform()
	if targetGOOS == "darwin" && targetGOARCH == "arm64" {
		return
	}
	msg := err.Error()
	for _, s := range []string{
		"MLX not available:",
		"failed to load MLX dynamic library",
		"failed to load MLX function symbols",
		"MLX on macOS requires Apple Silicon",
		"MLX is not supported on",
	} {
		if strings.Contains(msg, s) {
			t.Skipf("MLX not available on target %s/%s (runner %s/%s): %v", targetGOOS, targetGOARCH, runtime.GOOS, runtime.GOARCH, err)
		}
	}
}

func targetPlatform() (goos, goarch string) {
	goos = normalizeTargetGOOS(os.Getenv("OLLAMA_TEST_HOST_OS"))
	goarch = normalizeTargetGOARCH(os.Getenv("OLLAMA_TEST_HOST_ARCH"))
	if goos == "" {
		goos = runtime.GOOS
	}
	if goarch == "" {
		goarch = runtime.GOARCH
	}
	return goos, goarch
}

func normalizeTargetGOOS(goos string) string {
	switch strings.ToLower(goos) {
	case "darwin":
		return "darwin"
	case "linux":
		return "linux"
	case "windows", "win32nt":
		return "windows"
	default:
		return strings.ToLower(goos)
	}
}

func normalizeTargetGOARCH(goarch string) string {
	switch strings.ToLower(goarch) {
	case "aarch64", "arm64":
		return "arm64"
	case "x86_64", "amd64":
		return "amd64"
	default:
		return strings.ToLower(goarch)
	}
}

// vramGateWarning ensures the "VRAM gates disabled" warning is only
// emitted once per test run.
var vramGateWarning sync.Once

// maxVRAMBytes returns the VRAM budget the VRAM gates test against, from the
// OLLAMA_MAX_VRAM env var. Returns ok=false when it is unset, disabling the
// gates; a one-time log line makes that visible in the run logs.
// TODO derive this from a server API in the future.
func maxVRAMBytes(t *testing.T) (uint64, bool) {
	t.Helper()
	s := os.Getenv("OLLAMA_MAX_VRAM")
	if s == "" {
		vramGateWarning.Do(func() {
			slog.Warn("OLLAMA_MAX_VRAM not set - VRAM gates disabled, large models will be attempted")
		})
		return 0, false
	}
	maxVram, err := strconv.ParseUint(s, 10, 64)
	if err != nil {
		t.Fatalf("invalid OLLAMA_MAX_VRAM %v", err)
	}
	return maxVram, true
}

// modelNameMatches reports whether a model reference (possibly a bare name
// without a tag, e.g. "gemma4") matches a listed model name (which is always
// tagged, e.g. "gemma4:latest").
func modelNameMatches(model, name string) bool {
	if name == model {
		return true
	}
	return !strings.Contains(model, ":") && strings.HasPrefix(name, model+":")
}

// skipIfModelSizeExceedsVRAM is the shared implementation for the VRAM size
// gates: it skips the test when the model's on-disk size scaled by factor
// exceeds OLLAMA_MAX_VRAM. No-op (with a one-time warning) when
// OLLAMA_MAX_VRAM is unset.
func skipIfModelSizeExceedsVRAM(ctx context.Context, t *testing.T, client *api.Client, modelName string, factor float32) {
	t.Helper()
	maxVram, ok := maxVRAMBytes(t)
	if !ok {
		return
	}
	// OLLAMA_MAX_VRAM=0 marks a CPU-only host: GPU-fit gating doesn't apply
	// (the model runs from system RAM); the min-VRAM floors still bound
	// model sizes there.
	if maxVram == 0 {
		return
	}
	resp, err := client.List(ctx)
	if err != nil {
		t.Fatalf("list models failed %v", err)
	}
	for _, m := range resp.Models {
		if modelNameMatches(modelName, m.Name) && float32(m.Size)*factor > float32(maxVram) {
			t.Skipf("model %s is too large (%s on disk, x%.2f) for available VRAM %s", modelName, format.HumanBytes(m.Size), factor, format.HumanBytes(int64(maxVram)))
		}
	}
}

// skipIfModelTooLargeForVRAM skips the test when the model's on-disk size
// is larger than OLLAMA_MAX_VRAM by enough that even partial GPU offload
// won't help. The 0.75x factor keeps single-model tests (vision/audio)
// runnable on systems where the model is slightly over VRAM and a portion
// legitimately spills to CPU — partial offload is tolerable when only one
// model is being exercised. No-op when OLLAMA_MAX_VRAM is unset.
func skipIfModelTooLargeForVRAM(ctx context.Context, t *testing.T, client *api.Client, modelName string) {
	t.Helper()
	skipIfModelSizeExceedsVRAM(ctx, t, client, modelName, 0.75)
}

// skipIfModelTooLargeForSweepVRAM is the stricter gate used by model sweeps
// (chat/embed/context cases that iterate many models). The 1.2x factor adds
// headroom for KV cache and runtime overhead: sweeps must fit entirely in
// VRAM or they run unacceptably slowly and time out the whole sweep, so any
// model that would spill to CPU is skipped outright. No-op when
// OLLAMA_MAX_VRAM is unset.
func skipIfModelTooLargeForSweepVRAM(ctx context.Context, t *testing.T, client *api.Client, modelName string) {
	t.Helper()
	skipIfModelSizeExceedsVRAM(ctx, t, client, modelName, 1.2)
}

func skipUnderMinVRAM(t *testing.T, gb uint64) {
	// A value of 0 (CPU-only host) intentionally trips the floor, unlike
	// the size gates: the floors bound model sizes on CPU-only systems.
	if maxVram, ok := maxVRAMBytes(t); ok && maxVram < gb*format.GibiByte {
		// Don't hammer on small VRAM cards...
		t.Skip("skipping with small VRAM to avoid timeouts")
	}
}

// Skip if the target model isn't X% GPU loaded to avoid excessive runtime
func skipIfNotGPULoaded(ctx context.Context, t *testing.T, client *api.Client, model string, minPercent int) {
	gpuPercent := getGPUPercent(ctx, t, client, model)
	if gpuPercent < minPercent {
		// Unload the model if we're going to skip
		client.Generate(ctx, &api.GenerateRequest{Model: model, KeepAlive: &api.Duration{Duration: 0}}, func(rsp api.GenerateResponse) error { return nil })
		t.Skipf("test requires minimum %d%% GPU load, but model %s only has %d%%", minPercent, model, gpuPercent)
	}
}

func getGPUPercent(ctx context.Context, t *testing.T, client *api.Client, model string) int {
	models, err := client.ListRunning(ctx)
	if err != nil {
		t.Fatalf("failed to list running models: %s", err)
	}
	loaded := []string{}
	for _, m := range models.Models {
		loaded = append(loaded, m.Name)
		if !modelNameMatches(model, m.Name) {
			continue
		}
		gpuPercent := 0
		switch {
		case m.SizeVRAM == 0:
			gpuPercent = 0
		case m.SizeVRAM == m.Size:
			gpuPercent = 100
		case m.SizeVRAM > m.Size || m.Size == 0:
			t.Logf("unexpected size detected: %d", m.SizeVRAM)
		default:
			sizeCPU := m.Size - m.SizeVRAM
			cpuPercent := math.Round(float64(sizeCPU) / float64(m.Size) * 110)
			gpuPercent = int(100 - cpuPercent)
		}
		return gpuPercent
	}
	t.Fatalf("model %s not loaded - actually loaded: %v", model, loaded)
	return 0
}

func getTimeouts(t *testing.T) (soft time.Duration, hard time.Duration) {
	deadline, hasDeadline := t.Deadline()
	if !hasDeadline {
		return 8 * time.Minute, 10 * time.Minute
	} else if deadline.Compare(time.Now().Add(2*time.Minute)) <= 0 {
		t.Skip("too little time")
		return time.Duration(0), time.Duration(0)
	}
	return -time.Since(deadline.Add(-2 * time.Minute)), -time.Since(deadline.Add(-20 * time.Second))
}
