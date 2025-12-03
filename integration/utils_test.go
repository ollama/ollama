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
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/format"
)

var (
	smol   = "llama3.2:1b"
	stream = false
)

var (
	started = time.Now()

	// Note: add newer models at the top of the list to test them first
	ollamaEngineChatModels = []string{
		"ministral-3",
		"qwen3-coder:30b",
		"gpt-oss:20b",
		"gemma3n:e2b",
		"mistral-small3.2:latest",
		"deepseek-r1:1.5b",
		"llama3.2-vision:latest",
		"qwen2.5-coder:latest",
		"qwen2.5vl:3b",
		"qwen3:0.6b", // dense
		"qwen3:1.7b", // dense
		"qwen3:30b",  // MOE
		"gemma3:1b",
		"llama3.1:latest",
		"llama3.2:latest",
		"gemma2:latest",
		"minicpm-v:latest",    // arch=qwen2
		"granite-code:latest", // arch=llama
	}
	llamaRunnerChatModels = []string{
		"mistral:latest",
		"falcon3:latest",
		"granite3-moe:latest",
		"command-r:latest",
		"nemotron-mini:latest",
		"phi3.5:latest",
		"solar-pro:latest",
		"internlm2:latest",
		"codellama:latest", // arch=llama
		"phi3:latest",
		"falcon2:latest",
		"gemma:latest",
		"llama2:latest",
		"nous-hermes:latest",
		"orca-mini:latest",
		"qwen:latest",
		"stablelm2:latest", // Predictions are off, crashes on small VRAM GPUs
		"falcon:latest",
	}

	// Some library models are quite large - ensure large VRAM and sufficient disk space
	// before running scenarios based on this set
	libraryChatModels = []string{
		"alfred",
		"athene-v2",
		"aya-expanse",
		"aya",
		"bakllava",
		"bespoke-minicheck",
		"codebooga",
		"codegeex4",
		"codegemma",
		"codellama",
		"codeqwen",
		"codestral",
		"codeup",
		"cogito",
		"command-a",
		"command-r-plus",
		"command-r",
		"command-r7b-arabic",
		"command-r7b",
		"dbrx",
		"deepcoder",
		"deepscaler",
		"deepseek-coder-v2",
		"deepseek-coder",
		"deepseek-llm",
		"deepseek-r1",
		// "deepseek-v2.5", // requires 155 GB VRAM
		"deepseek-v2",
		// "deepseek-v3", // requires 482 GB VRAM
		"devstral",
		"dolphin-llama3",
		"dolphin-mistral",
		"dolphin-mixtral",
		"dolphin-phi",
		"dolphin3",
		"dolphincoder",
		"duckdb-nsql",
		"everythinglm",
		"exaone-deep",
		"exaone3.5",
		"falcon",
		"falcon2",
		"falcon3",
		"firefunction-v2",
		"gemma",
		"gemma2",
		"gemma3",
		"gemma3n",
		"glm4",
		"goliath",
		"gpt-oss:20b",
		"granite-code",
		"granite3-dense",
		"granite3-guardian",
		"granite3-moe",
		"granite3.1-dense",
		"granite3.1-moe",
		"granite3.2-vision",
		"granite3.2",
		"granite3.3",
		"hermes3",
		"internlm2",
		"llama-guard3",
		"llama-pro",
		"llama2-chinese",
		"llama2-uncensored",
		"llama2",
		"llama3-chatqa",
		"llama3-gradient",
		"llama3-groq-tool-use",
		"llama3.1",
		"llama3.2-vision",
		"llama3.2",
		"llama3.3",
		"llama3",
		"llama4",
		"llava-llama3",
		"llava-phi3",
		"llava",
		"magicoder",
		"magistral",
		"marco-o1",
		"mathstral",
		"meditron",
		"medllama2",
		"megadolphin",
		"minicpm-v",
		"ministral-3",
		"mistral-large",
		"mistral-nemo",
		"mistral-openorca",
		"mistral-small",
		"mistral-small3.1",
		"mistral-small3.2",
		"mistral",
		"mistrallite",
		"mixtral",
		"moondream",
		"nemotron-mini",
		"nemotron",
		"neural-chat",
		"nexusraven",
		"notus",
		"nous-hermes",
		"nous-hermes2-mixtral",
		"nous-hermes2",
		"nuextract",
		"olmo2",
		"open-orca-platypus2",
		"openchat",
		"opencoder",
		"openhermes",
		"openthinker",
		"orca-mini",
		"orca2",
		// "phi", // unreliable
		"phi3.5",
		"phi3",
		"phi4-mini-reasoning",
		"phi4-mini",
		"phi4-reasoning",
		"phi4",
		"phind-codellama",
		"qwen",
		"qwen2-math",
		"qwen2.5-coder",
		"qwen2.5",
		"qwen2.5vl",
		"qwen2",
		"qwen3:0.6b", // dense
		"qwen3:30b",  // MOE
		"qwq",
		"r1-1776",
		"reader-lm",
		"reflection",
		"sailor2",
		"samantha-mistral",
		"shieldgemma",
		"smallthinker",
		"smollm",
		"smollm2",
		"solar-pro",
		"solar",
		"sqlcoder",
		"stable-beluga",
		"stable-code",
		"stablelm-zephyr",
		"stablelm2",
		"starcoder",
		"starcoder2",
		"starling-lm",
		"tinydolphin",
		"tinyllama",
		"tulu3",
		"vicuna",
		"wizard-math",
		"wizard-vicuna-uncensored",
		"wizard-vicuna",
		"wizardcoder",
		"wizardlm-uncensored",
		"wizardlm2",
		"xwinlm",
		"yarn-llama2",
		"yarn-mistral",
		"yi-coder",
		"yi",
		"zephyr",
	}
	libraryEmbedModels = []string{
		"qwen3-embedding",
		"embeddinggemma",
		"nomic-embed-text",
		"all-minilm",
		"bge-large",
		"bge-m3",
		"granite-embedding",
		"mxbai-embed-large",
		"paraphrase-multilingual",
		"snowflake-arctic-embed",
		"snowflake-arctic-embed2",
	}
	libraryToolsModels = []string{
		"qwen3-vl",
		"gpt-oss:20b",
		"gpt-oss:120b",
		"qwen3",
		"llama3.1",
		"llama3.2",
		"mistral",
		"qwen2.5",
		"qwen2",
		"ministral-3",
		"mistral-nemo",
		"mistral-small",
		"mixtral:8x22b",
		"qwq",
		"granite3.3",
	}

	blueSkyPrompt   = "why is the sky blue? Be brief but factual in your reply"
	blueSkyExpected = []string{"rayleigh", "scatter", "atmosphere", "nitrogen", "oxygen", "wavelength", "interact"}

	rainbowPrompt    = "how do rainbows form? Be brief but factual in your reply"
	rainbowFollowups = []string{
		"Explain the physics involved in them.  Be breif in your reply",
		"Explain the chemistry involved in them.  Be breif in your reply",
		"What are common myths related to them? Be brief in your reply",
		"Can they form if there is no rain?  Be breif in your reply",
		"Can they form if there are no clouds?  Be breif in your reply",
		"Do they happen on other planets? Be brief in your reply",
	}
	rainbowExpected = []string{"water", "droplet", "mist", "glow", "refract", "reflect", "scatter", "particles", "wave", "color", "spectrum", "raindrop", "atmosphere", "frequency", "shower", "sky", "shimmer", "light", "storm", "sunny", "sunburst", "phenomenon", "mars", "venus", "jupiter"}
)

func init() {
	logger := slog.New(slog.NewTextHandler(os.Stdout, &slog.HandlerOptions{Level: slog.LevelDebug}))
	slog.SetDefault(logger)
	custom := os.Getenv("OLLAMA_TEST_DEFAULT_MODEL")
	if custom != "" {
		slog.Info("setting default test model to " + custom)
		smol = custom
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

	stallDuration := 60 * time.Second // This includes checksum verification, which can take a while on larger models, and slower systems
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

var serverProcMutex sync.Mutex

// Returns an Client, the testEndpoint, and a cleanup function, fails the test on errors
// Starts the server if needed
func InitServerConnection(ctx context.Context, t *testing.T) (*api.Client, string, func()) {
	client, testEndpoint := GetTestEndpoint()
	cleanup := func() {}
	if os.Getenv("OLLAMA_TEST_EXISTING") == "" && runtime.GOOS != "windows" {
		var err error
		err = startServer(t, ctx, testEndpoint)
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

			if t.Failed() {
				slog.Warn("SERVER LOG FOLLOWS")
				io.Copy(os.Stderr, &serverLog)
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
			break
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
	if err := PullIfMissing(ctx, client, req.Model); err != nil {
		t.Fatal(err)
	}
	DoChat(ctx, t, client, req, anyResp, 30*time.Second, 10*time.Second)
}

func DoGenerate(ctx context.Context, t *testing.T, client *api.Client, genReq api.GenerateRequest, anyResp []string, initialTimeout, streamTimeout time.Duration) []int {
	stallTimer := time.NewTimer(initialTimeout)
	var buf bytes.Buffer
	var context []int
	fn := func(response api.GenerateResponse) error {
		// fmt.Print(".")
		buf.Write([]byte(response.Response))
		if !stallTimer.Reset(streamTimeout) {
			return errors.New("stall was detected while streaming response, aborting")
		}
		if len(response.Context) > 0 {
			context = response.Context
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

	var response string
	verify := func() {
		// Verify the response contains the expected data
		response = buf.String()
		atLeastOne := false
		for _, resp := range anyResp {
			if strings.Contains(strings.ToLower(response), resp) {
				atLeastOne = true
				break
			}
		}
		if !atLeastOne {
			t.Fatalf("%s: none of %v found in %s", genReq.Model, anyResp, response)
		}
	}

	select {
	case <-stallTimer.C:
		if buf.Len() == 0 {
			t.Errorf("generate never started.  Timed out after :%s", initialTimeout.String())
		} else {
			t.Errorf("generate stalled.  Response so far:%s", buf.String())
		}
	case <-done:
		if genErr != nil && strings.Contains(genErr.Error(), "model requires more system memory") {
			slog.Warn("model is too large for the target test system", "model", genReq.Model, "error", genErr)
			return context
		}
		if genErr != nil {
			t.Fatalf("%s failed with %s request prompt %s", genErr, genReq.Model, genReq.Prompt)
		}
		verify()
		slog.Info("test pass", "model", genReq.Model, "prompt", genReq.Prompt, "contains", anyResp, "response", response)
	case <-ctx.Done():
		// On slow systems, we might timeout before some models finish rambling, so check what we have so far to see
		// if it's considered a pass - the stallTimer will detect hangs, but we want to consider slow systems a pass
		// if they are still generating valid responses
		slog.Warn("outer test context done while waiting for generate")
		verify()
	}
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

func DoChat(ctx context.Context, t *testing.T, client *api.Client, req api.ChatRequest, anyResp []string, initialTimeout, streamTimeout time.Duration) *api.Message {
	stallTimer := time.NewTimer(initialTimeout)
	var buf bytes.Buffer
	role := "assistant"
	fn := func(response api.ChatResponse) error {
		// fmt.Print(".")
		role = response.Message.Role
		buf.Write([]byte(response.Message.Content))
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

	var response string
	verify := func() {
		// Verify the response contains the expected data
		response = buf.String()
		atLeastOne := false
		for _, resp := range anyResp {
			if strings.Contains(strings.ToLower(response), resp) {
				atLeastOne = true
				break
			}
		}
		if !atLeastOne {
			t.Fatalf("%s: none of %v found in \"%s\" -- request was:%v", req.Model, anyResp, response, req.Messages)
		}
	}

	select {
	case <-stallTimer.C:
		if buf.Len() == 0 {
			t.Errorf("generate never started.  Timed out after :%s", initialTimeout.String())
		} else {
			t.Errorf("generate stalled.  Response so far:%s", buf.String())
		}
	case <-done:
		if genErr != nil && strings.Contains(genErr.Error(), "model requires more system memory") {
			slog.Warn("model is too large for the target test system", "model", req.Model, "error", genErr)
			return nil
		}
		if genErr != nil {
			t.Fatalf("%s failed with %s request prompt %v", genErr, req.Model, req.Messages)
		}
		verify()
		slog.Info("test pass", "model", req.Model, "messages", req.Messages, "contains", anyResp, "response", response)
	case <-ctx.Done():
		// On slow systems, we might timeout before some models finish rambling, so check what we have so far to see
		// if it's considered a pass - the stallTimer will detect hangs, but we want to consider slow systems a pass
		// if they are still generating valid responses
		slog.Warn("outer test context done while waiting for chat")
		verify()
	}
	return &api.Message{Role: role, Content: buf.String()}
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

func skipUnderMinVRAM(t *testing.T, gb uint64) {
	// TODO use info API in the future
	if s := os.Getenv("OLLAMA_MAX_VRAM"); s != "" {
		maxVram, err := strconv.ParseUint(s, 10, 64)
		if err != nil {
			t.Fatal(err)
		}
		// Don't hammer on small VRAM cards...
		if maxVram < gb*format.GibiByte {
			t.Skip("skipping with small VRAM to avoid timeouts")
		}
	}
}

// Skip if the target model isn't X% GPU loaded to avoid excessive runtime
func skipIfNotGPULoaded(ctx context.Context, t *testing.T, client *api.Client, model string, minPercent int) {
	gpuPercent := getGPUPercent(ctx, t, client, model)
	if gpuPercent < minPercent {
		t.Skip(fmt.Sprintf("test requires minimum %d%% GPU load, but model %s only has %d%%", minPercent, model, gpuPercent))
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
		if strings.Contains(model, ":") {
			if m.Name != model {
				continue
			}
		} else if strings.Contains(m.Name, ":") {
			if !strings.HasPrefix(m.Name, model+":") {
				continue
			}
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
