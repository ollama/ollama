//go:build integration

package integration

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"math/rand"
	"net"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/app/lifecycle"
	"github.com/ollama/ollama/format"
	"github.com/stretchr/testify/require"
)

var (
	smol = "llama3.2:1b"
)

var (
	started = time.Now()

	// Note: add newer models at the top of the list to test them first
	ollamaEngineChatModels = []string{
		"gpt-oss:20b",
		"gemma3n:e2b",
		"mistral-small3.2:latest",
		"deepseek-r1:1.5b",
		"llama3.2-vision:latest",
		"qwen2.5-coder:latest",
		"qwen2.5vl:3b",
		"qwen3:0.6b", // dense
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
		"all-minilm",
		"bge-large",
		"bge-m3",
		"granite-embedding",
		"mxbai-embed-large",
		"nomic-embed-text",
		"paraphrase-multilingual",
		"snowflake-arctic-embed",
		"snowflake-arctic-embed2",
	}
)

func init() {
	lifecycle.InitLogging()
	custom := os.Getenv("OLLAMA_TEST_SMOL_MODEL")
	if custom != "" {
		slog.Info("setting smol test model to " + custom)
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

	if os.Getenv("OLLAMA_TEST_EXISTING") == "" && port == defaultPort {
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

var serverMutex sync.Mutex
var serverReady bool

func startServer(t *testing.T, ctx context.Context, ollamaHost string) error {
	// Make sure the server has been built
	CLIName, err := filepath.Abs("../ollama")
	if err != nil {
		return err
	}

	if runtime.GOOS == "windows" {
		CLIName += ".exe"
	}
	_, err = os.Stat(CLIName)
	if err != nil {
		return fmt.Errorf("CLI missing, did you forget to build first?  %w", err)
	}
	serverMutex.Lock()
	defer serverMutex.Unlock()
	if serverReady {
		return nil
	}

	if tmp := os.Getenv("OLLAMA_HOST"); tmp != ollamaHost {
		slog.Info("setting env", "OLLAMA_HOST", ollamaHost)
		t.Setenv("OLLAMA_HOST", ollamaHost)
	}

	slog.Info("starting server", "url", ollamaHost)
	done, err := lifecycle.SpawnServer(ctx, "../ollama")
	if err != nil {
		return fmt.Errorf("failed to start server: %w", err)
	}

	go func() {
		<-ctx.Done()
		serverMutex.Lock()
		defer serverMutex.Unlock()
		exitCode := <-done
		if exitCode > 0 {
			slog.Warn("server failure", "exit", exitCode)
		}
		serverReady = false
	}()

	// TODO wait only long enough for the server to be responsive...
	time.Sleep(500 * time.Millisecond)

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
	if os.Getenv("OLLAMA_TEST_EXISTING") == "" {
		serverProcMutex.Lock()
		fp, err := os.CreateTemp("", "ollama-server-*.log")
		if err != nil {
			t.Fatalf("failed to generate log file: %s", err)
		}
		lifecycle.ServerLogFile = fp.Name()
		fp.Close()
		require.NoError(t, startServer(t, ctx, testEndpoint))
	}

	return client, testEndpoint, func() {
		if os.Getenv("OLLAMA_TEST_EXISTING") == "" {
			defer serverProcMutex.Unlock()
			if t.Failed() {
				fp, err := os.Open(lifecycle.ServerLogFile)
				if err != nil {
					slog.Error("failed to open server log", "logfile", lifecycle.ServerLogFile, "error", err)
					return
				}
				defer fp.Close()
				data, err := io.ReadAll(fp)
				if err != nil {
					slog.Error("failed to read server log", "logfile", lifecycle.ServerLogFile, "error", err)
					return
				}
				slog.Warn("SERVER LOG FOLLOWS")
				os.Stderr.Write(data)
				slog.Warn("END OF SERVER")
			}
			err := os.Remove(lifecycle.ServerLogFile)
			if err != nil && !os.IsNotExist(err) {
				slog.Warn("failed to cleanup", "logfile", lifecycle.ServerLogFile, "error", err)
			}
		}
	}
}

func GenerateTestHelper(ctx context.Context, t *testing.T, genReq api.GenerateRequest, anyResp []string) {
	client, _, cleanup := InitServerConnection(ctx, t)
	defer cleanup()
	require.NoError(t, PullIfMissing(ctx, client, genReq.Model))
	DoGenerate(ctx, t, client, genReq, anyResp, 30*time.Second, 10*time.Second)
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
		require.NoError(t, genErr, "failed with %s request prompt %s ", genReq.Model, genReq.Prompt)
		// Verify the response contains the expected data
		response := buf.String()
		atLeastOne := false
		for _, resp := range anyResp {
			if strings.Contains(strings.ToLower(response), resp) {
				atLeastOne = true
				break
			}
		}
		require.True(t, atLeastOne, "%s: none of %v found in %s", genReq.Model, anyResp, response)
		slog.Info("test pass", "model", genReq.Model, "prompt", genReq.Prompt, "contains", anyResp, "response", response)
	case <-ctx.Done():
		t.Error("outer test context done while waiting for generate")
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
				Prompt:    "what is the origin of the US thanksgiving holiday? Be brief but factual in your reply",
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
			{"sunlight", "scattering", "interact", "color", "surface", "depth", "red", "orange", "yellow", "absorbs", "wavelength"},
			{"soil", "organic", "earth", "black", "tan", "chemical", "processes", "pigments", "particles", "iron oxide", "rust", "air", "water", "mixture", "mixing"},
			{"england", "english", "massachusetts", "pilgrims", "colonists", "independence", "british", "feast", "family", "gatherings", "traditions", "turkey", "colonial", "period", "harvest", "agricultural", "european settlers", "american revolution", "civil war", "16th century", "17th century", "native american", "united states"},
			{"fourth", "july", "declaration", "independence"},
			{"nitrogen", "oxygen", "carbon", "dioxide"},
		}
}

func skipUnderMinVRAM(t *testing.T, gb uint64) {
	// TODO use info API in the future
	if s := os.Getenv("OLLAMA_MAX_VRAM"); s != "" {
		maxVram, err := strconv.ParseUint(s, 10, 64)
		require.NoError(t, err)
		// Don't hammer on small VRAM cards...
		if maxVram < gb*format.GibiByte {
			t.Skip("skipping with small VRAM to avoid timeouts")
		}
	}
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
