package cmd

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	"github.com/spf13/cobra"

	"github.com/ollama/ollama/api"
	appstore "github.com/ollama/ollama/app/store"
	"github.com/ollama/ollama/cmd/config"
	"github.com/ollama/ollama/types/model"
)

func TestShowInfo(t *testing.T) {
	t.Run("bare details", func(t *testing.T) {
		var b bytes.Buffer
		if err := showInfo(&api.ShowResponse{
			Details: api.ModelDetails{
				Family:            "test",
				ParameterSize:     "7B",
				QuantizationLevel: "FP16",
			},
		}, false, &b); err != nil {
			t.Fatal(err)
		}

		expect := `  Model
    architecture    test    
    parameters      7B      
    quantization    FP16    

`

		if diff := cmp.Diff(expect, b.String()); diff != "" {
			t.Errorf("unexpected output (-want +got):\n%s", diff)
		}
	})

	t.Run("bare model info", func(t *testing.T) {
		var b bytes.Buffer
		if err := showInfo(&api.ShowResponse{
			ModelInfo: map[string]any{
				"general.architecture":    "test",
				"general.parameter_count": float64(7_000_000_000),
				"test.context_length":     float64(0),
				"test.embedding_length":   float64(0),
			},
			Details: api.ModelDetails{
				Family:            "test",
				ParameterSize:     "7B",
				QuantizationLevel: "FP16",
			},
		}, false, &b); err != nil {
			t.Fatal(err)
		}

		expect := `  Model
    architecture        test    
    parameters          7B      
    context length      0       
    embedding length    0       
    quantization        FP16    

`
		if diff := cmp.Diff(expect, b.String()); diff != "" {
			t.Errorf("unexpected output (-want +got):\n%s", diff)
		}
	})

	t.Run("verbose model", func(t *testing.T) {
		var b bytes.Buffer
		if err := showInfo(&api.ShowResponse{
			Details: api.ModelDetails{
				Family:            "test",
				ParameterSize:     "8B",
				QuantizationLevel: "FP16",
			},
			Parameters: `
			stop up`,
			ModelInfo: map[string]any{
				"general.architecture":    "test",
				"general.parameter_count": float64(8_000_000_000),
				"some.true_bool":          true,
				"some.false_bool":         false,
				"test.context_length":     float64(1000),
				"test.embedding_length":   float64(11434),
			},
			Tensors: []api.Tensor{
				{Name: "blk.0.attn_k.weight", Type: "BF16", Shape: []uint64{42, 3117}},
				{Name: "blk.0.attn_q.weight", Type: "FP16", Shape: []uint64{3117, 42}},
			},
		}, true, &b); err != nil {
			t.Fatal(err)
		}

		expect := `  Model
    architecture        test     
    parameters          8B       
    context length      1000     
    embedding length    11434    
    quantization        FP16     

  Parameters
    stop    up    

  Metadata
    general.architecture       test     
    general.parameter_count    8e+09    
    some.false_bool            false    
    some.true_bool             true     
    test.context_length        1000     
    test.embedding_length      11434    

  Tensors
    blk.0.attn_k.weight    BF16    [42 3117]    
    blk.0.attn_q.weight    FP16    [3117 42]    

`
		if diff := cmp.Diff(expect, b.String()); diff != "" {
			t.Errorf("unexpected output (-want +got):\n%s", diff)
		}
	})

	t.Run("parameters", func(t *testing.T) {
		var b bytes.Buffer
		if err := showInfo(&api.ShowResponse{
			Details: api.ModelDetails{
				Family:            "test",
				ParameterSize:     "7B",
				QuantizationLevel: "FP16",
			},
			Parameters: `
			stop never
			stop gonna
			stop give
			stop you
			stop up
			temperature 99`,
		}, false, &b); err != nil {
			t.Fatal(err)
		}

		expect := `  Model
    architecture    test    
    parameters      7B      
    quantization    FP16    

  Parameters
    stop           never    
    stop           gonna    
    stop           give     
    stop           you      
    stop           up       
    temperature    99       

`
		if diff := cmp.Diff(expect, b.String()); diff != "" {
			t.Errorf("unexpected output (-want +got):\n%s", diff)
		}
	})

	t.Run("project info", func(t *testing.T) {
		var b bytes.Buffer
		if err := showInfo(&api.ShowResponse{
			Details: api.ModelDetails{
				Family:            "test",
				ParameterSize:     "7B",
				QuantizationLevel: "FP16",
			},
			ProjectorInfo: map[string]any{
				"general.architecture":         "clip",
				"general.parameter_count":      float64(133_700_000),
				"clip.vision.embedding_length": float64(0),
				"clip.vision.projection_dim":   float64(0),
			},
		}, false, &b); err != nil {
			t.Fatal(err)
		}

		expect := `  Model
    architecture    test    
    parameters      7B      
    quantization    FP16    

  Projector
    architecture        clip       
    parameters          133.70M    
    embedding length    0          
    dimensions          0          

`
		if diff := cmp.Diff(expect, b.String()); diff != "" {
			t.Errorf("unexpected output (-want +got):\n%s", diff)
		}
	})

	t.Run("system", func(t *testing.T) {
		var b bytes.Buffer
		if err := showInfo(&api.ShowResponse{
			Details: api.ModelDetails{
				Family:            "test",
				ParameterSize:     "7B",
				QuantizationLevel: "FP16",
			},
			System: `You are a pirate!
Ahoy, matey!
Weigh anchor!
			`,
		}, false, &b); err != nil {
			t.Fatal(err)
		}

		expect := `  Model
    architecture    test    
    parameters      7B      
    quantization    FP16    

  System
    You are a pirate!    
    Ahoy, matey!         
    ...                  

`
		if diff := cmp.Diff(expect, b.String()); diff != "" {
			t.Errorf("unexpected output (-want +got):\n%s", diff)
		}
	})

	t.Run("license", func(t *testing.T) {
		var b bytes.Buffer
		license := "MIT License\nCopyright (c) Ollama\n"
		if err := showInfo(&api.ShowResponse{
			Details: api.ModelDetails{
				Family:            "test",
				ParameterSize:     "7B",
				QuantizationLevel: "FP16",
			},
			License: license,
		}, false, &b); err != nil {
			t.Fatal(err)
		}

		expect := `  Model
    architecture    test    
    parameters      7B      
    quantization    FP16    

  License
    MIT License             
    Copyright (c) Ollama    

`
		if diff := cmp.Diff(expect, b.String()); diff != "" {
			t.Errorf("unexpected output (-want +got):\n%s", diff)
		}
	})

	t.Run("capabilities", func(t *testing.T) {
		var b bytes.Buffer
		if err := showInfo(&api.ShowResponse{
			Details: api.ModelDetails{
				Family:            "test",
				ParameterSize:     "7B",
				QuantizationLevel: "FP16",
			},
			Capabilities: []model.Capability{model.CapabilityVision, model.CapabilityTools},
		}, false, &b); err != nil {
			t.Fatal(err)
		}

		expect := "  Model\n" +
			"    architecture    test    \n" +
			"    parameters      7B      \n" +
			"    quantization    FP16    \n" +
			"\n" +
			"  Capabilities\n" +
			"    vision    \n" +
			"    tools     \n" +
			"\n"

		if diff := cmp.Diff(expect, b.String()); diff != "" {
			t.Errorf("unexpected output (-want +got):\n%s", diff)
		}
	})

	t.Run("min version", func(t *testing.T) {
		var b bytes.Buffer
		if err := showInfo(&api.ShowResponse{
			Details: api.ModelDetails{
				Family:            "test",
				ParameterSize:     "7B",
				QuantizationLevel: "FP16",
			},
			Requires: "0.19.0",
		}, false, &b); err != nil {
			t.Fatal(err)
		}

		expect := `  Model
    architecture    test      
    parameters      7B        
    quantization    FP16      
    requires        0.19.0

`
		trimLinePadding := func(s string) string {
			lines := strings.Split(s, "\n")
			for i, line := range lines {
				lines[i] = strings.TrimRight(line, " \t\r")
			}
			return strings.Join(lines, "\n")
		}
		if diff := cmp.Diff(trimLinePadding(expect), trimLinePadding(b.String())); diff != "" {
			t.Errorf("unexpected output (-want +got):\n%s", diff)
		}
	})
}

func TestContextWindowTokensForRunUsesRunningModel(t *testing.T) {
	mockServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/api/ps" || r.Method != http.MethodGet {
			t.Fatalf("unexpected request to %s %s", r.URL.Path, r.Method)
		}
		if err := json.NewEncoder(w).Encode(api.ProcessResponse{
			Models: []api.ProcessModelResponse{
				{Name: "llama3.2:latest", Model: "llama3.2:latest", ContextLength: 8192},
				{Name: "qwen3:8b", Model: "qwen3:8b", ContextLength: 16384},
			},
		}); err != nil {
			t.Fatal(err)
		}
	}))
	defer mockServer.Close()

	t.Setenv("OLLAMA_HOST", mockServer.URL)
	client, err := api.ClientFromEnvironment()
	if err != nil {
		t.Fatal(err)
	}

	if got := contextWindowTokensForRun(t.Context(), client, "qwen3:8b", 4096); got != 16384 {
		t.Fatalf("contextWindowTokensForRun exact = %d, want 16384", got)
	}
	if got := contextWindowTokensForRun(t.Context(), client, "llama3.2", 4096); got != 8192 {
		t.Fatalf("contextWindowTokensForRun default tag = %d, want 8192", got)
	}
	if got := contextWindowTokensForRun(t.Context(), client, "llama3.2:8b", 4096); got != 4096 {
		t.Fatalf("contextWindowTokensForRun different tag = %d, want fallback", got)
	}
}

func TestContextWindowTokensForRunUsesCloudShowFallback(t *testing.T) {
	mockServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch {
		case r.URL.Path == "/api/ps" && r.Method == http.MethodGet:
			if err := json.NewEncoder(w).Encode(api.ProcessResponse{}); err != nil {
				t.Fatal(err)
			}
		case r.URL.Path == "/api/show" && r.Method == http.MethodPost:
			if err := json.NewEncoder(w).Encode(api.ShowResponse{
				Details: api.ModelDetails{ContextLength: 262144},
			}); err != nil {
				t.Fatal(err)
			}
		default:
			t.Fatalf("unexpected request to %s %s", r.URL.Path, r.Method)
		}
	}))
	defer mockServer.Close()

	t.Setenv("OLLAMA_HOST", mockServer.URL)
	client, err := api.ClientFromEnvironment()
	if err != nil {
		t.Fatal(err)
	}

	if got := contextWindowTokensForRun(t.Context(), client, "qwen3.5:cloud", 4096); got != 262144 {
		t.Fatalf("contextWindowTokensForRun cloud show = %d, want 262144", got)
	}
}

func TestContextWindowTokensFromShowResponse(t *testing.T) {
	if got := contextWindowTokensFromShowResponse(&api.ShowResponse{
		Details: api.ModelDetails{ContextLength: 262144},
	}); got != 262144 {
		t.Fatalf("contextWindowTokensFromShowResponse details = %d, want 262144", got)
	}

	if got := contextWindowTokensFromShowResponse(&api.ShowResponse{
		ModelInfo: map[string]any{
			"general.architecture": "test",
			"test.context_length":  float64(131072),
		},
	}); got != 131072 {
		t.Fatalf("contextWindowTokensFromShowResponse model_info = %d, want 131072", got)
	}
}

func TestRunCommandArgsAllowsResumeWithoutModel(t *testing.T) {
	cmd := &cobra.Command{}
	cmd.Flags().Bool("resume", true, "")

	if err := runCommandArgs(cmd, nil); err != nil {
		t.Fatalf("runCommandArgs resume = %v, want nil", err)
	}

	cmd = &cobra.Command{}
	cmd.Flags().Bool("resume", false, "")
	if err := runCommandArgs(cmd, nil); err == nil {
		t.Fatal("runCommandArgs without model = nil, want error")
	}
}

func TestRunCommandFlags(t *testing.T) {
	root := NewCLI()
	runCmd, _, err := root.Find([]string{"run"})
	if err != nil {
		t.Fatal(err)
	}
	if runCmd == nil || runCmd.Name() != "run" {
		t.Fatalf("run command = %v, want run", runCmd)
	}

	if runCmd.Flags().Lookup("headless") != nil {
		t.Fatal("run command should not expose --headless")
	}
	if runCmd.Flags().Lookup("skill") != nil {
		t.Fatal("run command should not expose --skill")
	}
	if runCmd.Flags().Lookup("yolo") == nil {
		t.Fatal("run command should expose --yolo")
	}
	if runCmd.Flags().Lookup("auto-approve-tools") == nil {
		t.Fatal("run command should expose --auto-approve-tools")
	}
	if root.Flags().Lookup("model") == nil {
		t.Fatal("root command should expose --model")
	}
	if root.Flags().Lookup("format") != nil {
		t.Fatal("root command should not expose --format")
	}
	if root.Flags().Lookup("verbose") != nil {
		t.Fatal("root command should not expose --verbose")
	}
	for _, name := range []string{"think", "auto-approve-tools", "yolo", "keepalive"} {
		if root.Flags().Lookup(name) == nil {
			t.Fatalf("root command should expose --%s", name)
		}
	}
	for _, name := range []string{"format", "verbose"} {
		if runCmd.Flags().Lookup(name) == nil {
			t.Fatalf("run command should expose --%s", name)
		}
	}
}

func TestPrepareRootResumeRunCommand(t *testing.T) {
	rootCmd := &cobra.Command{}
	rootCmd.SetContext(t.Context())
	registerRootRunFlags(rootCmd)
	if err := rootCmd.Flags().Set("think", "high"); err != nil {
		t.Fatal(err)
	}
	if err := rootCmd.Flags().Set("yolo", "true"); err != nil {
		t.Fatal(err)
	}

	runCmd := &cobra.Command{}
	registerRunFlags(runCmd, true)

	if err := prepareRootResumeRunCommand(rootCmd, runCmd); err != nil {
		t.Fatal(err)
	}

	resume, err := runCmd.Flags().GetBool("resume")
	if err != nil {
		t.Fatal(err)
	}
	if !resume {
		t.Fatal("run resume flag = false, want true")
	}

	verbose, err := runCmd.Flags().GetBool("verbose")
	if err != nil {
		t.Fatal(err)
	}
	if verbose {
		t.Fatal("run verbose flag = true, want unchanged false")
	}

	format, err := runCmd.Flags().GetString("format")
	if err != nil {
		t.Fatal(err)
	}
	if format != "" {
		t.Fatalf("run format flag = %q, want unchanged empty", format)
	}
	think, err := runCmd.Flags().GetString("think")
	if err != nil {
		t.Fatal(err)
	}
	if think != "high" {
		t.Fatalf("run think flag = %q, want high", think)
	}
	autoApprove, err := autoApproveToolsFromFlags(runCmd)
	if err != nil {
		t.Fatal(err)
	}
	if !autoApprove {
		t.Fatal("run yolo flag did not enable auto approval")
	}
}

func TestApplyRunFlagsToOptions(t *testing.T) {
	cmd := &cobra.Command{}
	registerRunFlags(cmd, true)
	for name, value := range map[string]string{
		"format":             "json",
		"think":              "high",
		"keepalive":          "5m",
		"verbose":            "true",
		"hidethinking":       "true",
		"auto-approve-tools": "true",
	} {
		if err := cmd.Flags().Set(name, value); err != nil {
			t.Fatal(err)
		}
	}

	var opts runOptions
	thinkExplicit, err := applyRunFlagsToOptions(cmd, &opts)
	if err != nil {
		t.Fatal(err)
	}
	if !thinkExplicit {
		t.Fatal("thinkExplicit = false, want true")
	}
	if opts.Format != "json" {
		t.Fatalf("format = %q, want json", opts.Format)
	}
	if opts.Think == nil || opts.Think.Value != "high" {
		t.Fatalf("think = %#v, want high", opts.Think)
	}
	if !opts.AutoApproveTools || !opts.Verbose || !opts.HideThinking {
		t.Fatalf("flags not applied: %#v", opts)
	}
	if opts.KeepAlive == nil || opts.KeepAlive.Duration != 5*time.Minute {
		t.Fatalf("keepalive = %#v, want 5m", opts.KeepAlive)
	}
}

func TestRootModelRunsRunHandler(t *testing.T) {
	oldRunHandler := runHandler
	t.Cleanup(func() {
		runHandler = oldRunHandler
	})

	var gotArgs []string
	var gotVerbose bool
	var gotFormat string
	var gotThink string
	var gotAutoApprove bool
	runHandler = func(cmd *cobra.Command, args []string) error {
		gotArgs = append([]string(nil), args...)
		var err error
		gotVerbose, err = cmd.Flags().GetBool("verbose")
		if err != nil {
			return err
		}
		gotFormat, err = cmd.Flags().GetString("format")
		if err != nil {
			return err
		}
		gotThink, err = cmd.Flags().GetString("think")
		if err != nil {
			return err
		}
		gotAutoApprove, err = autoApproveToolsFromFlags(cmd)
		return err
	}

	root := NewCLI()
	root.SetContext(t.Context())
	runCmd, _, err := root.Find([]string{"run"})
	if err != nil {
		t.Fatal(err)
	}
	if runCmd == nil {
		t.Fatal("run command not found")
	}
	runCmd.PreRunE = nil

	root.SetArgs([]string{"--model", "llama3.2", "--think=high", "--yolo"})
	if err := root.Execute(); err != nil {
		t.Fatal(err)
	}

	if !reflect.DeepEqual(gotArgs, []string{"llama3.2"}) {
		t.Fatalf("run handler args = %#v, want %#v", gotArgs, []string{"llama3.2"})
	}
	if gotVerbose {
		t.Fatal("run handler verbose = true, want false")
	}
	if gotFormat != "" {
		t.Fatalf("run handler format = %q, want empty", gotFormat)
	}
	if gotThink != "high" {
		t.Fatalf("run handler think = %q, want high", gotThink)
	}
	if !gotAutoApprove {
		t.Fatal("run handler auto approve = false, want true")
	}
}

func TestRootDefaultRunsAgentModelPicker(t *testing.T) {
	oldRootAgentHandler := rootAgentHandler
	t.Cleanup(func() {
		rootAgentHandler = oldRootAgentHandler
	})

	var called bool
	rootAgentHandler = func(cmd *cobra.Command) {
		called = true
	}

	root := NewCLI()
	root.SetContext(t.Context())
	root.SetArgs(nil)
	if err := root.Execute(); err != nil {
		t.Fatal(err)
	}

	if !called {
		t.Fatal("root command should open the agent model picker by default")
	}
}

func TestAutoApproveToolsFromFlags(t *testing.T) {
	t.Run("yolo", func(t *testing.T) {
		cmd := &cobra.Command{}
		cmd.Flags().Bool("auto-approve-tools", false, "")
		cmd.Flags().Bool("yolo", false, "")
		if err := cmd.Flags().Set("yolo", "true"); err != nil {
			t.Fatal(err)
		}

		enabled, err := autoApproveToolsFromFlags(cmd)
		if err != nil {
			t.Fatal(err)
		}
		if !enabled {
			t.Fatal("autoApproveToolsFromFlags yolo = false, want true")
		}
	})

	t.Run("legacy experimental yolo", func(t *testing.T) {
		cmd := &cobra.Command{}
		cmd.Flags().Bool("experimental-yolo", false, "")
		if err := cmd.Flags().Set("experimental-yolo", "true"); err != nil {
			t.Fatal(err)
		}

		enabled, err := autoApproveToolsFromFlags(cmd)
		if err != nil {
			t.Fatal(err)
		}
		if !enabled {
			t.Fatal("autoApproveToolsFromFlags experimental-yolo = false, want true")
		}
	})
}

func TestResumeModelFromLatestChat(t *testing.T) {
	t.Setenv("HOME", t.TempDir())
	t.Setenv("LOCALAPPDATA", t.TempDir())

	store, err := appstore.New("")
	if err != nil {
		t.Fatal(err)
	}
	ctx := t.Context()
	if err := store.AppendAgentMessage(ctx, "chat-old", api.Message{Role: "assistant", Content: "old"}, "llama3.2"); err != nil {
		t.Fatal(err)
	}
	if err := store.AppendAgentMessage(ctx, "chat-new", api.Message{Role: "assistant", Content: "new"}, "qwen3:8b"); err != nil {
		t.Fatal(err)
	}
	if err := store.Close(); err != nil {
		t.Fatal(err)
	}

	got, err := resumeModelFromLatestChat(ctx)
	if err != nil {
		t.Fatal(err)
	}
	if got != "qwen3:8b" {
		t.Fatalf("resume model = %q, want qwen3:8b", got)
	}
}

func TestDeleteHandler(t *testing.T) {
	stopped := false
	mockServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/api/delete" && r.Method == http.MethodDelete {
			var req api.DeleteRequest
			if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
				http.Error(w, err.Error(), http.StatusBadRequest)
				return
			}
			if req.Name == "test-model" {
				w.WriteHeader(http.StatusOK)
			} else {
				w.WriteHeader(http.StatusNotFound)
				errPayload := `{"error":"model '%s' not found"}`
				w.Write([]byte(fmt.Sprintf(errPayload, req.Name)))
			}
			return
		}
		if r.URL.Path == "/api/generate" && r.Method == http.MethodPost {
			var req api.GenerateRequest
			if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
				http.Error(w, err.Error(), http.StatusBadRequest)
				return
			}
			if req.Model == "test-model" {
				w.WriteHeader(http.StatusOK)
				if err := json.NewEncoder(w).Encode(api.GenerateResponse{
					Done: true,
				}); err != nil {
					http.Error(w, err.Error(), http.StatusInternalServerError)
				}
				stopped = true
				return
			} else {
				w.WriteHeader(http.StatusNotFound)
				if err := json.NewEncoder(w).Encode(api.GenerateResponse{
					Done: false,
				}); err != nil {
					http.Error(w, err.Error(), http.StatusInternalServerError)
				}
			}
		}
	}))

	t.Setenv("OLLAMA_HOST", mockServer.URL)
	t.Setenv("HOME", t.TempDir())
	t.Setenv("LOCALAPPDATA", t.TempDir())
	t.Cleanup(mockServer.Close)

	cmd := &cobra.Command{}
	cmd.SetContext(t.Context())
	if err := DeleteHandler(cmd, []string{"test-model"}); err != nil {
		t.Fatalf("DeleteHandler failed: %v", err)
	}
	if !stopped {
		t.Fatal("Model was not stopped before deletion")
	}

	err := DeleteHandler(cmd, []string{"test-model-not-found"})
	if err == nil || !strings.Contains(err.Error(), "model 'test-model-not-found' not found") {
		t.Fatalf("DeleteHandler failed: expected error about stopping non-existent model, got %v", err)
	}
}

func TestRunEmbeddingModel(t *testing.T) {
	reqCh := make(chan api.EmbedRequest, 1)
	mockServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/api/show" && r.Method == http.MethodPost {
			w.Header().Set("Content-Type", "application/json")
			if err := json.NewEncoder(w).Encode(api.ShowResponse{
				Capabilities: []model.Capability{model.CapabilityEmbedding},
			}); err != nil {
				http.Error(w, err.Error(), http.StatusInternalServerError)
			}
			return
		}
		if r.URL.Path == "/api/embed" && r.Method == http.MethodPost {
			var req api.EmbedRequest
			if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
				http.Error(w, err.Error(), http.StatusBadRequest)
				return
			}
			reqCh <- req
			w.Header().Set("Content-Type", "application/json")
			if err := json.NewEncoder(w).Encode(api.EmbedResponse{
				Model:      "test-embedding-model",
				Embeddings: [][]float32{{0.1, 0.2, 0.3}},
			}); err != nil {
				http.Error(w, err.Error(), http.StatusInternalServerError)
			}
			return
		}
		http.NotFound(w, r)
	}))

	t.Setenv("OLLAMA_HOST", mockServer.URL)
	t.Setenv("HOME", t.TempDir())
	t.Setenv("LOCALAPPDATA", t.TempDir())
	t.Cleanup(mockServer.Close)

	cmd := &cobra.Command{}
	cmd.SetContext(t.Context())
	cmd.Flags().String("keepalive", "", "")
	cmd.Flags().Bool("truncate", false, "")
	cmd.Flags().Int("dimensions", 0, "")
	cmd.Flags().Bool("verbose", false, "")
	cmd.Flags().Bool("insecure", false, "")
	cmd.Flags().String("format", "", "")
	cmd.Flags().String("think", "", "")
	cmd.Flags().Bool("hidethinking", false, "")

	oldStdout := os.Stdout
	r, w, _ := os.Pipe()
	os.Stdout = w

	errCh := make(chan error, 1)
	go func() {
		errCh <- RunHandler(cmd, []string{"test-embedding-model", "hello", "world"})
	}()

	err := <-errCh
	w.Close()
	os.Stdout = oldStdout

	if err != nil {
		t.Fatalf("RunHandler returned error: %v", err)
	}

	var out bytes.Buffer
	io.Copy(&out, r)

	select {
	case req := <-reqCh:
		inputText, _ := req.Input.(string)
		if diff := cmp.Diff("hello world", inputText); diff != "" {
			t.Errorf("unexpected input (-want +got):\n%s", diff)
		}
		if req.Truncate != nil {
			t.Errorf("expected truncate to be nil, got %v", *req.Truncate)
		}
		if req.KeepAlive != nil {
			t.Errorf("expected keepalive to be nil, got %v", req.KeepAlive)
		}
		if req.Dimensions != 0 {
			t.Errorf("expected dimensions to be 0, got %d", req.Dimensions)
		}
	default:
		t.Fatal("server did not receive embed request")
	}

	expectOutput := "[0.1,0.2,0.3]\n"
	if diff := cmp.Diff(expectOutput, out.String()); diff != "" {
		t.Errorf("unexpected output (-want +got):\n%s", diff)
	}
}

func TestRunHandlerResumeUsesLatestChatInHeadlessMode(t *testing.T) {
	var chatReq api.ChatRequest
	mockServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/api/show":
			w.Header().Set("Content-Type", "application/json")
			if err := json.NewEncoder(w).Encode(api.ShowResponse{}); err != nil {
				t.Fatal(err)
			}
		case "/api/chat":
			if err := json.NewDecoder(r.Body).Decode(&chatReq); err != nil {
				http.Error(w, err.Error(), http.StatusBadRequest)
				return
			}
			w.Header().Set("Content-Type", "application/x-ndjson")
			enc := json.NewEncoder(w)
			if err := enc.Encode(api.ChatResponse{Message: api.Message{Role: "assistant", Content: "resumed"}}); err != nil {
				t.Fatal(err)
			}
			if flusher, ok := w.(http.Flusher); ok {
				flusher.Flush()
			}
			if err := enc.Encode(api.ChatResponse{Done: true, DoneReason: "stop"}); err != nil {
				t.Fatal(err)
			}
		case "/api/generate":
			if err := json.NewEncoder(w).Encode(api.GenerateResponse{Done: true}); err != nil {
				t.Fatal(err)
			}
		default:
			http.NotFound(w, r)
		}
	}))
	t.Cleanup(mockServer.Close)
	t.Setenv("OLLAMA_HOST", mockServer.URL)
	t.Setenv("HOME", t.TempDir())
	t.Setenv("LOCALAPPDATA", t.TempDir())

	store, err := appstore.New("")
	if err != nil {
		t.Fatal(err)
	}
	if err := store.AppendAgentMessage(t.Context(), "chat-1", api.Message{Role: "user", Content: "old prompt"}, "test-model"); err != nil {
		t.Fatal(err)
	}
	if err := store.Close(); err != nil {
		t.Fatal(err)
	}

	cmd := &cobra.Command{}
	cmd.SetContext(t.Context())
	cmd.Flags().String("format", "", "")
	cmd.Flags().String("think", "", "")
	cmd.Flags().Bool("hidethinking", false, "")
	cmd.Flags().Bool("resume", true, "")
	cmd.Flags().String("keepalive", "", "")
	cmd.Flags().Bool("verbose", false, "")

	oldStdin := os.Stdin
	stdinR, stdinW, _ := os.Pipe()
	os.Stdin = stdinR
	if _, err := stdinW.Write([]byte("follow up")); err != nil {
		t.Fatal(err)
	}
	stdinW.Close()

	oldStdout := os.Stdout
	stdoutR, stdoutW, _ := os.Pipe()
	os.Stdout = stdoutW

	errCh := make(chan error, 1)
	go func() {
		errCh <- RunHandler(cmd, nil)
	}()
	err = <-errCh
	stdoutW.Close()
	os.Stdout = oldStdout
	os.Stdin = oldStdin
	if err != nil {
		t.Fatalf("RunHandler returned error: %v", err)
	}

	var out bytes.Buffer
	if _, err := io.Copy(&out, stdoutR); err != nil {
		t.Fatal(err)
	}
	if out.String() != "resumed\n" {
		t.Fatalf("stdout = %q, want resumed newline", out.String())
	}
	if chatReq.Model != "test-model" {
		t.Fatalf("chat model = %q, want test-model", chatReq.Model)
	}
	if len(chatReq.Messages) != 3 ||
		chatReq.Messages[1].Content != "old prompt" ||
		chatReq.Messages[2].Content != "follow up" {
		t.Fatalf("chat messages = %#v", chatReq.Messages)
	}
}

func TestRunHandlerPromptRunsAgentHeadless(t *testing.T) {
	var chatReq api.ChatRequest
	mockServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/api/show":
			if r.Method != http.MethodPost {
				t.Errorf("show method = %s, want POST", r.Method)
			}
			w.Header().Set("Content-Type", "application/json")
			if err := json.NewEncoder(w).Encode(api.ShowResponse{}); err != nil {
				t.Fatal(err)
			}
		case "/api/chat":
			if r.Method != http.MethodPost {
				t.Errorf("chat method = %s, want POST", r.Method)
			}
			if err := json.NewDecoder(r.Body).Decode(&chatReq); err != nil {
				http.Error(w, err.Error(), http.StatusBadRequest)
				return
			}
			w.Header().Set("Content-Type", "application/x-ndjson")
			enc := json.NewEncoder(w)
			if err := enc.Encode(api.ChatResponse{Message: api.Message{Role: "assistant", Content: "hello"}}); err != nil {
				t.Fatal(err)
			}
			if flusher, ok := w.(http.Flusher); ok {
				flusher.Flush()
			}
			if err := enc.Encode(api.ChatResponse{Done: true, DoneReason: "stop"}); err != nil {
				t.Fatal(err)
			}
		case "/api/generate":
			if r.Method != http.MethodPost {
				t.Errorf("generate method = %s, want POST", r.Method)
			}
			if err := json.NewEncoder(w).Encode(api.GenerateResponse{Done: true}); err != nil {
				t.Fatal(err)
			}
		default:
			http.NotFound(w, r)
		}
	}))
	t.Cleanup(mockServer.Close)
	t.Setenv("OLLAMA_HOST", mockServer.URL)
	t.Setenv("HOME", t.TempDir())
	t.Setenv("LOCALAPPDATA", t.TempDir())
	if err := config.SetLastModel("previous-model"); err != nil {
		t.Fatal(err)
	}

	cmd := &cobra.Command{}
	cmd.SetContext(t.Context())
	cmd.Flags().String("format", "", "")
	cmd.Flags().String("think", "", "")
	cmd.Flags().Bool("hidethinking", false, "")
	cmd.Flags().Bool("resume", false, "")
	cmd.Flags().String("keepalive", "", "")
	cmd.Flags().Bool("verbose", false, "")

	oldStdout := os.Stdout
	r, w, _ := os.Pipe()
	os.Stdout = w

	err := RunHandler(cmd, []string{"test-model", "hello"})
	w.Close()
	os.Stdout = oldStdout
	if err != nil {
		t.Fatalf("RunHandler returned error: %v", err)
	}

	var out bytes.Buffer
	if _, err := io.Copy(&out, r); err != nil {
		t.Fatal(err)
	}
	if got := out.String(); got != "hello\n" {
		t.Fatalf("stdout = %q, want hello newline", got)
	}
	if chatReq.Model != "test-model" {
		t.Fatalf("chat model = %q, want test-model", chatReq.Model)
	}
	if len(chatReq.Messages) != 2 ||
		chatReq.Messages[0].Role != "system" ||
		!strings.Contains(chatReq.Messages[0].Content, "You are running in Ollama, in a harness to help the user accomplish tasks, and the model is test-model.") ||
		chatReq.Messages[1].Role != "user" ||
		chatReq.Messages[1].Content != "hello" {
		t.Fatalf("chat messages = %#v", chatReq.Messages)
	}
	if got := config.LastModel(); got != "previous-model" {
		t.Fatalf("headless run updated last model to %q, want previous-model", got)
	}
}

func TestRunHandlerHeadlessDeniedApprovalReturnsError(t *testing.T) {
	var chatCalls int
	mockServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/api/show":
			if err := json.NewEncoder(w).Encode(api.ShowResponse{
				Capabilities: []model.Capability{model.CapabilityTools},
			}); err != nil {
				t.Fatal(err)
			}
		case "/api/generate":
			if err := json.NewEncoder(w).Encode(api.GenerateResponse{Done: true}); err != nil {
				t.Fatal(err)
			}
		case "/api/ps":
			if err := json.NewEncoder(w).Encode(api.ProcessResponse{
				Models: []api.ProcessModelResponse{{
					Name:          "test-model:latest",
					Model:         "test-model:latest",
					ContextLength: 8192,
				}},
			}); err != nil {
				t.Fatal(err)
			}
		case "/api/chat":
			chatCalls++
			if chatCalls > 1 {
				t.Fatalf("chat calls = %d, want denied tool run to stop after first call", chatCalls)
			}
			args := api.NewToolCallFunctionArguments()
			args.Set("command", "pwd")
			w.Header().Set("Content-Type", "application/x-ndjson")
			enc := json.NewEncoder(w)
			if err := enc.Encode(api.ChatResponse{Message: api.Message{Role: "assistant", ToolCalls: []api.ToolCall{{
				ID: "call-1",
				Function: api.ToolCallFunction{
					Name:      "bash",
					Arguments: args,
				},
			}}}}); err != nil {
				t.Fatal(err)
			}
			if err := enc.Encode(api.ChatResponse{Done: true, DoneReason: "stop"}); err != nil {
				t.Fatal(err)
			}
		case "/api/status":
			if err := json.NewEncoder(w).Encode(api.StatusResponse{}); err != nil {
				t.Fatal(err)
			}
		default:
			http.NotFound(w, r)
		}
	}))
	t.Cleanup(mockServer.Close)
	t.Setenv("OLLAMA_HOST", mockServer.URL)
	t.Setenv("HOME", t.TempDir())
	t.Setenv("LOCALAPPDATA", t.TempDir())

	cmd := &cobra.Command{}
	cmd.SetContext(t.Context())
	cmd.Flags().String("format", "", "")
	cmd.Flags().String("think", "", "")
	cmd.Flags().Bool("hidethinking", false, "")
	cmd.Flags().Bool("resume", false, "")
	cmd.Flags().String("keepalive", "", "")
	cmd.Flags().Bool("verbose", false, "")

	oldStdout := os.Stdout
	stdoutR, stdoutW, _ := os.Pipe()
	os.Stdout = stdoutW
	oldStderr := os.Stderr
	stderrR, stderrW, _ := os.Pipe()
	os.Stderr = stderrW

	err := RunHandler(cmd, []string{"test-model", "run pwd"})
	stdoutW.Close()
	stderrW.Close()
	os.Stdout = oldStdout
	os.Stderr = oldStderr
	_, _ = io.Copy(io.Discard, stdoutR)
	_, _ = io.Copy(io.Discard, stderrR)
	if err == nil || !strings.Contains(err.Error(), "tool execution denied") {
		t.Fatalf("RunHandler error = %v, want tool execution denied", err)
	}
	if chatCalls != 1 {
		t.Fatalf("chat calls = %d, want 1", chatCalls)
	}
}

func TestRunHandlerHeadlessBudgetsAgainstLoadedContext(t *testing.T) {
	var chatCalled bool
	var generateCalled bool
	mockServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/api/show":
			w.Header().Set("Content-Type", "application/json")
			if err := json.NewEncoder(w).Encode(api.ShowResponse{
				Details: api.ModelDetails{ContextLength: 131072},
			}); err != nil {
				t.Fatal(err)
			}
		case "/api/generate":
			generateCalled = true
			if err := json.NewEncoder(w).Encode(api.GenerateResponse{Done: true}); err != nil {
				t.Fatal(err)
			}
		case "/api/ps":
			if err := json.NewEncoder(w).Encode(api.ProcessResponse{
				Models: []api.ProcessModelResponse{{
					Name:          "test-model:latest",
					Model:         "test-model:latest",
					ContextLength: 1024,
				}},
			}); err != nil {
				t.Fatal(err)
			}
		case "/api/chat":
			chatCalled = true
			http.Error(w, "chat should not be called after preflight fails", http.StatusInternalServerError)
		default:
			http.NotFound(w, r)
		}
	}))
	t.Cleanup(mockServer.Close)
	t.Setenv("OLLAMA_HOST", mockServer.URL)
	t.Setenv("HOME", t.TempDir())
	t.Setenv("LOCALAPPDATA", t.TempDir())

	cmd := &cobra.Command{}
	cmd.SetContext(t.Context())
	cmd.Flags().String("format", "", "")
	cmd.Flags().String("think", "", "")
	cmd.Flags().Bool("hidethinking", false, "")
	cmd.Flags().Bool("resume", false, "")
	cmd.Flags().String("keepalive", "", "")
	cmd.Flags().Bool("verbose", false, "")

	err := RunHandler(cmd, []string{"test-model", strings.Repeat("word ", 4000)})
	if err == nil {
		t.Fatal("expected preflight context error")
	}
	if !strings.Contains(err.Error(), "current context") {
		t.Fatalf("error = %q, want context preflight error", err.Error())
	}
	if !generateCalled {
		t.Fatal("expected model preload before context budget resolution")
	}
	if chatCalled {
		t.Fatal("chat should not be called when loaded context preflight fails")
	}
}

func TestRunHandlerPromptUsesAgentLoopByDefault(t *testing.T) {
	var chatReq api.ChatRequest
	var preloadCalled bool
	mockServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/api/show":
			w.Header().Set("Content-Type", "application/json")
			if err := json.NewEncoder(w).Encode(api.ShowResponse{}); err != nil {
				t.Fatal(err)
			}
		case "/api/chat":
			if err := json.NewDecoder(r.Body).Decode(&chatReq); err != nil {
				http.Error(w, err.Error(), http.StatusBadRequest)
				return
			}
			w.Header().Set("Content-Type", "application/x-ndjson")
			enc := json.NewEncoder(w)
			if err := enc.Encode(api.ChatResponse{Message: api.Message{Role: "assistant", Content: "hello"}}); err != nil {
				t.Fatal(err)
			}
			if err := enc.Encode(api.ChatResponse{Done: true, DoneReason: "stop"}); err != nil {
				t.Fatal(err)
			}
		case "/api/generate":
			preloadCalled = true
			if err := json.NewEncoder(w).Encode(api.GenerateResponse{Done: true}); err != nil {
				t.Fatal(err)
			}
		default:
			http.NotFound(w, r)
		}
	}))
	t.Cleanup(mockServer.Close)
	t.Setenv("OLLAMA_HOST", mockServer.URL)
	t.Setenv("HOME", t.TempDir())
	t.Setenv("LOCALAPPDATA", t.TempDir())

	cmd := &cobra.Command{}
	cmd.SetContext(t.Context())
	cmd.Flags().String("format", "", "")
	cmd.Flags().String("think", "", "")
	cmd.Flags().Bool("hidethinking", false, "")
	cmd.Flags().Bool("resume", false, "")
	cmd.Flags().String("keepalive", "", "")
	cmd.Flags().Bool("verbose", false, "")

	oldStdout := os.Stdout
	r, w, _ := os.Pipe()
	os.Stdout = w

	err := RunHandler(cmd, []string{"test-model", "hello"})
	w.Close()
	os.Stdout = oldStdout
	if err != nil {
		t.Fatalf("RunHandler returned error: %v", err)
	}
	if _, err := io.Copy(io.Discard, r); err != nil {
		t.Fatal(err)
	}

	if !preloadCalled {
		t.Fatal("expected default prompt path to preload the model")
	}
	if chatReq.Model != "test-model" {
		t.Fatalf("chat model = %q, want test-model", chatReq.Model)
	}
	if len(chatReq.Messages) != 2 ||
		chatReq.Messages[0].Role != "system" ||
		!strings.Contains(chatReq.Messages[0].Content, "You are running in Ollama, in a harness to help the user accomplish tasks, and the model is test-model.") ||
		chatReq.Messages[1].Role != "user" ||
		chatReq.Messages[1].Content != "hello" {
		t.Fatalf("chat messages = %#v", chatReq.Messages)
	}
}

func TestRunEmbeddingModelWithFlags(t *testing.T) {
	reqCh := make(chan api.EmbedRequest, 1)
	mockServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/api/show" && r.Method == http.MethodPost {
			w.Header().Set("Content-Type", "application/json")
			if err := json.NewEncoder(w).Encode(api.ShowResponse{
				Capabilities: []model.Capability{model.CapabilityEmbedding},
			}); err != nil {
				http.Error(w, err.Error(), http.StatusInternalServerError)
			}
			return
		}
		if r.URL.Path == "/api/embed" && r.Method == http.MethodPost {
			var req api.EmbedRequest
			if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
				http.Error(w, err.Error(), http.StatusBadRequest)
				return
			}
			reqCh <- req
			w.Header().Set("Content-Type", "application/json")
			if err := json.NewEncoder(w).Encode(api.EmbedResponse{
				Model:        "test-embedding-model",
				Embeddings:   [][]float32{{0.4, 0.5}},
				LoadDuration: 5 * time.Millisecond,
			}); err != nil {
				http.Error(w, err.Error(), http.StatusInternalServerError)
			}
			return
		}
		http.NotFound(w, r)
	}))

	t.Setenv("OLLAMA_HOST", mockServer.URL)
	t.Cleanup(mockServer.Close)

	cmd := &cobra.Command{}
	cmd.SetContext(t.Context())
	cmd.Flags().String("keepalive", "", "")
	cmd.Flags().Bool("truncate", false, "")
	cmd.Flags().Int("dimensions", 0, "")
	cmd.Flags().Bool("verbose", false, "")
	cmd.Flags().Bool("insecure", false, "")
	cmd.Flags().String("format", "", "")
	cmd.Flags().String("think", "", "")
	cmd.Flags().Bool("hidethinking", false, "")

	if err := cmd.Flags().Set("truncate", "true"); err != nil {
		t.Fatalf("failed to set truncate flag: %v", err)
	}
	if err := cmd.Flags().Set("dimensions", "2"); err != nil {
		t.Fatalf("failed to set dimensions flag: %v", err)
	}
	if err := cmd.Flags().Set("keepalive", "5m"); err != nil {
		t.Fatalf("failed to set keepalive flag: %v", err)
	}

	oldStdout := os.Stdout
	r, w, _ := os.Pipe()
	os.Stdout = w

	errCh := make(chan error, 1)
	go func() {
		errCh <- RunHandler(cmd, []string{"test-embedding-model", "test", "input"})
	}()

	err := <-errCh
	w.Close()
	os.Stdout = oldStdout

	if err != nil {
		t.Fatalf("RunHandler returned error: %v", err)
	}

	var out bytes.Buffer
	io.Copy(&out, r)

	select {
	case req := <-reqCh:
		inputText, _ := req.Input.(string)
		if diff := cmp.Diff("test input", inputText); diff != "" {
			t.Errorf("unexpected input (-want +got):\n%s", diff)
		}
		if req.Truncate == nil || !*req.Truncate {
			t.Errorf("expected truncate pointer true, got %v", req.Truncate)
		}
		if req.Dimensions != 2 {
			t.Errorf("expected dimensions 2, got %d", req.Dimensions)
		}
		if req.KeepAlive == nil || req.KeepAlive.Duration != 5*time.Minute {
			t.Errorf("unexpected keepalive duration: %v", req.KeepAlive)
		}
	default:
		t.Fatal("server did not receive embed request")
	}

	expectOutput := "[0.4,0.5]\n"
	if diff := cmp.Diff(expectOutput, out.String()); diff != "" {
		t.Errorf("unexpected output (-want +got):\n%s", diff)
	}
}

func TestRunEmbeddingModelPipedInput(t *testing.T) {
	reqCh := make(chan api.EmbedRequest, 1)
	mockServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/api/show" && r.Method == http.MethodPost {
			w.Header().Set("Content-Type", "application/json")
			if err := json.NewEncoder(w).Encode(api.ShowResponse{
				Capabilities: []model.Capability{model.CapabilityEmbedding},
			}); err != nil {
				http.Error(w, err.Error(), http.StatusInternalServerError)
			}
			return
		}
		if r.URL.Path == "/api/embed" && r.Method == http.MethodPost {
			var req api.EmbedRequest
			if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
				http.Error(w, err.Error(), http.StatusBadRequest)
				return
			}
			reqCh <- req
			w.Header().Set("Content-Type", "application/json")
			if err := json.NewEncoder(w).Encode(api.EmbedResponse{
				Model:      "test-embedding-model",
				Embeddings: [][]float32{{0.6, 0.7}},
			}); err != nil {
				http.Error(w, err.Error(), http.StatusInternalServerError)
			}
			return
		}
		http.NotFound(w, r)
	}))

	t.Setenv("OLLAMA_HOST", mockServer.URL)
	t.Cleanup(mockServer.Close)

	cmd := &cobra.Command{}
	cmd.SetContext(t.Context())
	cmd.Flags().String("keepalive", "", "")
	cmd.Flags().Bool("truncate", false, "")
	cmd.Flags().Int("dimensions", 0, "")
	cmd.Flags().Bool("verbose", false, "")
	cmd.Flags().Bool("insecure", false, "")
	cmd.Flags().String("format", "", "")
	cmd.Flags().String("think", "", "")
	cmd.Flags().Bool("hidethinking", false, "")

	// Capture stdin
	oldStdin := os.Stdin
	stdinR, stdinW, _ := os.Pipe()
	os.Stdin = stdinR
	stdinW.Write([]byte("piped text"))
	stdinW.Close()

	// Capture stdout
	oldStdout := os.Stdout
	stdoutR, stdoutW, _ := os.Pipe()
	os.Stdout = stdoutW

	errCh := make(chan error, 1)
	go func() {
		errCh <- RunHandler(cmd, []string{"test-embedding-model", "additional", "args"})
	}()

	err := <-errCh
	stdoutW.Close()
	os.Stdout = oldStdout
	os.Stdin = oldStdin

	if err != nil {
		t.Fatalf("RunHandler returned error: %v", err)
	}

	var out bytes.Buffer
	io.Copy(&out, stdoutR)

	select {
	case req := <-reqCh:
		inputText, _ := req.Input.(string)
		// Should combine piped input with command line args
		if diff := cmp.Diff("piped text additional args", inputText); diff != "" {
			t.Errorf("unexpected input (-want +got):\n%s", diff)
		}
	default:
		t.Fatal("server did not receive embed request")
	}

	expectOutput := "[0.6,0.7]\n"
	if diff := cmp.Diff(expectOutput, out.String()); diff != "" {
		t.Errorf("unexpected output (-want +got):\n%s", diff)
	}
}

func TestRunEmbeddingModelNoInput(t *testing.T) {
	mockServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/api/show" && r.Method == http.MethodPost {
			w.Header().Set("Content-Type", "application/json")
			if err := json.NewEncoder(w).Encode(api.ShowResponse{
				Capabilities: []model.Capability{model.CapabilityEmbedding},
			}); err != nil {
				http.Error(w, err.Error(), http.StatusInternalServerError)
			}
			return
		}
		http.NotFound(w, r)
	}))

	t.Setenv("OLLAMA_HOST", mockServer.URL)
	t.Cleanup(mockServer.Close)

	cmd := &cobra.Command{}
	cmd.SetContext(t.Context())
	cmd.Flags().String("keepalive", "", "")
	cmd.Flags().Bool("truncate", false, "")
	cmd.Flags().Int("dimensions", 0, "")
	cmd.Flags().Bool("verbose", false, "")
	cmd.Flags().Bool("insecure", false, "")
	cmd.Flags().String("format", "", "")
	cmd.Flags().String("think", "", "")
	cmd.Flags().Bool("hidethinking", false, "")

	cmd.SetOut(io.Discard)
	cmd.SetErr(io.Discard)

	// Test with no input arguments (only model name)
	err := RunHandler(cmd, []string{"test-embedding-model"})
	if err == nil || !strings.Contains(err.Error(), "embedding models require input text") {
		t.Fatalf("expected error about missing input, got %v", err)
	}
}

func TestRunHandler_CloudAuthErrorOnShow_PrintsSigninMessage(t *testing.T) {
	var generateCalled bool

	mockServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch {
		case r.URL.Path == "/api/show" && r.Method == http.MethodPost:
			w.WriteHeader(http.StatusUnauthorized)
			if err := json.NewEncoder(w).Encode(map[string]string{
				"error":      "unauthorized",
				"signin_url": "https://ollama.com/signin",
			}); err != nil {
				http.Error(w, err.Error(), http.StatusInternalServerError)
			}
			return
		case r.URL.Path == "/api/generate" && r.Method == http.MethodPost:
			generateCalled = true
			w.WriteHeader(http.StatusOK)
			if err := json.NewEncoder(w).Encode(api.GenerateResponse{Done: true}); err != nil {
				http.Error(w, err.Error(), http.StatusInternalServerError)
			}
			return
		default:
			http.NotFound(w, r)
		}
	}))

	t.Setenv("OLLAMA_HOST", mockServer.URL)
	t.Cleanup(mockServer.Close)

	cmd := &cobra.Command{}
	cmd.SetContext(t.Context())
	cmd.Flags().String("keepalive", "", "")
	cmd.Flags().Bool("truncate", false, "")
	cmd.Flags().Int("dimensions", 0, "")
	cmd.Flags().Bool("verbose", false, "")
	cmd.Flags().Bool("insecure", false, "")
	cmd.Flags().String("format", "", "")
	cmd.Flags().String("think", "", "")
	cmd.Flags().Bool("hidethinking", false, "")

	oldStdout := os.Stdout
	readOut, writeOut, _ := os.Pipe()
	os.Stdout = writeOut
	t.Cleanup(func() { os.Stdout = oldStdout })

	err := RunHandler(cmd, []string{"gpt-oss:20b:cloud", "hi"})

	_ = writeOut.Close()
	var out bytes.Buffer
	_, _ = io.Copy(&out, readOut)

	if err != nil {
		t.Fatalf("RunHandler returned error: %v", err)
	}

	if generateCalled {
		t.Fatal("expected run to stop before /api/generate after unauthorized /api/show")
	}

	if !strings.Contains(out.String(), "You need to be signed in to Ollama to run Cloud models.") {
		t.Fatalf("expected sign-in guidance message, got %q", out.String())
	}

	if !strings.Contains(out.String(), "https://ollama.com/signin") {
		t.Fatalf("expected signin_url in output, got %q", out.String())
	}
}

func TestRunHandler_CloudAuthErrorOnAgentChat_PrintsSigninMessage(t *testing.T) {
	mockServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch {
		case r.URL.Path == "/api/show" && r.Method == http.MethodPost:
			w.WriteHeader(http.StatusOK)
			if err := json.NewEncoder(w).Encode(api.ShowResponse{
				Capabilities: []model.Capability{model.CapabilityCompletion},
			}); err != nil {
				http.Error(w, err.Error(), http.StatusInternalServerError)
			}
			return
		case r.URL.Path == "/api/chat" && r.Method == http.MethodPost:
			w.WriteHeader(http.StatusUnauthorized)
			if err := json.NewEncoder(w).Encode(map[string]string{
				"error":      "unauthorized",
				"signin_url": "https://ollama.com/signin",
			}); err != nil {
				http.Error(w, err.Error(), http.StatusInternalServerError)
			}
			return
		default:
			http.NotFound(w, r)
		}
	}))

	t.Setenv("OLLAMA_HOST", mockServer.URL)
	t.Setenv("HOME", t.TempDir())
	t.Setenv("LOCALAPPDATA", t.TempDir())
	t.Cleanup(mockServer.Close)

	cmd := &cobra.Command{}
	cmd.SetContext(t.Context())
	cmd.Flags().String("keepalive", "", "")
	cmd.Flags().Bool("truncate", false, "")
	cmd.Flags().Int("dimensions", 0, "")
	cmd.Flags().Bool("verbose", false, "")
	cmd.Flags().Bool("insecure", false, "")
	cmd.Flags().String("format", "", "")
	cmd.Flags().String("think", "", "")
	cmd.Flags().Bool("hidethinking", false, "")

	oldStdout := os.Stdout
	readOut, writeOut, _ := os.Pipe()
	os.Stdout = writeOut
	t.Cleanup(func() { os.Stdout = oldStdout })

	err := RunHandler(cmd, []string{"gpt-oss:20b:cloud", "hi"})

	_ = writeOut.Close()
	var out bytes.Buffer
	_, _ = io.Copy(&out, readOut)

	if err != nil {
		t.Fatalf("RunHandler returned error: %v", err)
	}

	if !strings.Contains(out.String(), "You need to be signed in to Ollama to run Cloud models.") {
		t.Fatalf("expected sign-in guidance message, got %q", out.String())
	}

	if !strings.Contains(out.String(), "https://ollama.com/signin") {
		t.Fatalf("expected signin_url in output, got %q", out.String())
	}
}

func TestRunHandler_ExplicitCloudStubMissing_PullsNormalizedNameTEMP(t *testing.T) {
	var pulledModel string
	var chatCalled bool

	mockServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch {
		case r.URL.Path == "/api/show" && r.Method == http.MethodPost:
			w.WriteHeader(http.StatusOK)
			if err := json.NewEncoder(w).Encode(api.ShowResponse{
				Capabilities: []model.Capability{model.CapabilityCompletion},
				RemoteModel:  "gpt-oss:20b",
			}); err != nil {
				http.Error(w, err.Error(), http.StatusInternalServerError)
			}
			return
		case r.URL.Path == "/api/tags" && r.Method == http.MethodGet:
			w.WriteHeader(http.StatusOK)
			if err := json.NewEncoder(w).Encode(api.ListResponse{Models: nil}); err != nil {
				http.Error(w, err.Error(), http.StatusInternalServerError)
			}
			return
		case r.URL.Path == "/api/pull" && r.Method == http.MethodPost:
			var req api.PullRequest
			if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
				http.Error(w, err.Error(), http.StatusBadRequest)
				return
			}
			pulledModel = req.Model
			w.WriteHeader(http.StatusOK)
			if err := json.NewEncoder(w).Encode(api.ProgressResponse{Status: "success"}); err != nil {
				http.Error(w, err.Error(), http.StatusInternalServerError)
			}
			return
		case r.URL.Path == "/api/chat" && r.Method == http.MethodPost:
			chatCalled = true
			w.WriteHeader(http.StatusOK)
			if err := json.NewEncoder(w).Encode(api.ChatResponse{Done: true, DoneReason: "stop"}); err != nil {
				http.Error(w, err.Error(), http.StatusInternalServerError)
			}
			return
		default:
			http.NotFound(w, r)
		}
	}))

	t.Setenv("OLLAMA_HOST", mockServer.URL)
	t.Setenv("HOME", t.TempDir())
	t.Setenv("LOCALAPPDATA", t.TempDir())
	t.Cleanup(mockServer.Close)

	cmd := &cobra.Command{}
	cmd.SetContext(t.Context())
	cmd.Flags().String("keepalive", "", "")
	cmd.Flags().Bool("truncate", false, "")
	cmd.Flags().Int("dimensions", 0, "")
	cmd.Flags().Bool("verbose", false, "")
	cmd.Flags().Bool("insecure", false, "")
	cmd.Flags().String("format", "", "")
	cmd.Flags().String("think", "", "")
	cmd.Flags().Bool("hidethinking", false, "")

	err := RunHandler(cmd, []string{"gpt-oss:20b:cloud", "hi"})
	if err != nil {
		t.Fatalf("RunHandler returned error: %v", err)
	}

	if pulledModel != "gpt-oss:20b-cloud" {
		t.Fatalf("expected normalized pull model %q, got %q", "gpt-oss:20b-cloud", pulledModel)
	}

	if !chatCalled {
		t.Fatal("expected /api/chat to be called")
	}
}

func TestRunHandler_ExplicitCloudStubPresent_SkipsPullTEMP(t *testing.T) {
	var pullCalled bool
	var chatCalled bool

	mockServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch {
		case r.URL.Path == "/api/show" && r.Method == http.MethodPost:
			w.WriteHeader(http.StatusOK)
			if err := json.NewEncoder(w).Encode(api.ShowResponse{
				Capabilities: []model.Capability{model.CapabilityCompletion},
				RemoteModel:  "gpt-oss:20b",
			}); err != nil {
				http.Error(w, err.Error(), http.StatusInternalServerError)
			}
			return
		case r.URL.Path == "/api/tags" && r.Method == http.MethodGet:
			w.WriteHeader(http.StatusOK)
			if err := json.NewEncoder(w).Encode(api.ListResponse{
				Models: []api.ListModelResponse{{Name: "gpt-oss:20b-cloud"}},
			}); err != nil {
				http.Error(w, err.Error(), http.StatusInternalServerError)
			}
			return
		case r.URL.Path == "/api/pull" && r.Method == http.MethodPost:
			pullCalled = true
			w.WriteHeader(http.StatusOK)
			if err := json.NewEncoder(w).Encode(api.ProgressResponse{Status: "success"}); err != nil {
				http.Error(w, err.Error(), http.StatusInternalServerError)
			}
			return
		case r.URL.Path == "/api/chat" && r.Method == http.MethodPost:
			chatCalled = true
			w.WriteHeader(http.StatusOK)
			if err := json.NewEncoder(w).Encode(api.ChatResponse{Done: true, DoneReason: "stop"}); err != nil {
				http.Error(w, err.Error(), http.StatusInternalServerError)
			}
			return
		default:
			http.NotFound(w, r)
		}
	}))

	t.Setenv("OLLAMA_HOST", mockServer.URL)
	t.Setenv("HOME", t.TempDir())
	t.Setenv("LOCALAPPDATA", t.TempDir())
	t.Cleanup(mockServer.Close)

	cmd := &cobra.Command{}
	cmd.SetContext(t.Context())
	cmd.Flags().String("keepalive", "", "")
	cmd.Flags().Bool("truncate", false, "")
	cmd.Flags().Int("dimensions", 0, "")
	cmd.Flags().Bool("verbose", false, "")
	cmd.Flags().Bool("insecure", false, "")
	cmd.Flags().String("format", "", "")
	cmd.Flags().String("think", "", "")
	cmd.Flags().Bool("hidethinking", false, "")

	err := RunHandler(cmd, []string{"gpt-oss:20b:cloud", "hi"})
	if err != nil {
		t.Fatalf("RunHandler returned error: %v", err)
	}

	if pullCalled {
		t.Fatal("expected /api/pull not to be called when cloud stub already exists")
	}

	if !chatCalled {
		t.Fatal("expected /api/chat to be called")
	}
}

func TestRunHandler_ExplicitCloudStubPullFailure_IsBestEffortTEMP(t *testing.T) {
	var chatCalled bool

	mockServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch {
		case r.URL.Path == "/api/show" && r.Method == http.MethodPost:
			w.WriteHeader(http.StatusOK)
			if err := json.NewEncoder(w).Encode(api.ShowResponse{
				Capabilities: []model.Capability{model.CapabilityCompletion},
				RemoteModel:  "gpt-oss:20b",
			}); err != nil {
				http.Error(w, err.Error(), http.StatusInternalServerError)
			}
			return
		case r.URL.Path == "/api/tags" && r.Method == http.MethodGet:
			w.WriteHeader(http.StatusOK)
			if err := json.NewEncoder(w).Encode(api.ListResponse{Models: nil}); err != nil {
				http.Error(w, err.Error(), http.StatusInternalServerError)
			}
			return
		case r.URL.Path == "/api/pull" && r.Method == http.MethodPost:
			w.WriteHeader(http.StatusInternalServerError)
			if err := json.NewEncoder(w).Encode(map[string]string{"error": "pull failed"}); err != nil {
				http.Error(w, err.Error(), http.StatusInternalServerError)
			}
			return
		case r.URL.Path == "/api/chat" && r.Method == http.MethodPost:
			chatCalled = true
			w.WriteHeader(http.StatusOK)
			if err := json.NewEncoder(w).Encode(api.ChatResponse{Done: true, DoneReason: "stop"}); err != nil {
				http.Error(w, err.Error(), http.StatusInternalServerError)
			}
			return
		default:
			http.NotFound(w, r)
		}
	}))

	t.Setenv("OLLAMA_HOST", mockServer.URL)
	t.Setenv("HOME", t.TempDir())
	t.Setenv("LOCALAPPDATA", t.TempDir())
	t.Cleanup(mockServer.Close)

	cmd := &cobra.Command{}
	cmd.SetContext(t.Context())
	cmd.Flags().String("keepalive", "", "")
	cmd.Flags().Bool("truncate", false, "")
	cmd.Flags().Int("dimensions", 0, "")
	cmd.Flags().Bool("verbose", false, "")
	cmd.Flags().Bool("insecure", false, "")
	cmd.Flags().String("format", "", "")
	cmd.Flags().String("think", "", "")
	cmd.Flags().Bool("hidethinking", false, "")

	err := RunHandler(cmd, []string{"gpt-oss:20b:cloud", "hi"})
	if err != nil {
		t.Fatalf("RunHandler returned error: %v", err)
	}

	if !chatCalled {
		t.Fatal("expected /api/chat to be called despite pull failure")
	}
}

func TestGetModelfileName(t *testing.T) {
	tests := []struct {
		name          string
		modelfileName string
		fileExists    bool
		expectedName  string
		expectedErr   error
	}{
		{
			name:          "no modelfile specified, no modelfile exists",
			modelfileName: "",
			fileExists:    false,
			expectedName:  "",
			expectedErr:   os.ErrNotExist,
		},
		{
			name:          "no modelfile specified, modelfile exists",
			modelfileName: "",
			fileExists:    true,
			expectedName:  "Modelfile",
			expectedErr:   nil,
		},
		{
			name:          "modelfile specified, no modelfile exists",
			modelfileName: "crazyfile",
			fileExists:    false,
			expectedName:  "",
			expectedErr:   os.ErrNotExist,
		},
		{
			name:          "modelfile specified, modelfile exists",
			modelfileName: "anotherfile",
			fileExists:    true,
			expectedName:  "anotherfile",
			expectedErr:   nil,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cmd := &cobra.Command{
				Use: "fakecmd",
			}
			cmd.Flags().String("file", "", "path to modelfile")

			var expectedFilename string

			if tt.fileExists {
				var fn string
				if tt.modelfileName != "" {
					fn = tt.modelfileName
				} else {
					fn = "Modelfile"
				}

				tempFile, err := os.CreateTemp(t.TempDir(), fn)
				if err != nil {
					t.Fatalf("temp modelfile creation failed: %v", err)
				}
				defer tempFile.Close()

				expectedFilename = tempFile.Name()
				err = cmd.Flags().Set("file", expectedFilename)
				if err != nil {
					t.Fatalf("couldn't set file flag: %v", err)
				}
			} else {
				expectedFilename = tt.expectedName
				if tt.modelfileName != "" {
					err := cmd.Flags().Set("file", tt.modelfileName)
					if err != nil {
						t.Fatalf("couldn't set file flag: %v", err)
					}
				}
			}

			actualFilename, actualErr := getModelfileName(cmd)

			if actualFilename != expectedFilename {
				t.Errorf("expected filename: '%s' actual filename: '%s'", expectedFilename, actualFilename)
			}

			if tt.expectedErr != os.ErrNotExist {
				if actualErr != tt.expectedErr {
					t.Errorf("expected err: %v actual err: %v", tt.expectedErr, actualErr)
				}
			} else {
				if !os.IsNotExist(actualErr) {
					t.Errorf("expected err: %v actual err: %v", tt.expectedErr, actualErr)
				}
			}
		})
	}
}

func TestPushHandler(t *testing.T) {
	tests := []struct {
		name           string
		modelName      string
		serverResponse map[string]func(w http.ResponseWriter, r *http.Request)
		expectedError  string
		expectedOutput string
	}{
		{
			name:      "successful push",
			modelName: "test-model",
			serverResponse: map[string]func(w http.ResponseWriter, r *http.Request){
				"/api/push": func(w http.ResponseWriter, r *http.Request) {
					if r.Method != http.MethodPost {
						t.Errorf("expected POST request, got %s", r.Method)
					}

					var req api.PushRequest
					if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
						http.Error(w, err.Error(), http.StatusBadRequest)
						return
					}

					if req.Name != "test-model" {
						t.Errorf("expected model name 'test-model', got %s", req.Name)
					}

					// Simulate progress updates
					responses := []api.ProgressResponse{
						{Status: "preparing manifest"},
						{Digest: "sha256:abc123456789", Total: 100, Completed: 50},
						{Digest: "sha256:abc123456789", Total: 100, Completed: 100},
					}

					for _, resp := range responses {
						if err := json.NewEncoder(w).Encode(resp); err != nil {
							http.Error(w, err.Error(), http.StatusInternalServerError)
							return
						}
						w.(http.Flusher).Flush()
					}
				},
				"/api/me": func(w http.ResponseWriter, r *http.Request) {
					if r.Method != http.MethodPost {
						t.Errorf("expected POST request, got %s", r.Method)
					}
				},
			},
			expectedOutput: "\nYou can find your model at:\n\n\thttps://ollama.com/test-model\n",
		},
		{
			name:      "not signed in push",
			modelName: "notsignedin-model",
			serverResponse: map[string]func(w http.ResponseWriter, r *http.Request){
				"/api/me": func(w http.ResponseWriter, r *http.Request) {
					if r.Method != http.MethodPost {
						t.Errorf("expected POST request, got %s", r.Method)
					}
					w.Header().Set("Content-Type", "application/json")
					w.WriteHeader(http.StatusUnauthorized)
					err := json.NewEncoder(w).Encode(map[string]string{
						"error":      "unauthorized",
						"signin_url": "https://somethingsomething",
					})
					if err != nil {
						t.Fatal(err)
					}
				},
			},
			expectedOutput: "You need to be signed in to push",
		},
		{
			name:      "unauthorized push",
			modelName: "unauthorized-model",
			serverResponse: map[string]func(w http.ResponseWriter, r *http.Request){
				"/api/push": func(w http.ResponseWriter, r *http.Request) {
					w.Header().Set("Content-Type", "application/json")
					w.WriteHeader(http.StatusUnauthorized)
					err := json.NewEncoder(w).Encode(map[string]string{
						"error": "403: {\"errors\":[{\"code\":\"ACCESS DENIED\", \"message\":\"access denied\"}]}",
					})
					if err != nil {
						t.Fatal(err)
					}
				},
				"/api/me": func(w http.ResponseWriter, r *http.Request) {
					if r.Method != http.MethodPost {
						t.Errorf("expected POST request, got %s", r.Method)
					}
				},
			},
			expectedError: "you are not authorized to push to this namespace, create the model under a namespace you own",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			mockServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if handler, ok := tt.serverResponse[r.URL.Path]; ok {
					handler(w, r)
					return
				}
				http.Error(w, "not found", http.StatusNotFound)
			}))
			defer mockServer.Close()

			t.Setenv("OLLAMA_HOST", mockServer.URL)
			tmpDir := t.TempDir()
			t.Setenv("HOME", tmpDir)
			t.Setenv("USERPROFILE", tmpDir)
			initializeKeypair()

			cmd := &cobra.Command{}
			cmd.Flags().Bool("insecure", false, "")
			cmd.SetContext(t.Context())

			// Redirect stderr to capture progress output
			oldStderr := os.Stderr
			r, w, _ := os.Pipe()
			os.Stderr = w

			// Capture stdout for the "Model pushed" message
			oldStdout := os.Stdout
			outR, outW, _ := os.Pipe()
			os.Stdout = outW

			err := PushHandler(cmd, []string{tt.modelName})

			// Restore stderr
			w.Close()
			os.Stderr = oldStderr
			// drain the pipe
			if _, err := io.ReadAll(r); err != nil {
				t.Fatal(err)
			}

			// Restore stdout and get output
			outW.Close()
			os.Stdout = oldStdout
			stdout, _ := io.ReadAll(outR)

			if tt.expectedError == "" {
				if err != nil {
					t.Errorf("expected no error, got %v", err)
				}
				if tt.expectedOutput != "" {
					if got := string(stdout); !strings.Contains(got, tt.expectedOutput) {
						t.Errorf("expected output %q, got %q", tt.expectedOutput, got)
					}
				}
			} else {
				if err == nil || !strings.Contains(err.Error(), tt.expectedError) {
					t.Errorf("expected error containing %q, got %v", tt.expectedError, err)
				}
			}
		})
	}
}

func TestListHandler(t *testing.T) {
	tests := []struct {
		name           string
		args           []string
		serverResponse []api.ListModelResponse
		expectedError  string
		expectedOutput string
	}{
		{
			name: "list all models",
			args: []string{},
			serverResponse: []api.ListModelResponse{
				{Name: "model1", Digest: "sha256:abc123", Size: 1024, ModifiedAt: time.Now().Add(-24 * time.Hour)},
				{Name: "model2", Digest: "sha256:def456", Size: 2048, ModifiedAt: time.Now().Add(-48 * time.Hour)},
			},
			expectedOutput: "NAME      ID              SIZE      MODIFIED     \n" +
				"model1    sha256:abc12    1.0 KB    24 hours ago    \n" +
				"model2    sha256:def45    2.0 KB    2 days ago      \n",
		},
		{
			name: "filter models by prefix",
			args: []string{"model1"},
			serverResponse: []api.ListModelResponse{
				{Name: "model1", Digest: "sha256:abc123", Size: 1024, ModifiedAt: time.Now().Add(-24 * time.Hour)},
				{Name: "model2", Digest: "sha256:def456", Size: 2048, ModifiedAt: time.Now().Add(-24 * time.Hour)},
			},
			expectedOutput: "NAME      ID              SIZE      MODIFIED     \n" +
				"model1    sha256:abc12    1.0 KB    24 hours ago    \n",
		},
		{
			name:          "server error",
			args:          []string{},
			expectedError: "server error",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			mockServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if r.URL.Path != "/api/tags" || r.Method != http.MethodGet {
					t.Errorf("unexpected request to %s %s", r.Method, r.URL.Path)
					http.Error(w, "not found", http.StatusNotFound)
					return
				}

				if tt.expectedError != "" {
					http.Error(w, tt.expectedError, http.StatusInternalServerError)
					return
				}

				response := api.ListResponse{Models: tt.serverResponse}
				if err := json.NewEncoder(w).Encode(response); err != nil {
					t.Fatal(err)
				}
			}))
			defer mockServer.Close()

			t.Setenv("OLLAMA_HOST", mockServer.URL)

			cmd := &cobra.Command{}
			cmd.SetContext(t.Context())

			// Capture stdout
			oldStdout := os.Stdout
			r, w, _ := os.Pipe()
			os.Stdout = w

			err := ListHandler(cmd, tt.args)

			// Restore stdout and get output
			w.Close()
			os.Stdout = oldStdout
			output, _ := io.ReadAll(r)

			if tt.expectedError == "" {
				if err != nil {
					t.Errorf("expected no error, got %v", err)
				}
				if got := string(output); got != tt.expectedOutput {
					t.Errorf("expected output:\n%s\ngot:\n%s", tt.expectedOutput, got)
				}
			} else {
				if err == nil || !strings.Contains(err.Error(), tt.expectedError) {
					t.Errorf("expected error containing %q, got %v", tt.expectedError, err)
				}
			}
		})
	}
}

func TestCreateHandler(t *testing.T) {
	tests := []struct {
		name           string
		modelName      string
		modelFile      string
		serverResponse map[string]func(w http.ResponseWriter, r *http.Request)
		expectedError  string
		expectedOutput string
	}{
		{
			name:      "successful create",
			modelName: "test-model",
			modelFile: "FROM foo",
			serverResponse: map[string]func(w http.ResponseWriter, r *http.Request){
				"/api/create": func(w http.ResponseWriter, r *http.Request) {
					if r.Method != http.MethodPost {
						t.Errorf("expected POST request, got %s", r.Method)
					}

					req := api.CreateRequest{}
					if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
						http.Error(w, err.Error(), http.StatusBadRequest)
						return
					}

					if req.Model != "test-model" {
						t.Errorf("expected model name 'test-model', got %s", req.Name)
					}

					if req.From != "foo" {
						t.Errorf("expected from 'foo', got %s", req.From)
					}

					responses := []api.ProgressResponse{
						{Status: "using existing layer sha256:56bb8bd477a519ffa694fc449c2413c6f0e1d3b1c88fa7e3c9d88d3ae49d4dcb"},
						{Status: "writing manifest"},
						{Status: "success"},
					}

					for _, resp := range responses {
						if err := json.NewEncoder(w).Encode(resp); err != nil {
							http.Error(w, err.Error(), http.StatusInternalServerError)
							return
						}
						w.(http.Flusher).Flush()
					}
				},
			},
			expectedOutput: "",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			mockServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				handler, ok := tt.serverResponse[r.URL.Path]
				if !ok {
					t.Errorf("unexpected request to %s", r.URL.Path)
					http.Error(w, "not found", http.StatusNotFound)
					return
				}
				handler(w, r)
			}))
			t.Setenv("OLLAMA_HOST", mockServer.URL)
			t.Cleanup(mockServer.Close)
			tempFile, err := os.CreateTemp(t.TempDir(), "modelfile")
			if err != nil {
				t.Fatal(err)
			}
			defer os.Remove(tempFile.Name())

			if _, err := tempFile.WriteString(tt.modelFile); err != nil {
				t.Fatal(err)
			}
			if err := tempFile.Close(); err != nil {
				t.Fatal(err)
			}

			cmd := &cobra.Command{}
			cmd.Flags().String("file", "", "")
			if err := cmd.Flags().Set("file", tempFile.Name()); err != nil {
				t.Fatal(err)
			}

			cmd.Flags().Bool("insecure", false, "")
			cmd.SetContext(t.Context())

			// Redirect stderr to capture progress output
			oldStderr := os.Stderr
			r, w, _ := os.Pipe()
			os.Stderr = w

			// Capture stdout for the "Model pushed" message
			oldStdout := os.Stdout
			outR, outW, _ := os.Pipe()
			os.Stdout = outW

			err = CreateHandler(cmd, []string{tt.modelName})

			// Restore stderr
			w.Close()
			os.Stderr = oldStderr
			// drain the pipe
			if _, err := io.ReadAll(r); err != nil {
				t.Fatal(err)
			}

			// Restore stdout and get output
			outW.Close()
			os.Stdout = oldStdout
			stdout, _ := io.ReadAll(outR)

			if tt.expectedError == "" {
				if err != nil {
					t.Errorf("expected no error, got %v", err)
				}

				if tt.expectedOutput != "" {
					if got := string(stdout); got != tt.expectedOutput {
						t.Errorf("expected output %q, got %q", tt.expectedOutput, got)
					}
				}
			}
		})
	}
}

func TestCreateRequestFileNamesPreservesModelDirectoryLayout(t *testing.T) {
	root := t.TempDir()
	files := map[string]string{
		filepath.Join(root, "model.safetensors"):            "sha256:model",
		filepath.Join(root, "config.json"):                  "sha256:config",
		filepath.Join(root, "2_Dense", "config.json"):       "sha256:dense-config",
		filepath.Join(root, "2_Dense", "model.safetensors"): "sha256:dense-model",
	}

	got := createRequestFileNames(files)
	want := map[string]string{
		filepath.Join(root, "model.safetensors"):            "model.safetensors",
		filepath.Join(root, "config.json"):                  "config.json",
		filepath.Join(root, "2_Dense", "config.json"):       "2_Dense/config.json",
		filepath.Join(root, "2_Dense", "model.safetensors"): "2_Dense/model.safetensors",
	}

	if diff := cmp.Diff(want, got); diff != "" {
		t.Fatalf("mismatch (-want +got):\n%s", diff)
	}
}

func TestCreateRequestFileNamesPreservesRelativeModelDirectoryLayout(t *testing.T) {
	root := t.TempDir()
	t.Chdir(root)

	files := map[string]string{
		"model.safetensors":         "sha256:model",
		"config.json":               "sha256:config",
		"2_Dense/config.json":       "sha256:dense-config",
		"2_Dense/model.safetensors": "sha256:dense-model",
		"3_Dense/config.json":       "sha256:dense-config",
		"3_Dense/model.safetensors": "sha256:dense-model",
	}

	got := createRequestFileNames(files)
	for file := range files {
		if got[file] != filepath.ToSlash(file) {
			t.Fatalf("%s = %q, want %q", file, got[file], filepath.ToSlash(file))
		}
	}
}

func TestCreateHandlerDraftQuantizeRequiresDraft(t *testing.T) {
	dir := t.TempDir()
	modelfile := filepath.Join(dir, "Modelfile")
	if err := os.WriteFile(modelfile, []byte("FROM base\n"), 0o644); err != nil {
		t.Fatal(err)
	}

	cmd := &cobra.Command{}
	cmd.Flags().Bool("experimental", false, "")
	cmd.Flags().String("file", modelfile, "")
	cmd.Flags().String("draft-quantize", "mxfp8", "")
	cmd.SetContext(t.Context())

	err := CreateHandler(cmd, []string{"test-model"})
	if err == nil || !strings.Contains(err.Error(), "--draft-quantize requires a DRAFT model") {
		t.Fatalf("error = %v, want draft-quantize requires DRAFT", err)
	}
}

func TestResolveExperimentalLocalModelDir(t *testing.T) {
	dir := t.TempDir()
	modelfile := filepath.Join(dir, "Modelfile")
	modelDir := filepath.Join(dir, "model")
	if err := os.Mkdir(modelDir, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(modelDir, "config.json"), []byte(`{}`), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(modelDir, "model.safetensors"), []byte("dummy"), 0o644); err != nil {
		t.Fatal(err)
	}

	if got := resolveExperimentalLocalModelDir("gemma4", modelfile); got != "gemma4" {
		t.Fatalf("resolveExperimentalLocalModelDir(model name) = %q, want gemma4", got)
	}
	if got := resolveExperimentalLocalModelDir("./model", modelfile); got != modelDir {
		t.Fatalf("resolveExperimentalLocalModelDir(local dir) = %q, want %q", got, modelDir)
	}
}

func TestResolveExperimentalDraftDir(t *testing.T) {
	dir := t.TempDir()
	modelfile := filepath.Join(dir, "Modelfile")
	draftDir := filepath.Join(dir, "assistant")
	if err := os.Mkdir(draftDir, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(draftDir, "config.json"), []byte(`{}`), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(draftDir, "model.safetensors"), []byte("dummy"), 0o644); err != nil {
		t.Fatal(err)
	}

	got, err := resolveExperimentalDraftDir("./assistant", modelfile)
	if err != nil {
		t.Fatal(err)
	}
	if got != draftDir {
		t.Fatalf("resolveExperimentalDraftDir(local dir) = %q, want %q", got, draftDir)
	}

	_, err = resolveExperimentalDraftDir("assistant-model", modelfile)
	if err == nil || !strings.Contains(err.Error(), "DRAFT model references are not supported with --experimental yet") {
		t.Fatalf("error = %v, want unsupported draft model reference", err)
	}
}

func TestApplyShowResponseToRunOptions(t *testing.T) {
	opts := runOptions{}
	info := &api.ShowResponse{
		Details: api.ModelDetails{
			ParentModel: "parentmodel",
		},
		Messages: []api.Message{
			{Role: "assistant", Content: "loaded"},
		},
	}

	applyShowResponseToRunOptions(&opts, info)

	if opts.ParentModel != "parentmodel" {
		t.Fatalf("ParentModel = %q, want %q", opts.ParentModel, "parentmodel")
	}

	if !cmp.Equal(opts.LoadedMessages, info.Messages) {
		t.Fatalf("LoadedMessages = %#v, want %#v", opts.LoadedMessages, info.Messages)
	}

	info.Messages[0].Content = "modified"
	if opts.LoadedMessages[0].Content == "modified" {
		t.Fatal("LoadedMessages should be copied independently from ShowResponse")
	}
}

func TestRunOptions_Copy(t *testing.T) {
	// Setup test data
	originalKeepAlive := &api.Duration{Duration: 5 * time.Minute}
	originalThink := &api.ThinkValue{Value: "test reasoning"}

	original := runOptions{
		Model:          "test-model",
		ParentModel:    "parent-model",
		LoadedMessages: []api.Message{{Role: "assistant", Content: "loaded hello"}},
		Prompt:         "test prompt",
		Messages: []api.Message{
			{Role: "user", Content: "hello"},
			{Role: "assistant", Content: "hi there"},
		},
		Format: "json",
		System: "system prompt",
		Images: []api.ImageData{
			[]byte("image1"),
			[]byte("image2"),
		},
		Options: map[string]any{
			"temperature": 0.7,
			"max_tokens":  1000,
			"top_p":       0.9,
		},
		MultiModal:   true,
		KeepAlive:    originalKeepAlive,
		Think:        originalThink,
		HideThinking: false,
		ShowConnect:  true,
		Verbose:      true,
	}

	// Test the copy
	copied := original.Copy()

	// Test 1: Verify the copy is not the same instance
	if &copied == &original {
		t.Error("Copy should return a different instance")
	}

	// Test 2: Verify all fields are copied correctly
	tests := []struct {
		name string
		got  interface{}
		want interface{}
	}{
		{"Model", copied.Model, original.Model},
		{"ParentModel", copied.ParentModel, original.ParentModel},
		{"LoadedMessages", copied.LoadedMessages, original.LoadedMessages},
		{"Prompt", copied.Prompt, original.Prompt},
		{"Format", copied.Format, original.Format},
		{"System", copied.System, original.System},
		{"MultiModal", copied.MultiModal, original.MultiModal},
		{"HideThinking", copied.HideThinking, original.HideThinking},
		{"ShowConnect", copied.ShowConnect, original.ShowConnect},
		{"Verbose", copied.Verbose, original.Verbose},
	}

	for _, tt := range tests {
		if !reflect.DeepEqual(tt.got, tt.want) {
			t.Errorf("%s mismatch: got %v, want %v", tt.name, tt.got, tt.want)
		}
	}

	// Test 3: Verify Messages slice is deeply copied
	if len(copied.Messages) != len(original.Messages) {
		t.Errorf("Messages length mismatch: got %d, want %d", len(copied.Messages), len(original.Messages))
	}

	if len(copied.Messages) > 0 && &copied.Messages[0] == &original.Messages[0] {
		t.Error("Messages should be different instances")
	}

	// Modify original to verify independence
	if len(original.Messages) > 0 {
		originalContent := original.Messages[0].Content
		original.Messages[0].Content = "modified"
		if len(copied.Messages) > 0 && copied.Messages[0].Content == "modified" {
			t.Error("Messages should be independent after copy")
		}
		// Restore for other tests
		original.Messages[0].Content = originalContent
	}

	// Test 4: Verify Images slice is deeply copied
	if len(copied.Images) != len(original.Images) {
		t.Errorf("Images length mismatch: got %d, want %d", len(copied.Images), len(original.Images))
	}

	if len(copied.Images) > 0 && &copied.Images[0] == &original.Images[0] {
		t.Error("Images should be different instances")
	}

	// Modify original to verify independence
	if len(original.Images) > 0 {
		originalImage := original.Images[0]
		original.Images[0] = []byte("modified")
		if len(copied.Images) > 0 && string(copied.Images[0]) == "modified" {
			t.Error("Images should be independent after copy")
		}
		// Restore for other tests
		original.Images[0] = originalImage
	}

	// Test 5: Verify Options map is deeply copied
	if len(copied.Options) != len(original.Options) {
		t.Errorf("Options length mismatch: got %d, want %d", len(copied.Options), len(original.Options))
	}

	if len(copied.Options) > 0 && &copied.Options == &original.Options {
		t.Error("Options map should be different instances")
	}

	// Modify original to verify independence
	if len(original.Options) > 0 {
		originalTemp := original.Options["temperature"]
		original.Options["temperature"] = 0.9
		if copied.Options["temperature"] == 0.9 {
			t.Error("Options should be independent after copy")
		}
		// Restore for other tests
		original.Options["temperature"] = originalTemp
	}

	// Test 6: Verify KeepAlive pointer is copied (shallow copy)
	if copied.KeepAlive != original.KeepAlive {
		t.Error("KeepAlive pointer should be the same (shallow copy)")
	}

	// Test 7: Verify Think pointer creates a new instance
	if original.Think != nil && copied.Think == original.Think {
		t.Error("Think should be a different instance")
	}

	if original.Think != nil && copied.Think != nil {
		if !reflect.DeepEqual(copied.Think.Value, original.Think.Value) {
			t.Errorf("Think.Value mismatch: got %v, want %v", copied.Think.Value, original.Think.Value)
		}
	}

	// Test 8: Test with zero values
	zeroOriginal := runOptions{}
	zeroCopy := zeroOriginal.Copy()

	if !reflect.DeepEqual(zeroCopy, zeroOriginal) {
		fmt.Printf("orig: %#v\ncopy: %#v\n", zeroOriginal, zeroCopy)
		t.Error("Copy of zero value should equal original zero value")
	}
}

func TestAgentOptionsFromRunOptionsCopiesVerbose(t *testing.T) {
	opts := agentOptionsFromRunOptions(runOptions{Model: "llama3.2", Verbose: true})

	if !opts.Verbose {
		t.Fatal("Verbose should be copied into agent TUI options")
	}
}

func TestRunOptions_Copy_EmptySlicesAndMaps(t *testing.T) {
	// Test with empty slices and maps
	original := runOptions{
		LoadedMessages: []api.Message{},
		Messages:       []api.Message{},
		Images:         []api.ImageData{},
		Options:        map[string]any{},
	}

	copied := original.Copy()

	if copied.LoadedMessages == nil {
		t.Error("Empty LoadedMessages slice should remain empty, not nil")
	}

	if copied.Messages == nil {
		t.Error("Empty Messages slice should remain empty, not nil")
	}

	if copied.Images == nil {
		t.Error("Empty Images slice should remain empty, not nil")
	}

	if copied.Options == nil {
		t.Error("Empty Options map should remain empty, not nil")
	}

	if len(copied.Messages) != 0 {
		t.Error("Empty Messages slice should remain empty")
	}

	if len(copied.LoadedMessages) != 0 {
		t.Error("Empty LoadedMessages slice should remain empty")
	}

	if len(copied.Images) != 0 {
		t.Error("Empty Images slice should remain empty")
	}

	if len(copied.Options) != 0 {
		t.Error("Empty Options map should remain empty")
	}
}

func TestRunOptions_Copy_NilPointers(t *testing.T) {
	// Test with nil pointers
	original := runOptions{
		KeepAlive: nil,
		Think:     nil,
	}

	copied := original.Copy()

	if copied.KeepAlive != nil {
		t.Error("Nil KeepAlive should remain nil")
	}

	if copied.Think != nil {
		t.Error("Nil Think should remain nil")
	}
}

func TestRunOptions_Copy_ThinkValueVariants(t *testing.T) {
	tests := []struct {
		name  string
		think *api.ThinkValue
	}{
		{"nil Think", nil},
		{"bool true", &api.ThinkValue{Value: true}},
		{"bool false", &api.ThinkValue{Value: false}},
		{"string value", &api.ThinkValue{Value: "reasoning text"}},
		{"int value", &api.ThinkValue{Value: 42}},
		{"nil value", &api.ThinkValue{Value: nil}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			original := runOptions{Think: tt.think}
			copied := original.Copy()

			if tt.think == nil {
				if copied.Think != nil {
					t.Error("Nil Think should remain nil")
				}
				return
			}

			if copied.Think == nil {
				t.Error("Non-nil Think should not become nil")
				return
			}

			if copied.Think == original.Think {
				t.Error("Think should be a different instance")
			}

			if !reflect.DeepEqual(copied.Think.Value, original.Think.Value) {
				t.Errorf("Think.Value mismatch: got %v, want %v", copied.Think.Value, original.Think.Value)
			}
		})
	}
}

func TestShowInfoImageGen(t *testing.T) {
	var b bytes.Buffer
	err := showInfo(&api.ShowResponse{
		Details: api.ModelDetails{
			Family:            "ZImagePipeline",
			ParameterSize:     "10.3B",
			QuantizationLevel: "Q8",
		},
		Capabilities: []model.Capability{model.CapabilityImage},
		Requires:     "0.19.0",
	}, false, &b)
	if err != nil {
		t.Fatal(err)
	}

	expect := "  Model\n" +
		"    architecture    ZImagePipeline    \n" +
		"    parameters      10.3B             \n" +
		"    quantization    Q8                \n" +
		"    requires        0.19.0            \n" +
		"\n" +
		"  Capabilities\n" +
		"    image    \n" +
		"\n"
	if diff := cmp.Diff(expect, b.String()); diff != "" {
		t.Errorf("unexpected output (-want +got):\n%s", diff)
	}
}

func TestPushProgressMessage(t *testing.T) {
	tests := []struct {
		name    string
		status  string
		digest  string
		wantMsg string
	}{
		{
			name:    "uses status when provided",
			status:  "uploading model",
			digest:  "sha256:abc123456789def",
			wantMsg: "uploading model",
		},
		{
			name:    "falls back to digest when status empty",
			status:  "",
			digest:  "sha256:abc123456789def",
			wantMsg: "pushing abc123456789...",
		},
		{
			name:    "handles short digest gracefully",
			status:  "",
			digest:  "sha256:abc",
			wantMsg: "pushing sha256:abc...",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			msg := tt.status
			if msg == "" {
				if len(tt.digest) >= 19 {
					msg = fmt.Sprintf("pushing %s...", tt.digest[7:19])
				} else {
					msg = fmt.Sprintf("pushing %s...", tt.digest)
				}
			}
			if msg != tt.wantMsg {
				t.Errorf("got %q, want %q", msg, tt.wantMsg)
			}
		})
	}
}

func TestRunOptions_Copy_Independence(t *testing.T) {
	// Test that modifications to original don't affect copy
	originalThink := &api.ThinkValue{Value: "original"}
	original := runOptions{
		Model:          "original-model",
		LoadedMessages: []api.Message{{Role: "assistant", Content: "loaded"}},
		Messages:       []api.Message{{Role: "user", Content: "original"}},
		Options:        map[string]any{"key": "value"},
		Think:          originalThink,
	}

	copied := original.Copy()

	// Modify original
	original.Model = "modified-model"
	if len(original.LoadedMessages) > 0 {
		original.LoadedMessages[0].Content = "modified loaded"
	}
	if len(original.Messages) > 0 {
		original.Messages[0].Content = "modified"
	}
	original.Options["key"] = "modified"
	if original.Think != nil {
		original.Think.Value = "modified"
	}

	// Verify copy is unchanged
	if copied.Model == "modified-model" {
		t.Error("Copy Model should not be affected by original modification")
	}

	if len(copied.LoadedMessages) > 0 && copied.LoadedMessages[0].Content == "modified loaded" {
		t.Error("Copy LoadedMessages should not be affected by original modification")
	}

	if len(copied.Messages) > 0 && copied.Messages[0].Content == "modified" {
		t.Error("Copy Messages should not be affected by original modification")
	}

	if copied.Options["key"] == "modified" {
		t.Error("Copy Options should not be affected by original modification")
	}

	if copied.Think != nil && copied.Think.Value == "modified" {
		t.Error("Copy Think should not be affected by original modification")
	}
}

func TestLoadOrUnloadModel_CloudModelAuth(t *testing.T) {
	tests := []struct {
		name            string
		model           string
		showStatus      int
		remoteHost      string
		remoteModel     string
		whoamiStatus    int
		whoamiResp      any
		expectWhoami    bool
		expectedError   string
		expectAuthError bool
	}{
		{
			name:         "ollama.com cloud model - user signed in",
			model:        "test-cloud-model",
			remoteHost:   "https://ollama.com",
			remoteModel:  "test-model",
			whoamiStatus: http.StatusOK,
			whoamiResp:   api.UserResponse{Name: "testuser"},
			expectWhoami: true,
		},
		{
			name:         "ollama.com cloud model - user not signed in",
			model:        "test-cloud-model",
			remoteHost:   "https://ollama.com",
			remoteModel:  "test-model",
			whoamiStatus: http.StatusUnauthorized,
			whoamiResp: map[string]string{
				"error":      "unauthorized",
				"signin_url": "https://ollama.com/signin",
			},
			expectWhoami:    true,
			expectedError:   "unauthorized",
			expectAuthError: true,
		},
		{
			name:         "non-ollama.com remote - no auth check",
			model:        "test-cloud-model",
			remoteHost:   "https://other-remote.com",
			remoteModel:  "test-model",
			whoamiStatus: http.StatusUnauthorized, // should not be called
			whoamiResp:   nil,
		},
		{
			name:         "explicit :cloud model - auth check without remote metadata",
			model:        "kimi-k2.5:cloud",
			remoteHost:   "",
			remoteModel:  "",
			whoamiStatus: http.StatusOK,
			whoamiResp:   api.UserResponse{Name: "testuser"},
			expectWhoami: true,
		},
		{
			name:            "explicit :cloud model without local stub returns not found by default",
			model:           "minimax-m2.7:cloud",
			showStatus:      http.StatusNotFound,
			whoamiStatus:    http.StatusOK,
			whoamiResp:      api.UserResponse{Name: "testuser"},
			expectedError:   "not found",
			expectWhoami:    false,
			expectAuthError: false,
		},
		{
			name:         "explicit -cloud model - auth check without remote metadata",
			model:        "kimi-k2.5:latest-cloud",
			remoteHost:   "",
			remoteModel:  "",
			whoamiStatus: http.StatusOK,
			whoamiResp:   api.UserResponse{Name: "testuser"},
			expectWhoami: true,
		},
		{
			name:         "dash cloud-like name without explicit source does not require auth",
			model:        "test-cloud-model",
			remoteHost:   "",
			remoteModel:  "",
			whoamiStatus: http.StatusUnauthorized, // should not be called
			whoamiResp:   nil,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			whoamiCalled := false
			mockServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				switch r.URL.Path {
				case "/api/show":
					if tt.showStatus != 0 && tt.showStatus != http.StatusOK {
						w.WriteHeader(tt.showStatus)
						_ = json.NewEncoder(w).Encode(map[string]string{"error": "not found"})
						return
					}
					w.Header().Set("Content-Type", "application/json")
					if err := json.NewEncoder(w).Encode(api.ShowResponse{
						RemoteHost:  tt.remoteHost,
						RemoteModel: tt.remoteModel,
					}); err != nil {
						http.Error(w, err.Error(), http.StatusInternalServerError)
					}
				case "/api/me":
					whoamiCalled = true
					w.Header().Set("Content-Type", "application/json")
					w.WriteHeader(tt.whoamiStatus)
					if tt.whoamiResp != nil {
						if err := json.NewEncoder(w).Encode(tt.whoamiResp); err != nil {
							http.Error(w, err.Error(), http.StatusInternalServerError)
						}
					}
				case "/api/generate":
					w.WriteHeader(http.StatusOK)
				default:
					http.NotFound(w, r)
				}
			}))
			defer mockServer.Close()

			t.Setenv("OLLAMA_HOST", mockServer.URL)

			cmd := &cobra.Command{}
			cmd.SetContext(t.Context())

			opts := &runOptions{
				Model:       tt.model,
				ShowConnect: false,
			}

			err := loadOrUnloadModel(cmd, opts)

			if whoamiCalled != tt.expectWhoami {
				t.Errorf("whoami called = %v, want %v", whoamiCalled, tt.expectWhoami)
			}

			if tt.expectedError != "" {
				if err == nil {
					t.Errorf("expected error containing %q, got nil", tt.expectedError)
				} else {
					if !tt.expectAuthError && !strings.Contains(strings.ToLower(err.Error()), strings.ToLower(tt.expectedError)) {
						t.Errorf("expected error containing %q, got %v", tt.expectedError, err)
					}
					if tt.expectAuthError {
						var authErr api.AuthorizationError
						if !errors.As(err, &authErr) {
							t.Errorf("expected AuthorizationError, got %T: %v", err, err)
						}
					}
				}
			} else {
				if err != nil {
					t.Errorf("expected no error, got %v", err)
				}
			}
		})
	}
}

func TestLoadOrUnloadModel_CloudModelDoesNotPrintConnectBanner(t *testing.T) {
	mockServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/api/show":
			w.Header().Set("Content-Type", "application/json")
			_ = json.NewEncoder(w).Encode(api.ShowResponse{
				RemoteHost:  "https://ollama.com",
				RemoteModel: "minimax-m3:cloud",
			})
		case "/api/me":
			w.Header().Set("Content-Type", "application/json")
			_ = json.NewEncoder(w).Encode(api.UserResponse{Name: "testuser"})
		default:
			http.NotFound(w, r)
		}
	}))
	defer mockServer.Close()
	t.Setenv("OLLAMA_HOST", mockServer.URL)

	cmd := &cobra.Command{}
	cmd.SetContext(t.Context())

	oldStderr := os.Stderr
	readErr, writeErr, err := os.Pipe()
	if err != nil {
		t.Fatal(err)
	}
	os.Stderr = writeErr
	t.Cleanup(func() { os.Stderr = oldStderr })

	err = loadOrUnloadModel(cmd, &runOptions{Model: "minimax-m3:cloud", ShowConnect: true})
	_ = writeErr.Close()
	if err != nil {
		t.Fatal(err)
	}
	var stderr bytes.Buffer
	if _, err := io.Copy(&stderr, readErr); err != nil {
		t.Fatal(err)
	}
	if strings.Contains(stderr.String(), "Connecting to") || strings.Contains(stderr.String(), "ollama.com") {
		t.Fatalf("unexpected cloud connect banner: %q", stderr.String())
	}
}

func TestIsLocalhost(t *testing.T) {
	tests := []struct {
		name     string
		host     string
		expected bool
	}{
		{"default empty", "", true},
		{"localhost no port", "localhost", true},
		{"localhost with port", "localhost:11435", true},
		{"127.0.0.1 no port", "127.0.0.1", true},
		{"127.0.0.1 with port", "127.0.0.1:11434", true},
		{"0.0.0.0 no port", "0.0.0.0", true},
		{"0.0.0.0 with port", "0.0.0.0:11434", true},
		{"::1 no port", "::1", true},
		{"[::1] with port", "[::1]:11434", true},
		{"loopback with scheme", "http://localhost:11434", true},
		{"remote hostname", "example.com", false},
		{"remote hostname with port", "example.com:11434", false},
		{"remote IP", "192.168.1.1", false},
		{"remote IP with port", "192.168.1.1:11434", false},
		{"remote with scheme", "http://example.com:11434", false},
		{"https remote", "https://example.com:443", false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Setenv("OLLAMA_HOST", tt.host)
			got := isLocalhost()
			if got != tt.expected {
				t.Errorf("isLocalhost() with OLLAMA_HOST=%q = %v, want %v", tt.host, got, tt.expected)
			}
		})
	}
}
