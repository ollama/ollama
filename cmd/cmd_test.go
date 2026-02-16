package cmd

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	"github.com/spf13/cobra"

	"github.com/ollama/ollama/api"
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
			Requires: "0.14.0",
		}, false, &b); err != nil {
			t.Fatal(err)
		}

		expect := `  Model
    architecture    test      
    parameters      7B        
    quantization    FP16      
    requires        0.14.0    

`
		if diff := cmp.Diff(expect, b.String()); diff != "" {
			t.Errorf("unexpected output (-want +got):\n%s", diff)
		}
	})
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
	t.Cleanup(mockServer.Close)

	cmd := &cobra.Command{}
	cmd.SetContext(t.Context())
	cmd.Flags().String("keepalive", "", "")
	cmd.Flags().Bool("truncate", false, "")
	cmd.Flags().Int("dimensions", 0, "")
	cmd.Flags().Bool("verbose", false, "")
	cmd.Flags().Bool("insecure", false, "")
	cmd.Flags().Bool("nowordwrap", false, "")
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
	cmd.Flags().Bool("nowordwrap", false, "")
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
	cmd.Flags().Bool("nowordwrap", false, "")
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
	cmd.Flags().Bool("nowordwrap", false, "")
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

func TestNewCreateRequest(t *testing.T) {
	tests := []struct {
		name     string
		from     string
		opts     runOptions
		expected *api.CreateRequest
	}{
		{
			"basic test",
			"newmodel",
			runOptions{
				Model:       "mymodel",
				ParentModel: "",
				Prompt:      "You are a fun AI agent",
				Messages:    []api.Message{},
				WordWrap:    true,
			},
			&api.CreateRequest{
				From:  "mymodel",
				Model: "newmodel",
			},
		},
		{
			"parent model test",
			"newmodel",
			runOptions{
				Model:       "mymodel",
				ParentModel: "parentmodel",
				Messages:    []api.Message{},
				WordWrap:    true,
			},
			&api.CreateRequest{
				From:  "parentmodel",
				Model: "newmodel",
			},
		},
		{
			"parent model as filepath test",
			"newmodel",
			runOptions{
				Model:       "mymodel",
				ParentModel: "/some/file/like/etc/passwd",
				Messages:    []api.Message{},
				WordWrap:    true,
			},
			&api.CreateRequest{
				From:  "mymodel",
				Model: "newmodel",
			},
		},
		{
			"parent model as windows filepath test",
			"newmodel",
			runOptions{
				Model:       "mymodel",
				ParentModel: "D:\\some\\file\\like\\etc\\passwd",
				Messages:    []api.Message{},
				WordWrap:    true,
			},
			&api.CreateRequest{
				From:  "mymodel",
				Model: "newmodel",
			},
		},
		{
			"options test",
			"newmodel",
			runOptions{
				Model:       "mymodel",
				ParentModel: "parentmodel",
				Options: map[string]any{
					"temperature": 1.0,
				},
			},
			&api.CreateRequest{
				From:  "parentmodel",
				Model: "newmodel",
				Parameters: map[string]any{
					"temperature": 1.0,
				},
			},
		},
		{
			"messages test",
			"newmodel",
			runOptions{
				Model:       "mymodel",
				ParentModel: "parentmodel",
				System:      "You are a fun AI agent",
				Messages: []api.Message{
					{
						Role:    "user",
						Content: "hello there!",
					},
					{
						Role:    "assistant",
						Content: "hello to you!",
					},
				},
				WordWrap: true,
			},
			&api.CreateRequest{
				From:   "parentmodel",
				Model:  "newmodel",
				System: "You are a fun AI agent",
				Messages: []api.Message{
					{
						Role:    "user",
						Content: "hello there!",
					},
					{
						Role:    "assistant",
						Content: "hello to you!",
					},
				},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			actual := NewCreateRequest(tt.from, tt.opts)
			if !cmp.Equal(actual, tt.expected) {
				t.Errorf("expected output %#v, got %#v", tt.expected, actual)
			}
		})
	}
}

func TestRunOptions_Copy(t *testing.T) {
	// Setup test data
	originalKeepAlive := &api.Duration{Duration: 5 * time.Minute}
	originalThink := &api.ThinkValue{Value: "test reasoning"}

	original := runOptions{
		Model:       "test-model",
		ParentModel: "parent-model",
		Prompt:      "test prompt",
		Messages: []api.Message{
			{Role: "user", Content: "hello"},
			{Role: "assistant", Content: "hi there"},
		},
		WordWrap: true,
		Format:   "json",
		System:   "system prompt",
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
		{"Prompt", copied.Prompt, original.Prompt},
		{"WordWrap", copied.WordWrap, original.WordWrap},
		{"Format", copied.Format, original.Format},
		{"System", copied.System, original.System},
		{"MultiModal", copied.MultiModal, original.MultiModal},
		{"HideThinking", copied.HideThinking, original.HideThinking},
		{"ShowConnect", copied.ShowConnect, original.ShowConnect},
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

func TestRunOptions_Copy_EmptySlicesAndMaps(t *testing.T) {
	// Test with empty slices and maps
	original := runOptions{
		Messages: []api.Message{},
		Images:   []api.ImageData{},
		Options:  map[string]any{},
	}

	copied := original.Copy()

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
			QuantizationLevel: "FP8",
		},
		Capabilities: []model.Capability{model.CapabilityImage},
		Requires:     "0.14.0",
	}, false, &b)
	if err != nil {
		t.Fatal(err)
	}

	expect := "  Model\n" +
		"    architecture    ZImagePipeline    \n" +
		"    parameters      10.3B             \n" +
		"    quantization    FP8               \n" +
		"    requires        0.14.0            \n" +
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
		Model:    "original-model",
		Messages: []api.Message{{Role: "user", Content: "original"}},
		Options:  map[string]any{"key": "value"},
		Think:    originalThink,
	}

	copied := original.Copy()

	// Modify original
	original.Model = "modified-model"
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
