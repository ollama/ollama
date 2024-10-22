package cmd

import (
	"bytes"
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/google/go-cmp/cmp"
	"github.com/spf13/cobra"

	"github.com/ollama/ollama/api"
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
		}, &b); err != nil {
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
		}, &b); err != nil {
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
		}, &b); err != nil {
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
		}, &b); err != nil {
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
		}, &b); err != nil {
			t.Fatal(err)
		}

		expect := `  Model
    architecture    test    
    parameters      7B      
    quantization    FP16    

  System
    You are a pirate!    
    Ahoy, matey!         

`
		if diff := cmp.Diff(expect, b.String()); diff != "" {
			t.Errorf("unexpected output (-want +got):\n%s", diff)
		}
	})

	t.Run("license", func(t *testing.T) {
		var b bytes.Buffer
		license, err := os.ReadFile(filepath.Join("..", "LICENSE"))
		if err != nil {
			t.Fatal(err)
		}

		if err := showInfo(&api.ShowResponse{
			Details: api.ModelDetails{
				Family:            "test",
				ParameterSize:     "7B",
				QuantizationLevel: "FP16",
			},
			License: string(license),
		}, &b); err != nil {
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
	cmd.SetContext(context.TODO())
	if err := DeleteHandler(cmd, []string{"test-model"}); err != nil {
		t.Fatalf("DeleteHandler failed: %v", err)
	}
	if !stopped {
		t.Fatal("Model was not stopped before deletion")
	}

	err := DeleteHandler(cmd, []string{"test-model-not-found"})
	if err == nil || !strings.Contains(err.Error(), "unable to stop existing running model \"test-model-not-found\"") {
		t.Fatalf("DeleteHandler failed: expected error about stopping non-existent model, got %v", err)
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
			expectedName:  "crazyfile",
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
				tempDir, err := os.MkdirTemp("", "modelfiledir")
				defer os.RemoveAll(tempDir)
				if err != nil {
					t.Fatalf("temp modelfile dir creation failed: %v", err)
				}
				var fn string
				if tt.modelfileName != "" {
					fn = tt.modelfileName
				} else {
					fn = "Modelfile"
				}

				tempFile, err := os.CreateTemp(tempDir, fn)
				if err != nil {
					t.Fatalf("temp modelfile creation failed: %v", err)
				}

				expectedFilename = tempFile.Name()
				err = cmd.Flags().Set("file", expectedFilename)
				if err != nil {
					t.Fatalf("couldn't set file flag: %v", err)
				}
			} else {
				if tt.modelfileName != "" {
					expectedFilename = tt.modelfileName
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
