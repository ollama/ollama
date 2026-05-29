package server

import (
	"crypto/sha256"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/ollama/ollama/fs/ggml"
	"github.com/ollama/ollama/manifest"
	"github.com/ollama/ollama/template"
	"github.com/ollama/ollama/types/model"
)

func TestPruneLayersSkipsRecentOrphans(t *testing.T) {
	t.Setenv("OLLAMA_MODELS", t.TempDir())

	recentDigest := "sha256:0000000000000000000000000000000000000000000000000000000000000001"
	oldDigest := "sha256:0000000000000000000000000000000000000000000000000000000000000002"

	for _, digest := range []string{recentDigest, oldDigest} {
		p, err := manifest.BlobsPath(digest)
		if err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(p, nil, 0o644); err != nil {
			t.Fatal(err)
		}
	}

	oldPath, err := manifest.BlobsPath(oldDigest)
	if err != nil {
		t.Fatal(err)
	}
	oldTime := time.Now().Add(-layerPruneGracePeriod - time.Hour)
	if err := os.Chtimes(oldPath, oldTime, oldTime); err != nil {
		t.Fatal(err)
	}

	if err := PruneLayers(); err != nil {
		t.Fatal(err)
	}

	recentPath, err := manifest.BlobsPath(recentDigest)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(recentPath); err != nil {
		t.Fatalf("recent orphan was pruned: %v", err)
	}
	if _, err := os.Stat(oldPath); !os.IsNotExist(err) {
		t.Fatalf("old orphan still exists: %v", err)
	}
}

func TestGetModelTemplateMetadata(t *testing.T) {
	customTemplate := "CUSTOM {{ .Prompt }}"

	t.Run("records chat template and Go TEMPLATE layer", func(t *testing.T) {
		t.Setenv("OLLAMA_MODELS", t.TempDir())
		t.Setenv("OLLAMA_GO_TEMPLATE", "")

		_, digest := createBinFile(t, ggml.KV{
			"general.architecture":    "llama",
			"tokenizer.chat_template": "{{ bos_token }}{{ messages[0]['content'] }}",
		}, nil)
		writeTestModelManifest(t, "template-disabled", digest, customTemplate)

		m, err := GetModel("template-disabled")
		if err != nil {
			t.Fatal(err)
		}
		if !m.HasChatTemplate {
			t.Fatal("expected GGUF chat template to be detected")
		}
		if !m.HasGoTemplate {
			t.Fatal("expected Go TEMPLATE layer to be detected")
		}
		if got := m.Template.String(); got != customTemplate {
			t.Fatalf("template = %q, want %q", got, customTemplate)
		}
	})

	t.Run("prefers chat template when Go TEMPLATE has fewer capabilities", func(t *testing.T) {
		t.Setenv("OLLAMA_MODELS", t.TempDir())
		t.Setenv("OLLAMA_GO_TEMPLATE", "")

		_, digest := createBinFile(t, ggml.KV{
			"general.architecture":    "llama",
			"tokenizer.chat_template": "{% if tools %}{{ tools }}{% endif %}{{ messages[0]['content'] }}",
		}, nil)
		writeTestModelManifest(t, "chat-template-tools", digest, customTemplate)

		m, err := GetModel("chat-template-tools")
		if err != nil {
			t.Fatal(err)
		}
		if !m.PreferChatTemplate {
			t.Fatal("expected chat template to be preferred")
		}
		if got := m.CheckCapabilities(model.CapabilityTools); got != nil {
			t.Fatalf("expected tools capability, got %v", got)
		}
	})

	t.Run("prefers Qwen chat template with tools and inferred thinking", func(t *testing.T) {
		t.Setenv("OLLAMA_MODELS", t.TempDir())
		t.Setenv("OLLAMA_GO_TEMPLATE", "")

		_, digest := createBinFile(t, ggml.KV{
			"general.architecture":    "llama",
			"tokenizer.chat_template": "{% if tools %}{{ tools }}{% endif %}{% set content = (content.split('</think>')|last) %}",
		}, nil)
		writeTestModelManifest(t, "chat-template-tools-thinking", digest, "{{ range .Messages }}{{ if .Thinking }}<think>{{ .Thinking }}</think>{{ end }}{{ .Content }}{{ end }}")

		m, err := GetModel("chat-template-tools-thinking")
		if err != nil {
			t.Fatal(err)
		}
		if !m.PreferChatTemplate {
			t.Fatal("expected chat template to be preferred")
		}
		if got := m.CheckCapabilities(model.CapabilityTools); got != nil {
			t.Fatalf("expected tools capability, got %v", got)
		}
		if got := m.CheckCapabilities(model.CapabilityThinking); got != nil {
			t.Fatalf("expected thinking capability, got %v", got)
		}
	})

	t.Run("keeps Go TEMPLATE when chat template has weaker tool support", func(t *testing.T) {
		t.Setenv("OLLAMA_MODELS", t.TempDir())
		t.Setenv("OLLAMA_GO_TEMPLATE", "")

		_, digest := createBinFile(t, ggml.KV{
			"general.architecture": "llama",
			"tokenizer.chat_template": `{%- if tools and not available_tools -%}
{{- set available_tools = tools -}}
{%- endif -%}
{%- if available_tools -%}
{{ '<|start_of_role|>available_tools<|end_of_role|>' }}{{ available_tools | tojson }}{{ '<|end_of_text|>' }}
{%- endif -%}
{%- if thinking -%}<think></think><response></response>{%- endif -%}
{%- for message in messages -%}
{{ '<|start_of_role|>' + message['role'] + '<|end_of_role|>' + message['content'] + '<|end_of_text|>' }}
{%- endfor -%}`,
		}, nil)
		writeTestModelManifest(t, "chat-template-weaker-tools", digest, `{{ if .Tools }}tools{{ end }}
{{ range .Messages }}
{{ if eq .Role "tool" }}tool_response{{ else }}{{ .Role }}{{ end }}
{{ if .ToolCalls }}<|tool_call|>{{ range .ToolCalls }}{{ .Function.Name }}{{ end }}{{ else }}{{ .Content }}{{ end }}
{{ end }}`)

		m, err := GetModel("chat-template-weaker-tools")
		if err != nil {
			t.Fatal(err)
		}
		if m.PreferChatTemplate {
			t.Fatal("expected Go TEMPLATE to be preferred")
		}
		if got := m.CheckCapabilities(model.CapabilityTools); got != nil {
			t.Fatalf("expected tools capability, got %v", got)
		}
		if got := m.CheckCapabilities(model.CapabilityThinking); got == nil {
			t.Fatal("expected thinking capability to remain unavailable on Go TEMPLATE path")
		}
	})

	t.Run("respects explicit Go TEMPLATE enablement", func(t *testing.T) {
		t.Setenv("OLLAMA_MODELS", t.TempDir())
		t.Setenv("OLLAMA_GO_TEMPLATE", "1")

		_, digest := createBinFile(t, ggml.KV{
			"general.architecture":    "llama",
			"tokenizer.chat_template": "{% if tools %}{{ tools }}{% endif %}{{ messages[0]['content'] }}",
		}, nil)
		writeTestModelManifest(t, "go-template-forced", digest, customTemplate)

		m, err := GetModel("go-template-forced")
		if err != nil {
			t.Fatal(err)
		}
		if m.PreferChatTemplate {
			t.Fatal("expected explicit Go TEMPLATE setting to suppress chat_template preference")
		}
		if got := m.CheckCapabilities(model.CapabilityTools); got == nil {
			t.Fatal("expected tools capability to be unavailable when Go TEMPLATE is explicitly enabled")
		}
	})

	t.Run("respects explicit Go TEMPLATE disablement", func(t *testing.T) {
		t.Setenv("OLLAMA_MODELS", t.TempDir())
		t.Setenv("OLLAMA_GO_TEMPLATE", "0")

		_, digest := createBinFile(t, ggml.KV{
			"general.architecture":    "llama",
			"tokenizer.chat_template": "{% if tools %}{{ tools }}{% endif %}{{ messages[0]['content'] }}",
		}, nil)
		writeTestModelManifest(t, "go-template-disabled", digest, customTemplate)

		m, err := GetModel("go-template-disabled")
		if err != nil {
			t.Fatal(err)
		}
		if m.PreferChatTemplate {
			t.Fatal("expected explicit Go TEMPLATE setting to suppress chat_template preference")
		}
		if got := m.CheckCapabilities(model.CapabilityTools); got != nil {
			t.Fatalf("expected tools capability from GGUF chat_template, got %v", got)
		}
	})

	t.Run("records missing chat template", func(t *testing.T) {
		t.Setenv("OLLAMA_MODELS", t.TempDir())
		t.Setenv("OLLAMA_GO_TEMPLATE", "")

		_, digest := createBinFile(t, ggml.KV{
			"general.architecture": "llama",
		}, nil)
		writeTestModelManifest(t, "missing-chat-template", digest, customTemplate)

		m, err := GetModel("missing-chat-template")
		if err != nil {
			t.Fatal(err)
		}
		if m.HasChatTemplate {
			t.Fatal("expected missing GGUF chat template")
		}
		if !m.HasGoTemplate {
			t.Fatal("expected Go TEMPLATE layer to be detected")
		}
	})
}

func writeTestModelManifest(t *testing.T, name, digest, tmpl string) {
	t.Helper()

	modelLayer, err := manifest.NewLayerFromLayer(digest, "application/vnd.ollama.image.model", "")
	if err != nil {
		t.Fatal(err)
	}
	templateLayer, err := manifest.NewLayer(strings.NewReader(tmpl), "application/vnd.ollama.image.template")
	if err != nil {
		t.Fatal(err)
	}

	layers := []manifest.Layer{modelLayer, templateLayer}
	configLayer, err := createConfigLayer(layers, model.ConfigV2{
		ModelFormat:   "gguf",
		ModelFamily:   "llama",
		ModelFamilies: []string{"llama"},
	})
	if err != nil {
		t.Fatal(err)
	}
	if err := manifest.WriteManifest(model.ParseName(name), *configLayer, layers); err != nil {
		t.Fatal(err)
	}
}

func TestModelCapabilities(t *testing.T) {
	// Create completion model (llama architecture without vision)
	completionModelPath, _ := createBinFile(t, ggml.KV{
		"general.architecture": "llama",
	}, []*ggml.Tensor{})

	ggufToolTemplateModelPath, _ := createBinFile(t, ggml.KV{
		"general.architecture":    "llama",
		"tokenizer.chat_template": `{% if tools %}<tool_call>{{ tools }}</tool_call>{% endif %}<think>{{ messages[0]['content'] }}</think>`,
	}, []*ggml.Tensor{})

	// Create vision model (llama architecture with vision block count)
	visionModelPath, _ := createBinFile(t, ggml.KV{
		"general.architecture":     "llama",
		"llama.vision.block_count": uint32(1),
	}, []*ggml.Tensor{})

	// Create embedding model (bert architecture with pooling type)
	embeddingModelPath, _ := createBinFile(t, ggml.KV{
		"general.architecture": "bert",
		"bert.pooling_type":    uint32(1),
	}, []*ggml.Tensor{})

	audioProjectorPath, _ := createBinFile(t, ggml.KV{
		"general.architecture":    "clip",
		"clip.has_audio_encoder":  true,
		"vision.projector_type":   "pixtral",
		"clip.vision.block_count": uint32(1),
	}, []*ggml.Tensor{})

	nemotronOmniModelPath, _ := createBinFile(t, ggml.KV{
		"general.architecture":                 "nemotron_h_omni",
		"nemotron_h_omni.vision.block_count":   uint32(1),
		"nemotron_h_omni.audio.block_count":    uint32(1),
		"nemotron_h_omni.embedding_length":     uint32(1),
		"nemotron_h_omni.attention.head_count": uint32(1),
	}, []*ggml.Tensor{})

	suppressedAudioProjectorPath, _ := createBinFile(t, ggml.KV{
		"general.architecture":    "clip",
		"clip.has_audio_encoder":  true,
		"vision.projector_type":   "gemma4v",
		"clip.vision.block_count": uint32(1),
	}, []*ggml.Tensor{})

	toolsInsertTemplate, err := template.Parse("{{ .prompt }}{{ if .tools }}{{ .tools }}{{ end }}{{ if .suffix }}{{ .suffix }}{{ end }}")
	if err != nil {
		t.Fatalf("Failed to parse template: %v", err)
	}

	chatTemplate, err := template.Parse("{{ .prompt }}")
	if err != nil {
		t.Fatalf("Failed to parse template: %v", err)
	}

	toolsTemplate, err := template.Parse("{{ .prompt }}{{ if .tools }}{{ .tools }}{{ end }}")
	if err != nil {
		t.Fatalf("Failed to parse template: %v", err)
	}

	testModels := []struct {
		name         string
		model        Model
		expectedCaps []model.Capability
	}{
		{
			name: "model with image generation capability via config",
			model: Model{
				Config: model.ConfigV2{
					Capabilities: []string{"image"},
				},
			},
			expectedCaps: []model.Capability{model.CapabilityImage},
		},
		{
			name: "model with image and vision capability (image editing)",
			model: Model{
				Config: model.ConfigV2{
					Capabilities: []string{"image", "vision"},
				},
			},
			expectedCaps: []model.Capability{model.CapabilityImage, model.CapabilityVision},
		},
		{
			name: "model with completion capability",
			model: Model{
				ModelPath: completionModelPath,
				Template:  chatTemplate,
			},
			expectedCaps: []model.Capability{model.CapabilityCompletion},
		},

		{
			name: "model with completion, tools, and insert capability",
			model: Model{
				ModelPath: completionModelPath,
				Template:  toolsInsertTemplate,
			},
			expectedCaps: []model.Capability{model.CapabilityCompletion, model.CapabilityTools, model.CapabilityInsert},
		},
		{
			name: "model with tools capability",
			model: Model{
				ModelPath: completionModelPath,
				Template:  toolsTemplate,
			},
			expectedCaps: []model.Capability{model.CapabilityCompletion, model.CapabilityTools},
		},
		{
			name: "model with GGUF chat_template tools and thinking",
			model: Model{
				ModelPath: ggufToolTemplateModelPath,
			},
			expectedCaps: []model.Capability{model.CapabilityCompletion, model.CapabilityTools, model.CapabilityThinking},
		},
		{
			name: "model with Go TEMPLATE ignores GGUF chat_template capabilities",
			model: Model{
				ModelPath:       ggufToolTemplateModelPath,
				Template:        chatTemplate,
				HasGoTemplate:   true,
				HasChatTemplate: true,
			},
			expectedCaps: []model.Capability{model.CapabilityCompletion},
		},
		{
			name: "model with tools capability from config and parser",
			model: Model{
				Config: model.ConfigV2{
					Capabilities: []string{"completion", "tools"},
					Parser:       "qwen3-coder",
				},
				Template: chatTemplate,
			},
			expectedCaps: []model.Capability{model.CapabilityCompletion, model.CapabilityTools},
		},
		{
			name: "model with vision capability",
			model: Model{
				ModelPath: visionModelPath,
				Template:  chatTemplate,
			},
			expectedCaps: []model.Capability{model.CapabilityCompletion, model.CapabilityVision},
		},
		{
			name: "model with vision, tools, and insert capability",
			model: Model{
				ModelPath: visionModelPath,
				Template:  toolsInsertTemplate,
			},
			expectedCaps: []model.Capability{model.CapabilityCompletion, model.CapabilityVision, model.CapabilityTools, model.CapabilityInsert},
		},
		{
			name: "model with embedding capability",
			model: Model{
				ModelPath: embeddingModelPath,
				Template:  chatTemplate,
			},
			expectedCaps: []model.Capability{model.CapabilityEmbedding},
		},
		{
			name: "model with audio projector capability",
			model: Model{
				ModelPath:      completionModelPath,
				ProjectorPaths: []string{audioProjectorPath},
				Template:       chatTemplate,
			},
			expectedCaps: []model.Capability{model.CapabilityCompletion, model.CapabilityVision, model.CapabilityAudio},
		},
		{
			name: "model with parser and projector capabilities without template",
			model: Model{
				ModelPath:      completionModelPath,
				ProjectorPaths: []string{audioProjectorPath},
				Config: model.ConfigV2{
					Parser: "functiongemma",
				},
			},
			expectedCaps: []model.Capability{model.CapabilityCompletion, model.CapabilityVision, model.CapabilityAudio, model.CapabilityTools},
		},
		{
			name: "gemma4 projector exposes audio capability",
			model: Model{
				ModelPath:      completionModelPath,
				ProjectorPaths: []string{suppressedAudioProjectorPath},
				Template:       chatTemplate,
			},
			expectedCaps: []model.Capability{model.CapabilityCompletion, model.CapabilityVision, model.CapabilityAudio},
		},
		{
			name: "gemma4 gguf exposes audio capability",
			model: Model{
				ModelPath:      completionModelPath,
				ProjectorPaths: []string{audioProjectorPath},
				Config: model.ConfigV2{
					Renderer:     gemma4RendererSmall,
					Capabilities: []string{"audio"},
				},
				Template: chatTemplate,
			},
			expectedCaps: []model.Capability{model.CapabilityAudio, model.CapabilityCompletion, model.CapabilityVision},
		},
		{
			name: "nemotron3 gguf suppresses audio capability",
			model: Model{
				ModelPath: nemotronOmniModelPath,
				Template:  chatTemplate,
			},
			expectedCaps: []model.Capability{model.CapabilityCompletion, model.CapabilityVision},
		},
		{
			name: "nemotron3 projector suppresses audio capability",
			model: Model{
				ModelPath:      completionModelPath,
				ProjectorPaths: []string{audioProjectorPath},
				Config: model.ConfigV2{
					ModelFamily: "nemotron_h_omni",
				},
				Template: chatTemplate,
			},
			expectedCaps: []model.Capability{model.CapabilityCompletion, model.CapabilityVision},
		},
		{
			name: "gemma4 small safetensors suppresses vision and audio",
			model: Model{
				Config: model.ConfigV2{
					ModelFormat:  "safetensors",
					Renderer:     gemma4RendererSmall,
					Capabilities: []string{"vision", "audio"},
				},
				Template: chatTemplate,
			},
		},
		{
			name: "gemma4 large safetensors suppresses vision and audio",
			model: Model{
				Config: model.ConfigV2{
					ModelFormat:  "safetensors",
					Renderer:     gemma4RendererLarge,
					Capabilities: []string{"vision", "audio"},
				},
				Template: chatTemplate,
			},
		},
		{
			name: "default gemma4 safetensors suppresses vision and audio",
			model: Model{
				Config: model.ConfigV2{
					ModelFormat:  "safetensors",
					Renderer:     gemma4RendererLegacy,
					Capabilities: []string{"vision", "audio"},
				},
				Template: chatTemplate,
			},
		},
	}

	// compare two slices of model.Capability regardless of order
	compareCapabilities := func(a, b []model.Capability) bool {
		if len(a) != len(b) {
			return false
		}

		aCount := make(map[model.Capability]int)
		for _, cap := range a {
			aCount[cap]++
		}

		bCount := make(map[model.Capability]int)
		for _, cap := range b {
			bCount[cap]++
		}

		for cap, count := range aCount {
			if bCount[cap] != count {
				return false
			}
		}

		return true
	}

	for _, tt := range testModels {
		t.Run(tt.name, func(t *testing.T) {
			// Test Capabilities method
			caps := tt.model.Capabilities()
			if !compareCapabilities(caps, tt.expectedCaps) {
				t.Errorf("Expected capabilities %v, got %v", tt.expectedCaps, caps)
			}
		})
	}
}

func TestModelCheckCapabilities(t *testing.T) {
	// Create simple model file for tests that don't depend on GGUF content
	completionModelPath, _ := createBinFile(t, ggml.KV{
		"general.architecture": "llama",
	}, []*ggml.Tensor{})

	// Create vision model (llama architecture with vision block count)
	visionModelPath, _ := createBinFile(t, ggml.KV{
		"general.architecture":     "llama",
		"llama.vision.block_count": uint32(1),
	}, []*ggml.Tensor{})

	// Create embedding model (bert architecture with pooling type)
	embeddingModelPath, _ := createBinFile(t, ggml.KV{
		"general.architecture": "bert",
		"bert.pooling_type":    uint32(1),
	}, []*ggml.Tensor{})

	toolsInsertTemplate, err := template.Parse("{{ .prompt }}{{ if .tools }}{{ .tools }}{{ end }}{{ if .suffix }}{{ .suffix }}{{ end }}")
	if err != nil {
		t.Fatalf("Failed to parse template: %v", err)
	}

	chatTemplate, err := template.Parse("{{ .prompt }}")
	if err != nil {
		t.Fatalf("Failed to parse template: %v", err)
	}

	toolsTemplate, err := template.Parse("{{ .prompt }}{{ if .tools }}{{ .tools }}{{ end }}")
	if err != nil {
		t.Fatalf("Failed to parse template: %v", err)
	}

	tests := []struct {
		name           string
		model          Model
		checkCaps      []model.Capability
		expectedErrMsg string
	}{
		{
			name: "completion model without tools capability",
			model: Model{
				ModelPath: completionModelPath,
				Template:  chatTemplate,
			},
			checkCaps:      []model.Capability{model.CapabilityTools},
			expectedErrMsg: "does not support tools",
		},
		{
			name: "model with all needed capabilities",
			model: Model{
				ModelPath: completionModelPath,
				Template:  toolsInsertTemplate,
			},
			checkCaps: []model.Capability{model.CapabilityTools, model.CapabilityInsert},
		},
		{
			name: "model missing insert capability",
			model: Model{
				ModelPath: completionModelPath,
				Template:  toolsTemplate,
			},
			checkCaps:      []model.Capability{model.CapabilityInsert},
			expectedErrMsg: "does not support insert",
		},
		{
			name: "model missing vision capability",
			model: Model{
				ModelPath: completionModelPath,
				Template:  toolsTemplate,
			},
			checkCaps:      []model.Capability{model.CapabilityVision},
			expectedErrMsg: "does not support vision",
		},
		{
			name: "model with vision capability",
			model: Model{
				ModelPath: visionModelPath,
				Template:  chatTemplate,
			},
			checkCaps: []model.Capability{model.CapabilityVision},
		},
		{
			name: "model with embedding capability",
			model: Model{
				ModelPath: embeddingModelPath,
				Template:  chatTemplate,
			},
			checkCaps: []model.Capability{model.CapabilityEmbedding},
		},
		{
			name: "unknown capability",
			model: Model{
				ModelPath: completionModelPath,
				Template:  chatTemplate,
			},
			checkCaps:      []model.Capability{"unknown"},
			expectedErrMsg: "unknown capability",
		},
		{
			name: "model missing image generation capability",
			model: Model{
				ModelPath: completionModelPath,
				Template:  chatTemplate,
			},
			checkCaps:      []model.Capability{model.CapabilityImage},
			expectedErrMsg: "does not support image generation",
		},
		{
			name: "model with image generation capability",
			model: Model{
				Config: model.ConfigV2{
					Capabilities: []string{"image"},
				},
			},
			checkCaps: []model.Capability{model.CapabilityImage},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Test CheckCapabilities method
			err := tt.model.CheckCapabilities(tt.checkCaps...)
			if tt.expectedErrMsg == "" {
				if err != nil {
					t.Errorf("Expected no error, got: %v", err)
				}
			} else {
				if err == nil {
					t.Errorf("Expected error containing %q, got nil", tt.expectedErrMsg)
				} else if !strings.Contains(err.Error(), tt.expectedErrMsg) {
					t.Errorf("Expected error containing %q, got: %v", tt.expectedErrMsg, err)
				}
			}
		})
	}
}

func TestPullModelManifest(t *testing.T) {
	cases := []struct {
		name     string
		manifest string
	}{
		{
			name: "pretty printed",
			manifest: `{  "schemaVersion": 2,  "mediaType": "application/vnd.docker.distribution.manifest.v2+json",
  "config": { "digest": "sha256:abc", "mediaType": "application/vnd.docker.container.image.v1+json", "size": 50 },
  "layers": [{ "digest": "sha256:t1", "mediaType": "application/vnd.ollama.image.tensor", "size": 1024, "name": "model.weight" }]
}`,
		},
		{
			name:     "non-standard field order",
			manifest: `{"layers":[{"size":999,"digest":"sha256:def","mediaType":"application/vnd.ollama.image.model"}],"schemaVersion":2,"config":{"size":50,"digest":"sha256:abc","mediaType":"application/vnd.docker.container.image.v1+json"},"mediaType":"application/vnd.docker.distribution.manifest.v2+json"}`,
		},
	}

	for _, tt := range cases {
		t.Run(tt.name, func(t *testing.T) {
			ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				w.Write([]byte(tt.manifest))
			}))
			defer ts.Close()

			n := model.ParseName("test/model:latest")
			n.ProtocolScheme = "http"
			n.Host = strings.TrimPrefix(ts.URL, "http://")

			mf, data, err := pullModelManifest(t.Context(), n, &registryOptions{})
			if err != nil {
				t.Fatal(err)
			}

			// Raw bytes must be byte-for-byte identical to what the server sent
			if string(data) != tt.manifest {
				t.Fatalf("raw bytes differ from server response")
			}

			// SHA256 of returned data must match the expected registry digest
			expectedDigest := fmt.Sprintf("%x", sha256.Sum256([]byte(tt.manifest)))
			gotDigest := fmt.Sprintf("%x", sha256.Sum256(data))
			if gotDigest != expectedDigest {
				t.Fatalf("digest mismatch\ngot:  %s\nwant: %s", gotDigest, expectedDigest)
			}

			// Parsed manifest must still be usable
			if mf.SchemaVersion != 2 {
				t.Fatalf("schemaVersion = %d, want 2", mf.SchemaVersion)
			}
			if mf.Config.Digest == "" {
				t.Fatal("config digest is empty")
			}
			if len(mf.Layers) == 0 {
				t.Fatal("expected at least one layer")
			}
		})
	}
}
