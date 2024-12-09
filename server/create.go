package server

import (
	"bytes"
	"cmp"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"os"
	"path/filepath"
	"slices"
	"strings"

	"github.com/gin-gonic/gin"
	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/convert"
	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/format"
	"github.com/ollama/ollama/llama"
	"github.com/ollama/ollama/llm"
	"github.com/ollama/ollama/template"
	"github.com/ollama/ollama/types/errtypes"
	"github.com/ollama/ollama/types/model"
)

func (s *Server) CreateHandler(c *gin.Context) {
	var r api.CreateRequest
	if err := c.ShouldBindJSON(&r); errors.Is(err, io.EOF) {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": "missing request body"})
		return
	} else if err != nil {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	name := model.ParseName(cmp.Or(r.Model, r.Name))
	if !name.IsValid() {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": errtypes.InvalidModelNameErrMsg})
		return
	}

	name, err := getExistingName(name)
	if err != nil {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	ch := make(chan any)
	go func() {
		defer close(ch)
		fn := func(resp api.ProgressResponse) {
			ch <- resp
		}

		ctx, cancel := context.WithCancel(c.Request.Context())
		defer cancel()

		var baseLayers []*layerGGML
		var err error

		oldManifest, _ := ParseNamedManifest(name)

		var cfr *api.CreateFromRequest
		switch v := r.From.(type) {
		case string:
			slog.Debug("create model from model name")
			fromName := model.ParseName(v)
			if !fromName.IsValid() {
				ch <- gin.H{"error": errtypes.InvalidModelNameErrMsg, "status": http.StatusBadRequest}
				return
			}

			baseLayers, err = parseFromModel(ctx, fromName, fn)
			if err != nil {
				ch <- gin.H{"error": err.Error()}
			}
		case map[string]any:
			b, _ := json.Marshal(v) // re-marshal to JSON
			if err := json.Unmarshal(b, &cfr); err == nil {
				switch cfr.Type {
				case "safetensors":
					slog.Debug("create model from safetensors")
					baseLayers, err = convertModelFromSafetensors(r, cfr, name, fn)
					if err != nil {
						ch <- gin.H{"error": err.Error()}
						return
					}
				case "gguf":
					slog.Debug("create model from gguf")
					baseLayers, err = convertModelFromGGUF(r, cfr, name, fn)
					if err != nil {
						ch <- gin.H{"error": err.Error()}
						return
					}
				default:
					ch <- gin.H{"error": fmt.Sprintf("unknown from type: %s", cfr.Type), "status": http.StatusBadRequest}
					return
				}
			}
		default:
			ch <- gin.H{"error": fmt.Sprintf("unknown from type: %T", v), "status": http.StatusBadRequest}
			return
		}

		if err := createModel(r, name, baseLayers, fn); err != nil {
			if errors.Is(err, errBadTemplate) {
				ch <- gin.H{"error": err.Error(), "status": http.StatusBadRequest}
				return
			}
			ch <- gin.H{"error": err.Error()}
			return
		}

		if !envconfig.NoPrune() && oldManifest != nil {
			if err := oldManifest.RemoveLayers(); err != nil {
				ch <- gin.H{"error": err.Error()}
			}
		}

		ch <- api.ProgressResponse{Status: "success"}
	}()

	if r.Stream != nil && !*r.Stream {
		waitForStream(c, ch)
		return
	}

	streamResponse(c, ch)
}

func convertModelFromSafetensors(r api.CreateRequest, cfr *api.CreateFromRequest, name model.Name, fn func(resp api.ProgressResponse)) ([]*layerGGML, error) {
	tmpDir, err := os.MkdirTemp("", "ollama-safetensors")
	if err != nil {
		return nil, err
	}
	defer os.RemoveAll(tmpDir)

	for _, file := range cfr.Files {
		blobPath, err := GetBlobsPath(file.Digest)
		if err != nil {
			return nil, err
		}
		if err := createLink(blobPath, filepath.Join(tmpDir, file.Path)); err != nil {
			return nil, err
		}
	}

	t, err := os.CreateTemp(tmpDir, "fp16")
	if err != nil {
		return nil, err
	}
	defer t.Close()
	fn(api.ProgressResponse{Status: "converting the model"})

	if err := convert.ConvertModel(os.DirFS(tmpDir), t); err != nil {
		return nil, err
	}

	if _, err := t.Seek(0, io.SeekStart); err != nil {
		return nil, err
	}

	layer, err := NewLayer(t, "application/vnd.ollama.image.model")
	if err != nil {
		return nil, err
	}

	bin, err := layer.Open()
	if err != nil {
		return nil, err
	}

	ggml, _, err := llm.DecodeGGML(bin, 0)
	if err != nil {
		return nil, err
	}
	layers := []*layerGGML{{layer, ggml}}

	return detectChatTemplate(layers)
}

func convertModelFromGGUF(r api.CreateRequest, cfr *api.CreateFromRequest, name model.Name, fn func(resp api.ProgressResponse)) ([]*layerGGML, error) {
	if len(cfr.Files) == 0 {
		return nil, fmt.Errorf("no files provided")
	} else if len(cfr.Files) > 1 {
		return nil, fmt.Errorf("only one file is supported")
	}

	file := cfr.Files[0]
	return getGGUFLayers(file, fn)
}

func createModel(r api.CreateRequest, name model.Name, baseLayers []*layerGGML, fn func(resp api.ProgressResponse)) (err error) {
	// todo move to the new config type
	config := ConfigV2{
		OS:           "linux",
		Architecture: "amd64",
		RootFS: RootFS{
			Type: "layers",
		},
	}

	if r.Adapters != nil {
		var adapters []api.File
		switch v := r.Adapters.(type) {
		case []any:
			b, _ := json.Marshal(v) // re-marshal to JSON
			if err := json.Unmarshal(b, &adapters); err != nil {
				return err
			}
			if len(adapters) == 0 {
				return fmt.Errorf("no adapters found")
			} else if len(adapters) > 1 {
				return fmt.Errorf("only one adapter is supported")
			}
			layers, err := getGGUFLayers(adapters[0], fn)
			if err != nil {
				return err
			}
			baseLayers = append(baseLayers, layers...)
		default:
			return fmt.Errorf("adapters is not a valid type")
		}
	}

	var layers []Layer
	for _, layer := range baseLayers {
		if layer.GGML != nil {
			quantType := strings.ToUpper(cmp.Or(r.Quantize, r.Quantization))
			if quantType != "" && layer.GGML.Name() == "gguf" && layer.MediaType == "application/vnd.ollama.image.model" {
				want, err := llm.ParseFileType(quantType)
				if err != nil {
					return err
				}

				ft := layer.GGML.KV().FileType()
				if !slices.Contains([]string{"F16", "F32"}, ft.String()) {
					return errors.New("quantization is only supported for F16 and F32 models")
				} else if ft != want {
					layer, err = quantizeLayer(layer, quantType, fn)
					if err != nil {
						return err
					}
				}
			}
			config.ModelFormat = cmp.Or(config.ModelFormat, layer.GGML.Name())
			config.ModelFamily = cmp.Or(config.ModelFamily, layer.GGML.KV().Architecture())
			config.ModelType = cmp.Or(config.ModelType, format.HumanNumber(layer.GGML.KV().ParameterCount()))
			config.FileType = cmp.Or(config.FileType, layer.GGML.KV().FileType().String())
			config.ModelFamilies = append(config.ModelFamilies, layer.GGML.KV().Architecture())
		}
		layers = append(layers, layer.Layer)
	}

	if r.Template != "" {
		layers, err = setTemplate(layers, r.Template)
		if err != nil {
			return err
		}
	}

	if r.System != "" {
		layers, err = setSystem(layers, r.System)
		if err != nil {
			return err
		}
	}

	if r.License != nil {
		switch l := r.License.(type) {
		case string:
			if l != "" {
				layers, err = setLicense(layers, l)
				if err != nil {
					return err
				}
			}
		case any:
			var licenses []string
			b, _ := json.Marshal(l) // re-marshal to JSON
			if err := json.Unmarshal(b, &licenses); err != nil {
				return err
			}
			for _, v := range licenses {
				layers, err = setLicense(layers, v)
				if err != nil {
					return err
				}
			}
		default:
			return fmt.Errorf("unknown license type: %T", l)
		}
	}

	layers, err = setParameters(layers, r.Parameters)
	if err != nil {
		return err
	}

	layers, err = setMessages(layers, r.Messages)
	if err != nil {
		return err
	}

	configLayer, err := getConfigLayer(layers, config)
	if err != nil {
		return err
	}

	for _, layer := range layers {
		if layer.status != "" {
			fn(api.ProgressResponse{Status: layer.status})
		}
	}

	fn(api.ProgressResponse{Status: "writing manifest"})
	if err := WriteManifest(name, *configLayer, layers); err != nil {
		return err
	}

	return nil
}

func quantizeLayer(layer *layerGGML, quantizeType string, fn func(resp api.ProgressResponse)) (*layerGGML, error) {
	ft := layer.GGML.KV().FileType()
	fn(api.ProgressResponse{Status: fmt.Sprintf("quantizing %s model to %s", ft, quantizeType)})

	want, err := llm.ParseFileType(quantizeType)
	if err != nil {
		return nil, err
	}

	blob, err := GetBlobsPath(layer.Digest)
	if err != nil {
		return nil, err
	}

	temp, err := os.CreateTemp(filepath.Dir(blob), quantizeType)
	if err != nil {
		return nil, err
	}
	defer temp.Close()
	defer os.Remove(temp.Name())

	if err := llama.Quantize(blob, temp.Name(), uint32(want)); err != nil {
		return nil, err
	}

	newLayer, err := NewLayer(temp, layer.MediaType)
	if err != nil {
		return nil, err
	}

	if _, err := temp.Seek(0, io.SeekStart); err != nil {
		return nil, err
	}

	ggml, _, err := llm.DecodeGGML(temp, 0)
	if err != nil {
		fmt.Printf("error decoding ggml: %s\n", err)
		return nil, err
	}

	return &layerGGML{newLayer, ggml}, nil
}

func getGGUFLayers(file api.File, fn func(resp api.ProgressResponse)) ([]*layerGGML, error) {
	var layers []*layerGGML

	fn(api.ProgressResponse{Status: "parsing GGUF"})
	blobPath, err := GetBlobsPath(file.Digest)
	if err != nil {
		return nil, err
	}

	blob, err := os.Open(blobPath)
	if err != nil {
		return nil, err
	}
	defer blob.Close()

	sr := io.NewSectionReader(blob, 0, 512)
	contentType, err := detectContentType(sr)
	if err != nil {
		return nil, err
	}

	if contentType != "gguf" {
		return nil, fmt.Errorf("unsupported content type: %s", contentType)
	}

	stat, err := blob.Stat()
	if err != nil {
		return nil, err
	}

	var offset int64
	for offset < stat.Size() {
		ggml, n, err := llm.DecodeGGML(blob, 0)
		if errors.Is(err, io.EOF) {
			break
		} else if err != nil {
			return nil, err
		}

		mediatype := "application/vnd.ollama.image.model"
		if ggml.KV().Kind() == "adapter" {
			mediatype = "application/vnd.ollama.image.adapter"
		} else if _, ok := ggml.KV()[fmt.Sprintf("%s.vision.block_count", ggml.KV().Architecture())]; ok || ggml.KV().Kind() == "projector" {
			mediatype = "application/vnd.ollama.image.projector"
		}

		var layer Layer
		if file.Digest != "" && n == stat.Size() && offset == 0 {
			layer, err = NewLayerFromLayer(file.Digest, mediatype, blob.Name())
			if err != nil {
				slog.Debug("could not create new layer from layer", "error", err)
			}
		}

		// Fallback to creating layer from file copy (either NewLayerFromLayer failed, or digest empty/n != stat.Size())
		if layer.Digest == "" {
			layer, err = NewLayer(io.NewSectionReader(blob, offset, n), mediatype)
			if err != nil {
				return nil, err
			}
		}

		layers = append(layers, &layerGGML{layer, ggml})
		offset = n
	}

	return detectChatTemplate(layers)
}

func removeLayer(layers []Layer, mediatype string) []Layer {
	return slices.DeleteFunc(layers, func(layer Layer) bool {
		if layer.MediaType != mediatype {
			return false
		}

		if err := layer.Remove(); err != nil {
			return false
		}

		return true
	})
}

func setTemplate(layers []Layer, t string) ([]Layer, error) {
	layers = removeLayer(layers, "application/vnd.ollama.image.template")
	if _, err := template.Parse(t); err != nil {
		return nil, fmt.Errorf("%w: %s", errBadTemplate, err)
	}
	if _, err := template.Parse(t); err != nil {
		return nil, fmt.Errorf("%w: %s", errBadTemplate, err)
	}

	blob := strings.NewReader(t)
	layer, err := NewLayer(blob, "application/vnd.ollama.image.template")
	if err != nil {
		return nil, err
	}

	layers = append(layers, layer)
	return layers, nil
}

func setSystem(layers []Layer, s string) ([]Layer, error) {
	layers = removeLayer(layers, "application/vnd.ollama.image.system")
	if s != "" {
		blob := strings.NewReader(s)
		layer, err := NewLayer(blob, "application/vnd.ollama.image.system")
		if err != nil {
			return nil, err
		}
		layers = append(layers, layer)
	}
	return layers, nil
}

func setLicense(layers []Layer, l string) ([]Layer, error) {
	blob := strings.NewReader(l)
	layer, err := NewLayer(blob, "application/vnd.ollama.image.license")
	if err != nil {
		return nil, err
	}
	layers = append(layers, layer)
	return layers, nil
}

func setParameters(layers []Layer, p map[string]any) ([]Layer, error) {
	if p == nil {
		p = make(map[string]any)
	}
	for _, layer := range layers {
		if layer.MediaType != "application/vnd.ollama.image.params" {
			continue
		}

		digestPath, err := GetBlobsPath(layer.Digest)
		if err != nil {
			return nil, err
		}

		fn, err := os.Open(digestPath)
		if err != nil {
			return nil, err
		}
		defer fn.Close()

		var existing map[string]any
		if err := json.NewDecoder(fn).Decode(&existing); err != nil {
			return nil, err
		}

		for k, v := range existing {
			if _, exists := p[k]; exists {
				continue
			}
			p[k] = v
		}
	}

	if len(p) == 0 {
		return layers, nil
	}

	layers = removeLayer(layers, "application/vnd.ollama.image.params")

	var b bytes.Buffer
	if err := json.NewEncoder(&b).Encode(p); err != nil {
		return nil, err
	}
	layer, err := NewLayer(&b, "application/vnd.ollama.image.params")
	if err != nil {
		return nil, err
	}
	layers = append(layers, layer)
	return layers, nil
}

func setMessages(layers []Layer, m []api.Message) ([]Layer, error) {
	/*
		// this merges the old messages with the new messages
		for _, layer := range layers {
			if layer.MediaType != "application/vnd.ollama.image.messages" {
				continue
			}

			digestPath, err := GetBlobsPath(layer.Digest)
			if err != nil {
				return nil, err
			}

			fn, err := os.Open(digestPath)
			if err != nil {
				return nil, err
			}
			defer fn.Close()

			var existing []api.Message
			if err := json.NewDecoder(fn).Decode(&existing); err != nil {
				return nil, err
			}
			m = append(existing, m...)
		}
	*/
	for _, layer := range layers {
		fmt.Printf("layer mediatype: %s\n", layer.MediaType)
	}

	// this leaves the old messages intact if no new messages were specified
	// which may not be the correct behaviour
	if len(m) == 0 {
		return layers, nil
	}

	fmt.Printf("removing old messages\n")
	layers = removeLayer(layers, "application/vnd.ollama.image.messages")
	var b bytes.Buffer
	if err := json.NewEncoder(&b).Encode(m); err != nil {
		return nil, err
	}
	layer, err := NewLayer(&b, "application/vnd.ollama.image.messages")
	if err != nil {
		return nil, err
	}
	layers = append(layers, layer)
	return layers, nil
}

func getConfigLayer(layers []Layer, config ConfigV2) (*Layer, error) {
	digests := make([]string, len(layers))
	for i, layer := range layers {
		digests[i] = layer.Digest
	}
	config.RootFS.DiffIDs = digests

	var b bytes.Buffer
	if err := json.NewEncoder(&b).Encode(config); err != nil {
		return nil, err
	}
	layer, err := NewLayer(&b, "application/vnd.docker.container.image.v1+json")
	if err != nil {
		return nil, err
	}
	return &layer, nil
}

func createLink(src, dst string) error {
	// make any subdirs for dst
	if err := os.MkdirAll(filepath.Dir(dst), 0755); err != nil {
		return err
	}

	_ = os.Remove(dst)
	if err := os.Symlink(src, dst); err != nil {
		if err := copyFile(src, dst); err != nil {
			return err
		}
	}
	return nil
}

func copyFile(src, dst string) error {
	srcFile, err := os.Open(src)
	if err != nil {
		return err
	}
	defer srcFile.Close()

	dstFile, err := os.Create(dst)
	if err != nil {
		return err
	}
	defer dstFile.Close()

	_, err = io.Copy(dstFile, srcFile)
	return err
}
