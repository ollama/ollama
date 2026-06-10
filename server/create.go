package server

import (
	"bytes"
	"cmp"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"log/slog"
	"maps"
	"net"
	"net/http"
	"net/url"
	"os"
	"path"
	"path/filepath"
	"regexp"
	"slices"
	"strconv"
	"strings"
	"sync/atomic"

	"github.com/gin-gonic/gin"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/convert"
	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/format"
	ofs "github.com/ollama/ollama/fs"
	"github.com/ollama/ollama/fs/ggml"
	"github.com/ollama/ollama/manifest"
	"github.com/ollama/ollama/template"
	"github.com/ollama/ollama/types/errtypes"
	"github.com/ollama/ollama/types/model"
)

var (
	errNoFilesProvided         = errors.New("no files provided to convert")
	errOnlyOneAdapterSupported = errors.New("only one adapter is currently supported")
	errOnlyGGUFSupported       = errors.New("supplied file was not in GGUF format")
	errUnknownType             = errors.New("unknown type")
	errNeitherFromOrFiles      = errors.New("neither 'from' or 'files' was specified")
	errFilePath                = errors.New("file path must be relative")
	errRemoteDraftUnsupported  = errors.New("DRAFT cannot be used with remote models")
)

func (s *Server) CreateHandler(c *gin.Context) {
	config := &model.ConfigV2{
		OS:           "linux",
		Architecture: "amd64",
		RootFS: model.RootFS{
			Type: "layers",
		},
	}

	var r api.CreateRequest
	if err := c.ShouldBindJSON(&r); errors.Is(err, io.EOF) {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": "missing request body"})
		return
	} else if err != nil {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	config.Renderer = r.Renderer
	config.Parser = r.Parser
	config.Requires = r.Requires

	for v, digest := range r.Files {
		if !fs.ValidPath(v) {
			c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": errFilePath.Error()})
			return
		}
		if digest == "" {
			c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": manifest.ErrInvalidDigestFormat.Error()})
			return
		}
	}

	for v, digest := range r.DraftFiles {
		if !fs.ValidPath(v) {
			c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": errFilePath.Error()})
			return
		}
		if digest == "" {
			c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": manifest.ErrInvalidDigestFormat.Error()})
			return
		}
	}
	if r.DraftQuantize != "" && len(r.DraftFiles) == 0 {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": "--draft-quantize requires a DRAFT model"})
		return
	}

	for _, digest := range r.Adapters {
		if digest == "" {
			c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": manifest.ErrInvalidDigestFormat.Error()})
			return
		}
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

		oldManifest, _ := manifest.ParseNamedManifest(name)

		var baseLayers []*layerGGML
		var err error
		var remote bool

		if r.From != "" {
			slog.Debug("create model from model name", "from", r.From)
			fromRef, err := parseAndValidateModelRef(r.From)
			if err != nil {
				ch <- gin.H{"error": errtypes.InvalidModelNameErrMsg, "status": http.StatusBadRequest}
				return
			}

			fromName := fromRef.Name
			remoteHost := r.RemoteHost
			if fromRef.Source == modelSourceCloud && remoteHost == "" {
				remoteHost = cloudProxyBaseURL
			}

			if remoteHost != "" {
				ru, err := remoteURL(remoteHost)
				if err != nil {
					ch <- gin.H{"error": "bad remote", "status": http.StatusBadRequest}
					return
				}

				config.RemoteModel = fromRef.Base
				config.RemoteHost = ru
				remote = true
			} else {
				ctx, cancel := context.WithCancel(c.Request.Context())
				defer cancel()

				baseLayers, err = parseFromModel(ctx, fromName, fn)
				if err != nil {
					ch <- gin.H{"error": err.Error()}
				}

				if err == nil && !remote {
					mf, mErr := manifest.ParseNamedManifest(fromName)
					if mErr == nil && mf.Config.Digest != "" {
						configPath, pErr := manifest.BlobsPath(mf.Config.Digest)
						if pErr == nil {
							if cfgFile, fErr := os.Open(configPath); fErr == nil {
								var baseConfig model.ConfigV2
								if decErr := json.NewDecoder(cfgFile).Decode(&baseConfig); decErr == nil {
									if config.Renderer == "" {
										config.Renderer = baseConfig.Renderer
									}
									if config.Parser == "" {
										config.Parser = baseConfig.Parser
									}
									if config.Requires == "" {
										config.Requires = baseConfig.Requires
									}
									if config.ModelFormat == "" {
										config.ModelFormat = baseConfig.ModelFormat
									}
									if len(config.Capabilities) == 0 {
										config.Capabilities = baseConfig.Capabilities
									}
								}
								cfgFile.Close()
							}
						}
					}
				}
			}
		} else if r.Files != nil {
			baseLayers, err = convertModelFromFiles(r.Files, baseLayers, false, fn)
			if err != nil {
				for _, badReq := range []error{errNoFilesProvided, errOnlyGGUFSupported, errUnknownType} {
					if errors.Is(err, badReq) {
						ch <- gin.H{"error": err.Error(), "status": http.StatusBadRequest}
						return
					}
				}
				ch <- gin.H{"error": err.Error()}
				return
			}
		} else {
			ch <- gin.H{"error": errNeitherFromOrFiles.Error(), "status": http.StatusBadRequest}
			return
		}

		if remote && len(r.DraftFiles) > 0 {
			ch <- gin.H{"error": errRemoteDraftUnsupported.Error(), "status": http.StatusBadRequest}
			return
		}

		var draftLayers []*layerGGML
		if !remote && r.DraftFiles != nil {
			draftLayers, err = convertDraftModelFromFiles(r.DraftFiles, baseLayers, fn)
			if err != nil {
				for _, badReq := range []error{errNoFilesProvided, errOnlyGGUFSupported, errUnknownType, errFilePath} {
					if errors.Is(err, badReq) {
						ch <- gin.H{"error": err.Error(), "status": http.StatusBadRequest}
						return
					}
				}
				ch <- gin.H{"error": err.Error(), "status": http.StatusBadRequest}
				return
			}
		}

		var adapterLayers []*layerGGML
		if !remote && r.Adapters != nil {
			adapterLayers, err = convertModelFromFiles(r.Adapters, baseLayers, true, fn)
			if err != nil {
				for _, badReq := range []error{errNoFilesProvided, errOnlyOneAdapterSupported, errOnlyGGUFSupported, errUnknownType, errFilePath} {
					if errors.Is(err, badReq) {
						ch <- gin.H{"error": err.Error(), "status": http.StatusBadRequest}
						return
					}
				}
				ch <- gin.H{"error": err.Error(), "status": http.StatusBadRequest}
				return
			}
		}

		if len(adapterLayers) > 0 {
			baseLayers = append(baseLayers, adapterLayers...)
		}
		if len(draftLayers) > 0 {
			baseLayers = append(baseLayers, draftLayers...)
		}

		// Info is not currently exposed by Modelfiles, but allows overriding various
		// config values
		if r.Info != nil {
			caps, ok := r.Info["capabilities"]
			if ok {
				switch tcaps := caps.(type) {
				case []any:
					caps := make([]string, len(tcaps))
					for i, c := range tcaps {
						str, ok := c.(string)
						if !ok {
							continue
						}
						caps[i] = str
					}
					config.Capabilities = append(config.Capabilities, caps...)
				}
			}

			strFromInfo := func(k string) string {
				v, ok := r.Info[k]
				if ok {
					val := v.(string)
					return val
				}
				return ""
			}

			vFromInfo := func(k string) float64 {
				v, ok := r.Info[k]
				if ok {
					val := v.(float64)
					return val
				}
				return 0
			}

			config.ModelFamily = strFromInfo("model_family")
			if config.ModelFamily != "" {
				config.ModelFamilies = []string{config.ModelFamily}
			}

			config.BaseName = strFromInfo("base_name")
			config.FileType = strFromInfo("quantization_level")
			config.ModelType = strFromInfo("parameter_size")
			config.ContextLen = int(vFromInfo("context_length"))
			config.EmbedLen = int(vFromInfo("embedding_length"))
		}

		if err := createModel(r, name, baseLayers, config, fn); err != nil {
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

		s.refreshModelListCache(name)

		ch <- api.ProgressResponse{Status: "success"}
	}()

	if r.Stream != nil && !*r.Stream {
		waitForStream(c, ch)
		return
	}

	streamResponse(c, ch)
}

func remoteURL(raw string) (string, error) {
	// Special‑case: user supplied only a path ("/foo/bar").
	if strings.HasPrefix(raw, "/") {
		return (&url.URL{
			Scheme: "http",
			Host:   net.JoinHostPort("localhost", "11434"),
			Path:   path.Clean(raw),
		}).String(), nil
	}

	if !strings.Contains(raw, "://") {
		raw = "http://" + raw
	}

	if raw == "ollama.com" || raw == "http://ollama.com" {
		raw = "https://ollama.com:443"
	}

	u, err := url.Parse(raw)
	if err != nil {
		return "", fmt.Errorf("parse error: %w", err)
	}

	if u.Host == "" {
		u.Host = "localhost"
	}

	hostPart, portPart, err := net.SplitHostPort(u.Host)
	if err == nil {
		u.Host = net.JoinHostPort(hostPart, portPart)
	} else {
		u.Host = net.JoinHostPort(u.Host, "11434")
	}

	if u.Path != "" {
		u.Path = path.Clean(u.Path)
	}

	if u.Path == "/" {
		u.Path = ""
	}

	return u.String(), nil
}

func convertModelFromFiles(files map[string]string, baseLayers []*layerGGML, isAdapter bool, fn func(resp api.ProgressResponse)) ([]*layerGGML, error) {
	return convertModelFromFilesWithMediaType(files, baseLayers, isAdapter, "", true, fn)
}

func convertDraftModelFromFiles(files map[string]string, baseLayers []*layerGGML, fn func(resp api.ProgressResponse)) ([]*layerGGML, error) {
	return convertModelFromFilesWithMediaType(files, baseLayers, false, manifest.MediaTypeImageDraft, false, fn)
}

func convertModelFromFilesWithMediaType(files map[string]string, baseLayers []*layerGGML, isAdapter bool, mediaType string, detectTemplate bool, fn func(resp api.ProgressResponse)) ([]*layerGGML, error) {
	switch detectModelTypeFromFiles(files) {
	case "safetensors":
		layers, err := convertFromSafetensors(files, baseLayers, isAdapter, mediaType, detectTemplate, fn)
		if err != nil {
			slog.Error("error converting from safetensors", "error", err)
			return nil, err
		}
		return layers, nil
	case "gguf":
		if len(files) == 0 {
			return nil, errNoFilesProvided
		} else if len(files) > 1 && isAdapter {
			return nil, errOnlyOneAdapterSupported
		}

		filePaths := make([]string, 0, len(files))
		for filePath := range files {
			filePaths = append(filePaths, filePath)
		}
		slices.Sort(filePaths)

		var allLayers []*layerGGML
		var splitGroupKeys []string
		splitGroups := map[string][]*layerGGML{}
		for _, filePath := range filePaths {
			layers, err := ggufLayersWithMediaType(files[filePath], filePath, mediaType, fn)
			if err != nil {
				return nil, err
			}
			for _, layer := range layers {
				if key, ok, err := splitGGUFGroupKey(layer); err != nil {
					return nil, err
				} else if ok {
					if _, exists := splitGroups[key]; !exists {
						splitGroupKeys = append(splitGroupKeys, key)
					}
					splitGroups[key] = append(splitGroups[key], layer)
					continue
				}
				allLayers = append(allLayers, layer)
			}
		}

		for _, key := range splitGroupKeys {
			layer, err := mergeSplitGGUFLayers(splitGroups[key])
			if err != nil {
				return nil, err
			}
			allLayers = append(allLayers, layer)
		}

		if detectTemplate {
			return detectChatTemplate(allLayers)
		}
		return allLayers, nil
	default:
		return nil, errUnknownType
	}
}

func detectModelTypeFromFiles(files map[string]string) string {
	for fn := range files {
		if strings.HasSuffix(fn, ".safetensors") {
			return "safetensors"
		} else if strings.HasSuffix(fn, ".gguf") {
			return "gguf"
		} else {
			// try to see if we can find a gguf file even without the file extension
			blobPath, err := manifest.BlobsPath(files[fn])
			if err != nil {
				slog.Error("error getting blobs path", "file", fn)
				return ""
			}

			f, err := os.Open(blobPath)
			if err != nil {
				slog.Error("error reading file", "error", err)
				return ""
			}
			defer f.Close()

			buf := make([]byte, 4)
			_, err = f.Read(buf)
			if err != nil {
				slog.Error("error reading file", "error", err)
				return ""
			}

			ct := ggml.DetectContentType(buf)
			if ct == "gguf" {
				return "gguf"
			}
		}
	}

	return ""
}

func convertFromSafetensors(files map[string]string, baseLayers []*layerGGML, isAdapter bool, mediaType string, detectTemplate bool, fn func(resp api.ProgressResponse)) ([]*layerGGML, error) {
	tmpDir, err := os.MkdirTemp(envconfig.Models(), "ollama-safetensors")
	if err != nil {
		return nil, err
	}
	defer os.RemoveAll(tmpDir)
	// Set up a root to validate paths
	root, err := os.OpenRoot(tmpDir)
	if err != nil {
		return nil, err
	}
	defer root.Close()

	for fp, digest := range files {
		if !fs.ValidPath(fp) {
			return nil, fmt.Errorf("%w: %s", errFilePath, fp)
		}
		if _, err := root.Stat(fp); err != nil && !errors.Is(err, fs.ErrNotExist) {
			// Path is likely outside the root
			return nil, fmt.Errorf("%w: %s: %s", errFilePath, err, fp)
		}

		blobPath, err := manifest.BlobsPath(digest)
		if err != nil {
			return nil, err
		}
		if err := createLink(blobPath, filepath.Join(tmpDir, fp)); err != nil {
			return nil, err
		}
	}

	t, err := os.CreateTemp(tmpDir, "fp16")
	if err != nil {
		return nil, err
	}
	defer t.Close()

	var projFile *os.File
	if !isAdapter {
		projFile, err = os.CreateTemp(tmpDir, "projector")
		if err != nil {
			return nil, err
		}
		defer projFile.Close()
	}

	if !isAdapter {
		fn(api.ProgressResponse{Status: "converting model"})
		mediaType = cmp.Or(mediaType, "application/vnd.ollama.image.model")
		if mediaType == manifest.MediaTypeImageDraft {
			if err := convertMTPDraftFromSafetensors(os.DirFS(tmpDir), t, baseLayers); err != nil {
				return nil, err
			}
		} else {
			if err := convert.ConvertModel(os.DirFS(tmpDir), t, projFile); err != nil {
				return nil, err
			}
		}
	} else {
		kv, err := kvFromLayers(baseLayers)
		if err != nil {
			return nil, err
		}
		fn(api.ProgressResponse{Status: "converting adapter"})
		mediaType = "application/vnd.ollama.image.adapter"
		if err := convert.ConvertAdapter(os.DirFS(tmpDir), t, kv); err != nil {
			return nil, err
		}
	}

	if _, err := t.Seek(0, io.SeekStart); err != nil {
		return nil, err
	}

	layer, err := manifest.NewLayer(t, mediaType)
	if err != nil {
		return nil, err
	}

	bin, err := layer.Open()
	if err != nil {
		return nil, err
	}
	defer bin.Close()

	f, err := ggml.Decode(bin, -1)
	if err != nil {
		return nil, err
	}
	layers := []*layerGGML{{Layer: layer, GGML: f, rewriteForCreate: true}}

	if !isAdapter {
		projSize, err := projFile.Seek(0, io.SeekEnd)
		if err != nil {
			return nil, err
		}
		if projSize > 0 {
			if _, err := projFile.Seek(0, io.SeekStart); err != nil {
				return nil, err
			}
			projLayer, err := manifest.NewLayer(projFile, "application/vnd.ollama.image.projector")
			if err != nil {
				return nil, err
			}
			projBin, err := projLayer.Open()
			if err != nil {
				return nil, err
			}
			defer projBin.Close()
			projGGML, err := ggml.Decode(projBin, -1)
			if err != nil {
				return nil, err
			}
			projectorLayer := &layerGGML{Layer: projLayer, GGML: projGGML, rewriteForCreate: true}
			if needsDefaultLlavaProjectorType(projGGML) {
				projectorLayer, err = addDefaultLlavaProjectorType(projectorLayer)
				if err != nil {
					return nil, err
				}
			}
			layers = append(layers, projectorLayer)
		}
		if detectTemplate {
			return detectChatTemplate(layers)
		}
	}
	return layers, nil
}

func convertMTPDraftFromSafetensors(fsys fs.FS, out *os.File, baseLayers []*layerGGML) error {
	baseLayer, err := baseModelLayer(baseLayers)
	if err != nil {
		return err
	}

	tensors, cleanup, err := baseLayerTensors(baseLayer)
	if err != nil {
		return err
	}
	defer cleanup()

	return convert.ConvertQwen35MTPDraft(fsys, out, baseLayer.GGML.KV(), tensors)
}

func baseLayerTensors(layer *layerGGML) ([]*ggml.Tensor, func(), error) {
	if len(layer.splitParts) == 0 {
		blobPath, err := manifest.BlobsPath(layer.Digest)
		if err != nil {
			return nil, nil, err
		}
		blob, err := os.Open(blobPath)
		if err != nil {
			return nil, nil, err
		}
		tensors := tensorsFromGGUFFile(blob, layer.GGML)
		return tensors, func() { blob.Close() }, nil
	}

	var files []*os.File
	tensors := make([]*ggml.Tensor, 0, len(layer.GGML.Tensors().Items()))
	cleanup := func() {
		for _, f := range files {
			f.Close()
		}
	}

	for _, part := range layer.splitParts {
		blobPath, err := manifest.BlobsPath(part.Digest)
		if err != nil {
			cleanup()
			return nil, nil, err
		}
		blob, err := os.Open(blobPath)
		if err != nil {
			cleanup()
			return nil, nil, err
		}
		files = append(files, blob)
		tensors = append(tensors, tensorsFromGGUFFile(blob, part.GGML)...)
	}

	return tensors, cleanup, nil
}

func tensorsFromGGUFFile(file *os.File, f *ggml.GGML) []*ggml.Tensor {
	tensors := make([]*ggml.Tensor, 0, len(f.Tensors().Items()))
	for _, tensor := range f.Tensors().Items() {
		tensors = append(tensors, tensorFromFile(file, f.Tensors().Offset+tensor.Offset, tensor))
	}
	return tensors
}

func baseModelLayer(layers []*layerGGML) (*layerGGML, error) {
	for _, layer := range layers {
		if layer.GGML != nil && layer.MediaType == "application/vnd.ollama.image.model" {
			return layer, nil
		}
	}
	return nil, fmt.Errorf("no base model was found")
}

func kvFromLayers(baseLayers []*layerGGML) (ofs.Config, error) {
	for _, l := range baseLayers {
		if l.GGML != nil {
			return l.KV(), nil
		}
	}
	return ggml.KV{}, fmt.Errorf("no base model was found")
}

func createModel(r api.CreateRequest, name model.Name, baseLayers []*layerGGML, config *model.ConfigV2, fn func(resp api.ProgressResponse)) (err error) {
	var layers []manifest.Layer
	for _, layer := range baseLayers {
		if layer.GGML != nil {
			if layer.rewriteForCreate && layer.GGML.Name() == "gguf" && len(layer.splitParts) > 0 && layerHasEmbeddedCompatibilityTensors(layer) {
				var err error
				layer, err = copySplitLayerPreservingTensors(layer)
				if err != nil {
					return err
				}
			}

			quantType := ""
			if layer.MediaType == "application/vnd.ollama.image.model" {
				quantType = strings.ToUpper(cmp.Or(r.Quantize, r.Quantization))
			} else if layer.MediaType == manifest.MediaTypeImageDraft {
				quantType = strings.ToUpper(r.DraftQuantize)
			}
			ft := layer.GGML.KV().FileType()
			rewroteLayer := false
			if quantType == "" && hasSourceFP8Tensors(layer.GGML.KV()) && layer.GGML.Name() == "gguf" && layer.MediaType == "application/vnd.ollama.image.model" && slices.Contains([]string{"F16", "BF16", "F32"}, ft.String()) {
				quantType = "Q8_0"
			}
			if quantType != "" && layer.GGML.Name() == "gguf" && slices.Contains([]string{"application/vnd.ollama.image.model", manifest.MediaTypeImageDraft}, layer.MediaType) {
				want, err := ggml.ParseFileType(quantType)
				if err != nil {
					return err
				}

				if layer.MediaType == manifest.MediaTypeImageDraft && ft.ToTensorType().IsQuantized() {
					return fmt.Errorf("draft quantization requires an unquantized draft model, got %s", ft)
				} else if !slices.Contains([]string{"F16", "BF16", "F32"}, ft.String()) {
					return errors.New("quantization is only supported for F16, BF16 and F32 models")
				} else if ft != want {
					layer, err = quantizeLayer(layer, quantType, fn)
					if err != nil {
						return err
					}
					rewroteLayer = true
				}
			}
			if !rewroteLayer && layer.rewriteForCreate && layer.GGML.Name() == "gguf" && layer.MediaType == "application/vnd.ollama.image.model" && !hasEmbeddedCompatibilityTensors(layer.GGML) {
				var err error
				layer, err = copyLayerWithLlamaQuantize(layer, fn)
				if err != nil {
					return err
				}
			}
			if !rewroteLayer && layer.rewriteForCreate && layer.GGML.Name() == "gguf" && layer.MediaType == manifest.MediaTypeImageDraft && len(layer.splitParts) > 0 {
				var err error
				layer, err = copyLayerWithLlamaQuantize(layer, fn)
				if err != nil {
					return err
				}
			}
			if layer.rewriteForCreate && layer.GGML.Name() == "gguf" && layer.MediaType == "application/vnd.ollama.image.projector" && needsDefaultLlavaProjectorType(layer.GGML) {
				var err error
				fn(api.ProgressResponse{Status: "updating GGUF projector metadata"})
				layer, err = addDefaultLlavaProjectorType(layer)
				if err != nil {
					return err
				}
			}
			switch layer.MediaType {
			case "application/vnd.ollama.image.model":
				config.ModelFormat = cmp.Or(config.ModelFormat, layer.GGML.Name())
				config.ModelFamily = cmp.Or(config.ModelFamily, layer.GGML.KV().Architecture())
				config.ModelType = cmp.Or(config.ModelType, format.HumanNumber(layer.GGML.KV().ParameterCount()))
				config.FileType = cmp.Or(config.FileType, layer.GGML.KV().FileType().String())
				config.ModelFamilies = append(config.ModelFamilies, layer.GGML.KV().Architecture())

				// Auto-detect renderer, parser, and stop tokens from GGUF architecture.
				// TODO: abstract this into a registry/lookup table when multiple models
				// need architecture-based renderer/parser/stop defaults.
				if config.Renderer == "" || config.Parser == "" {
					arch := layer.GGML.KV().Architecture()
					switch arch {
					case "gemma4":
						config.Renderer = cmp.Or(config.Renderer, gemma4RendererLegacy)
						config.Parser = cmp.Or(config.Parser, "gemma4")
						if _, ok := r.Parameters["stop"]; !ok {
							if r.Parameters == nil {
								r.Parameters = make(map[string]any)
							}
							r.Parameters["stop"] = []string{"<turn|>"}
						}
					case "laguna":
						config.Renderer = cmp.Or(config.Renderer, "laguna")
						config.Parser = cmp.Or(config.Parser, "laguna")
					case "nemotron_h", "nemotron_h_moe", "nemotron_h_omni":
						config.Renderer = cmp.Or(config.Renderer, "nemotron-3-nano")
						config.Parser = cmp.Or(config.Parser, "nemotron-3-nano")
					}
				}
			case manifest.MediaTypeImageDraft:
				config.Draft = &model.Draft{
					ModelFormat:  layer.GGML.Name(),
					Architecture: layer.GGML.KV().Architecture(),
				}
			}
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

	configLayer, err := createConfigLayer(layers, *config)
	if err != nil {
		return err
	}

	for _, layer := range layers {
		if layer.Status != "" {
			fn(api.ProgressResponse{Status: layer.Status})
		}
	}

	fn(api.ProgressResponse{Status: "writing manifest"})
	if err := manifest.WriteManifest(name, *configLayer, layers); err != nil {
		return err
	}

	return nil
}

func hasSourceFP8Tensors(kv ggml.KV) bool {
	return kv.String("source_quantization") == "hf_fp8" && len(kv.Strings("source_fp8_tensors")) > 0
}

func hasEmbeddedCompatibilityTensors(f *ggml.GGML) bool {
	for _, t := range f.Tensors().Items() {
		if isEmbeddedCompatibilityTensor(t.Name) {
			return true
		}
	}
	return false
}

func layerHasEmbeddedCompatibilityTensors(layer *layerGGML) bool {
	if hasEmbeddedCompatibilityTensors(layer.GGML) {
		return true
	}
	for _, part := range layer.splitParts {
		if part.GGML != nil && hasEmbeddedCompatibilityTensors(part.GGML) {
			return true
		}
	}
	return false
}

func isEmbeddedCompatibilityTensor(name string) bool {
	for _, prefix := range []string{"a.", "mm.", "mtp.", "s.", "v."} {
		if strings.HasPrefix(name, prefix) {
			return true
		}
	}
	return false
}

func quantizeLayer(layer *layerGGML, quantizeType string, fn func(resp api.ProgressResponse)) (*layerGGML, error) {
	ftype, err := ggml.ParseFileType(quantizeType)
	if err != nil {
		return nil, err
	}

	return rewriteLayerWithLlamaQuantize(layer, quantizeType, fn, func(in, out *os.File, progressFn func(uint64)) error {
		return quantize(in, out, layer.GGML, ftype, progressFn)
	})
}

func copyLayerWithLlamaQuantize(layer *layerGGML, fn func(resp api.ProgressResponse)) (*layerGGML, error) {
	newLayer, err := rewriteLayerWithLlamaQuantize(layer, "COPY", fn, func(in, out *os.File, progressFn func(uint64)) error {
		return copyGGUFWithLlamaQuantize(in, out, layer.GGML, progressFn)
	})
	if err != nil {
		return nil, fmt.Errorf("failed to validate GGUF with llama-quantize without compatibility patches: %w", err)
	}
	return newLayer, nil
}

func copySplitLayerPreservingTensors(layer *layerGGML) (*layerGGML, error) {
	blob, err := manifest.BlobsPath(layer.Digest)
	if err != nil {
		return nil, err
	}

	tensors, cleanup, err := baseLayerTensors(layer)
	if err != nil {
		return nil, err
	}
	defer cleanup()

	kv := maps.Clone(layer.GGML.KV())
	removeSplitMetadata(kv, layer.GGML.KV().Architecture())

	temp, err := os.CreateTemp(filepath.Dir(blob), "split-copy")
	if err != nil {
		return nil, err
	}
	defer os.Remove(temp.Name())
	defer temp.Close()

	if err := ggml.WriteGGUF(temp, kv, tensors); err != nil {
		return nil, err
	}
	if _, err := temp.Seek(0, io.SeekStart); err != nil {
		return nil, err
	}

	newLayer, err := manifest.NewLayer(temp, layer.MediaType)
	if err != nil {
		return nil, err
	}
	if _, err := temp.Seek(0, io.SeekStart); err != nil {
		return nil, err
	}

	f, err := ggml.Decode(temp, 1024)
	if err != nil {
		return nil, err
	}
	return &layerGGML{Layer: newLayer, GGML: f}, nil
}

func removeSplitMetadata(kv ggml.KV, arch string) {
	for _, key := range []string{
		"split.no",
		"split.count",
		"split.tensors.count",
	} {
		delete(kv, key)
		delete(kv, arch+"."+key)
	}
}

func rewriteLayerWithLlamaQuantize(layer *layerGGML, typeName string, fn func(resp api.ProgressResponse), rewrite func(in, out *os.File, progressFn func(uint64)) error) (*layerGGML, error) {
	ft := layer.GGML.KV().FileType()
	var doneBytes atomic.Uint64
	totalBytes := uint64(layer.Size) - layer.GGML.Tensors().Offset
	fnWrap := func(n uint64) {
		done := doneBytes.Add(n)
		progress := float32(done) / float32(totalBytes)
		status := fmt.Sprintf("quantizing %s model to %s", ft, typeName)
		if typeName == "COPY" {
			status = "validating GGUF model"
		}
		fn(api.ProgressResponse{Status: status, Digest: "0000000000000000000", Total: layer.Size, Completed: int64(progress * float32(layer.Size))})
	}

	blob, err := manifest.BlobsPath(layer.Digest)
	if err != nil {
		return nil, err
	}
	fp, err := os.Open(blob)
	if err != nil {
		return nil, err
	}
	defer fp.Close()

	in := fp
	if len(layer.splitParts) > 0 {
		splitInput, cleanup, err := prepareSplitGGUFInput(layer, filepath.Dir(blob))
		if err != nil {
			return nil, err
		}
		defer cleanup()
		defer splitInput.Close()
		in = splitInput
	}

	temp, err := os.CreateTemp(filepath.Dir(blob), typeName)
	if err != nil {
		return nil, err
	}
	defer os.Remove(temp.Name())
	defer temp.Close()

	if err := rewrite(in, temp, fnWrap); err != nil {
		return nil, err
	}
	if _, err := temp.Seek(0, io.SeekStart); err != nil {
		return nil, err
	}
	fn(api.ProgressResponse{Status: "verifying conversion"})
	newLayer, err := manifest.NewLayer(temp, layer.MediaType)
	if err != nil {
		return nil, err
	}
	if _, err := temp.Seek(0, io.SeekStart); err != nil {
		return nil, err
	}

	f, err := ggml.Decode(temp, 1024)
	if err != nil {
		slog.Error(fmt.Sprintf("error decoding ggml: %s\n", err))
		return nil, err
	}
	return &layerGGML{Layer: newLayer, GGML: f}, nil
}

func prepareSplitGGUFInput(layer *layerGGML, dir string) (*os.File, func(), error) {
	tempDir, err := os.MkdirTemp(dir, "split-gguf-")
	if err != nil {
		return nil, nil, err
	}

	cleanup := func() {
		if err := os.RemoveAll(tempDir); err != nil {
			slog.Warn("failed to remove temporary split GGUF links", "dir", tempDir, "error", err)
		}
	}

	var firstPath string
	for i, part := range layer.splitParts {
		blobPath, err := manifest.BlobsPath(part.Digest)
		if err != nil {
			cleanup()
			return nil, nil, err
		}
		linkPath := filepath.Join(tempDir, path.Base(part.Name))
		if err := os.Link(blobPath, linkPath); err != nil {
			cleanup()
			return nil, nil, err
		}
		if i == 0 {
			firstPath = linkPath
		}
	}

	f, err := os.Open(firstPath)
	if err != nil {
		cleanup()
		return nil, nil, err
	}
	return f, cleanup, nil
}

var splitGGUFNameRe = regexp.MustCompile(`^(.*)-(\d{5})-of-(\d{5})\.gguf$`)

func splitGGUFName(name string) (prefix string, index, count uint16, ok bool) {
	matches := splitGGUFNameRe.FindStringSubmatch(path.Base(name))
	if len(matches) != 4 {
		return "", 0, 0, false
	}

	idx, err := strconv.ParseUint(matches[2], 10, 16)
	if err != nil || idx == 0 {
		return "", 0, 0, false
	}
	n, err := strconv.ParseUint(matches[3], 10, 16)
	if err != nil || n == 0 {
		return "", 0, 0, false
	}
	return matches[1], uint16(idx - 1), uint16(n), true
}

func splitGGUFGroupKey(layer *layerGGML) (string, bool, error) {
	count, ok := splitGGUFUint(layer.GGML.KV(), "split.count")
	if !ok {
		return "", false, nil
	}
	if count <= 1 {
		return "", false, nil
	}

	prefix, index, nameCount, ok := splitGGUFName(layer.From)
	if !ok {
		return "", false, fmt.Errorf("split GGUF %q must use llama.cpp split filename pattern", layer.From)
	}
	if nameCount != count {
		return "", false, fmt.Errorf("split GGUF %q filename count %d does not match metadata count %d", layer.From, nameCount, count)
	}
	splitNo, ok := splitGGUFUint(layer.GGML.KV(), "split.no")
	if !ok {
		return "", false, fmt.Errorf("split GGUF %q is missing split.no metadata", layer.From)
	}
	if splitNo != index {
		return "", false, fmt.Errorf("split GGUF %q filename index %d does not match metadata index %d", layer.From, index, splitNo)
	}

	return fmt.Sprintf("%s:%s:%d", layer.MediaType, prefix, count), true, nil
}

func mergeSplitGGUFLayers(layers []*layerGGML) (*layerGGML, error) {
	if len(layers) == 0 {
		return nil, fmt.Errorf("empty split GGUF group")
	}

	count, ok := splitGGUFUint(layers[0].GGML.KV(), "split.count")
	if !ok {
		return nil, fmt.Errorf("split GGUF %q is missing split.count metadata", layers[0].From)
	}
	if int(count) != len(layers) {
		return nil, fmt.Errorf("split GGUF %q has %d shards, expected %d", layers[0].From, len(layers), count)
	}

	byIndex := make([]*layerGGML, count)
	for _, layer := range layers {
		index, ok := splitGGUFUint(layer.GGML.KV(), "split.no")
		if !ok {
			return nil, fmt.Errorf("split GGUF %q is missing split.no metadata", layer.From)
		}
		if index >= count {
			return nil, fmt.Errorf("split GGUF %q has invalid shard index %d", layer.From, index)
		}
		if byIndex[index] != nil {
			return nil, fmt.Errorf("split GGUF has duplicate shard index %d", index)
		}
		byIndex[index] = layer
	}

	primary := byIndex[0]
	if primary == nil {
		return nil, fmt.Errorf("split GGUF is missing first shard")
	}

	primary.splitParts = make([]splitGGUFPart, 0, count)
	for i, layer := range byIndex {
		if layer == nil {
			return nil, fmt.Errorf("split GGUF %q is missing shard %d", primary.From, i)
		}
		primary.splitParts = append(primary.splitParts, splitGGUFPart{Digest: layer.Digest, Name: layer.From, GGML: layer.GGML})
	}

	return primary, nil
}

func splitGGUFUint(kv ggml.KV, key string) (uint16, bool) {
	keys := []string{key}
	if !strings.HasPrefix(key, "tokenizer.") && !strings.HasPrefix(key, "general.") {
		keys = append(keys, kv.Architecture()+"."+key)
	}
	for _, k := range keys {
		switch v := kv.Value(k).(type) {
		case uint16:
			return v, true
		case uint32:
			if v <= uint32(^uint16(0)) {
				return uint16(v), true
			}
		case uint64:
			if v <= uint64(^uint16(0)) {
				return uint16(v), true
			}
		case int32:
			if v >= 0 && v <= int32(^uint16(0)) {
				return uint16(v), true
			}
		case int64:
			if v >= 0 && v <= int64(^uint16(0)) {
				return uint16(v), true
			}
		}
	}
	return 0, false
}

func ggufLayers(digest, sourceName string, fn func(resp api.ProgressResponse)) ([]*layerGGML, error) {
	return ggufLayersWithMediaType(digest, sourceName, "", fn)
}

func ggufLayersWithMediaType(digest, sourceName, mediaType string, fn func(resp api.ProgressResponse)) ([]*layerGGML, error) {
	var layers []*layerGGML

	fn(api.ProgressResponse{Status: "parsing GGUF"})
	blobPath, err := manifest.BlobsPath(digest)
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
		slog.Error(fmt.Sprintf("unsupported content type: %s", contentType))
		return nil, errOnlyGGUFSupported
	}

	f, err := ggml.Decode(blob, -1)
	if err != nil {
		return nil, err
	}

	if mediaType == "" {
		mediaType = "application/vnd.ollama.image.model"
		if f.KV().Kind() == "adapter" {
			mediaType = "application/vnd.ollama.image.adapter"
		} else if isProjectorGGUF(f.KV()) {
			mediaType = "application/vnd.ollama.image.projector"
		}
	}

	layer, err := manifest.NewLayerFromLayer(digest, mediaType, sourceName)
	if err != nil {
		slog.Debug("could not create new layer from layer", "error", err)
		return nil, err
	}

	layers = append(layers, &layerGGML{Layer: layer, GGML: f, rewriteForCreate: true})

	return layers, nil
}

func isProjectorGGUF(kv ggml.KV) bool {
	switch kv.Kind() {
	case "projector", "mmproj":
		return true
	}

	// If a model has vision.block_count but not block_count, it is a standalone vision model.
	if kv.Uint("block_count") == 0 && kv.Uint("vision.block_count") > 0 {
		return true
	}

	return kv.Architecture() == "clip" && kv.Uint("block_count") == 0 && (kv.Bool("has_vision_encoder") || kv.Bool("has_audio_encoder"))
}

func removeLayer(layers []manifest.Layer, mediatype string) []manifest.Layer {
	return slices.DeleteFunc(layers, func(layer manifest.Layer) bool {
		if layer.MediaType != mediatype {
			return false
		}

		if err := layer.Remove(); err != nil {
			slog.Warn("couldn't remove blob", "digest", layer.Digest, "error", err)
			return true
		}

		return true
	})
}

func setTemplate(layers []manifest.Layer, t string) ([]manifest.Layer, error) {
	layers = removeLayer(layers, "application/vnd.ollama.image.template")
	if _, err := template.Parse(t); err != nil {
		return nil, fmt.Errorf("%w: %s", errBadTemplate, err)
	}

	blob := strings.NewReader(t)
	layer, err := manifest.NewLayer(blob, "application/vnd.ollama.image.template")
	if err != nil {
		return nil, err
	}

	layers = append(layers, layer)
	return layers, nil
}

func setSystem(layers []manifest.Layer, s string) ([]manifest.Layer, error) {
	layers = removeLayer(layers, "application/vnd.ollama.image.system")
	if s != "" {
		blob := strings.NewReader(s)
		layer, err := manifest.NewLayer(blob, "application/vnd.ollama.image.system")
		if err != nil {
			return nil, err
		}
		layers = append(layers, layer)
	}
	return layers, nil
}

func setLicense(layers []manifest.Layer, l string) ([]manifest.Layer, error) {
	blob := strings.NewReader(l)
	layer, err := manifest.NewLayer(blob, "application/vnd.ollama.image.license")
	if err != nil {
		return nil, err
	}
	layers = append(layers, layer)
	return layers, nil
}

func setParameters(layers []manifest.Layer, p map[string]any) ([]manifest.Layer, error) {
	if p == nil {
		p = make(map[string]any)
	}
	for _, layer := range layers {
		if layer.MediaType != "application/vnd.ollama.image.params" {
			continue
		}

		digestPath, err := manifest.BlobsPath(layer.Digest)
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
	layer, err := manifest.NewLayer(&b, "application/vnd.ollama.image.params")
	if err != nil {
		return nil, err
	}
	layers = append(layers, layer)
	return layers, nil
}

func setMessages(layers []manifest.Layer, m []api.Message) ([]manifest.Layer, error) {
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
	layer, err := manifest.NewLayer(&b, "application/vnd.ollama.image.messages")
	if err != nil {
		return nil, err
	}
	layers = append(layers, layer)
	return layers, nil
}

func createConfigLayer(layers []manifest.Layer, config model.ConfigV2) (*manifest.Layer, error) {
	digests := make([]string, len(layers))
	for i, layer := range layers {
		digests[i] = layer.Digest
	}
	config.RootFS.DiffIDs = digests

	var b bytes.Buffer
	if err := json.NewEncoder(&b).Encode(config); err != nil {
		return nil, err
	}
	layer, err := manifest.NewLayer(&b, "application/vnd.docker.container.image.v1+json")
	if err != nil {
		return nil, err
	}
	return &layer, nil
}

func createLink(src, dst string) error {
	// make any subdirs for dst
	if err := os.MkdirAll(filepath.Dir(dst), 0o755); err != nil {
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
