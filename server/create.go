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
	"math"
	"net"
	"net/http"
	"net/url"
	"os"
	"path"
	"path/filepath"
	"runtime/debug"
	"slices"
	"strings"

	"github.com/gin-gonic/gin"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/format"
	"github.com/ollama/ollama/fs/ggml"
	"github.com/ollama/ollama/manifest"
	"github.com/ollama/ollama/template"
	"github.com/ollama/ollama/types/errtypes"
	"github.com/ollama/ollama/types/model"
	xcreate "github.com/ollama/ollama/x/create"
	"github.com/ollama/ollama/x/quant"
)

var (
	errNoFilesProvided         = errors.New("no files provided to convert")
	errOnlyOneAdapterSupported = errors.New("only one adapter is currently supported")
	errOnlyGGUFSupported       = errors.New("supplied file was not in GGUF format")
	errUnknownType             = errors.New("unknown type")
	errNeitherFromOrFiles      = errors.New("neither 'from' or 'files' was specified")
	errFilePath                = errors.New("file path must be relative")
	errRemoteDraftUnsupported  = errors.New("DRAFT cannot be used with remote models")
	errSafetensorsFrom         = errors.New("safetensors imports do not support FROM model overlays")
	errSafetensorsAdapters     = errors.New("safetensors imports do not support adapters")
	errInvalidSplitGGUF        = errors.New("invalid split GGUF")
	errMixedModelTypes         = errors.New("mixed model file types")
)

func (s *Server) CreateHandler(c *gin.Context) {
	config := &model.ConfigV2{
		OS:           "linux",
		Architecture: "amd64",
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

	if err := validateCreateFiles(r.Files); err != nil {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}
	if err := validateCreateFiles(r.DraftFiles); err != nil {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}
	if err := validateCreateFiles(r.Adapters); err != nil {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}
	if r.DraftQuantize != "" && len(r.DraftFiles) == 0 {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": "--draft-quantize requires a DRAFT model"})
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

	fileType, err := detectModelTypeFromFiles(r.Files)
	if err != nil {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}
	if err := validateCreateOptions(r, fileType); err != nil {
		c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	reqCtx := c.Request.Context()
	ch := make(chan any)
	go func() {
		send := func(resp any) bool {
			select {
			case ch <- resp:
				return true
			case <-reqCtx.Done():
				return false
			}
		}
		defer close(ch)
		defer recoverCreatePanic(send)

		fn := func(resp api.ProgressResponse) {
			send(resp)
		}

		oldManifest, _ := manifest.ParseNamedManifest(name)

		if fileType == "safetensors" {
			if err := applyCreateInfo(config, r.Info); err != nil {
				send(gin.H{"error": err.Error(), "status": http.StatusBadRequest})
				return
			}
			if err := createSafetensorsModel(reqCtx, r, name, config, fn); err != nil {
				send(createSafetensorsErrorResponse(err))
				return
			}
			if err := pruneOldManifestLayers(oldManifest); err != nil {
				send(gin.H{"error": err.Error()})
				return
			}
			s.refreshModelListCache(name)
			send(api.ProgressResponse{Status: "success"})
			return
		}

		var baseLayers []*layerGGML
		var err error
		var remote bool

		if r.From != "" {
			slog.Debug("create model from model name", "from", r.From)
			fromRef, err := parseAndValidateModelRef(r.From)
			if err != nil {
				send(gin.H{"error": errtypes.InvalidModelNameErrMsg, "status": http.StatusBadRequest})
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
					send(gin.H{"error": "bad remote", "status": http.StatusBadRequest})
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
					send(gin.H{"error": err.Error()})
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
				for _, badReq := range []error{errNoFilesProvided, errOnlyGGUFSupported, errUnknownType, errInvalidSplitGGUF, errMixedModelTypes} {
					if errors.Is(err, badReq) {
						send(gin.H{"error": err.Error(), "status": http.StatusBadRequest})
						return
					}
				}
				send(gin.H{"error": err.Error()})
				return
			}
		} else {
			send(gin.H{"error": errNeitherFromOrFiles.Error(), "status": http.StatusBadRequest})
			return
		}

		if remote && len(r.DraftFiles) > 0 {
			send(gin.H{"error": errRemoteDraftUnsupported.Error(), "status": http.StatusBadRequest})
			return
		}

		var draftLayers []*layerGGML
		if !remote && r.DraftFiles != nil {
			draftLayers, err = convertDraftModelFromFiles(r.DraftFiles, baseLayers, fn)
			if err != nil {
				for _, badReq := range []error{errNoFilesProvided, errOnlyGGUFSupported, errUnknownType, errFilePath, errInvalidSplitGGUF, errMixedModelTypes} {
					if errors.Is(err, badReq) {
						send(gin.H{"error": err.Error(), "status": http.StatusBadRequest})
						return
					}
				}
				send(gin.H{"error": err.Error(), "status": http.StatusBadRequest})
				return
			}
		}

		var adapterLayers []*layerGGML
		if !remote && r.Adapters != nil {
			adapterLayers, err = convertModelFromFiles(r.Adapters, baseLayers, true, fn)
			if err != nil {
				for _, badReq := range []error{errNoFilesProvided, errOnlyOneAdapterSupported, errOnlyGGUFSupported, errUnknownType, errFilePath, errInvalidSplitGGUF, errMixedModelTypes} {
					if errors.Is(err, badReq) {
						send(gin.H{"error": err.Error(), "status": http.StatusBadRequest})
						return
					}
				}
				send(gin.H{"error": err.Error(), "status": http.StatusBadRequest})
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
		// config values.
		if err := applyCreateInfo(config, r.Info); err != nil {
			send(gin.H{"error": err.Error(), "status": http.StatusBadRequest})
			return
		}

		if err := createModel(r, name, baseLayers, config, fn); err != nil {
			if errors.Is(err, errBadTemplate) || errors.Is(err, errInvalidSplitGGUF) {
				send(gin.H{"error": err.Error(), "status": http.StatusBadRequest})
				return
			}
			send(gin.H{"error": err.Error()})
			return
		}

		if err := pruneOldManifestLayers(oldManifest); err != nil {
			send(gin.H{"error": err.Error()})
		}

		s.refreshModelListCache(name)

		send(api.ProgressResponse{Status: "success"})
	}()

	if r.Stream != nil && !*r.Stream {
		waitForStream(c, ch)
		return
	}

	streamResponse(c, ch)
}

func pruneOldManifestLayers(oldManifest *manifest.Manifest) error {
	if envconfig.NoPrune() || oldManifest == nil {
		return nil
	}
	return oldManifest.RemoveLayers()
}

func recoverCreatePanic(send func(any) bool) {
	if r := recover(); r != nil {
		slog.Error("panic in create background goroutine", "panic", r, "stack", string(debug.Stack()))
		send(gin.H{"error": "internal server error"})
	}
}

// createSafetensorsModel imports uploaded raw safetensors source files by
// staging them as a normal model directory and running the shared x/create
// pipeline on the server.
func createSafetensorsModel(ctx context.Context, r api.CreateRequest, name model.Name, config *model.ConfigV2, fn func(resp api.ProgressResponse)) error {
	if len(r.Files) == 0 {
		return errNoFilesProvided
	}
	if r.From != "" {
		return errSafetensorsFrom
	}
	if len(r.Adapters) > 0 {
		return errSafetensorsAdapters
	}
	if err := ctx.Err(); err != nil {
		return err
	}

	modelDir, cleanup, err := stageSafetensorsSourceFiles(r.Files)
	if err != nil {
		return err
	}
	defer cleanup()
	if err := ctx.Err(); err != nil {
		return err
	}

	progressFn := func(status string) {
		fn(api.ProgressResponse{Status: status})
	}
	store := xcreate.StoreFromLayerCreator(createSafetensorsLayer)

	var draftDir string
	var draftCleanup func()
	var draftLayers []xcreate.LayerInfo
	if len(r.DraftFiles) > 0 {
		if err := ctx.Err(); err != nil {
			return err
		}
		draftDir, draftCleanup, err = stageSafetensorsSourceFiles(r.DraftFiles)
		if err != nil {
			return err
		}
		defer draftCleanup()

		draftLayers, err = xcreate.CreateDraftLayers(ctx, draftDir, "draft.", "draft/", r.DraftQuantize, store, progressFn)
		if err != nil {
			return err
		}
	}

	writeManifest := writeSafetensorsManifest(config, r, draftDir, fn)
	if len(draftLayers) > 0 {
		next := writeManifest
		writeManifest = func(modelName string, info xcreate.ManifestInfo) error {
			info.Layers = append(info.Layers, draftLayers...)
			return next(modelName, info)
		}
	}

	return xcreate.Create(ctx, name.String(), modelDir, cmp.Or(r.Quantize, r.Quantization), store, writeManifest, progressFn)
}

func createSafetensorsErrorResponse(err error) gin.H {
	status := http.StatusInternalServerError
	for _, badReq := range []error{errNoFilesProvided, errFilePath, errSafetensorsFrom, errSafetensorsAdapters, manifest.ErrInvalidDigestFormat} {
		if errors.Is(err, badReq) {
			status = http.StatusBadRequest
			break
		}
	}
	return gin.H{"error": err.Error(), "status": status}
}

func createSafetensorsLayer(r io.Reader, mediaType, name string) (xcreate.LayerInfo, error) {
	layer, err := manifest.NewLayer(r, mediaType)
	if err != nil {
		return xcreate.LayerInfo{}, err
	}
	return xcreate.LayerInfo{
		Digest:    layer.Digest,
		Size:      layer.Size,
		MediaType: layer.MediaType,
		Name:      name,
	}, nil
}

func writeSafetensorsManifest(config *model.ConfigV2, r api.CreateRequest, draftDir string, fn func(resp api.ProgressResponse)) xcreate.ManifestWriter {
	capabilities := append([]string(nil), config.Capabilities...)
	if len(capabilities) == 0 {
		capabilities = []string{"completion"}
	}
	baseConfig := *config
	return xcreate.NewSafetensorsManifestWriter(xcreate.SafetensorsManifestOptions{
		BaseConfig:          &baseConfig,
		Capabilities:        capabilities,
		MinVersion:          xcreate.SafetensorsMinOllamaVersion,
		Requires:            config.Requires,
		Parser:              config.Parser,
		Renderer:            config.Renderer,
		DraftDir:            draftDir,
		Template:            r.Template,
		System:              r.System,
		License:             r.License,
		Parameters:          r.Parameters,
		ExtraLayers:         func(layers []manifest.Layer) ([]manifest.Layer, error) { return setMessages(layers, r.Messages) },
		BeforeWriteManifest: func() { fn(api.ProgressResponse{Status: "writing manifest"}) },
	})
}

func stageSafetensorsSourceFiles(files map[string]string) (string, func(), error) {
	dir, err := os.MkdirTemp("", "ollama-create-safetensors-*")
	if err != nil {
		return "", nil, err
	}
	cleanup := func() {
		if err := os.RemoveAll(dir); err != nil {
			slog.Warn("failed to remove staged safetensors source", "dir", dir, "error", err)
		}
	}

	for filePath, digest := range files {
		if err := validateCreateFilePath(filePath); err != nil {
			cleanup()
			return "", nil, err
		}
		blobPath, err := manifest.BlobsPath(digest)
		if err != nil {
			cleanup()
			return "", nil, fmt.Errorf("invalid digest for %s: %w", filePath, err)
		}
		if _, err := os.Stat(blobPath); err != nil {
			cleanup()
			return "", nil, fmt.Errorf("blob not found for %s (digest %s): %w", filePath, digest, err)
		}

		dst := filepath.Join(dir, filepath.FromSlash(filePath))
		if err := linkOrCopyFile(blobPath, dst); err != nil {
			cleanup()
			return "", nil, fmt.Errorf("stage %s: %w", filePath, err)
		}
	}
	return dir, cleanup, nil
}

func linkOrCopyFile(src, dst string) error {
	if err := os.MkdirAll(filepath.Dir(dst), 0o755); err != nil {
		return err
	}
	if err := os.Link(src, dst); err == nil {
		return nil
	}
	if err := os.Symlink(src, dst); err == nil {
		return nil
	}

	in, err := os.Open(src)
	if err != nil {
		return err
	}
	defer in.Close()

	out, err := os.OpenFile(dst, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0o644)
	if err != nil {
		return err
	}
	_, copyErr := io.Copy(out, in)
	closeErr := out.Close()
	if copyErr != nil {
		return copyErr
	}
	return closeErr
}

func applyCreateInfo(config *model.ConfigV2, info map[string]any) error {
	if info == nil {
		return nil
	}
	if caps, ok := info["capabilities"]; ok {
		parsed, err := capabilitiesFromInfo(caps)
		if err != nil {
			return err
		}
		config.Capabilities = append(config.Capabilities, parsed...)
	}

	strFromInfo := func(k string) (string, error) {
		v, ok := info[k]
		if !ok {
			return "", nil
		}
		val, ok := v.(string)
		if !ok {
			return "", fmt.Errorf("info field %q must be a string", k)
		}
		return val, nil
	}
	intFromInfo := func(k string) (int, error) {
		v, ok := info[k]
		if !ok {
			return 0, nil
		}
		val, ok := v.(float64)
		if !ok {
			return 0, fmt.Errorf("info field %q must be a number", k)
		}
		if val < 0 || math.Trunc(val) != val || val > float64(maxCreateInfoInt()) {
			return 0, fmt.Errorf("info field %q must be a non-negative integer", k)
		}
		return int(val), nil
	}

	var err error
	if config.ModelFamily, err = strFromInfo("model_family"); err != nil {
		return err
	}
	if config.ModelFamily != "" {
		config.ModelFamilies = []string{config.ModelFamily}
	}
	if config.BaseName, err = strFromInfo("base_name"); err != nil {
		return err
	}
	if config.FileType, err = strFromInfo("quantization_level"); err != nil {
		return err
	}
	if config.ModelType, err = strFromInfo("parameter_size"); err != nil {
		return err
	}
	if config.ContextLen, err = intFromInfo("context_length"); err != nil {
		return err
	}
	if config.EmbedLen, err = intFromInfo("embedding_length"); err != nil {
		return err
	}
	return nil
}

func capabilitiesFromInfo(v any) ([]string, error) {
	switch caps := v.(type) {
	case []string:
		return append([]string(nil), caps...), nil
	case []any:
		out := make([]string, len(caps))
		for i, c := range caps {
			str, ok := c.(string)
			if !ok {
				return nil, fmt.Errorf("info field %q element %d must be a string", "capabilities", i)
			}
			out[i] = str
		}
		return out, nil
	default:
		return nil, fmt.Errorf("info field %q must be an array of strings", "capabilities")
	}
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
	modelType, err := detectModelTypeFromFiles(files)
	if err != nil {
		return nil, err
	}
	switch modelType {
	case "safetensors":
		return nil, errOnlyGGUFSupported
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

		splitCollector := newSplitGGUFCollector()
		for _, filePath := range filePaths {
			layers, err := ggufLayersWithMediaType(files[filePath], filePath, mediaType, fn)
			if err != nil {
				return nil, err
			}
			for _, layer := range layers {
				if err := splitCollector.Add(layer); err != nil {
					return nil, err
				}
			}
		}

		allLayers, err := splitCollector.Layers()
		if err != nil {
			return nil, err
		}

		if detectTemplate {
			return detectChatTemplate(allLayers)
		}
		return allLayers, nil
	default:
		return nil, errUnknownType
	}
}

func validateCreateFiles(files map[string]string) error {
	for filePath, digest := range files {
		if err := validateCreateFilePath(filePath); err != nil {
			return err
		}
		if digest == "" {
			return manifest.ErrInvalidDigestFormat
		}
		if _, err := manifest.BlobsPath(digest); err != nil {
			return fmt.Errorf("invalid digest for %s: %w", filePath, err)
		}
	}
	return nil
}

func validateCreateFilePath(filePath string) error {
	if filePath == "." || !fs.ValidPath(filePath) || strings.ContainsAny(filePath, `\:`) {
		return fmt.Errorf("%w: %s", errFilePath, filePath)
	}
	return nil
}

func validateCreateOptions(r api.CreateRequest, modelType string) error {
	quantize := cmp.Or(r.Quantize, r.Quantization)
	if modelType == "gguf" || (modelType == "" && r.From != "") {
		if quantize != "" {
			return fmt.Errorf("create-time quantization is only supported for safetensors imports; quantize GGUF models with llama.cpp tools before importing")
		}
		if r.DraftQuantize != "" {
			return fmt.Errorf("draft quantization during create is only supported for safetensors imports; quantize GGUF draft models with llama.cpp tools before importing")
		}
		return nil
	}

	if quantize != "" && quant.Canonical(quantize) == "" {
		return fmt.Errorf("unsupported quantize type %q: supported types are int4, int8, nvfp4, mxfp4, mxfp8", quantize)
	}
	if r.DraftQuantize != "" && quant.Canonical(r.DraftQuantize) == "" {
		return fmt.Errorf("unsupported draft quantize type %q: supported types are int4, int8, nvfp4, mxfp4, mxfp8", r.DraftQuantize)
	}
	return nil
}

func maxCreateInfoInt() int {
	return int(^uint(0) >> 1)
}

func detectModelTypeFromFiles(files map[string]string) (string, error) {
	filePaths := make([]string, 0, len(files))
	for filePath := range files {
		filePaths = append(filePaths, filePath)
	}
	slices.Sort(filePaths)

	var modelType string
	for _, filePath := range filePaths {
		t, err := detectModelTypeFromFile(filePath, files[filePath])
		if err != nil {
			return "", err
		}
		if t == "" {
			continue
		}
		if modelType != "" && modelType != t {
			return "", fmt.Errorf("%w: found both %s and %s inputs", errMixedModelTypes, modelType, t)
		}
		modelType = t
	}

	return modelType, nil
}

func detectModelTypeFromFile(filePath, digest string) (string, error) {
	if strings.HasSuffix(filePath, ".safetensors") {
		return "safetensors", nil
	}
	if strings.HasSuffix(filePath, ".gguf") {
		return "gguf", nil
	}

	// Try to detect GGUF files even when the source name has no extension.
	blobPath, err := manifest.BlobsPath(digest)
	if err != nil {
		return "", fmt.Errorf("blob path for %s: %w", filePath, err)
	}

	f, err := os.Open(blobPath)
	if err != nil {
		return "", fmt.Errorf("read %s: %w", filePath, err)
	}
	defer f.Close()

	buf := make([]byte, 4)
	if _, err := io.ReadFull(f, buf); err != nil {
		if !errors.Is(err, io.EOF) && !errors.Is(err, io.ErrUnexpectedEOF) {
			return "", fmt.Errorf("read %s: %w", filePath, err)
		}
		return "", nil
	}
	if ggml.DetectContentType(buf) == "gguf" {
		return "gguf", nil
	}
	return "", nil
}

func createModel(r api.CreateRequest, name model.Name, baseLayers []*layerGGML, config *model.ConfigV2, fn func(resp api.ProgressResponse)) (err error) {
	if quantize := cmp.Or(r.Quantize, r.Quantization); quantize != "" {
		return fmt.Errorf("create-time quantization is only supported for safetensors imports; quantize GGUF models with llama.cpp tools before importing")
	}
	if r.DraftQuantize != "" {
		return fmt.Errorf("draft quantization during create is only supported for safetensors imports; quantize GGUF draft models with llama.cpp tools before importing")
	}

	var layers []manifest.Layer
	for _, layer := range baseLayers {
		if len(layer.splitParts) > 0 {
			fn(api.ProgressResponse{Status: "merging split GGUF"})
			layer, err = mergeSplitGGUFToLayer(layer)
			if err != nil {
				return err
			}
		}

		if layer.GGML != nil {
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

	configLayer, err := createConfigLayer(*config)
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

	layers = append(layers, &layerGGML{Layer: layer, GGML: f})

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

func createConfigLayer(config model.ConfigV2) (*manifest.Layer, error) {
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
