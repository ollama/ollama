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
	"slices"
	"strings"

	"github.com/gin-gonic/gin"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/compatmigrate"
	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/format"
	"github.com/ollama/ollama/fs/ggml"
	"github.com/ollama/ollama/manifest"
	"github.com/ollama/ollama/template"
	"github.com/ollama/ollama/types/errtypes"
	"github.com/ollama/ollama/types/model"
	xcreate "github.com/ollama/ollama/x/create"
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
	errMixedModelFiles         = errors.New("mixed model file types are not supported")
	errInvalidSplitGGUF        = errors.New("invalid split GGUF")

	errGGUFQuantizeUnsupported      = errors.New("create-time quantization is only supported for safetensors imports; quantize GGUF models with llama.cpp tools before importing")
	errGGUFDraftQuantizeUnsupported = errors.New("draft quantization during create is only supported for safetensors imports; quantize GGUF draft models with llama.cpp tools before importing")
)

type manifestListRequestError struct {
	err error
}

func (e manifestListRequestError) Error() string {
	return e.err.Error()
}

func newManifestListRequestError(format string, args ...any) error {
	return manifestListRequestError{err: fmt.Errorf(format, args...)}
}

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

	reqCtx := c.Request.Context()
	ch := make(chan any)
	go func() {
		defer close(ch)
		send := func(resp any) bool {
			select {
			case ch <- resp:
				return true
			case <-reqCtx.Done():
				return false
			}
		}
		fn := func(resp api.ProgressResponse) {
			send(resp)
		}

		oldManifestDigests, _ := manifest.ReferencedBlobDigestsForName(name)

		if len(r.List) > 0 {
			if err := createManifestList(r, name, fn); err != nil {
				status := http.StatusInternalServerError
				var requestErr manifestListRequestError
				if errors.As(err, &requestErr) {
					status = http.StatusBadRequest
				}
				send(gin.H{"error": err.Error(), "status": status})
				return
			}

			if !envconfig.NoPrune() && len(oldManifestDigests) > 0 {
				if _, err := manifest.RemoveUnreferencedBlobs(oldManifestDigests...); err != nil {
					send(gin.H{"error": err.Error()})
					return
				}
			}

			send(api.ProgressResponse{Status: "success"})
			return
		}

		if detectModelTypeFromFiles(r.Files) == "safetensors" {
			if err := applyCreateInfo(config, r.Info); err != nil {
				send(gin.H{"error": err.Error(), "status": http.StatusBadRequest})
				return
			}
			if err := createSafetensorsModel(reqCtx, r, name, config, fn); err != nil {
				send(createSafetensorsErrorResponse(err))
				return
			}
			if !envconfig.NoPrune() && len(oldManifestDigests) > 0 {
				if _, err := manifest.RemoveUnreferencedBlobs(oldManifestDigests...); err != nil {
					send(gin.H{"error": err.Error()})
					return
				}
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
				for _, badReq := range []error{errNoFilesProvided, errOnlyGGUFSupported, errMixedModelFiles, errUnknownType, errInvalidSplitGGUF} {
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
				for _, badReq := range []error{errNoFilesProvided, errOnlyGGUFSupported, errMixedModelFiles, errUnknownType, errFilePath, errInvalidSplitGGUF} {
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
				for _, badReq := range []error{errNoFilesProvided, errOnlyOneAdapterSupported, errOnlyGGUFSupported, errSafetensorsAdapters, errMixedModelFiles, errUnknownType, errFilePath, errInvalidSplitGGUF} {
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
			if errors.Is(err, errBadTemplate) || errors.Is(err, errInvalidSplitGGUF) ||
				errors.Is(err, errGGUFQuantizeUnsupported) || errors.Is(err, errGGUFDraftQuantizeUnsupported) {
				send(gin.H{"error": err.Error(), "status": http.StatusBadRequest})
				return
			}
			send(gin.H{"error": err.Error()})
			return
		}

		if !envconfig.NoPrune() && len(oldManifestDigests) > 0 {
			if _, err := manifest.RemoveUnreferencedBlobs(oldManifestDigests...); err != nil {
				send(gin.H{"error": err.Error()})
			}
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

const safetensorsCreateMinVersion = "0.19.0"

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
	store := xcreate.StoreFromLayerCreator(xcreate.ManifestLayerCreator())

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
		writeManifest = func(modelName string, cfg xcreate.LayerInfo, layers []xcreate.LayerInfo) error {
			layers = append(layers, draftLayers...)
			return next(modelName, cfg, layers)
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

func writeSafetensorsManifest(config *model.ConfigV2, r api.CreateRequest, draftDir string, fn func(resp api.ProgressResponse)) xcreate.ManifestWriter {
	capabilities := append([]string(nil), config.Capabilities...)
	if len(capabilities) == 0 {
		capabilities = []string{"completion"}
	}
	baseConfig := *config
	return xcreate.NewSafetensorsManifestWriter(xcreate.SafetensorsManifestOptions{
		BaseConfig:          &baseConfig,
		Quantize:            cmp.Or(r.Quantize, r.Quantization),
		Capabilities:        capabilities,
		MinVersion:          safetensorsCreateMinVersion,
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
		IncludeRootFSDiffs:  true,
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
		if !fs.ValidPath(filePath) || strings.Contains(filePath, `\`) {
			cleanup()
			return "", nil, fmt.Errorf("%w: %s", errFilePath, filePath)
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
	if caps, ok := info["capabilities"].([]any); ok {
		for _, c := range caps {
			if str, ok := c.(string); ok {
				config.Capabilities = append(config.Capabilities, str)
			}
		}
	} else if caps, ok := info["capabilities"].([]string); ok {
		config.Capabilities = append(config.Capabilities, caps...)
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
		if isAdapter {
			return nil, errSafetensorsAdapters
		}
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
	case "mixed":
		return nil, errMixedModelFiles
	default:
		return nil, errUnknownType
	}
}

func maxCreateInfoInt() int {
	return int(^uint(0) >> 1)
}

func detectModelTypeFromFiles(files map[string]string) string {
	var detected string
	fileNames := make([]string, 0, len(files))
	for name := range files {
		fileNames = append(fileNames, name)
	}
	slices.Sort(fileNames)

	for _, name := range fileNames {
		fileType := detectModelTypeFromFile(name, files[name])
		if fileType == "" {
			continue
		}
		if detected == "" {
			detected = fileType
			continue
		}
		if detected != fileType {
			return "mixed"
		}
	}

	return detected
}

func detectModelTypeFromFile(name, digest string) string {
	if strings.HasSuffix(name, ".safetensors") {
		return "safetensors"
	}
	if strings.HasSuffix(name, ".gguf") {
		return "gguf"
	}

	// Try to detect GGUF inputs even when the file extension is absent.
	blobPath, err := manifest.BlobsPath(digest)
	if err != nil {
		slog.Error("error getting blobs path", "file", name)
		return ""
	}

	f, err := os.Open(blobPath)
	if err != nil {
		slog.Error("error reading file", "error", err)
		return ""
	}
	defer f.Close()

	buf := make([]byte, 4)
	if _, err := f.Read(buf); err != nil {
		slog.Error("error reading file", "error", err)
		return ""
	}

	return ggml.DetectContentType(buf)
}

func createModel(r api.CreateRequest, name model.Name, baseLayers []*layerGGML, config *model.ConfigV2, fn func(resp api.ProgressResponse)) (err error) {
	if quantize := cmp.Or(r.Quantize, r.Quantization); quantize != "" {
		return errGGUFQuantizeUnsupported
	}
	if r.DraftQuantize != "" {
		return errGGUFDraftQuantizeUnsupported
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
			case "application/vnd.ollama.image.model", manifest.MediaTypeImageDraft:
				rewritten, changed, err := compatmigrate.RewriteLlama3MetadataLayer(layer.Layer)
				if err != nil {
					return err
				}
				if changed {
					layer.Layer = rewritten
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

	configLayer, err := manifest.NewConfigLayer(*config, layers, manifest.IncludeRootFSDiffIDs)
	if err != nil {
		return err
	}

	for _, layer := range layers {
		if layer.Status != "" {
			fn(api.ProgressResponse{Status: layer.Status})
		}
	}

	fn(api.ProgressResponse{Status: "writing manifest"})
	runner, format := manifest.MetadataForConfig(*config)
	if err := manifest.WriteManifestWithMetadata(name, configLayer, layers, runner, format); err != nil {
		return err
	}

	return nil
}

func createManifestList(r api.CreateRequest, name model.Name, fn func(resp api.ProgressResponse)) error {
	if err := validateCreateManifestListRequest(r); err != nil {
		return err
	}

	manifests := make([]manifest.Manifest, 0, len(r.List))
	seenDigests := make(map[string]string)
	seenRunners := make(map[string]string)
	for _, ref := range r.List {
		ref = strings.TrimSpace(ref)
		if ref == "" {
			return newManifestListRequestError("manifest list contains an empty model")
		}

		fn(api.ProgressResponse{Status: fmt.Sprintf("reading manifest %s", ref)})

		modelRef, err := parseAndValidateModelRef(ref)
		if err != nil {
			return err
		}
		if modelRef.Source == modelSourceCloud {
			return newManifestListRequestError("manifest list entries must be local models: %s", ref)
		}

		childName, err := getExistingName(modelRef.Name)
		if err != nil {
			return err
		}

		data, err := manifest.ReadManifestData(childName)
		if err != nil {
			return fmt.Errorf("read manifest %s: %w", ref, err)
		}

		var child manifest.Manifest
		if err := json.Unmarshal(data, &child); err != nil {
			return err
		}
		if child.MediaType == manifest.MediaTypeManifestList {
			return newManifestListRequestError("manifest list entry %s is already a manifest list", ref)
		}

		if err := manifest.FillMetadata(&child); err != nil {
			return fmt.Errorf("manifest list entry %s: %w", ref, err)
		}

		childData, err := json.Marshal(child)
		if err != nil {
			return err
		}
		childDigest, err := manifest.WriteManifestBlob(childData)
		if err != nil {
			return err
		}

		if previous, ok := seenDigests[childDigest]; ok {
			return newManifestListRequestError("manifest list entries %s and %s resolve to the same manifest", previous, ref)
		}
		runner := strings.ToLower(strings.TrimSpace(child.Runner))
		if previous, ok := seenRunners[runner]; ok {
			return newManifestListRequestError("manifest list entries %s and %s use the same runner %q", previous, ref, child.Runner)
		}
		seenDigests[childDigest] = ref
		seenRunners[runner] = ref

		childRef, err := manifest.NewManifestReference(childDigest, child.Runner, child.Format)
		if err != nil {
			return err
		}

		manifests = append(manifests, childRef)
	}

	fn(api.ProgressResponse{Status: "writing manifest list"})
	return manifest.WriteManifestList(name, manifests)
}

func validateCreateManifestListRequest(r api.CreateRequest) error {
	if len(r.List) == 0 {
		return newManifestListRequestError("manifest list must contain at least one model")
	}

	switch {
	case r.From != "", r.RemoteHost != "", len(r.Files) > 0, len(r.Adapters) > 0, len(r.DraftFiles) > 0:
		return newManifestListRequestError("manifest list creation cannot be combined with model creation options")
	case r.Template != "", r.System != "", r.License != nil, len(r.Parameters) > 0, len(r.Messages) > 0:
		return newManifestListRequestError("manifest list creation cannot be combined with model creation options")
	case r.Renderer != "", r.Parser != "", r.Requires != "", len(r.Info) > 0:
		return newManifestListRequestError("manifest list creation cannot be combined with model creation options")
	case r.Quantize != "", r.Quantization != "", r.DraftQuantize != "":
		return newManifestListRequestError("manifest list creation cannot be combined with model creation options")
	default:
		return nil
	}
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
