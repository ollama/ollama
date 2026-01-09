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
	"net"
	"net/http"
	"net/url"
	"os"
	"path"
	"path/filepath"
	"slices"
	"strings"
	"sync/atomic"

	"github.com/gin-gonic/gin"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/convert"
	"github.com/ollama/ollama/envconfig"
	"github.com/ollama/ollama/format"
	ofs "github.com/ollama/ollama/fs"
	"github.com/ollama/ollama/fs/ggml"
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

	for v := range r.Files {
		if !fs.ValidPath(v) {
			c.AbortWithStatusJSON(http.StatusBadRequest, gin.H{"error": errFilePath.Error()})
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

		oldManifest, _ := ParseNamedManifest(name)

		var baseLayers []*layerGGML
		var err error
		var remote bool

		if r.From != "" {
			slog.Debug("create model from model name", "from", r.From)
			fromName := model.ParseName(r.From)
			if !fromName.IsValid() {
				ch <- gin.H{"error": errtypes.InvalidModelNameErrMsg, "status": http.StatusBadRequest}
				return
			}
			if r.RemoteHost != "" {
				ru, err := remoteURL(r.RemoteHost)
				if err != nil {
					ch <- gin.H{"error": "bad remote", "status": http.StatusBadRequest}
					return
				}

				config.RemoteModel = r.From
				config.RemoteHost = ru
				remote = true
			} else {
				ctx, cancel := context.WithCancel(c.Request.Context())
				defer cancel()

				baseLayers, err = parseFromModel(ctx, fromName, fn)
				if err != nil {
					ch <- gin.H{"error": err.Error()}
				}

				if err == nil && !remote && (config.Renderer == "" || config.Parser == "" || config.Requires == "") {
					manifest, mErr := ParseNamedManifest(fromName)
					if mErr == nil && manifest.Config.Digest != "" {
						configPath, pErr := GetBlobsPath(manifest.Config.Digest)
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
	switch detectModelTypeFromFiles(files) {
	case "safetensors":
		layers, err := convertFromSafetensors(files, baseLayers, isAdapter, fn)
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

		var digest string
		var allLayers []*layerGGML
		for _, v := range files {
			digest = v
			layers, err := ggufLayers(digest, fn)
			if err != nil {
				return nil, err
			}
			allLayers = append(allLayers, layers...)
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
			blobPath, err := GetBlobsPath(files[fn])
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

func convertFromSafetensors(files map[string]string, baseLayers []*layerGGML, isAdapter bool, fn func(resp api.ProgressResponse)) ([]*layerGGML, error) {
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

		blobPath, err := GetBlobsPath(digest)
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

	var mediaType string
	if !isAdapter {
		fn(api.ProgressResponse{Status: "converting model"})
		mediaType = "application/vnd.ollama.image.model"
		if err := convert.ConvertModel(os.DirFS(tmpDir), t); err != nil {
			return nil, err
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

	layer, err := NewLayer(t, mediaType)
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
	layers := []*layerGGML{{layer, f}}

	if !isAdapter {
		return detectChatTemplate(layers)
	}
	return layers, nil
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
	var layers []Layer
	for _, layer := range baseLayers {
		if layer.GGML != nil {
			quantType := strings.ToUpper(cmp.Or(r.Quantize, r.Quantization))
			if quantType != "" && layer.GGML.Name() == "gguf" && layer.MediaType == "application/vnd.ollama.image.model" {
				want, err := ggml.ParseFileType(quantType)
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

	configLayer, err := createConfigLayer(layers, *config)
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
	var doneBytes atomic.Uint64
	totalBytes := uint64(layer.Size) - layer.GGML.Tensors().Offset
	fnWrap := func(n uint64) {
		done := doneBytes.Add(n)
		progress := float32(done) / float32(totalBytes)
		fn(api.ProgressResponse{Status: fmt.Sprintf("quantizing %s model to %s", ft, quantizeType), Digest: "0000000000000000000", Total: layer.Size, Completed: int64(progress * float32(layer.Size))})
	}
	ftype, err := ggml.ParseFileType(quantizeType)
	if err != nil {
		return nil, err
	}

	blob, err := GetBlobsPath(layer.Digest)
	if err != nil {
		return nil, err
	}
	fp, err := os.Open(blob)
	if err != nil {
		return nil, err
	}
	defer fp.Close()

	temp, err := os.CreateTemp(filepath.Dir(blob), quantizeType)
	if err != nil {
		return nil, err
	}
	defer temp.Close()
	defer os.Remove(temp.Name())

	if err := quantize(fp, temp, layer.GGML, ftype, fnWrap); err != nil {
		return nil, err
	}
	temp.Seek(0, io.SeekStart)
	fn(api.ProgressResponse{Status: "verifying conversion"})
	newLayer, err := NewLayer(temp, layer.MediaType)
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
	return &layerGGML{newLayer, f}, nil
}

func ggufLayers(digest string, fn func(resp api.ProgressResponse)) ([]*layerGGML, error) {
	var layers []*layerGGML

	fn(api.ProgressResponse{Status: "parsing GGUF"})
	blobPath, err := GetBlobsPath(digest)
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

	mediatype := "application/vnd.ollama.image.model"
	if f.KV().Kind() == "adapter" {
		mediatype = "application/vnd.ollama.image.adapter"
	} else if (f.KV().Uint("block_count") == 0 && f.KV().Uint("vision.block_count") > 0) || f.KV().Kind() == "projector" {
		// if a model has vision.block_count but not block_count, it is a standalone vision model
		mediatype = "application/vnd.ollama.image.projector"
	}

	layer, err := NewLayerFromLayer(digest, mediatype, blob.Name())
	if err != nil {
		slog.Debug("could not create new layer from layer", "error", err)
		return nil, err
	}

	layers = append(layers, &layerGGML{layer, f})

	return detectChatTemplate(layers)
}

func removeLayer(layers []Layer, mediatype string) []Layer {
	return slices.DeleteFunc(layers, func(layer Layer) bool {
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

func createConfigLayer(layers []Layer, config model.ConfigV2) (*Layer, error) {
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
