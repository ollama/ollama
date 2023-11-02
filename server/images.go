package server

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"reflect"
	"runtime"
	"strconv"
	"strings"
	"text/template"

	"golang.org/x/exp/slices"

	"github.com/jmorganca/ollama/api"
	"github.com/jmorganca/ollama/llm"
	"github.com/jmorganca/ollama/parser"
	"github.com/jmorganca/ollama/version"
)

type RegistryOptions struct {
	Insecure bool
	Username string
	Password string
	Token    string
}

type Model struct {
	Name          string `json:"name"`
	ShortName     string
	ModelPath     string
	OriginalModel string
	AdapterPaths  []string
	Template      string
	System        string
	License       []string
	Digest        string
	Options       map[string]interface{}
}

func (m *Model) Prompt(request api.GenerateRequest) (string, error) {
	t := m.Template
	if request.Template != "" {
		t = request.Template
	}

	tmpl, err := template.New("").Parse(t)
	if err != nil {
		return "", err
	}

	var vars struct {
		First  bool
		System string
		Prompt string

		// deprecated: versions <= 0.0.7 used this to omit the system prompt
		Context []int
	}

	vars.First = len(request.Context) == 0
	vars.System = m.System
	vars.Prompt = request.Prompt
	vars.Context = request.Context

	if request.System != "" {
		vars.System = request.System
	}

	var sb strings.Builder
	if err := tmpl.Execute(&sb, vars); err != nil {
		return "", err
	}

	return sb.String(), nil
}

type ManifestV2 struct {
	SchemaVersion int      `json:"schemaVersion"`
	MediaType     string   `json:"mediaType"`
	Config        Layer    `json:"config"`
	Layers        []*Layer `json:"layers"`
}

type Layer struct {
	MediaType string `json:"mediaType"`
	Digest    string `json:"digest"`
	Size      int64  `json:"size"`
	From      string `json:"from,omitempty"`
}

type LayerReader struct {
	Layer
	io.Reader
}

type ConfigV2 struct {
	ModelFormat string `json:"model_format"`
	ModelFamily string `json:"model_family"`
	ModelType   string `json:"model_type"`
	FileType    string `json:"file_type"`
	RootFS      RootFS `json:"rootfs"`

	// required by spec
	Architecture string `json:"architecture"`
	OS           string `json:"os"`
}

type RootFS struct {
	Type    string   `json:"type"`
	DiffIDs []string `json:"diff_ids"`
}

func (m *ManifestV2) GetTotalSize() (total int64) {
	for _, layer := range m.Layers {
		total += layer.Size
	}

	total += m.Config.Size
	return total
}

func GetManifest(mp ModelPath) (*ManifestV2, string, error) {
	fp, err := mp.GetManifestPath()
	if err != nil {
		return nil, "", err
	}

	if _, err = os.Stat(fp); err != nil {
		return nil, "", err
	}

	var manifest *ManifestV2

	bts, err := os.ReadFile(fp)
	if err != nil {
		return nil, "", fmt.Errorf("couldn't open file '%s'", fp)
	}

	shaSum := sha256.Sum256(bts)
	shaStr := hex.EncodeToString(shaSum[:])

	if err := json.Unmarshal(bts, &manifest); err != nil {
		return nil, "", err
	}

	return manifest, shaStr, nil
}

func GetModel(name string) (*Model, error) {
	mp := ParseModelPath(name)
	manifest, digest, err := GetManifest(mp)
	if err != nil {
		return nil, err
	}

	model := &Model{
		Name:      mp.GetFullTagname(),
		ShortName: mp.GetShortTagname(),
		Digest:    digest,
		Template:  "{{ .Prompt }}",
		License:   []string{},
	}

	for _, layer := range manifest.Layers {
		filename, err := GetBlobsPath(layer.Digest)
		if err != nil {
			return nil, err
		}

		switch layer.MediaType {
		case "application/vnd.ollama.image.model":
			model.ModelPath = filename
			model.OriginalModel = layer.From
		case "application/vnd.ollama.image.embed":
			// Deprecated in versions  > 0.1.2
			// TODO: remove this warning in a future version
			log.Print("WARNING: model contains embeddings, but embeddings in modelfiles have been deprecated and will be ignored.")
		case "application/vnd.ollama.image.adapter":
			model.AdapterPaths = append(model.AdapterPaths, filename)
		case "application/vnd.ollama.image.template":
			bts, err := os.ReadFile(filename)
			if err != nil {
				return nil, err
			}

			model.Template = string(bts)
		case "application/vnd.ollama.image.system":
			bts, err := os.ReadFile(filename)
			if err != nil {
				return nil, err
			}

			model.System = string(bts)
		case "application/vnd.ollama.image.prompt":
			bts, err := os.ReadFile(filename)
			if err != nil {
				return nil, err
			}

			model.Template = string(bts)
		case "application/vnd.ollama.image.params":
			params, err := os.Open(filename)
			if err != nil {
				return nil, err
			}
			defer params.Close()

			// parse model options parameters into a map so that we can see which fields have been specified explicitly
			if err = json.NewDecoder(params).Decode(&model.Options); err != nil {
				return nil, err
			}
		case "application/vnd.ollama.image.license":
			bts, err := os.ReadFile(filename)
			if err != nil {
				return nil, err
			}
			model.License = append(model.License, string(bts))
		}
	}

	return model, nil
}

func filenameWithPath(path, f string) (string, error) {
	// if filePath starts with ~/, replace it with the user's home directory.
	if strings.HasPrefix(f, fmt.Sprintf("~%s", string(os.PathSeparator))) {
		parts := strings.Split(f, string(os.PathSeparator))
		home, err := os.UserHomeDir()
		if err != nil {
			return "", fmt.Errorf("failed to open file: %v", err)
		}

		f = filepath.Join(home, filepath.Join(parts[1:]...))
	}

	// if filePath is not an absolute path, make it relative to the modelfile path
	if !filepath.IsAbs(f) {
		f = filepath.Join(filepath.Dir(path), f)
	}

	return f, nil
}

func CreateModel(ctx context.Context, name string, path string, fn func(resp api.ProgressResponse)) error {
	mp := ParseModelPath(name)

	var manifest *ManifestV2
	var err error
	var noprune string

	// build deleteMap to prune unused layers
	deleteMap := make(map[string]bool)

	if noprune = os.Getenv("OLLAMA_NOPRUNE"); noprune == "" {
		manifest, _, err = GetManifest(mp)
		if err != nil && !errors.Is(err, os.ErrNotExist) {
			return err
		}

		if manifest != nil {
			for _, l := range manifest.Layers {
				deleteMap[l.Digest] = true
			}
			deleteMap[manifest.Config.Digest] = true
		}
	}

	mf, err := os.Open(path)
	if err != nil {
		fn(api.ProgressResponse{Status: fmt.Sprintf("couldn't open modelfile '%s'", path)})
		return fmt.Errorf("failed to open file: %w", err)
	}
	defer mf.Close()

	fn(api.ProgressResponse{Status: "parsing modelfile"})
	commands, err := parser.Parse(mf)
	if err != nil {
		return err
	}

	config := ConfigV2{
		Architecture: "amd64",
		OS:           "linux",
	}

	var layers []*LayerReader
	params := make(map[string][]string)
	var sourceParams map[string]any
	for _, c := range commands {
		log.Printf("[%s] - %s\n", c.Name, c.Args)
		switch c.Name {
		case "model":
			fn(api.ProgressResponse{Status: "looking for model"})

			mp := ParseModelPath(c.Args)
			mf, _, err := GetManifest(mp)
			if err != nil {
				modelFile, err := filenameWithPath(path, c.Args)
				if err != nil {
					return err
				}
				if _, err := os.Stat(modelFile); err != nil {
					// the model file does not exist, try pulling it
					if errors.Is(err, os.ErrNotExist) {
						fn(api.ProgressResponse{Status: "pulling model file"})
						if err := PullModel(ctx, c.Args, &RegistryOptions{}, fn); err != nil {
							return err
						}
						mf, _, err = GetManifest(mp)
						if err != nil {
							return fmt.Errorf("failed to open file after pull: %v", err)
						}
					} else {
						return err
					}
				} else {
					// create a model from this specified file
					fn(api.ProgressResponse{Status: "creating model layer"})
					file, err := os.Open(modelFile)
					if err != nil {
						return fmt.Errorf("failed to open file: %v", err)
					}
					defer file.Close()

					ggml, err := llm.DecodeGGML(file)
					if err != nil {
						return err
					}

					config.ModelFormat = ggml.Name()
					config.ModelFamily = ggml.ModelFamily()
					config.ModelType = ggml.ModelType()
					config.FileType = ggml.FileType()

					// reset the file
					file.Seek(0, io.SeekStart)

					l, err := CreateLayer(file)
					if err != nil {
						return fmt.Errorf("failed to create layer: %v", err)
					}
					l.MediaType = "application/vnd.ollama.image.model"
					layers = append(layers, l)
				}
			}

			if mf != nil {
				fn(api.ProgressResponse{Status: "reading model metadata"})
				sourceBlobPath, err := GetBlobsPath(mf.Config.Digest)
				if err != nil {
					return err
				}

				sourceBlob, err := os.Open(sourceBlobPath)
				if err != nil {
					return err
				}
				defer sourceBlob.Close()

				var source ConfigV2
				if err := json.NewDecoder(sourceBlob).Decode(&source); err != nil {
					return err
				}

				// copy the model metadata
				config.ModelFamily = source.ModelFamily
				config.ModelType = source.ModelType
				config.ModelFormat = source.ModelFormat
				config.FileType = source.FileType

				for _, l := range mf.Layers {
					if l.MediaType == "application/vnd.ollama.image.params" {
						sourceParamsBlobPath, err := GetBlobsPath(l.Digest)
						if err != nil {
							return err
						}

						sourceParamsBlob, err := os.Open(sourceParamsBlobPath)
						if err != nil {
							return err
						}
						defer sourceParamsBlob.Close()

						if err := json.NewDecoder(sourceParamsBlob).Decode(&sourceParams); err != nil {
							return err
						}
					}

					newLayer, err := GetLayerWithBufferFromLayer(l)
					if err != nil {
						return err
					}
					newLayer.From = mp.GetNamespaceRepository()
					layers = append(layers, newLayer)
				}
			}
		case "adapter":
			fn(api.ProgressResponse{Status: fmt.Sprintf("creating model %s layer", c.Name)})

			fp, err := filenameWithPath(path, c.Args)
			if err != nil {
				return err
			}

			// create a model from this specified file
			fn(api.ProgressResponse{Status: "creating model layer"})

			file, err := os.Open(fp)
			if err != nil {
				return fmt.Errorf("failed to open file: %v", err)
			}
			defer file.Close()

			l, err := CreateLayer(file)
			if err != nil {
				return fmt.Errorf("failed to create layer: %v", err)
			}
			l.MediaType = "application/vnd.ollama.image.adapter"
			layers = append(layers, l)
		case "license":
			fn(api.ProgressResponse{Status: fmt.Sprintf("creating model %s layer", c.Name)})
			mediaType := fmt.Sprintf("application/vnd.ollama.image.%s", c.Name)

			layer, err := CreateLayer(strings.NewReader(c.Args))
			if err != nil {
				return err
			}

			if layer.Size > 0 {
				layer.MediaType = mediaType
				layers = append(layers, layer)
			}
		case "template", "system", "prompt":
			fn(api.ProgressResponse{Status: fmt.Sprintf("creating model %s layer", c.Name)})
			// remove the layer if one exists
			mediaType := fmt.Sprintf("application/vnd.ollama.image.%s", c.Name)
			layers = removeLayerFromLayers(layers, mediaType)

			layer, err := CreateLayer(strings.NewReader(c.Args))
			if err != nil {
				return err
			}

			if layer.Size > 0 {
				layer.MediaType = mediaType
				layers = append(layers, layer)
			}
		default:
			// runtime parameters, build a list of args for each parameter to allow multiple values to be specified (ex: multiple stop sequences)
			params[c.Name] = append(params[c.Name], c.Args)
		}
	}

	// Create a single layer for the parameters
	if len(params) > 0 {
		fn(api.ProgressResponse{Status: "creating parameter layer"})

		layers = removeLayerFromLayers(layers, "application/vnd.ollama.image.params")
		formattedParams, err := formatParams(params)
		if err != nil {
			return fmt.Errorf("couldn't create params json: %v", err)
		}

		for k, v := range sourceParams {
			if _, ok := formattedParams[k]; !ok {
				formattedParams[k] = v
			}
		}

		if config.ModelType == "65B" {
			if numGQA, ok := formattedParams["num_gqa"].(int); ok && numGQA == 8 {
				config.ModelType = "70B"
			}
		}

		bts, err := json.Marshal(formattedParams)
		if err != nil {
			return err
		}

		l, err := CreateLayer(bytes.NewReader(bts))
		if err != nil {
			return fmt.Errorf("failed to create layer: %v", err)
		}
		l.MediaType = "application/vnd.ollama.image.params"
		layers = append(layers, l)
	}

	digests, err := getLayerDigests(layers)
	if err != nil {
		return err
	}

	var manifestLayers []*Layer
	for _, l := range layers {
		manifestLayers = append(manifestLayers, &l.Layer)
		delete(deleteMap, l.Layer.Digest)
	}

	// Create a layer for the config object
	fn(api.ProgressResponse{Status: "creating config layer"})
	cfg, err := createConfigLayer(config, digests)
	if err != nil {
		return err
	}
	layers = append(layers, cfg)
	delete(deleteMap, cfg.Layer.Digest)

	if err := SaveLayers(layers, fn, false); err != nil {
		return err
	}

	// Create the manifest
	fn(api.ProgressResponse{Status: "writing manifest"})
	err = CreateManifest(name, cfg, manifestLayers)
	if err != nil {
		return err
	}

	if noprune == "" {
		fn(api.ProgressResponse{Status: "removing any unused layers"})
		err = deleteUnusedLayers(nil, deleteMap, false)
		if err != nil {
			return err
		}
	}

	fn(api.ProgressResponse{Status: "success"})
	return nil
}

func removeLayerFromLayers(layers []*LayerReader, mediaType string) []*LayerReader {
	return slices.DeleteFunc(layers, func(layer *LayerReader) bool {
		return layer.MediaType == mediaType
	})
}

func SaveLayers(layers []*LayerReader, fn func(resp api.ProgressResponse), force bool) error {
	// Write each of the layers to disk
	for _, layer := range layers {
		fp, err := GetBlobsPath(layer.Digest)
		if err != nil {
			return err
		}

		_, err = os.Stat(fp)
		if os.IsNotExist(err) || force {
			fn(api.ProgressResponse{Status: fmt.Sprintf("writing layer %s", layer.Digest)})

			out, err := os.Create(fp)
			if err != nil {
				log.Printf("couldn't create %s", fp)
				return err
			}
			defer out.Close()

			if _, err = io.Copy(out, layer.Reader); err != nil {
				return err
			}

		} else {
			fn(api.ProgressResponse{Status: fmt.Sprintf("using already created layer %s", layer.Digest)})
		}
	}

	return nil
}

func CreateManifest(name string, cfg *LayerReader, layers []*Layer) error {
	mp := ParseModelPath(name)
	manifest := ManifestV2{
		SchemaVersion: 2,
		MediaType:     "application/vnd.docker.distribution.manifest.v2+json",
		Config: Layer{
			MediaType: cfg.MediaType,
			Size:      cfg.Size,
			Digest:    cfg.Digest,
		},
		Layers: layers,
	}

	manifestJSON, err := json.Marshal(manifest)
	if err != nil {
		return err
	}

	fp, err := mp.GetManifestPath()
	if err != nil {
		return err
	}
	if err := os.MkdirAll(filepath.Dir(fp), 0o755); err != nil {
		return err
	}
	return os.WriteFile(fp, manifestJSON, 0o644)
}

func GetLayerWithBufferFromLayer(layer *Layer) (*LayerReader, error) {
	fp, err := GetBlobsPath(layer.Digest)
	if err != nil {
		return nil, err
	}

	file, err := os.Open(fp)
	if err != nil {
		return nil, fmt.Errorf("could not open blob: %w", err)
	}
	defer file.Close()

	newLayer, err := CreateLayer(file)
	if err != nil {
		return nil, err
	}
	newLayer.MediaType = layer.MediaType
	return newLayer, nil
}

// formatParams converts specified parameter options to their correct types
func formatParams(params map[string][]string) (map[string]interface{}, error) {
	opts := api.Options{}
	valueOpts := reflect.ValueOf(&opts).Elem() // names of the fields in the options struct
	typeOpts := reflect.TypeOf(opts)           // types of the fields in the options struct

	// build map of json struct tags to their types
	jsonOpts := make(map[string]reflect.StructField)
	for _, field := range reflect.VisibleFields(typeOpts) {
		jsonTag := strings.Split(field.Tag.Get("json"), ",")[0]
		if jsonTag != "" {
			jsonOpts[jsonTag] = field
		}
	}

	out := make(map[string]interface{})
	// iterate params and set values based on json struct tags
	for key, vals := range params {
		if opt, ok := jsonOpts[key]; ok {
			field := valueOpts.FieldByName(opt.Name)
			if field.IsValid() && field.CanSet() {
				switch field.Kind() {
				case reflect.Float32:
					floatVal, err := strconv.ParseFloat(vals[0], 32)
					if err != nil {
						return nil, fmt.Errorf("invalid float value %s", vals)
					}

					out[key] = float32(floatVal)
				case reflect.Int:
					intVal, err := strconv.ParseInt(vals[0], 10, 64)
					if err != nil {
						return nil, fmt.Errorf("invalid int value %s", vals)
					}

					out[key] = intVal
				case reflect.Bool:
					boolVal, err := strconv.ParseBool(vals[0])
					if err != nil {
						return nil, fmt.Errorf("invalid bool value %s", vals)
					}

					out[key] = boolVal
				case reflect.String:
					out[key] = vals[0]
				case reflect.Slice:
					// TODO: only string slices are supported right now
					out[key] = vals
				default:
					return nil, fmt.Errorf("unknown type %s for %s", field.Kind(), key)
				}
			}
		}
	}

	return out, nil
}

func getLayerDigests(layers []*LayerReader) ([]string, error) {
	var digests []string
	for _, l := range layers {
		if l.Digest == "" {
			return nil, fmt.Errorf("layer is missing a digest")
		}
		digests = append(digests, l.Digest)
	}
	return digests, nil
}

// CreateLayer creates a Layer object from a given file
func CreateLayer(f io.ReadSeeker) (*LayerReader, error) {
	digest, size := GetSHA256Digest(f)
	f.Seek(0, io.SeekStart)

	layer := &LayerReader{
		Layer: Layer{
			MediaType: "application/vnd.docker.image.rootfs.diff.tar",
			Digest:    digest,
			Size:      size,
		},
		Reader: f,
	}

	return layer, nil
}

func CopyModel(src, dest string) error {
	srcModelPath := ParseModelPath(src)
	srcPath, err := srcModelPath.GetManifestPath()
	if err != nil {
		return err
	}

	destModelPath := ParseModelPath(dest)
	destPath, err := destModelPath.GetManifestPath()
	if err != nil {
		return err
	}
	if err := os.MkdirAll(filepath.Dir(destPath), 0o755); err != nil {
		return err
	}

	// copy the file
	input, err := os.ReadFile(srcPath)
	if err != nil {
		fmt.Println("Error reading file:", err)
		return err
	}

	err = os.WriteFile(destPath, input, 0o644)
	if err != nil {
		fmt.Println("Error reading file:", err)
		return err
	}

	return nil
}

func deleteUnusedLayers(skipModelPath *ModelPath, deleteMap map[string]bool, dryRun bool) error {
	fp, err := GetManifestPath()
	if err != nil {
		return err
	}

	walkFunc := func(path string, info os.FileInfo, _ error) error {
		if info.IsDir() {
			return nil
		}

		dir, file := filepath.Split(path)
		dir = strings.Trim(strings.TrimPrefix(dir, fp), string(os.PathSeparator))
		tag := strings.Join([]string{dir, file}, ":")
		fmp := ParseModelPath(tag)

		// skip the manifest we're trying to delete
		if skipModelPath != nil && skipModelPath.GetFullTagname() == fmp.GetFullTagname() {
			return nil
		}

		// save (i.e. delete from the deleteMap) any files used in other manifests
		manifest, _, err := GetManifest(fmp)
		if err != nil {
			return nil
		}

		for _, layer := range manifest.Layers {
			delete(deleteMap, layer.Digest)
		}

		delete(deleteMap, manifest.Config.Digest)
		return nil
	}

	if err := filepath.Walk(fp, walkFunc); err != nil {
		return err
	}

	// only delete the files which are still in the deleteMap
	for k, v := range deleteMap {
		if v {
			fp, err := GetBlobsPath(k)
			if err != nil {
				log.Printf("couldn't get file path for '%s': %v", k, err)
				continue
			}
			if !dryRun {
				if err := os.Remove(fp); err != nil {
					log.Printf("couldn't remove file '%s': %v", fp, err)
					continue
				}
			} else {
				log.Printf("wanted to remove: %s", fp)
			}
		}
	}

	return nil
}

func PruneLayers() error {
	deleteMap := make(map[string]bool)
	p, err := GetBlobsPath("")
	if err != nil {
		return err
	}

	blobs, err := os.ReadDir(p)
	if err != nil {
		log.Printf("couldn't read dir '%s': %v", p, err)
		return err
	}

	for _, blob := range blobs {
		name := blob.Name()
		if runtime.GOOS == "windows" {
			name = strings.ReplaceAll(name, "-", ":")
		}
		deleteMap[name] = true
	}

	log.Printf("total blobs: %d", len(deleteMap))

	err = deleteUnusedLayers(nil, deleteMap, false)
	if err != nil {
		return err
	}

	log.Printf("total unused blobs removed: %d", len(deleteMap))

	return nil
}

func PruneDirectory(path string) error {
	info, err := os.Lstat(path)
	if err != nil {
		return err
	}

	if info.IsDir() && info.Mode()&os.ModeSymlink == 0 {
		entries, err := os.ReadDir(path)
		if err != nil {
			return err
		}

		for _, entry := range entries {
			if err := PruneDirectory(filepath.Join(path, entry.Name())); err != nil {
				return err
			}
		}

		entries, err = os.ReadDir(path)
		if err != nil {
			return err
		}

		if len(entries) > 0 {
			return nil
		}

		return os.Remove(path)
	}

	return nil
}

func DeleteModel(name string) error {
	mp := ParseModelPath(name)
	manifest, _, err := GetManifest(mp)
	if err != nil {
		return err
	}

	deleteMap := make(map[string]bool)
	for _, layer := range manifest.Layers {
		deleteMap[layer.Digest] = true
	}
	deleteMap[manifest.Config.Digest] = true

	err = deleteUnusedLayers(&mp, deleteMap, false)
	if err != nil {
		return err
	}

	fp, err := mp.GetManifestPath()
	if err != nil {
		return err
	}
	err = os.Remove(fp)
	if err != nil {
		log.Printf("couldn't remove manifest file '%s': %v", fp, err)
		return err
	}

	return nil
}

func ShowModelfile(model *Model) (string, error) {
	var mt struct {
		*Model
		From       string
		Parameters map[string][]any
	}

	mt.Parameters = make(map[string][]any)
	for k, v := range model.Options {
		if s, ok := v.([]any); ok {
			mt.Parameters[k] = s
			continue
		}

		mt.Parameters[k] = []any{v}
	}

	mt.Model = model
	mt.From = model.ModelPath

	if model.OriginalModel != "" {
		mt.From = model.OriginalModel
	}

	modelFile := `# Modelfile generated by "ollama show"
# To build a new Modelfile based on this one, replace the FROM line with:
# FROM {{ .ShortName }}

FROM {{ .From }}
TEMPLATE """{{ .Template }}"""

{{- if .System }}
SYSTEM """{{ .System }}"""
{{- end }}

{{- range $adapter := .AdapterPaths }}
ADAPTER {{ $adapter }}
{{- end }}

{{- range $k, $v := .Parameters }}
{{- range $parameter := $v }}
PARAMETER {{ $k }} {{ printf "%#v" $parameter }}
{{- end }}
{{- end }}`

	tmpl, err := template.New("").Parse(modelFile)
	if err != nil {
		log.Printf("error parsing template: %q", err)
		return "", err
	}

	var buf bytes.Buffer

	if err = tmpl.Execute(&buf, mt); err != nil {
		log.Printf("error executing template: %q", err)
		return "", err
	}

	return buf.String(), nil
}

func PushModel(ctx context.Context, name string, regOpts *RegistryOptions, fn func(api.ProgressResponse)) error {
	mp := ParseModelPath(name)
	fn(api.ProgressResponse{Status: "retrieving manifest"})

	if mp.ProtocolScheme == "http" && !regOpts.Insecure {
		return fmt.Errorf("insecure protocol http")
	}

	manifest, _, err := GetManifest(mp)
	if err != nil {
		fn(api.ProgressResponse{Status: "couldn't retrieve manifest"})
		return err
	}

	var layers []*Layer
	layers = append(layers, manifest.Layers...)
	layers = append(layers, &manifest.Config)

	for _, layer := range layers {
		if err := uploadBlob(ctx, mp, layer, regOpts, fn); err != nil {
			log.Printf("error uploading blob: %v", err)
			return err
		}
	}

	fn(api.ProgressResponse{Status: "pushing manifest"})
	requestURL := mp.BaseURL()
	requestURL = requestURL.JoinPath("v2", mp.GetNamespaceRepository(), "manifests", mp.Tag)

	manifestJSON, err := json.Marshal(manifest)
	if err != nil {
		return err
	}

	headers := make(http.Header)
	headers.Set("Content-Type", "application/vnd.docker.distribution.manifest.v2+json")
	resp, err := makeRequestWithRetry(ctx, http.MethodPut, requestURL, headers, bytes.NewReader(manifestJSON), regOpts)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	fn(api.ProgressResponse{Status: "success"})

	return nil
}

func PullModel(ctx context.Context, name string, regOpts *RegistryOptions, fn func(api.ProgressResponse)) error {
	mp := ParseModelPath(name)

	var manifest *ManifestV2
	var err error
	var noprune string

	// build deleteMap to prune unused layers
	deleteMap := make(map[string]bool)

	if noprune = os.Getenv("OLLAMA_NOPRUNE"); noprune == "" {
		manifest, _, err = GetManifest(mp)
		if err != nil && !errors.Is(err, os.ErrNotExist) {
			return err
		}

		if manifest != nil {
			for _, l := range manifest.Layers {
				deleteMap[l.Digest] = true
			}
			deleteMap[manifest.Config.Digest] = true
		}
	}

	if mp.ProtocolScheme == "http" && !regOpts.Insecure {
		return fmt.Errorf("insecure protocol http")
	}

	fn(api.ProgressResponse{Status: "pulling manifest"})

	manifest, err = pullModelManifest(ctx, mp, regOpts)
	if err != nil {
		return fmt.Errorf("pull model manifest: %s", err)
	}

	var layers []*Layer
	layers = append(layers, manifest.Layers...)
	layers = append(layers, &manifest.Config)

	for _, layer := range layers {
		if err := downloadBlob(
			ctx,
			downloadOpts{
				mp:      mp,
				digest:  layer.Digest,
				regOpts: regOpts,
				fn:      fn,
			}); err != nil {
			return err
		}
		delete(deleteMap, layer.Digest)
	}
	delete(deleteMap, manifest.Config.Digest)

	fn(api.ProgressResponse{Status: "verifying sha256 digest"})
	for _, layer := range layers {
		if err := verifyBlob(layer.Digest); err != nil {
			if errors.Is(err, errDigestMismatch) {
				// something went wrong, delete the blob
				fp, err := GetBlobsPath(layer.Digest)
				if err != nil {
					return err
				}
				if err := os.Remove(fp); err != nil {
					// log this, but return the original error
					log.Printf("couldn't remove file with digest mismatch '%s': %v", fp, err)
				}
			}
			return err
		}
	}

	fn(api.ProgressResponse{Status: "writing manifest"})

	manifestJSON, err := json.Marshal(manifest)
	if err != nil {
		return err
	}

	fp, err := mp.GetManifestPath()
	if err != nil {
		return err
	}
	if err := os.MkdirAll(filepath.Dir(fp), 0o755); err != nil {
		return err
	}

	err = os.WriteFile(fp, manifestJSON, 0o644)
	if err != nil {
		log.Printf("couldn't write to %s", fp)
		return err
	}

	if noprune == "" {
		fn(api.ProgressResponse{Status: "removing any unused layers"})
		err = deleteUnusedLayers(nil, deleteMap, false)
		if err != nil {
			return err
		}
	}

	fn(api.ProgressResponse{Status: "success"})

	return nil
}

func pullModelManifest(ctx context.Context, mp ModelPath, regOpts *RegistryOptions) (*ManifestV2, error) {
	requestURL := mp.BaseURL().JoinPath("v2", mp.GetNamespaceRepository(), "manifests", mp.Tag)

	headers := make(http.Header)
	headers.Set("Accept", "application/vnd.docker.distribution.manifest.v2+json")
	resp, err := makeRequestWithRetry(ctx, http.MethodGet, requestURL, headers, nil, regOpts)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	var m *ManifestV2
	if err := json.NewDecoder(resp.Body).Decode(&m); err != nil {
		return nil, err
	}

	return m, err
}

func createConfigLayer(config ConfigV2, layers []string) (*LayerReader, error) {
	config.RootFS = RootFS{
		Type:    "layers",
		DiffIDs: layers,
	}

	configJSON, err := json.Marshal(config)
	if err != nil {
		return nil, err
	}

	digest, size := GetSHA256Digest(bytes.NewBuffer(configJSON))

	layer := &LayerReader{
		Layer: Layer{
			MediaType: "application/vnd.docker.container.image.v1+json",
			Digest:    digest,
			Size:      size,
		},
		Reader: bytes.NewBuffer(configJSON),
	}
	return layer, nil
}

// GetSHA256Digest returns the SHA256 hash of a given buffer and returns it, and the size of buffer
func GetSHA256Digest(r io.Reader) (string, int64) {
	h := sha256.New()
	n, err := io.Copy(h, r)
	if err != nil {
		log.Fatal(err)
	}

	return fmt.Sprintf("sha256:%x", h.Sum(nil)), n
}

func makeRequestWithRetry(ctx context.Context, method string, requestURL *url.URL, headers http.Header, body io.ReadSeeker, regOpts *RegistryOptions) (*http.Response, error) {
	for try := 0; try < maxRetries; try++ {
		resp, err := makeRequest(ctx, method, requestURL, headers, body, regOpts)
		if err != nil {
			log.Printf("couldn't start upload: %v", err)
			return nil, err
		}

		switch {
		case resp.StatusCode == http.StatusUnauthorized:
			auth := resp.Header.Get("www-authenticate")
			authRedir := ParseAuthRedirectString(auth)
			token, err := getAuthToken(ctx, authRedir)
			if err != nil {
				return nil, err
			}

			regOpts.Token = token
			if body != nil {
				if _, err := body.Seek(0, io.SeekStart); err != nil {
					return nil, err
				}
			}

			continue
		case resp.StatusCode >= http.StatusBadRequest:
			body, _ := io.ReadAll(resp.Body)
			return nil, fmt.Errorf("on upload registry responded with code %d: %s", resp.StatusCode, body)
		default:
			return resp, nil
		}
	}

	return nil, errMaxRetriesExceeded
}

func makeRequest(ctx context.Context, method string, requestURL *url.URL, headers http.Header, body io.Reader, regOpts *RegistryOptions) (*http.Response, error) {
	if requestURL.Scheme != "http" && regOpts != nil && regOpts.Insecure {
		requestURL.Scheme = "http"
	}

	req, err := http.NewRequestWithContext(ctx, method, requestURL.String(), body)
	if err != nil {
		return nil, err
	}

	if headers != nil {
		req.Header = headers
	}

	if regOpts != nil {
		if regOpts.Token != "" {
			req.Header.Set("Authorization", "Bearer "+regOpts.Token)
		} else if regOpts.Username != "" && regOpts.Password != "" {
			req.SetBasicAuth(regOpts.Username, regOpts.Password)
		}
	}

	req.Header.Set("User-Agent", fmt.Sprintf("ollama/%s (%s %s) Go/%s", version.Version, runtime.GOARCH, runtime.GOOS, runtime.Version()))

	if s := req.Header.Get("Content-Length"); s != "" {
		contentLength, err := strconv.ParseInt(s, 10, 64)
		if err != nil {
			return nil, err
		}

		req.ContentLength = contentLength
	}

	proxyURL, err := http.ProxyFromEnvironment(req)
	if err != nil {
		return nil, err
	}

	client := http.Client{
		Transport: &http.Transport{
			Proxy: http.ProxyURL(proxyURL),
		},
	}

	resp, err := client.Do(req)
	if err != nil {
		return nil, err
	}

	return resp, nil
}

func getValue(header, key string) string {
	startIdx := strings.Index(header, key+"=")
	if startIdx == -1 {
		return ""
	}

	// Move the index to the starting quote after the key.
	startIdx += len(key) + 2
	endIdx := startIdx

	for endIdx < len(header) {
		if header[endIdx] == '"' {
			if endIdx+1 < len(header) && header[endIdx+1] != ',' { // If the next character isn't a comma, continue
				endIdx++
				continue
			}
			break
		}
		endIdx++
	}
	return header[startIdx:endIdx]
}

func ParseAuthRedirectString(authStr string) AuthRedirect {
	authStr = strings.TrimPrefix(authStr, "Bearer ")

	return AuthRedirect{
		Realm:   getValue(authStr, "realm"),
		Service: getValue(authStr, "service"),
		Scope:   getValue(authStr, "scope"),
	}
}

var errDigestMismatch = fmt.Errorf("digest mismatch, file must be downloaded again")

func verifyBlob(digest string) error {
	fp, err := GetBlobsPath(digest)
	if err != nil {
		return err
	}

	f, err := os.Open(fp)
	if err != nil {
		return err
	}
	defer f.Close()

	fileDigest, _ := GetSHA256Digest(f)
	if digest != fileDigest {
		return fmt.Errorf("%w: want %s, got %s", errDigestMismatch, digest, fileDigest)
	}

	return nil
}
