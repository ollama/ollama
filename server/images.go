package server

import (
	"bufio"
	"bytes"
	"crypto/sha256"
	"encoding/json"
	"errors"
	"fmt"
	"html/template"
	"io"
	"log"
	"math"
	"net/http"
	"os"
	"path"
	"path/filepath"
	"reflect"
	"strconv"
	"strings"

	"github.com/jmorganca/ollama/api"
	"github.com/jmorganca/ollama/llama"
	"github.com/jmorganca/ollama/parser"
	"github.com/jmorganca/ollama/vector"
)

type RegistryOptions struct {
	Insecure bool
	Username string
	Password string
}

type Model struct {
	Name       string `json:"name"`
	ModelPath  string
	Template   string
	System     string
	Digest     string
	Options    map[string]interface{}
	Embeddings []vector.Embedding
}

func (m *Model) Prompt(request api.GenerateRequest, embedding string) (string, error) {
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
		Embed  string

		// deprecated: versions <= 0.0.7 used this to omit the system prompt
		Context []int
	}

	vars.First = len(request.Context) == 0
	vars.System = m.System
	vars.Prompt = request.Prompt
	vars.Context = request.Context
	vars.Embed = embedding

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
	Size      int    `json:"size"`
}

type LayerReader struct {
	Layer
	io.Reader
}

type ConfigV2 struct {
	Architecture string `json:"architecture"`
	OS           string `json:"os"`
	RootFS       RootFS `json:"rootfs"`
}

type RootFS struct {
	Type    string   `json:"type"`
	DiffIDs []string `json:"diff_ids"`
}

func (m *ManifestV2) GetTotalSize() int {
	var total int
	for _, layer := range m.Layers {
		total += layer.Size
	}
	total += m.Config.Size
	return total
}

func GetManifest(mp ModelPath) (*ManifestV2, error) {
	fp, err := mp.GetManifestPath(false)
	if err != nil {
		return nil, err
	}

	if _, err = os.Stat(fp); err != nil {
		return nil, err
	}

	var manifest *ManifestV2

	bts, err := os.ReadFile(fp)
	if err != nil {
		return nil, fmt.Errorf("couldn't open file '%s'", fp)
	}

	if err := json.Unmarshal(bts, &manifest); err != nil {
		return nil, err
	}

	return manifest, nil
}

func GetModel(name string) (*Model, error) {
	mp := ParseModelPath(name)

	manifest, err := GetManifest(mp)
	if err != nil {
		return nil, err
	}

	model := &Model{
		Name:   mp.GetFullTagname(),
		Digest: manifest.Config.Digest,
	}

	for _, layer := range manifest.Layers {
		filename, err := GetBlobsPath(layer.Digest)
		if err != nil {
			return nil, err
		}

		switch layer.MediaType {
		case "application/vnd.ollama.image.model":
			model.ModelPath = filename
		case "application/vnd.ollama.image.embed":
			file, err := os.Open(filename)
			if err != nil {
				return nil, fmt.Errorf("failed to open file: %s", filename)
			}
			defer file.Close()

			if err = json.NewDecoder(file).Decode(&model.Embeddings); err != nil {
				return nil, err
			}
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
		}
	}

	return model, nil
}

func filenameWithPath(path, f string) (string, error) {
	// if filePath starts with ~/, replace it with the user's home directory.
	if strings.HasPrefix(f, "~/") {
		parts := strings.Split(f, "/")
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

func CreateModel(name string, path string, fn func(resp api.ProgressResponse)) error {
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

	var layers []*LayerReader
	params := make(map[string][]string)
	embed := EmbeddingParams{fn: fn, opts: api.DefaultOptions()}
	for _, c := range commands {
		log.Printf("[%s] - %s\n", c.Name, c.Args)
		switch c.Name {
		case "model":
			fn(api.ProgressResponse{Status: "looking for model"})
			embed.model = c.Args
			mf, err := GetManifest(ParseModelPath(c.Args))
			if err != nil {
				modelFile, err := filenameWithPath(path, c.Args)
				if err != nil {
					return err
				}
				if _, err := os.Stat(modelFile); err != nil {
					// the model file does not exist, try pulling it
					if errors.Is(err, os.ErrNotExist) {
						fn(api.ProgressResponse{Status: "pulling model file"})
						if err := PullModel(c.Args, &RegistryOptions{}, fn); err != nil {
							return err
						}
						mf, err = GetManifest(ParseModelPath(c.Args))
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

					l, err := CreateLayer(file)
					if err != nil {
						return fmt.Errorf("failed to create layer: %v", err)
					}
					l.MediaType = "application/vnd.ollama.image.model"
					layers = append(layers, l)
				}
			}
			if mf != nil {
				log.Printf("manifest = %#v", mf)
				for _, l := range mf.Layers {
					newLayer, err := GetLayerWithBufferFromLayer(l)
					if err != nil {
						return err
					}
					layers = append(layers, newLayer)
				}
			}
		case "embed":
			embedFilePath, err := filenameWithPath(path, c.Args)
			if err != nil {
				return err
			}
			embed.files = append(embed.files, embedFilePath)
		case "license":
			fn(api.ProgressResponse{Status: fmt.Sprintf("creating model %s layer", c.Name)})
			mediaType := fmt.Sprintf("application/vnd.ollama.image.%s", c.Name)

			layer, err := CreateLayer(strings.NewReader(c.Args))
			if err != nil {
				return err
			}

			layer.MediaType = mediaType
			layers = append(layers, layer)
		case "template", "system", "prompt":
			fn(api.ProgressResponse{Status: fmt.Sprintf("creating model %s layer", c.Name)})
			// remove the prompt layer if one exists
			mediaType := fmt.Sprintf("application/vnd.ollama.image.%s", c.Name)
			layers = removeLayerFromLayers(layers, mediaType)

			layer, err := CreateLayer(strings.NewReader(c.Args))
			if err != nil {
				return err
			}

			layer.MediaType = mediaType
			layers = append(layers, layer)
		default:
			// runtime parameters, build a list of args for each parameter to allow multiple values to be specified (ex: multiple stop tokens)
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

		// apply these parameters to the embedding options, in case embeddings need to be generated using this model
		embed.opts = api.DefaultOptions()
		embed.opts.FromMap(formattedParams)
	}

	// generate the embedding layers
	embeddingLayers, err := embeddingLayers(embed)
	if err != nil {
		return err
	}
	layers = append(layers, embeddingLayers...)

	digests, err := getLayerDigests(layers)
	if err != nil {
		return err
	}

	var manifestLayers []*Layer
	for _, l := range layers {
		manifestLayers = append(manifestLayers, &l.Layer)
	}

	// Create a layer for the config object
	fn(api.ProgressResponse{Status: "creating config layer"})
	cfg, err := createConfigLayer(digests)
	if err != nil {
		return err
	}
	layers = append(layers, cfg)

	err = SaveLayers(layers, fn, false)
	if err != nil {
		return err
	}

	// Create the manifest
	fn(api.ProgressResponse{Status: "writing manifest"})
	err = CreateManifest(name, cfg, manifestLayers)
	if err != nil {
		return err
	}

	fn(api.ProgressResponse{Status: "success"})
	return nil
}

type EmbeddingParams struct {
	model string
	opts  api.Options
	files []string // paths to files to embed
	fn    func(resp api.ProgressResponse)
}

// embeddingLayers loads the associated LLM and generates the embeddings to be stored from an input file
func embeddingLayers(e EmbeddingParams) ([]*LayerReader, error) {
	layers := []*LayerReader{}
	if len(e.files) > 0 {
		if _, err := os.Stat(e.model); err != nil {
			if os.IsNotExist(err) {
				// this is a model name rather than the file
				model, err := GetModel(e.model)
				if err != nil {
					return nil, fmt.Errorf("failed to get model to generate embeddings: %v", err)
				}
				e.model = model.ModelPath
			} else {
				return nil, fmt.Errorf("failed to get model file to generate embeddings: %v", err)
			}
		}

		e.opts.EmbeddingOnly = true
		llm, err := llama.New(e.model, e.opts)
		if err != nil {
			return nil, fmt.Errorf("load model to generate embeddings: %v", err)
		}
		defer func() {
			if llm != nil {
				llm.Close()
			}
		}()

		addedFiles := make(map[string]bool) // keep track of files that have already been added
		for _, filePattern := range e.files {
			matchingFiles, err := filepath.Glob(filePattern)
			if err != nil {
				return nil, fmt.Errorf("could not find files with pattern %s: %w", filePattern, err)
			}

			for _, filePath := range matchingFiles {
				if addedFiles[filePath] {
					continue
				}
				addedFiles[filePath] = true
				// TODO: check file type
				f, err := os.Open(filePath)
				if err != nil {
					return nil, fmt.Errorf("could not open embed file: %w", err)
				}
				scanner := bufio.NewScanner(f)
				scanner.Split(bufio.ScanLines)

				data := []string{}
				for scanner.Scan() {
					data = append(data, scanner.Text())
				}
				f.Close()

				// the digest of the file is set here so that the client knows a new operation is in progress
				fileDigest, _ := GetSHA256Digest(bytes.NewReader([]byte(filePath)))

				embeddings := []vector.Embedding{}
				for i, d := range data {
					if strings.TrimSpace(d) == "" {
						continue
					}
					e.fn(api.ProgressResponse{
						Status:    fmt.Sprintf("creating embeddings for file %s", filePath),
						Digest:    fileDigest,
						Total:     len(data) - 1,
						Completed: i,
					})
					retry := 0
				generate:
					if retry > 3 {
						log.Printf("failed to generate embedding for '%s' line %d: %v", filePath, i+1, err)
						continue
					}
					embed, err := llm.Embedding(d)
					if err != nil {
						log.Printf("retrying embedding generation for '%s' line %d: %v", filePath, i+1, err)
						retry++
						goto generate
					}
					// Check for NaN and Inf in the embedding, which can't be stored
					for _, value := range embed {
						if math.IsNaN(value) || math.IsInf(value, 0) {
							log.Printf("reloading model, embedding contains NaN or Inf")
							// reload the model to get a new embedding, the seed can effect these outputs and reloading changes it
							llm.Close()
							llm, err = llama.New(e.model, e.opts)
							if err != nil {
								return nil, fmt.Errorf("load model to generate embeddings: %v", err)
							}
							retry++
							goto generate
						}
					}
					embeddings = append(embeddings, vector.Embedding{Data: d, Vector: embed})
				}

				b, err := json.Marshal(embeddings)
				if err != nil {
					return nil, fmt.Errorf("failed to encode embeddings: %w", err)
				}
				r := bytes.NewReader(b)

				digest, size := GetSHA256Digest(r)
				// Reset the position of the reader after calculating the digest
				if _, err := r.Seek(0, io.SeekStart); err != nil {
					return nil, fmt.Errorf("could not reset embed reader: %w", err)
				}

				layer := &LayerReader{
					Layer: Layer{
						MediaType: "application/vnd.ollama.image.embed",
						Digest:    digest,
						Size:      size,
					},
					Reader: r,
				}

				layers = append(layers, layer)
			}
		}
	}
	return layers, nil
}

func removeLayerFromLayers(layers []*LayerReader, mediaType string) []*LayerReader {
	j := 0
	for _, l := range layers {
		if l.MediaType != mediaType {
			layers[j] = l
			j++
		}
	}
	return layers[:j]
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

	fp, err := mp.GetManifestPath(true)
	if err != nil {
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

					out[key] = floatVal
				case reflect.Int:
					intVal, err := strconv.ParseInt(vals[0], 10, 0)
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
	f.Seek(0, 0)

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
	srcPath, err := ParseModelPath(src).GetManifestPath(false)
	if err != nil {
		return err
	}
	destPath, err := ParseModelPath(dest).GetManifestPath(true)
	if err != nil {
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

func DeleteModel(name string) error {
	mp := ParseModelPath(name)

	manifest, err := GetManifest(mp)
	if err != nil {
		return err
	}
	deleteMap := make(map[string]bool)
	for _, layer := range manifest.Layers {
		deleteMap[layer.Digest] = true
	}
	deleteMap[manifest.Config.Digest] = true

	fp, err := GetManifestPath()
	if err != nil {
		return err
	}
	err = filepath.Walk(fp, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}
		if !info.IsDir() {
			path := path[len(fp)+1:]
			slashIndex := strings.LastIndex(path, "/")
			if slashIndex == -1 {
				return nil
			}
			tag := path[:slashIndex] + ":" + path[slashIndex+1:]
			fmp := ParseModelPath(tag)

			// skip the manifest we're trying to delete
			if mp.GetFullTagname() == fmp.GetFullTagname() {
				return nil
			}

			// save (i.e. delete from the deleteMap) any files used in other manifests
			manifest, err := GetManifest(fmp)
			if err != nil {
				log.Printf("skipping file: %s", fp)
				return nil
			}
			for _, layer := range manifest.Layers {
				delete(deleteMap, layer.Digest)
			}
			delete(deleteMap, manifest.Config.Digest)
		}
		return nil
	})
	if err != nil {
		return err
	}

	if err != nil {
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
			if err := os.Remove(fp); err != nil {
				log.Printf("couldn't remove file '%s': %v", fp, err)
				continue
			}
		}
	}

	fp, err = mp.GetManifestPath(false)
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

func PushModel(name string, regOpts *RegistryOptions, fn func(api.ProgressResponse)) error {
	mp := ParseModelPath(name)

	fn(api.ProgressResponse{Status: "retrieving manifest"})

	manifest, err := GetManifest(mp)
	if err != nil {
		fn(api.ProgressResponse{Status: "couldn't retrieve manifest"})
		return err
	}

	var layers []*Layer
	layers = append(layers, manifest.Layers...)
	layers = append(layers, &manifest.Config)

	for _, layer := range layers {
		exists, err := checkBlobExistence(mp, layer.Digest, regOpts)
		if err != nil {
			return err
		}

		if exists {
			fn(api.ProgressResponse{
				Status:    "using existing layer",
				Digest:    layer.Digest,
				Total:     layer.Size,
				Completed: layer.Size,
			})
			log.Printf("Layer %s already exists", layer.Digest)
			continue
		}

		fn(api.ProgressResponse{
			Status: "starting upload",
			Digest: layer.Digest,
			Total:  layer.Size,
		})

		location, err := startUpload(mp, regOpts)
		if err != nil {
			log.Printf("couldn't start upload: %v", err)
			return err
		}

		err = uploadBlobChunked(mp, location, layer, regOpts, fn)
		if err != nil {
			log.Printf("error uploading blob: %v", err)
			return err
		}
	}

	fn(api.ProgressResponse{Status: "pushing manifest"})
	url := fmt.Sprintf("%s/v2/%s/manifests/%s", mp.Registry, mp.GetNamespaceRepository(), mp.Tag)
	headers := map[string]string{
		"Content-Type": "application/vnd.docker.distribution.manifest.v2+json",
	}

	manifestJSON, err := json.Marshal(manifest)
	if err != nil {
		return err
	}

	resp, err := makeRequest("PUT", url, headers, bytes.NewReader(manifestJSON), regOpts)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	// Check for success: For a successful upload, the Docker registry will respond with a 201 Created
	if resp.StatusCode != http.StatusCreated {
		body, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("on push registry responded with code %d: %v", resp.StatusCode, string(body))
	}

	fn(api.ProgressResponse{Status: "success"})

	return nil
}

func PullModel(name string, regOpts *RegistryOptions, fn func(api.ProgressResponse)) error {
	mp := ParseModelPath(name)

	fn(api.ProgressResponse{Status: "pulling manifest"})

	manifest, err := pullModelManifest(mp, regOpts)
	if err != nil {
		return fmt.Errorf("pull model manifest: %s", err)
	}

	var layers []*Layer
	layers = append(layers, manifest.Layers...)
	layers = append(layers, &manifest.Config)

	for _, layer := range layers {
		if err := downloadBlob(mp, layer.Digest, regOpts, fn); err != nil {
			return err
		}
	}

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

	fp, err := mp.GetManifestPath(true)
	if err != nil {
		return err
	}

	err = os.WriteFile(fp, manifestJSON, 0o644)
	if err != nil {
		log.Printf("couldn't write to %s", fp)
		return err
	}

	fn(api.ProgressResponse{Status: "success"})

	return nil
}

func pullModelManifest(mp ModelPath, regOpts *RegistryOptions) (*ManifestV2, error) {
	url := fmt.Sprintf("%s/v2/%s/manifests/%s", mp.Registry, mp.GetNamespaceRepository(), mp.Tag)
	headers := map[string]string{
		"Accept": "application/vnd.docker.distribution.manifest.v2+json",
	}

	resp, err := makeRequest("GET", url, headers, nil, regOpts)
	if err != nil {
		log.Printf("couldn't get manifest: %v", err)
		return nil, err
	}
	defer resp.Body.Close()

	// Check for success: For a successful upload, the Docker registry will respond with a 201 Created
	if resp.StatusCode != http.StatusOK {
		if resp.StatusCode == http.StatusNotFound {
			return nil, fmt.Errorf("model not found")
		}
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("on pull registry responded with code %d: %s", resp.StatusCode, body)
	}

	var m *ManifestV2
	if err := json.NewDecoder(resp.Body).Decode(&m); err != nil {
		return nil, err
	}

	return m, err
}

func createConfigLayer(layers []string) (*LayerReader, error) {
	// TODO change architecture and OS
	config := ConfigV2{
		Architecture: "arm64",
		OS:           "linux",
		RootFS: RootFS{
			Type:    "layers",
			DiffIDs: layers,
		},
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
func GetSHA256Digest(r io.Reader) (string, int) {
	h := sha256.New()
	n, err := io.Copy(h, r)
	if err != nil {
		log.Fatal(err)
	}

	return fmt.Sprintf("sha256:%x", h.Sum(nil)), int(n)
}

func startUpload(mp ModelPath, regOpts *RegistryOptions) (string, error) {
	url := fmt.Sprintf("%s/v2/%s/blobs/uploads/", mp.Registry, mp.GetNamespaceRepository())

	resp, err := makeRequest("POST", url, nil, nil, regOpts)
	if err != nil {
		log.Printf("couldn't start upload: %v", err)
		return "", err
	}
	defer resp.Body.Close()

	// Check for success
	if resp.StatusCode != http.StatusAccepted {
		body, _ := io.ReadAll(resp.Body)
		return "", fmt.Errorf("on upload registry responded with code %d: %s", resp.StatusCode, body)
	}

	// Extract UUID location from header
	location := resp.Header.Get("Location")
	if location == "" {
		return "", fmt.Errorf("location header is missing in response")
	}

	return location, nil
}

// Function to check if a blob already exists in the Docker registry
func checkBlobExistence(mp ModelPath, digest string, regOpts *RegistryOptions) (bool, error) {
	url := fmt.Sprintf("%s/v2/%s/blobs/%s", mp.Registry, mp.GetNamespaceRepository(), digest)

	resp, err := makeRequest("HEAD", url, nil, nil, regOpts)
	if err != nil {
		log.Printf("couldn't check for blob: %v", err)
		return false, err
	}
	defer resp.Body.Close()

	// Check for success: If the blob exists, the Docker registry will respond with a 200 OK
	return resp.StatusCode == http.StatusOK, nil
}

func uploadBlobChunked(mp ModelPath, url string, layer *Layer, regOpts *RegistryOptions, fn func(api.ProgressResponse)) error {
	// TODO allow resumability
	// TODO allow canceling uploads via DELETE
	// TODO allow cross repo blob mount

	fp, err := GetBlobsPath(layer.Digest)
	if err != nil {
		return err
	}

	f, err := os.Open(fp)
	if err != nil {
		return err
	}

	totalUploaded := 0

	r, w := io.Pipe()
	defer r.Close()

	go func() {
		defer w.Close()
		for {
			n, err := io.CopyN(w, f, 1024*1024)
			if err != nil && !errors.Is(err, io.EOF) {
				fn(api.ProgressResponse{
					Status:    fmt.Sprintf("error copying pipe: %v", err),
					Digest:    layer.Digest,
					Total:     layer.Size,
					Completed: totalUploaded,
				})
				return
			}

			totalUploaded += int(n)

			fn(api.ProgressResponse{
				Status:    fmt.Sprintf("uploading %s", layer.Digest),
				Digest:    layer.Digest,
				Total:     layer.Size,
				Completed: totalUploaded,
			})

			if totalUploaded >= layer.Size {
				return
			}
		}
	}()

	url = fmt.Sprintf("%s&digest=%s", url, layer.Digest)

	headers := make(map[string]string)
	headers["Content-Type"] = "application/octet-stream"
	headers["Content-Range"] = fmt.Sprintf("0-%d", layer.Size-1)
	headers["Content-Length"] = strconv.Itoa(int(layer.Size))

	// finish the upload
	resp, err := makeRequest("PUT", url, headers, r, regOpts)
	if err != nil {
		log.Printf("couldn't finish upload: %v", err)
		return err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusCreated {
		body, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("on finish upload registry responded with code %d: %v", resp.StatusCode, string(body))
	}
	return nil
}

func downloadBlob(mp ModelPath, digest string, regOpts *RegistryOptions, fn func(api.ProgressResponse)) error {
	fp, err := GetBlobsPath(digest)
	if err != nil {
		return err
	}

	if fi, _ := os.Stat(fp); fi != nil {
		// we already have the file, so return
		fn(api.ProgressResponse{
			Digest:    digest,
			Total:     int(fi.Size()),
			Completed: int(fi.Size()),
		})

		return nil
	}

	var size int64
	chunkSize := 1024 * 1024 // 1 MiB in bytes

	fi, err := os.Stat(fp + "-partial")
	switch {
	case errors.Is(err, os.ErrNotExist):
		// noop, file doesn't exist so create it
	case err != nil:
		return fmt.Errorf("stat: %w", err)
	default:
		size = fi.Size()
		// Ensure the size is divisible by the chunk size by removing excess bytes
		size -= size % int64(chunkSize)

		err := os.Truncate(fp+"-partial", size)
		if err != nil {
			return fmt.Errorf("truncate: %w", err)
		}
	}

	url := fmt.Sprintf("%s/v2/%s/blobs/%s", mp.Registry, mp.GetNamespaceRepository(), digest)
	headers := map[string]string{
		"Range": fmt.Sprintf("bytes=%d-", size),
	}

	resp, err := makeRequest("GET", url, headers, nil, regOpts)
	if err != nil {
		log.Printf("couldn't download blob: %v", err)
		return err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK && resp.StatusCode != http.StatusPartialContent {
		body, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("on download registry responded with code %d: %v", resp.StatusCode, string(body))
	}

	err = os.MkdirAll(path.Dir(fp), 0o700)
	if err != nil {
		return fmt.Errorf("make blobs directory: %w", err)
	}

	out, err := os.OpenFile(fp+"-partial", os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0o644)
	if err != nil {
		return fmt.Errorf("open file: %w", err)
	}
	defer out.Close()

	remaining, _ := strconv.ParseInt(resp.Header.Get("Content-Length"), 10, 64)
	completed := size
	total := remaining + completed

	for {
		fn(api.ProgressResponse{
			Status:    fmt.Sprintf("pulling %s...", digest[7:19]),
			Digest:    digest,
			Total:     int(total),
			Completed: int(completed),
		})

		if completed >= total {
			if err := out.Close(); err != nil {
				return err
			}

			if err := os.Rename(fp+"-partial", fp); err != nil {
				fn(api.ProgressResponse{
					Status:    fmt.Sprintf("error renaming file: %v", err),
					Digest:    digest,
					Total:     int(total),
					Completed: int(completed),
				})
				return err
			}

			break
		}

		n, err := io.CopyN(out, resp.Body, int64(chunkSize))
		if err != nil && !errors.Is(err, io.EOF) {
			return err
		}
		completed += n
	}

	log.Printf("success getting %s\n", digest)
	return nil
}

func makeRequest(method, url string, headers map[string]string, body io.Reader, regOpts *RegistryOptions) (*http.Response, error) {
	if !strings.HasPrefix(url, "http") {
		if regOpts.Insecure {
			url = "http://" + url
		} else {
			url = "https://" + url
		}
	}

	req, err := http.NewRequest(method, url, body)
	if err != nil {
		return nil, err
	}

	for k, v := range headers {
		req.Header.Set(k, v)
	}

	// TODO: better auth
	if regOpts.Username != "" && regOpts.Password != "" {
		req.SetBasicAuth(regOpts.Username, regOpts.Password)
	}

	client := &http.Client{
		CheckRedirect: func(req *http.Request, via []*http.Request) error {
			if len(via) >= 10 {
				return fmt.Errorf("too many redirects")
			}
			log.Printf("redirected to: %s\n", req.URL)
			return nil
		},
	}
	resp, err := client.Do(req)
	if err != nil {
		return nil, err
	}

	return resp, nil
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
