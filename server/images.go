package server

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"io/ioutil"
	"log"
	"net/http"
	"os"
	"path"
	"strings"

	"github.com/jmorganca/ollama/api"
	"github.com/jmorganca/ollama/parser"
)

var DefaultRegistry string = "http://localhost:6000"

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

type LayerWithBuffer struct {
	Layer

	Buffer *bytes.Buffer
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

func GetManifest(name string) (*ManifestV2, error) {
	home, err := os.UserHomeDir()
	if err != nil {
		return nil, err
	}

	filepath := path.Join(home, ".ollama/models/manifests", name)
	_, err = os.Stat(filepath)
	if os.IsNotExist(err) {
		return nil, fmt.Errorf("couldn't find model '%s'", name)
	}

	var manifest *ManifestV2

	f, err := os.Open(filepath)
	if err != nil {
		return nil, fmt.Errorf("couldn't open file '%s'", filepath)
	}

	decoder := json.NewDecoder(f)
	err = decoder.Decode(&manifest)
	if err != nil {
		return nil, err
	}

	return manifest, nil
}

func GetModel(name string) (*Model, error) {
	home, err := os.UserHomeDir()
	if err != nil {
		return nil, err
	}

	manifest, err := GetManifest(name)
	if err != nil {
		return nil, err
	}

	model := &Model{
		Name: name,
	}

	for _, layer := range manifest.Layers {
		filename := path.Join(home, ".ollama/models/blobs", layer.Digest)
		switch layer.MediaType {
		case "application/vnd.ollama.image.model":
			model.ModelPath = filename
		case "application/vnd.ollama.image.prompt":
			f, err := os.Open(filename)
			if err != nil {
				return nil, err
			}
			data, err := ioutil.ReadAll(f)
			if err != nil {
				return nil, err
			}
			model.Prompt = string(data)
		case "application/vnd.ollama.image.params":
			/*
				f, err = os.Open(filename)
				if err != nil {
					return nil, err
				}
			*/

			var opts api.Options
			/*
				decoder = json.NewDecoder(f)
				err = decoder.Decode(&opts)
				if err != nil {
					return nil, err
				}
			*/
			model.Options = opts
		}
	}

	return model, nil
}

func CreateModel(name string, mf io.Reader, fn func(status string)) error {
	fn("parsing modelfile")
	commands, err := parser.Parse(mf)
	if err != nil {
		return err
	}

	var layers []*LayerWithBuffer
	var param map[string]string
	param = make(map[string]string)

	for _, c := range commands {
		log.Printf("[%s] - %s\n", c.Name, c.Arg)
		switch c.Name {
		case "model":
			fn("creating model layer")
			file, err := os.Open(c.Arg)
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
		case "prompt":
			fn("creating prompt layer")
			prompt := strings.NewReader(c.Arg)
			l, err := CreateLayer(prompt)
			if err != nil {
				return fmt.Errorf("failed to create layer: %v", err)
			}
			l.MediaType = "application/vnd.ollama.image.prompt"
			layers = append(layers, l)
		default:
			param[c.Name] = c.Arg
		}
	}

	// Create a single layer for the parameters
	fn("creating parameter layer")
	paramData, err := paramsToReader(param)
	if err != nil {
		return fmt.Errorf("couldn't create params json: %v", err)
	}
	l, err := CreateLayer(paramData)
	if err != nil {
		return fmt.Errorf("failed to create layer: %v", err)
	}
	l.MediaType = "application/vnd.ollama.image.params"
	layers = append(layers, l)

	digests, err := getLayerDigests(layers)
	if err != nil {
		return err
	}

	// Create a layer for the config object
	fn("creating config layer")
	cfg, err := createConfigLayer(digests)
	if err != nil {
		return err
	}
	layers = append(layers, cfg)

	home, err := os.UserHomeDir()
	if err != nil {
		return err
	}

	var manifestLayers []*Layer

	// Write each of the layers to disk
	for _, layer := range layers {
		filepath := path.Join(home, ".ollama/models/blobs", layer.Digest)

		// TODO add a force flag to always write out the layers

		_, err = os.Stat(filepath)
		if os.IsNotExist(err) {
			fn(fmt.Sprintf("writing layer %s", layer.Digest))
			out, err := os.Create(filepath)
			if err != nil {
				log.Printf("couldn't create %s", filepath)
				return err
			}
			defer out.Close()

			_, err = io.Copy(out, layer.Buffer)
			if err != nil {
				return err
			}
		} else {
			fn(fmt.Sprintf("using already created layer %s", layer.Digest))
		}

		if layer.MediaType == "application/vnd.docker.container.image.v1+json" {
			continue
		}

		manifestLayer := &Layer{
			MediaType: layer.MediaType,
			Size:      layer.Size,
			Digest:    layer.Digest,
		}

		manifestLayers = append(manifestLayers, manifestLayer)
	}

	// Create the manifest
	fn("writing manifest")
	manifest := ManifestV2{
		SchemaVersion: 2,
		MediaType:     "application/vnd.docker.distribution.manifest.v2+json",
		Config: Layer{
			MediaType: cfg.MediaType,
			Size:      cfg.Size,
			Digest:    cfg.Digest,
		},
		Layers: manifestLayers,
	}

	manifestJSON, err := json.Marshal(manifest)
	if err != nil {
		return err
	}

	filepath := path.Join(home, ".ollama/models/manifests", name)
	err = os.WriteFile(filepath, manifestJSON, 0644)
	if err != nil {
		log.Printf("couldn't write to %s", filepath)
		return err
	}

	fn("success")
	return nil
}

func paramsToReader(m map[string]string) (io.Reader, error) {
	data, err := json.MarshalIndent(m, "", "  ")
	if err != nil {
		return nil, err
	}

	return strings.NewReader(string(data)), nil
}

func getLayerDigests(layers []*LayerWithBuffer) ([]string, error) {
	var digests []string
	for _, l := range layers {
		if l.Digest == "" {
			return nil, fmt.Errorf("layer is missing a digest!")
		}
		digests = append(digests, l.Digest)
	}
	return digests, nil
}

// CreateLayer creates a Layer object from a given file
func CreateLayer(f io.Reader) (*LayerWithBuffer, error) {
	buf := new(bytes.Buffer)
	_, err := io.Copy(buf, f)
	if err != nil {
		return nil, err
	}

	digest, size := GetSHA256Digest(buf)

	layer := &LayerWithBuffer{
		Layer: Layer{
			MediaType: "application/vnd.docker.image.rootfs.diff.tar",
			Digest:    digest,
			Size:      size,
		},
		Buffer: buf,
	}

	return layer, nil
}

func PushModel(name, username, password string, fn func(status, digest string, Total, Completed int, Percent float64)) error {
	fn("retrieving manifest", "", 0, 0, 0)
	manifest, err := GetManifest(name)
	if err != nil {
		fn("couldn't retrieve manifest", "", 0, 0, 0)
		return err
	}

	var repoName string
	var tag string

	comps := strings.Split(name, ":")
	switch {
	case len(comps) < 1 || len(comps) > 2:
		return fmt.Errorf("repository name was invalid")
	case len(comps) == 1:
		repoName = comps[0]
		tag = "latest"
	case len(comps) == 2:
		repoName = comps[0]
		tag = comps[1]
	}

	var layers []*Layer
	var total int
	var completed int
	for _, layer := range manifest.Layers {
		layers = append(layers, layer)
		total += layer.Size
	}
	layers = append(layers, &manifest.Config)
	total += manifest.Config.Size

	for _, layer := range layers {
		exists, err := checkBlobExistence(DefaultRegistry, repoName, layer.Digest, username, password)
		if err != nil {
			return err
		}

		if exists {
			completed += layer.Size
			fn("using existing layer", layer.Digest, total, completed, float64(completed)/float64(total))
			continue
		}

		fn("starting upload", layer.Digest, total, completed, float64(completed)/float64(total))

		location, err := startUpload(DefaultRegistry, repoName, username, password)
		if err != nil {
			log.Printf("couldn't start upload: %v", err)
			return err
		}

		err = uploadBlob(location, layer, username, password)
		if err != nil {
			log.Printf("error uploading blob: %v", err)
			return err
		}
		completed += layer.Size
		fn("upload complete", layer.Digest, total, completed, float64(completed)/float64(total))
	}

	fn("pushing manifest", "", total, completed, float64(completed/total))
	url := fmt.Sprintf("%s/v2/%s/manifests/%s", DefaultRegistry, repoName, tag)
	headers := map[string]string{
		"Content-Type": "application/vnd.docker.distribution.manifest.v2+json",
	}

	manifestJSON, err := json.Marshal(manifest)
	if err != nil {
		return err
	}

	resp, err := makeRequest("PUT", url, headers, bytes.NewReader(manifestJSON), username, password)
	defer resp.Body.Close()

	// Check for success: For a successful upload, the Docker registry will respond with a 201 Created
	if resp.StatusCode != http.StatusCreated {
		body, _ := ioutil.ReadAll(resp.Body)
		return fmt.Errorf("registry responded with code %d: %v", resp.StatusCode, string(body))
	}

	fn("success", "", total, completed, 1.0)

	return nil
}

func PullModel(name, username, password string, fn func(status, digest string, Total, Completed int, Percent float64)) error {
	var repoName string
	var tag string

	comps := strings.Split(name, ":")
	switch {
	case len(comps) < 1 || len(comps) > 2:
		return fmt.Errorf("repository name was invalid")
	case len(comps) == 1:
		repoName = comps[0]
		tag = "latest"
	case len(comps) == 2:
		repoName = comps[0]
		tag = comps[1]
	}

	fn("pulling manifest", "", 0, 0, 0)

	manifest, err := pullModelManifest(DefaultRegistry, repoName, tag, username, password)
	if err != nil {
		fmt.Errorf("Error: %q", err)
		return err
	}

	var layers []*Layer
	var total int
	var completed int
	for _, layer := range manifest.Layers {
		layers = append(layers, layer)
		total += layer.Size
	}
	layers = append(layers, &manifest.Config)
	total += manifest.Config.Size

	for _, layer := range layers {
		fn("starting download", layer.Digest, total, completed, float64(completed)/float64(total))
		if err := downloadBlob(DefaultRegistry, repoName, layer.Digest, username, password); err != nil {
			return err
		}
		completed += layer.Size
		fn("download complete", layer.Digest, total, completed, float64(completed)/float64(total))
	}

	fn("writing manifest", "", total, completed, 1.0)

	home, err := os.UserHomeDir()
	if err != nil {
		return err
	}

	manifestJSON, err := json.Marshal(manifest)
	if err != nil {
		return err
	}

	filepath := path.Join(home, ".ollama/models/manifests", name)
	err = os.WriteFile(filepath, manifestJSON, 0644)
	if err != nil {
		log.Printf("couldn't write to %s", filepath)
		return err
	}

	fn("success", "", total, completed, 1.0)

	return nil
}

func pullModelManifest(registryURL, repoName, tag, username, password string) (*ManifestV2, error) {
	url := fmt.Sprintf("%s/v2/%s/manifests/%s", registryURL, repoName, tag)
	headers := map[string]string{
		"Accept": "application/vnd.docker.distribution.manifest.v2+json",
	}

	resp, err := makeRequest("GET", url, headers, nil, username, password)
	defer resp.Body.Close()

	// Check for success: For a successful upload, the Docker registry will respond with a 201 Created
	if resp.StatusCode != http.StatusOK {
		body, _ := ioutil.ReadAll(resp.Body)
		return nil, fmt.Errorf("registry responded with code %d: %v", resp.StatusCode, string(body))
	}

	var m *ManifestV2
	if err := json.NewDecoder(resp.Body).Decode(&m); err != nil {
		return nil, err
	}

	return m, err
}

func createConfigLayer(layers []string) (*LayerWithBuffer, error) {
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

	buf := bytes.NewBuffer(configJSON)
	digest, size := GetSHA256Digest(buf)

	layer := &LayerWithBuffer{
		Layer: Layer{
			MediaType: "application/vnd.docker.container.image.v1+json",
			Digest:    digest,
			Size:      size,
		},
		Buffer: buf,
	}
	return layer, nil
}

// GetSHA256Digest returns the SHA256 hash of a given buffer and returns it, and the size of buffer
func GetSHA256Digest(data *bytes.Buffer) (string, int) {
	layerBytes := data.Bytes()
	hash := sha256.Sum256(layerBytes)
	return "sha256:" + hex.EncodeToString(hash[:]), len(layerBytes)
}

func startUpload(registryURL string, repositoryName string, username string, password string) (string, error) {
	url := fmt.Sprintf("%s/v2/%s/blobs/uploads/", registryURL, repositoryName)

	resp, err := makeRequest("POST", url, nil, nil, username, password)
	defer resp.Body.Close()

	if err != nil {
		return "", fmt.Errorf("failed to create request: %v", err)
	}

	// Check for success
	if resp.StatusCode != http.StatusAccepted {
		body, _ := ioutil.ReadAll(resp.Body)
		return "", fmt.Errorf("registry responded with code %d: %v", resp.StatusCode, string(body))
	}

	// Extract UUID location from header
	location := resp.Header.Get("Location")
	if location == "" {
		return "", fmt.Errorf("Location header is missing in response")
	}

	return location, nil
}

// Function to check if a blob already exists in the Docker registry
func checkBlobExistence(registryURL string, repositoryName string, digest string, username string, password string) (bool, error) {
	url := fmt.Sprintf("%s/v2/%s/blobs/%s", registryURL, repositoryName, digest)

	resp, err := makeRequest("HEAD", url, nil, nil, username, password)
	defer resp.Body.Close()

	if err != nil {
		return false, fmt.Errorf("failed to create request: %v", err)
	}

	// Check for success: If the blob exists, the Docker registry will respond with a 200 OK
	return resp.StatusCode == http.StatusOK, nil
}

func uploadBlob(location string, layer *Layer, username string, password string) error {
	home, err := os.UserHomeDir()
	if err != nil {
		return err
	}

	// Create URL
	url := fmt.Sprintf("%s&digest=%s", location, layer.Digest)

	headers := make(map[string]string)
	headers["Content-Length"] = fmt.Sprintf("%d", layer.Size)
	headers["Content-Type"] = "application/octet-stream"

	// TODO change from monolithic uploads to chunked uploads
	// TODO allow resumability
	// TODO allow canceling uploads via DELETE
	// TODO allow cross repo blob mount

	filepath := path.Join(home, ".ollama/models/blobs", layer.Digest)
	f, err := os.Open(filepath)
	if err != nil {
		return err
	}

	resp, err := makeRequest("PUT", url, headers, f, username, password)
	defer resp.Body.Close()

	if err != nil {
		return err
	}

	// Check for success: For a successful upload, the Docker registry will respond with a 201 Created
	if resp.StatusCode != http.StatusCreated {
		body, _ := ioutil.ReadAll(resp.Body)
		return fmt.Errorf("registry responded with code %d: %v", resp.StatusCode, string(body))
	}

	return nil
}

func downloadBlob(registryURL, repoName, digest, username, password string) error {
	home, err := os.UserHomeDir()
	if err != nil {
		return err
	}

	filepath := path.Join(home, ".ollama/models/blobs", digest)

	_, err = os.Stat(filepath)
	if !os.IsNotExist(err) {
		// we already have the file, so return
		log.Printf("already have %s\n", digest)
		return nil
	}

	url := fmt.Sprintf("%s/v2/%s/blobs/%s", registryURL, repoName, digest)
	headers := map[string]string{}

	resp, err := makeRequest("GET", url, headers, nil, username, password)
	defer resp.Body.Close()

	// TODO: handle 307 redirects
	// TODO: handle range requests to make this resumable

	if resp.StatusCode != http.StatusOK {
		body, _ := ioutil.ReadAll(resp.Body)
		return fmt.Errorf("registry responded with code %d: %v", resp.StatusCode, string(body))
	}

	out, err := os.Create(filepath)
	if err != nil {
		log.Printf("couldn't create %s", filepath)
		return err
	}
	defer out.Close()

	_, err = io.Copy(out, resp.Body)
	if err != nil {
		return err
	}

	log.Printf("success getting %s\n", digest)
	return nil
}

func makeRequest(method, url string, headers map[string]string, body io.Reader, username, password string) (*http.Response, error) {
	req, err := http.NewRequest(method, url, body)
	if err != nil {
		return nil, err
	}

	for k, v := range headers {
		req.Header.Set(k, v)
	}

	// TODO: better auth
	if username != "" && password != "" {
		req.SetBasicAuth(username, password)
	}

	client := &http.Client{}
	resp, err := client.Do(req)
	if err != nil {
		return nil, err
	}

	return resp, nil
}
