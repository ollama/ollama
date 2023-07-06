package server

import (
	"encoding/json"
	"fmt"
	"io"
	"io/ioutil"
	"net/http"
	"os"
	"path"
	"strconv"

	"github.com/jmorganca/ollama/api"
)

// const directoryURL = "https://ollama.ai/api/models"
// TODO
const directoryURL = "https://raw.githubusercontent.com/jmorganca/ollama/go/models.json"

type Model struct {
	Name             string `json:"name"`
	DisplayName      string `json:"display_name"`
	Parameters       string `json:"parameters"`
	URL              string `json:"url"`
	ShortDescription string `json:"short_description"`
	Description      string `json:"description"`
	PublishedBy      string `json:"published_by"`
	OriginalAuthor   string `json:"original_author"`
	OriginalURL      string `json:"original_url"`
	License          string `json:"license"`
}

func pull(model string, progressCh chan<- api.PullProgress) error {
	remote, err := getRemote(model)
	if err != nil {
		return fmt.Errorf("failed to pull model: %w", err)
	}
	return saveModel(remote, progressCh)
}

func getRemote(model string) (*Model, error) {
	// resolve the model download from our directory
	resp, err := http.Get(directoryURL)
	if err != nil {
		return nil, fmt.Errorf("failed to get directory: %w", err)
	}
	defer resp.Body.Close()
	body, err := ioutil.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read directory: %w", err)
	}
	var models []Model
	err = json.Unmarshal(body, &models)
	if err != nil {
		return nil, fmt.Errorf("failed to parse directory: %w", err)
	}
	for _, m := range models {
		if m.Name == model {
			return &m, nil
		}
	}
	return nil, fmt.Errorf("model not found in directory: %s", model)
}

func saveModel(model *Model, progressCh chan<- api.PullProgress) error {
	// this models cache directory is created by the server on startup
	home, err := os.UserHomeDir()
	if err != nil {
		return fmt.Errorf("failed to get home directory: %w", err)
	}
	modelsCache := path.Join(home, ".ollama", "models")

	fileName := path.Join(modelsCache, model.Name+".bin")

	client := &http.Client{}
	req, err := http.NewRequest("GET", model.URL, nil)
	if err != nil {
		return fmt.Errorf("failed to download model: %w", err)
	}
	// check for resume
	alreadyDownloaded := 0
	fileInfo, err := os.Stat(fileName)
	if err != nil {
		if !os.IsNotExist(err) {
			return fmt.Errorf("failed to check resume model file: %w", err)
		}
		// file doesn't exist, create it now
	} else {
		alreadyDownloaded = int(fileInfo.Size())
		req.Header.Add("Range", "bytes="+strconv.Itoa(alreadyDownloaded)+"-")
	}

	resp, err := client.Do(req)
	if err != nil {
		return fmt.Errorf("failed to download model: %w", err)
	}

	defer resp.Body.Close()

	if resp.StatusCode == http.StatusRequestedRangeNotSatisfiable {
		// already downloaded
		progressCh <- api.PullProgress{
			Total:     alreadyDownloaded,
			Completed: alreadyDownloaded,
			Percent:   100,
		}
		return nil
	}

	if resp.StatusCode != http.StatusOK && resp.StatusCode != http.StatusPartialContent {
		return fmt.Errorf("failed to download model: %s", resp.Status)
	}

	out, err := os.OpenFile(fileName, os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0o644)
	if err != nil {
		panic(err)
	}
	defer out.Close()

	totalSize, _ := strconv.Atoi(resp.Header.Get("Content-Length"))

	buf := make([]byte, 1024)
	totalBytes := alreadyDownloaded
	totalSize += alreadyDownloaded

	for {
		n, err := resp.Body.Read(buf)
		if err != nil && err != io.EOF {
			return err
		}
		if n == 0 {
			break
		}
		if _, err := out.Write(buf[:n]); err != nil {
			return err
		}
		totalBytes += n

		// send progress updates
		progressCh <- api.PullProgress{
			Total:     totalSize,
			Completed: totalBytes,
			Percent:   float64(totalBytes) / float64(totalSize) * 100,
		}
	}

	progressCh <- api.PullProgress{
		Total:     totalSize,
		Completed: totalSize,
		Percent:   100,
	}

	return nil
}
