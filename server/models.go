package server

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"path"
	"strconv"
)

const directoryURL = "https://ollama.ai/api/models"

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

func (m *Model) FullName() string {
	home, err := os.UserHomeDir()
	if err != nil {
		panic(err)
	}

	return path.Join(home, ".ollama", "models", m.Name+".bin")
}

func (m *Model) TempFile() string {
	fullName := m.FullName()
	return path.Join(
		path.Dir(fullName),
		fmt.Sprintf(".%s.part", path.Base(fullName)),
	)
}

func getRemote(model string) (*Model, error) {
	// resolve the model download from our directory
	resp, err := http.Get(directoryURL)
	if err != nil {
		return nil, fmt.Errorf("failed to get directory: %w", err)
	}
	defer resp.Body.Close()
	body, err := io.ReadAll(resp.Body)
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

func saveModel(model *Model, fn func(total, completed int64)) error {
	// this models cache directory is created by the server on startup

	client := &http.Client{}
	req, err := http.NewRequest("GET", model.URL, nil)
	if err != nil {
		return fmt.Errorf("failed to download model: %w", err)
	}

	// check if completed file exists
	fi, err := os.Stat(model.FullName())
	switch {
	case errors.Is(err, os.ErrNotExist):
		// noop, file doesn't exist so create it
	case err != nil:
		return fmt.Errorf("stat: %w", err)
	default:
		fn(fi.Size(), fi.Size())
		return nil
	}

	var size int64

	// completed file doesn't exist, check partial file
	fi, err = os.Stat(model.TempFile())
	switch {
	case errors.Is(err, os.ErrNotExist):
		// noop, file doesn't exist so create it
	case err != nil:
		return fmt.Errorf("stat: %w", err)
	default:
		size = fi.Size()
	}

	req.Header.Add("Range", fmt.Sprintf("bytes=%d-", size))

	resp, err := client.Do(req)
	if err != nil {
		return fmt.Errorf("failed to download model: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode >= 400 {
		return fmt.Errorf("failed to download model: %s", resp.Status)
	}

	out, err := os.OpenFile(model.TempFile(), os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0o644)
	if err != nil {
		panic(err)
	}
	defer out.Close()

	totalSize, _ := strconv.ParseInt(resp.Header.Get("Content-Length"), 10, 64)

	totalBytes := size
	totalSize += size

	for {
		n, err := io.CopyN(out, resp.Body, 8192)
		if err != nil && !errors.Is(err, io.EOF) {
			return err
		}

		if n == 0 {
			break
		}

		totalBytes += n
		fn(totalSize, totalBytes)
	}

	fn(totalSize, totalSize)
	return os.Rename(model.TempFile(), model.FullName())
}
