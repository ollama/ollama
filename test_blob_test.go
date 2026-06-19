package main_test

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"
)

func getOllamaBlobURL(model string, tag string) (string, error) {
	manifestURL := fmt.Sprintf("https://registry.ollama.ai/v2/library/%s/manifests/%s", model, tag)
	
	client := &http.Client{Timeout: 10 * time.Second}
	req, err := http.NewRequest("GET", manifestURL, nil)
	if err != nil {
		return "", fmt.Errorf("req: %v", err)
	}
	req.Header.Set("Accept", "application/vnd.docker.distribution.manifest.v2+json, application/vnd.ollama.image.model")
	resp, err := client.Do(req)
	if err != nil {
		return "", fmt.Errorf("do: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("status: %v", resp.StatusCode)
	}

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", fmt.Errorf("read: %v", err)
	}
	
	var manifest struct {
		Layers []struct {
			Digest string `json:"digest"`
		} `json:"layers"`
	}
	if err := json.Unmarshal(body, &manifest); err != nil {
		return "", fmt.Errorf("json: %v", err)
	}
	
	if len(manifest.Layers) == 0 {
		return "", fmt.Errorf("no layers")
	}
	
	return fmt.Sprintf("https://registry.ollama.ai/v2/library/%s/blobs/%s", model, manifest.Layers[0].Digest), nil
}

func main() {
	url, err := getOllamaBlobURL("llama3.1", "8b-instruct-q4_k_m")
	if err != nil {
		fmt.Println("ERROR:", err)
		return
	}
	fmt.Println("BLOB URL:", url)

	// test download
	client := &http.Client{Timeout: 10 * time.Second}
	req, err := http.NewRequest("GET", url, nil)
	if err != nil {
		fmt.Println("ERROR req:", err)
		return
	}
	req.Header.Set("Range", "bytes=0-1048575")

	start := time.Now()
	resp, err := client.Do(req)
	if err != nil {
		fmt.Println("ERROR do:", err)
		return
	}
	defer resp.Body.Close()
	if resp.StatusCode >= 400 {
		fmt.Println("ERROR STATUS:", resp.StatusCode)
		return
	}
	
	n, err := io.CopyN(io.Discard, resp.Body, 1024*1024)
	fmt.Printf("Copied %d bytes in %v, err=%v\n", n, time.Since(start), err)
}
