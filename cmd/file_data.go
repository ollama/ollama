package cmd

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/cmd/internal/filedata"
)

func normalizeFilePath(fp string) string {
	return filedata.NormalizePath(fp)
}

func extractFileNames(input string) []string {
	return filedata.ExtractNames(input)
}

func extractFileData(input string) (string, []api.ImageData, error) {
	cleaned, files, err := filedata.ExtractWithFiles(input)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Couldn't process file: %q\n", err)
		return "", nil, err
	}

	imgs := make([]api.ImageData, 0, len(files))
	for _, file := range files {
		ext := strings.ToLower(filepath.Ext(file.Path))
		switch ext {
		case ".wav":
			fmt.Fprintf(os.Stderr, "Added audio '%s'\n", file.Path)
		default:
			fmt.Fprintf(os.Stderr, "Added image '%s'\n", file.Path)
		}
		imgs = append(imgs, file.Data)
	}
	return cleaned, imgs, nil
}

func getImageData(filePath string) ([]byte, error) {
	return filedata.GetData(filePath)
}
