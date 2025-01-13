package main

import (
	"fmt"
	"os"
	"path/filepath"
	"runtime"

	"github.com/ollama/ollama/convert"
)

func main() {
	// Check if directory path is provided
	if len(os.Args) != 2 {
		fmt.Printf("expected one argument (directory path), got %d\n", len(os.Args)-1)
		os.Exit(1)
	}

	dirPath := os.Args[1]

	if err := convertFromDirectory(dirPath); err != nil {
		fmt.Fprintf(os.Stderr, "error: %v\n", err)
		os.Exit(1)
	}
	fmt.Println("conversion completed successfully")
}

func convertFromDirectory(dirPath string) error {
	// Verify the directory exists and is accessible
	info, err := os.Stat(dirPath)
	if err != nil {
		if os.IsNotExist(err) {
			return fmt.Errorf("directory does not exist: %s", dirPath)
		}
		if os.IsPermission(err) {
			return fmt.Errorf("permission denied accessing directory: %s", dirPath)
		}
		return fmt.Errorf("error accessing directory: %v", err)
	}
	if !info.IsDir() {
		return fmt.Errorf("%s is not a directory", dirPath)
	}

	// Get the directory where the script is located
	_, scriptPath, _, ok := runtime.Caller(0)
	if !ok {
		return fmt.Errorf("could not determine script location")
	}
	scriptDir := filepath.Dir(scriptPath)

	// Create out directory relative to the script location
	outDir := filepath.Join(scriptDir, "out")
	if err := os.MkdirAll(outDir, 0755); err != nil {
		return fmt.Errorf("failed to create output directory: %v", err)
	}

	// Create output file in the out directory
	outFile := filepath.Join(outDir, "model.fp16")
	fmt.Printf("writing output to: %s\n", outFile)

	t, err := os.Create(outFile)
	if err != nil {
		return fmt.Errorf("failed to create output file: %v", err)
	}
	defer t.Close()

	// Use standard os.DirFS to read from directory
	if err := convert.ConvertModel(os.DirFS(dirPath), t); err != nil {
		// Clean up the output file if conversion fails
		os.Remove(outFile)
		return fmt.Errorf("model conversion failed: %v", err)
	}

	return nil
}
