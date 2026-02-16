package main

import (
	"bufio"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"strings"
)

func main() {
	if len(os.Args) < 2 {
		fmt.Fprintln(os.Stderr, "Usage: go run extract-examples.go <mdx-file>")
		os.Exit(1)
	}

	mdxFile := os.Args[1]

	f, err := os.Open(mdxFile)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		os.Exit(1)
	}
	defer f.Close()

	// Create temp directory
	tempDir, err := os.MkdirTemp("", "mdx-examples-*")
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error creating temp dir: %v\n", err)
		os.Exit(1)
	}

	fmt.Printf("Extracting code examples to: %s\n\n", tempDir)

	// Patterns
	codeBlockStart := regexp.MustCompile("^```([a-zA-Z0-9_-]+)\\s+([^\\s]+)$")
	codeGroupStart := regexp.MustCompile("^<CodeGroup")
	codeGroupEnd := regexp.MustCompile("^</CodeGroup>")

	scanner := bufio.NewScanner(f)
	inCodeBlock := false
	inCodeGroup := false
	var currentFile string
	var content strings.Builder
	count := 0
	codeGroupNum := 0

	for scanner.Scan() {
		line := scanner.Text()

		// Track CodeGroup boundaries
		if codeGroupStart.MatchString(line) {
			inCodeGroup = true
			codeGroupNum++
			continue
		}
		if codeGroupEnd.MatchString(line) {
			inCodeGroup = false
			continue
		}

		if inCodeBlock {
			if line == "```" {
				// End of code block - write file
				if currentFile != "" {
					outPath := filepath.Join(tempDir, currentFile)
					if err := os.WriteFile(outPath, []byte(content.String()), 0o644); err != nil {
						fmt.Fprintf(os.Stderr, "Error writing %s: %v\n", currentFile, err)
					} else {
						fmt.Printf("  - %s\n", currentFile)
						count++
					}
				}
				inCodeBlock = false
				currentFile = ""
				content.Reset()
			} else {
				content.WriteString(line)
				content.WriteString("\n")
			}
		} else {
			if matches := codeBlockStart.FindStringSubmatch(line); matches != nil {
				inCodeBlock = true
				filename := matches[2]
				// Prefix with CodeGroup number if inside a CodeGroup
				if inCodeGroup {
					currentFile = fmt.Sprintf("%02d_%s", codeGroupNum, filename)
				} else {
					currentFile = filename
				}
				content.Reset()
			}
		}
	}

	if err := scanner.Err(); err != nil {
		fmt.Fprintf(os.Stderr, "Error reading file: %v\n", err)
		os.Exit(1)
	}

	// Write package.json for JavaScript dependencies
	packageJSON := `{
  "name": "mdx-examples",
  "type": "module",
  "dependencies": {
    "openai": "^4",
    "ollama": "^0.5"
  }
}
`
	if err := os.WriteFile(filepath.Join(tempDir, "package.json"), []byte(packageJSON), 0o644); err != nil {
		fmt.Fprintf(os.Stderr, "Error writing package.json: %v\n", err)
	}

	// Write pyproject.toml for Python dependencies
	pyprojectTOML := `[project]
name = "mdx-examples"
version = "0.0.0"
dependencies = [
    "openai",
    "ollama",
]
`
	if err := os.WriteFile(filepath.Join(tempDir, "pyproject.toml"), []byte(pyprojectTOML), 0o644); err != nil {
		fmt.Fprintf(os.Stderr, "Error writing pyproject.toml: %v\n", err)
	}

	fmt.Printf("\n")
	fmt.Printf("Extracted %d file(s) to %s\n", count, tempDir)
	fmt.Printf("\n")
	fmt.Printf("To run examples:\n")
	fmt.Printf("\n")
	fmt.Printf("  cd %s\n  npm install   # for JS examples\n", tempDir)
	fmt.Printf("\n")
	fmt.Printf("then run individual files with `node file.js`, `python file.py`, `bash file.sh`\n")
}
