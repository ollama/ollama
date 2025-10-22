package tools

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/ledongthuc/pdf"
)

// FileInfo represents information about a single file or directory
type FileInfo struct {
	// BasePath string `json:"base_path"`
	RelPath string `json:"rel_path"`
	IsDir   bool   `json:"is_dir"`
}

// FileListResult represents the result of a directory listing operation
type FileListResult struct {
	BasePath string     `json:"base_path"`
	Files    []FileInfo `json:"files"`
	Count    int        `json:"count"`
}

// FileReadResult represents the result of a file read operation
type FileReadResult struct {
	Path       string `json:"path"`
	TotalLines int    `json:"total_lines"`
	LinesRead  int    `json:"lines_read"`
	Content    string `json:"content"`
}

// FileWriteResult represents the result of a file write operation
type FileWriteResult struct {
	Path     string `json:"path"`
	Size     int64  `json:"size,omitempty"`
	Written  int    `json:"written"`
	Mode     string `json:"mode,omitempty"`
	Modified int64  `json:"modified,omitempty"`
}

// FileReader implements the file reading functionality
type FileReader struct {
	workingDir string
}

func (f *FileReader) SetWorkingDir(dir string) {
	f.workingDir = dir
}

func (f *FileReader) Name() string {
	return "file_read"
}

func (f *FileReader) Description() string {
	return "Read the contents of a file from the file system"
}

func (f *FileReader) Prompt() string {
	// TODO: read iteratively in agent mode, full in single shot - control with prompt?
	return `Use the file_read tool to read the contents of a file using the path parameter. read_full is false by default and will return the first 100 lines of the file, if the user requires more information about the file, set read_full to true`
}

func (f *FileReader) Schema() map[string]any {
	schemaBytes := []byte(`{
		"type": "object",
		"properties": {
			"path": {
				"type": "string",
				"description": "The path to the file to read"
			},
			"read_full": {
				"type": "boolean", 
				"description": "returns the first 100 lines of the file when set to false (default: false)",
				"default": false 
			}
		},
		"required": ["path"]
	}`)
	var schema map[string]any
	if err := json.Unmarshal(schemaBytes, &schema); err != nil {
		return nil
	}
	return schema
}

func (f *FileReader) Execute(ctx context.Context, args map[string]any) (any, error) {
	fmt.Println("file_read tool called", args)
	path, ok := args["path"].(string)
	if !ok {
		return nil, fmt.Errorf("path parameter is required and must be a string")
	}

	// If path is not absolute and working directory is set, make it relative to working directory
	if !filepath.IsAbs(path) && f.workingDir != "" {
		path = filepath.Join(f.workingDir, path)
	}

	// Security: Clean and validate the path
	cleanPath := filepath.Clean(path)
	if strings.Contains(cleanPath, "..") {
		return nil, fmt.Errorf("path traversal not allowed")
	}

	// Get max size limit
	maxSize := int64(1024 * 1024) // 1MB default
	if ms, ok := args["max_size"]; ok {
		switch v := ms.(type) {
		case float64:
			maxSize = int64(v)
		case int:
			maxSize = int64(v)
		case int64:
			maxSize = v
		}
	}

	// Check if file exists and get info
	info, err := os.Stat(cleanPath)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, fmt.Errorf("file does not exist: %s", cleanPath)
		}
		return nil, fmt.Errorf("error accessing file: %w", err)
	}

	// Check if it's a directory
	if info.IsDir() {
		return nil, fmt.Errorf("path is a directory, not a file: %s", cleanPath)
	}

	// Check file size
	if info.Size() > maxSize {
		return nil, fmt.Errorf("file too large (%d bytes), maximum allowed: %d bytes", info.Size(), maxSize)
	}

	if strings.HasSuffix(strings.ToLower(cleanPath), ".pdf") {
		return f.readPDFFile(cleanPath, args)
	}

	// Check read_full parameter
	readFull := false // default to false
	if rf, ok := args["read_full"]; ok {
		readFull, _ = rf.(bool)
	}

	// Open and read the file
	file, err := os.Open(cleanPath)
	if err != nil {
		return nil, fmt.Errorf("error opening file: %w", err)
	}
	defer file.Close()

	// Read file content
	scanner := bufio.NewScanner(file)
	var lines []string
	totalLines := 0

	// Read content, keeping track of total lines but only storing up to 100 if !readFull
	for scanner.Scan() {
		totalLines++
		if readFull || totalLines <= 100 {
			lines = append(lines, scanner.Text())
		}
	}

	if err := scanner.Err(); err != nil {
		return nil, fmt.Errorf("error reading file: %w", err)
	}

	content := strings.Join(lines, "\n")

	return &FileReadResult{
		Path:       cleanPath,
		LinesRead:  len(lines),
		TotalLines: totalLines,
		Content:    content,
	}, nil
}

// readPDFFile extracts text from a PDF file
func (f *FileReader) readPDFFile(cleanPath string, args map[string]any) (any, error) {
	// Open the PDF file
	pdfFile, r, err := pdf.Open(cleanPath)
	if err != nil {
		return nil, fmt.Errorf("error opening PDF: %w", err)
	}
	defer pdfFile.Close()

	// Get total number of pages
	totalPages := r.NumPage()

	// Check read_full parameter - for PDFs, this controls whether to read all pages
	readFull := false
	if rf, ok := args["read_full"]; ok {
		readFull, _ = rf.(bool)
	}

	// Extract text from pages
	var allText strings.Builder
	maxPages := 10 // Default to first 10 pages if not read_full
	if readFull {
		maxPages = totalPages
	}

	linesExtracted := 0
	for pageNum := 1; pageNum <= totalPages && pageNum <= maxPages; pageNum++ {
		// Get page
		page := r.Page(pageNum)
		if page.V.IsNull() {
			continue
		}

		// Use the built-in GetPlainText method which handles text extraction better
		pageText, err := page.GetPlainText(nil)
		if err != nil {
			// If GetPlainText fails, fall back to manual extraction
			pageText = f.extractTextFromPage(page)
		}

		pageText = strings.TrimSpace(pageText)
		if pageText != "" {
			if allText.Len() > 0 {
				allText.WriteString("\n\n")
			}
			allText.WriteString(fmt.Sprintf("--- Page %d ---\n", pageNum))
			allText.WriteString(pageText)

			// Count lines for reporting
			linesExtracted += strings.Count(pageText, "\n") + 1
		}
	}

	content := strings.TrimSpace(allText.String())

	// If no text was extracted, return a helpful message
	if content == "" {
		content = "[PDF file contains no extractable text - it may contain only images or use complex encoding]"
		linesExtracted = 1
	}

	return &FileReadResult{
		Path:       cleanPath,
		LinesRead:  linesExtracted,
		TotalLines: totalPages, // For PDFs, we report pages as "lines"
		Content:    content,
	}, nil
}

// extractTextFromPage extracts text from a single PDF page
func (f *FileReader) extractTextFromPage(page pdf.Page) string {
	var buf bytes.Buffer

	// Get page contents
	contents := page.Content()

	// Group text elements that appear to be part of the same word/line
	var currentLine strings.Builder
	lastX := -1.0

	for i, t := range contents.Text {
		// Skip empty text
		if t.S == "" {
			continue
		}

		// Check if this text element is on a new line or far from the previous one
		// If X position is significantly different or we've reset to the beginning, it's likely a new word
		if lastX >= 0 && (t.X < lastX-10 || t.X > lastX+50) {
			// Add the accumulated line to buffer with a space
			if currentLine.Len() > 0 {
				buf.WriteString(currentLine.String())
				buf.WriteString(" ")
				currentLine.Reset()
			}
		}

		// Add the text without extra spaces
		currentLine.WriteString(t.S)
		lastX = t.X

		// Check if next element exists and has significantly different Y position (new line)
		if i+1 < len(contents.Text) && contents.Text[i+1].Y > t.Y+5 {
			if currentLine.Len() > 0 {
				buf.WriteString(currentLine.String())
				buf.WriteString("\n")
				currentLine.Reset()
				lastX = -1.0
			}
		}
	}

	// Add any remaining text
	if currentLine.Len() > 0 {
		buf.WriteString(currentLine.String())
	}

	return strings.TrimSpace(buf.String())
}

// FileList implements the directory listing functionality
type FileList struct {
	workingDir string
}

func (f *FileList) SetWorkingDir(dir string) {
	f.workingDir = dir
}

func (f *FileList) Name() string {
	return "file_list"
}

func (f *FileList) Description() string {
	return "List the contents of a directory"
}

func (f *FileList) Prompt() string {
	return `Use the file_list tool to list the contents of a directory using the path parameter`
}

func (f *FileList) Schema() map[string]any {
	schemaBytes := []byte(`{
		"type": "object",
		"properties": {
			"path": {
				"type":        "string", 
				"description": "The path to the directory to list (default: current directory)",
				"default":     "."
			},
			"show_hidden": {
				"type":        "boolean",
				"description": "Whether to show hidden files (starting with .)",
				"default":     false
			},
			"depth": {
				"type":        "integer",
				"description": "How many directory levels deep to list (default: 1)",
				"default":     1
			}
		},
		"required": []
	}`)
	var schema map[string]any
	if err := json.Unmarshal(schemaBytes, &schema); err != nil {
		return nil
	}
	return schema
}

func (f *FileList) Execute(ctx context.Context, args map[string]any) (any, error) {
	path := "."
	if p, ok := args["path"].(string); ok {
		path = p
	}

	// If path is not absolute and working directory is set, make it relative to working directory
	if !filepath.IsAbs(path) && f.workingDir != "" {
		path = filepath.Join(f.workingDir, path)
	}

	// Security: Clean and validate the path
	cleanPath := filepath.Clean(path)
	if strings.Contains(cleanPath, "..") {
		return nil, fmt.Errorf("path traversal not allowed")
	}

	// Get optional parameters
	showHidden := false
	if sh, ok := args["show_hidden"].(bool); ok {
		showHidden = sh
	}

	maxDepth := 1
	if md, ok := args["depth"].(float64); ok {
		maxDepth = int(md)
	}

	// Check if directory exists
	info, err := os.Stat(cleanPath)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, fmt.Errorf("directory does not exist: %s", cleanPath)
		}
		return nil, fmt.Errorf("error accessing directory: %w", err)
	}

	if !info.IsDir() {
		return nil, fmt.Errorf("path is not a directory: %s", cleanPath)
	}

	var files []FileInfo

	files, err = f.listRecursive(cleanPath, showHidden, maxDepth, 0)

	if err != nil {
		return nil, err
	}

	return &FileListResult{
		BasePath: cleanPath,
		Files:    files,
		Count:    len(files),
	}, nil
}

func (f *FileList) listDirectory(path string, showHidden bool) ([]FileInfo, error) {
	entries, err := os.ReadDir(path)
	if err != nil {
		return nil, fmt.Errorf("error reading directory: %w", err)
	}

	var files []FileInfo
	for _, entry := range entries {
		name := entry.Name()

		// Skip hidden files if not requested
		if !showHidden && strings.HasPrefix(name, ".") {
			continue
		}

		fileInfo := FileInfo{
			RelPath: name,
			IsDir:   entry.IsDir(),
		}

		files = append(files, fileInfo)
	}

	return files, nil
}

func (f *FileList) listRecursive(path string, showHidden bool, maxDepth, currentDepth int) ([]FileInfo, error) {
	if currentDepth >= maxDepth {
		return nil, nil
	}

	files, err := f.listDirectory(path, showHidden)
	if err != nil {
		return nil, err
	}

	var allFiles []FileInfo
	for _, file := range files {
		// For the first level, use the file name as is
		// For deeper levels, join with parent directory
		if currentDepth != 0 {
			// Get the relative part of the path by removing the base path
			rel, err := filepath.Rel(filepath.Dir(path), path)
			if err == nil {
				file.RelPath = filepath.Join(rel, file.RelPath)
			}
		}
		allFiles = append(allFiles, file)

		if file.IsDir {
			subFiles, err := f.listRecursive(filepath.Join(path, file.RelPath), showHidden, maxDepth, currentDepth+1)
			if err != nil {
				continue // Skip directories we can't read
			}
			allFiles = append(allFiles, subFiles...)
		}
	}

	return allFiles, nil
}

// FileWriter implements the file writing functionality
// TODO(parthsareen): max file size limit
type FileWriter struct {
	workingDir string
}

func (f *FileWriter) SetWorkingDir(dir string) {
	f.workingDir = dir
}

func (f *FileWriter) Name() string {
	return "file_write"
}

func (f *FileWriter) Description() string {
	return "Write content to a file on the file system"
}

func (f *FileWriter) Prompt() string {
	return `Use the file_write tool to write content to a file using the path parameter`
}

func (f *FileWriter) Schema() map[string]any {
	schemaBytes := []byte(`{
		"type": "object",
		"properties": {
			"path": {
				"type":        "string",
				"description": "The path to the file to write"
			},
			"content": {
				"type":        "string",
				"description": "The content to write to the file"
			},
			"append": {
				"type":        "boolean",
				"description": "Whether to append to the file instead of overwriting (default: false)",
				"default":     false
			},
			"create_dirs": {
				"type":        "boolean",
				"description": "Whether to create parent directories if they don't exist (default: false)",
				"default":     false
			},
			"max_size": {
				"type":        "integer",
				"description": "Maximum content size to write in bytes (default: 1MB)",
				"default":     1024 * 1024
			}
		},
		"required": ["path", "content"]
	}`)
	var schema map[string]any
	if err := json.Unmarshal(schemaBytes, &schema); err != nil {
		return nil
	}
	return schema
}

func (f *FileWriter) Execute(ctx context.Context, args map[string]any) (any, error) {
	path, ok := args["path"].(string)
	if !ok {
		return nil, fmt.Errorf("path parameter is required and must be a string")
	}

	// If path is not absolute and working directory is set, make it relative to working directory
	if !filepath.IsAbs(path) && f.workingDir != "" {
		path = filepath.Join(f.workingDir, path)
	}

	// Extract required parameters
	content, ok := args["content"].(string)
	if !ok {
		return nil, fmt.Errorf("content parameter is required and must be a string")
	}

	// Get optional parameters with defaults
	append := true // Always append by default
	if a, ok := args["append"].(bool); ok && !a {
		return nil, fmt.Errorf("overwriting existing files is not allowed - must use append mode")
	}

	createDirs := false
	if cd, ok := args["create_dirs"].(bool); ok {
		createDirs = cd
	}

	maxSize := int64(1024 * 1024) // 1MB default
	if ms, ok := args["max_size"].(float64); ok {
		maxSize = int64(ms)
	}

	// Security: Clean and validate the path
	cleanPath := filepath.Clean(path)
	if strings.Contains(cleanPath, "..") {
		return nil, fmt.Errorf("path traversal not allowed")
	}

	// Check content size
	if int64(len(content)) > maxSize {
		return nil, fmt.Errorf("content too large (%d bytes), maximum allowed: %d bytes", len(content), maxSize)
	}

	// Create parent directories if requested
	if createDirs {
		dir := filepath.Dir(cleanPath)
		if err := os.MkdirAll(dir, 0755); err != nil {
			return nil, fmt.Errorf("failed to create parent directories: %w", err)
		}
	}

	// Check if file exists - if it does, we must append
	fileInfo, err := os.Stat(cleanPath)
	if err == nil && fileInfo.Size() > 0 {
		// File exists and has content
		if !append {
			return nil, fmt.Errorf("file %s already exists - cannot overwrite, must use append mode", cleanPath)
		}
	}

	// Open file in append mode
	flag := os.O_WRONLY | os.O_CREATE | os.O_APPEND
	file, err := os.OpenFile(cleanPath, flag, 0644)
	if err != nil {
		return nil, fmt.Errorf("error opening file for writing: %w", err)
	}
	defer file.Close()

	// Write content
	n, err := file.WriteString(content)
	if err != nil {
		return nil, fmt.Errorf("error writing to file: %w", err)
	}

	// Get file info for response
	info, err := file.Stat()
	if err != nil {
		// Return basic success info if we can't get file stats
		return &FileWriteResult{
			Path:    cleanPath,
			Written: n,
		}, nil
	}

	return &FileWriteResult{
		Path:     cleanPath,
		Size:     info.Size(),
		Written:  n,
		Mode:     info.Mode().String(),
		Modified: info.ModTime().Unix(),
	}, nil
}
