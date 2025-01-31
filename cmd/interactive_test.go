package cmd

import (
    "testing"
)



// Test generated using Keploy
func TestNormalizeFilePath_EscapedCharacters(t *testing.T) {
    input := `C:\\path\\to\\file\\with\\escaped\\characters\\file\\name\\space\\test\\file.txt`
    expected := `C:\path\to\file\with\escaped\characters\file\name\space\test\file.txt`
    result := normalizeFilePath(input)

    if result != expected {
        t.Errorf("Expected '%s', got '%s'", expected, result)
    }
}

// Test generated using Keploy
func TestExtractFileNames_ValidExtensions(t *testing.T) {
    input := "Here are some files: /path/to/image1.jpg ./image2.png C:\\images\\image3.jpeg"
    expected := []string{"/path/to/image1.jpg", "./image2.png", "C:\\images\\image3.jpeg"}
    result := extractFileNames(input)

    if len(result) != len(expected) {
        t.Fatalf("Expected %d file paths, got %d", len(expected), len(result))
    }
    for i, file := range result {
        if file != expected[i] {
            t.Errorf("Expected file path '%s', got '%s'", expected[i], file)
        }
    }
}

