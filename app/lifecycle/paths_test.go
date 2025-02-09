package lifecycle

import (
    "strings"
    "testing"
)


// Test generated using Keploy
func TestInitialize_WindowsPathAlreadyContainsAppDir(t *testing.T) {
    // Save original values
    originalGetEnv := getEnv
    defer func() {
        getEnv = originalGetEnv
    }()

    // Mock getEnv
    getEnv = func(key string) string {
        switch key {
        case "PATH":
            return "C:\\Windows\\System32;" + AppDir
        case "LOCALAPPDATA":
            return "C:\\Users\\TestUser\\AppData\\Local"
        default:
            return ""
        }
    }

    // Mock getExecutable
    getExecutable = func() (string, error) {
        return "C:\\Program Files\\Ollama\\ollama.exe", nil
    }

    // Set AppDir
    AppDir = "C:\\Program Files\\Ollama"

    // Call initialize
    initialize("windows")

    // Check that AppDir is not duplicated in PATH
    pathEnv := getEnv("PATH")
    if strings.Count(pathEnv, AppDir) != 1 {
        t.Errorf("Expected PATH to contain AppDir only once, but found multiple occurrences")
    }
}
