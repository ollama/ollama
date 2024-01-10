package cmd

import (
	"fmt"
	"github.com/joho/godotenv"
	"os"
	"path/filepath"
)

// LoadDotEnvFromOllamaFolder loads environment variables from a .env file located in the ~/.ollama directory.
// If the file does not exist, the function returns nil without an error.
func LoadDotEnvFromOllamaFolder() error {
	home, err := os.UserHomeDir()
	if err != nil {
		return fmt.Errorf("failed to get user home directory: %w", err)
	}

	envPath := filepath.Join(home, ".ollama", ".env")

	// Check if the .env file exists
	if _, err := os.Stat(envPath); os.IsNotExist(err) {
		// If the file does not exist, return nil without an error
		return nil
	} else if err != nil {
		// If there is another error when checking the file, return that error
		return fmt.Errorf("failed to check if .env file exists: %w", err)
	}

	// Load the .env file
	if err := godotenv.Load(envPath); err != nil {
		return fmt.Errorf("could not load %s: %w", envPath, err)
	}

	return nil
}
