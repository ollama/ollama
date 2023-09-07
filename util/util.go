package util

import "os"

// UserHomeDir returns the path to the user's home directory.
//
// It first checks if the "OLLAMA_HOME" environment variable is set and returns its value if so.
// If the environment variable is not set, it uses the os.UserHomeDir() function to get the path.
// The function returns the path as a string and an error if there was any issue getting the path.
func UserHomeDir() (string, error) {
	envHomePath := os.Getenv("OLLAMA_HOME")
	if envHomePath != "" {
		return envHomePath, nil
	}

	home, err := os.UserHomeDir()
	if err != nil {
		return "", err
	}

	return home, nil
}
