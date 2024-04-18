package main

import (
	"bytes"
	"context"
	"encoding/base64"
	"fmt"
	"log/slog"
	"os"
	"os/exec"
	"regexp"
	"runtime"
	"strings"

	"github.com/ollama/ollama/auth"
	"github.com/ollama/ollama/cmd"
	"github.com/pkg/browser"
)

func main() {
	if err := cmd.NewCLI().ExecuteContext(context.Background()); err != nil {
		fmt.Fprintln(os.Stderr, "Error:", err)

		if strings.Contains(err.Error(), "unknown ollama key") {
			if err := handleUnkownKeyError(err); err != nil {
				fmt.Fprintln(os.Stderr, "Error:", err)
			}
		}

		os.Exit(1)
	}
}

func handleUnkownKeyError(accessErr error) error {
	// find SSH public key in the error message
	sshKeyPattern := `ssh-\w+ [^\s"]+`
	re := regexp.MustCompile(sshKeyPattern)
	matches := re.FindStringSubmatch(accessErr.Error())

	if len(matches) > 0 {
		serverPubKey := matches[0]

		localPubKey, err := auth.GetPublicKey()
		if err != nil {
			return accessErr
		}

		if runtime.GOOS == "linux" {
			// check if the ollama service is active
			if isOllamaServiceActive() {
				svcPubKey, err := os.ReadFile("/usr/share/ollama/.ollama/id_ed25519.pub")
				if err != nil {
					slog.Info(fmt.Sprintf("Failed to load public key: %v", err))
					return accessErr
				}
				localPubKey = strings.TrimSpace(string(svcPubKey))
			}
		}

		// check if the returned public key matches the local public key, this prevents adding a remote key to the user's account
		if serverPubKey != localPubKey {
			return accessErr
		}

		fmt.Println("")
		fmt.Println("Add your key at: https://ollama.com/settings/keys")
		fmt.Println(localPubKey)

		//lint:ignore errcheck this is optional, it may not work on all systems
		browser.OpenURL("https://ollama.com/settings/keys?add=" + base64.StdEncoding.EncodeToString([]byte(localPubKey)))

		return nil
	}

	return accessErr
}

// isOllamaServiceActive uses systemctl to check if the ollama service is active on linux
func isOllamaServiceActive() bool {
	cmd := exec.Command("systemctl", "is-active", "ollama")

	var out bytes.Buffer
	cmd.Stdout = &out

	err := cmd.Run()
	status := strings.TrimSpace(out.String())

	if err != nil {
		// either the service is not found or systemctl is not available
		return false
	}

	// Check if the service is active
	return status == "active"
}
