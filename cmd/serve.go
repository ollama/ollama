package cmd

import (
	"crypto/ed25519"
	"crypto/rand"
	"encoding/pem"
	"fmt"
	"github.com/jmorganca/ollama/server"
	"github.com/spf13/cobra"
	"golang.org/x/crypto/ssh"
	"net"
	"os"
	"path/filepath"
	"strings"
)

var serveCmd = &cobra.Command{
	Use:     "serve",
	Aliases: []string{"start"},
	Short:   "Start ollama",
	Args:    cobra.ExactArgs(0),
	RunE:    RunServer,
}

func init() {
	serveCmd.SetUsageTemplate(serveCmd.UsageTemplate() + `
Environment Variables:

    OLLAMA_HOST         The host:port to bind to (default "127.0.0.1:11434")
    OLLAMA_ORIGINS      A comma separated list of allowed origins.
    OLLAMA_MODELS       The path to the models directory (default is "~/.ollama/models")
    OLLAMA_KEEP_ALIVE   The duration that models stay loaded in memory (default is "5m")
`)
	rootCmd.AddCommand(serveCmd)
}

func RunServer(cmd *cobra.Command, _ []string) error {
	host, port, err := net.SplitHostPort(strings.Trim(os.Getenv("OLLAMA_HOST"), "\"'"))
	if err != nil {
		host, port = "127.0.0.1", "11434"
		if ip := net.ParseIP(strings.Trim(os.Getenv("OLLAMA_HOST"), "[]")); ip != nil {
			host = ip.String()
		}
	}

	if err := initializeKeypair(); err != nil {
		return err
	}

	ln, err := net.Listen("tcp", net.JoinHostPort(host, port))
	if err != nil {
		return err
	}

	return server.Serve(ln)
}

func initializeKeypair() error {
	home, err := os.UserHomeDir()
	if err != nil {
		return err
	}

	privKeyPath := filepath.Join(home, ".ollama", "id_ed25519")
	pubKeyPath := filepath.Join(home, ".ollama", "id_ed25519.pub")

	_, err = os.Stat(privKeyPath)
	if os.IsNotExist(err) {
		fmt.Printf("Couldn't find '%s'. Generating new private key.\n", privKeyPath)
		cryptoPublicKey, cryptoPrivateKey, err := ed25519.GenerateKey(rand.Reader)
		if err != nil {
			return err
		}

		privateKeyBytes, err := ssh.MarshalPrivateKey(cryptoPrivateKey, "")
		if err != nil {
			return err
		}

		if err := os.MkdirAll(filepath.Dir(privKeyPath), 0o755); err != nil {
			return fmt.Errorf("could not create directory %w", err)
		}

		if err := os.WriteFile(privKeyPath, pem.EncodeToMemory(privateKeyBytes), 0o600); err != nil {
			return err
		}

		sshPublicKey, err := ssh.NewPublicKey(cryptoPublicKey)
		if err != nil {
			return err
		}

		publicKeyBytes := ssh.MarshalAuthorizedKey(sshPublicKey)

		if err := os.WriteFile(pubKeyPath, publicKeyBytes, 0o644); err != nil {
			return err
		}

		fmt.Printf("Your new public key is: \n\n%s\n", publicKeyBytes)
	}
	return nil
}
