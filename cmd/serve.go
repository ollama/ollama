package cmd

import (
	"github.com/ollama/ollama/server"
	"github.com/spf13/cobra"
	"net"
	"os"
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
    OLLAMA_DEBUG        Set to 1 to enable additional debug logging
`)
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
