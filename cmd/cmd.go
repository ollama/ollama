package cmd

import (
	"context"
	"errors"
	"fmt"
	"log"
	"os"
	"runtime"
	"strings"
	"time"

	"github.com/containerd/console"

	"github.com/spf13/cobra"
	"golang.org/x/term"

	"github.com/jmorganca/ollama/api"
)

type generateContextKey string

type runOptions struct {
	Model       string
	ParentModel string
	Prompt      string
	Messages    []api.Message
	WordWrap    bool
	Format      string
	System      string
	Template    string
	Images      []api.ImageData
	Options     map[string]interface{}
	MultiModal  bool
}

type displayResponseState struct {
	lineLength int
	wordBuffer string
}

func displayResponse(content string, wordWrap bool, state *displayResponseState) {
	termWidth, _, _ := term.GetSize(int(os.Stdout.Fd()))
	if wordWrap && termWidth >= 10 {
		for _, ch := range content {
			if state.lineLength+1 > termWidth-5 {
				if len(state.wordBuffer) > termWidth-10 {
					fmt.Printf("%s%c", state.wordBuffer, ch)
					state.wordBuffer = ""
					state.lineLength = 0
					continue
				}

				// backtrack the length of the last word and clear to the end of the line
				fmt.Printf("\x1b[%dD\x1b[K\n", len(state.wordBuffer))
				fmt.Printf("%s%c", state.wordBuffer, ch)
				state.lineLength = len(state.wordBuffer) + 1
			} else {
				fmt.Print(string(ch))
				state.lineLength += 1

				switch ch {
				case ' ':
					state.wordBuffer = ""
				case '\n':
					state.lineLength = 0
				default:
					state.wordBuffer += string(ch)
				}
			}
		}
	} else {
		fmt.Printf("%s%s", state.wordBuffer, content)
		if len(state.wordBuffer) > 0 {
			state.wordBuffer = ""
		}
	}
}

//nolint:unused
func waitForServer(ctx context.Context, client *api.Client) error {
	// wait for the server to start
	timeout := time.After(5 * time.Second)
	tick := time.Tick(500 * time.Millisecond)
	for {
		select {
		case <-timeout:
			return errors.New("timed out waiting for server to start")
		case <-tick:
			if err := client.Heartbeat(ctx); err == nil {
				return nil // server has started
			}
		}
	}

}

func checkServerHeartbeat(cmd *cobra.Command, _ []string) error {
	client, err := api.ClientFromEnvironment()
	if err != nil {
		return err
	}
	if err := client.Heartbeat(cmd.Context()); err != nil {
		if !strings.Contains(err.Error(), " refused") {
			return err
		}
		if err := startApp(cmd.Context(), client); err != nil {
			return fmt.Errorf("could not connect to ollama app, is it running?")
		}
	}
	return nil
}

func appendHostEnvDocs(cmd *cobra.Command) {
	const hostEnvDocs = `
Environment Variables:
      OLLAMA_HOST        The host:port or base URL of the Ollama server (e.g. http://localhost:11434)
`
	cmd.SetUsageTemplate(cmd.UsageTemplate() + hostEnvDocs)
}

func NewCLI() *cobra.Command {
	log.SetFlags(log.LstdFlags | log.Lshortfile)
	cobra.EnableCommandSorting = false

	if runtime.GOOS == "windows" {
		console.ConsoleFromFile(os.Stdin) //nolint:errcheck
	}

	return rootCmd
}
