package launch

import (
	"fmt"
	"os"
	"os/exec"

	"github.com/ollama/ollama/envconfig"
)

// Gemini implements Runner for Gemini CLI integration.
type Gemini struct{}

func (g *Gemini) String() string { return "Gemini CLI" }

func (g *Gemini) args(model string, extra []string) []string {
	var args []string
	if model != "" {
		args = append(args, "--model", model)
	}
	args = append(args, extra...)
	return args
}

func (g *Gemini) findPath() (string, error) {
	if p, err := exec.LookPath("gemini"); err == nil {
		return p, nil
	}
	return "", fmt.Errorf("gemini binary not found")
}

func (g *Gemini) Run(model string, args []string) error {
	geminiPath, err := g.findPath()
	if err != nil {
		return fmt.Errorf("gemini is not installed, install with: npm install -g @google/gemini-cli")
	}

	cmd := exec.Command(geminiPath, g.args(model, args)...)
	cmd.Stdin = os.Stdin
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr

	cmd.Env = append(os.Environ(),
		"GOOGLE_API_BASE_URL="+envconfig.Host().String()+"/v1",
		"GOOGLE_API_KEY=ollama",
	)

	return cmd.Run()
}
