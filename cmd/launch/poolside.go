package launch

import (
	"fmt"
	"os"
	"os/exec"
	"runtime"

	"github.com/ollama/ollama/envconfig"
)

// Poolside implements Runner for Poolside's CLI.
type Poolside struct{}

var poolsideGOOS = runtime.GOOS

func (p *Poolside) String() string { return "Poolside" }

func poolsideUnsupportedError() error {
	return fmt.Errorf("Warning: Poolside is not currently supported on Windows")
}

func (p *Poolside) args(model string, extra []string) []string {
	var args []string
	if model != "" {
		args = append(args, "-m", model)
	}
	args = append(args, extra...)
	return args
}

func (p *Poolside) Run(model string, args []string) error {
	if poolsideGOOS == "windows" {
		return poolsideUnsupportedError()
	}

	bin, err := exec.LookPath("pool")
	if err != nil {
		return fmt.Errorf("pool is not installed")
	}

	cmd := exec.Command(bin, p.args(model, args)...)
	cmd.Stdin = os.Stdin
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	cmd.Env = append(os.Environ(),
		"POOLSIDE_STANDALONE_BASE_URL="+envconfig.Host().String()+"/v1",
		"POOLSIDE_API_KEY=ollama",
	)
	return cmd.Run()
}
