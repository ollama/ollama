package launch

import (
	"fmt"
	"os"
	"os/exec"

	"github.com/ollama/ollama/envconfig"
)

// Poolside implements Runner for Poolside's CLI.
type Poolside struct{}

func (p *Poolside) String() string { return "Pool" }

func (p *Poolside) args(model string, extra []string) []string {
	var args []string
	if model != "" {
		args = append(args, "-m", model)
	}
	args = append(args, extra...)
	return args
}

func (p *Poolside) Run(model string, _ []LaunchModel, args []string) error {
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
