package launch

import (
	"fmt"
	"os"
	"os/exec"
	"strings"

	"github.com/ollama/ollama/envconfig"
)

// OMP implements Runner for OMP CLI integration.
type OMP struct{}

func (o *OMP) String() string { return "OMP" }

func (o *OMP) args(model string, extra []string) []string {
	var args []string
	if model != "" {
		args = append(args, "--model", ompModelName(model))
	}
	args = append(args, extra...)
	return args
}

func (o *OMP) Run(model string, args []string) error {
	bin, err := exec.LookPath("omp")
	if err != nil {
		return fmt.Errorf("omp is not installed")
	}

	cmd := exec.Command(bin, o.args(model, args)...)
	cmd.Stdin = os.Stdin
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	cmd.Env = append(os.Environ(),
		"OPENAI_BASE_URL="+strings.TrimRight(envconfig.ConnectableHost().String(), "/")+"/v1",
		"OPENAI_API_KEY=ollama",
	)

	return cmd.Run()
}

func ompModelName(model string) string {
	if strings.HasPrefix(model, "ollama/") {
		return model
	}
	return "ollama/" + model
}
