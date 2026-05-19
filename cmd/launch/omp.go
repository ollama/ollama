package launch

import (
	"fmt"
	"os"
	"os/exec"

	"github.com/ollama/ollama/envconfig"
)

// OMP implements Runner for OMP CLI integration.
type OMP struct{}

func (o *OMP) String() string { return "OMP" }

func (o *OMP) Run(model string, args []string) error {
	bin, err := exec.LookPath("omp")
	if err != nil {
		return fmt.Errorf("omp is not installed")
	}

	cmd := exec.Command(bin, args...)
	cmd.Stdin = os.Stdin
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	cmd.Env = append(os.Environ(),
		"OPENAI_BASE_URL="+envconfig.Host().String()+"/v1",
		"OPENAI_API_KEY=ollama",
	)

	return cmd.Run()
}
