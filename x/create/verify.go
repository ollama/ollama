package create

import (
	"errors"
	"fmt"
	"log/slog"

	"github.com/ollama/ollama/x/mlxrunner"
)

var ErrUnsupportedMLXArchitecture = errors.New("unsupported MLX architecture")

// MLXValidationOptions controls failures that can be downgraded while
// developing support for a new model architecture.
type MLXValidationOptions struct {
	Force   bool
	Warning func(string)
}

func validateMLXSource(cfg sourceModelConfig, draft bool, opts MLXValidationOptions) error {
	var arch string
	if len(cfg.Architectures) > 0 {
		arch = cfg.Architectures[0]
	}
	if draft && arch == "" {
		arch = cfg.ModelType
	}

	var architectureErr error
	if arch == "" {
		architectureErr = fmt.Errorf("%w: config.json does not name an architecture", ErrUnsupportedMLXArchitecture)
	} else if draft {
		if !mlxrunner.SupportsDraftArchitecture(arch) {
			architectureErr = fmt.Errorf("%w: draft model %q", ErrUnsupportedMLXArchitecture, arch)
		}
	} else if !mlxrunner.SupportsArchitecture(arch) {
		architectureErr = fmt.Errorf("%w: model %q", ErrUnsupportedMLXArchitecture, arch)
	}

	for _, checkErr := range []error{architectureErr, mlxrunner.CheckRuntime()} {
		if checkErr == nil {
			continue
		}
		if !opts.Force {
			return checkErr
		}
		warnMLX(opts.Warning, checkErr.Error())
	}
	return nil
}

func warnMLX(warn func(string), message string) {
	if warn != nil {
		warn(message)
		return
	}
	slog.Warn(message)
}
