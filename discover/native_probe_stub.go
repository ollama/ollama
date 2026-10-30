//go:build !linux && !windows

package discover

import (
	"context"
	"errors"
)

func runPlatformNativeProbe(context.Context, []string) ([]nativeProbeDevice, error) {
	return nil, errors.New("native GPU discovery is not implemented on this platform")
}
