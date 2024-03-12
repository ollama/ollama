//go:build !windows

package lifecycle

import (
	"context"
	"fmt"
)

func DoUpgrade(cancel context.CancelFunc, done chan int) error {
	return fmt.Errorf("DoUpgrade not yet implemented")
}
