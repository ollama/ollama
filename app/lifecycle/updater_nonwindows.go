//go:build !windows

package lifecycle

import (
	"context"
	"errors"
)

func DoUpgrade(cancel context.CancelFunc, done chan int) error {
	return errors.New("not implemented")
}
