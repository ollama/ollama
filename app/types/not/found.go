//go:build windows || darwin

package not

import (
	"errors"
)

// Found is an error that indicates that a value was not found. It
// may be used by low-level packages to signal to higher-level
// packages that a value was not found.
//
// It exists to avoid using errors.New("not found") in multiple
// packages to mean the same thing.
//
// Found should not be used directly. Instead it should be wrapped
// or joined using errors.Join or fmt.Errorf, etc.
//
// Errors wrapping Found should provide additional context, e.g.
// fmt.Errorf("%w: %s", not.Found, key)
//
//lint:ignore ST1012 This is a sentinel error intended to be read like not.Found.
var Found = errors.New("not found")

// Available is an error that indicates that a value is not available.
//
//lint:ignore ST1012 This is a sentinel error intended to be read like not.Available.
var Available = errors.New("not available")
