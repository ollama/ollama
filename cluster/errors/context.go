package cerrors

import (
	"context"
	"errors"
	"fmt"
	"time"
)

// ContextError represents an error with additional context information
type ContextError struct {
	// Context is the additional context information about the error
	Context string
	
	// Err is the underlying error
	Err error
}

// NewContextError creates a new ContextError
func NewContextError(ctx string, err error) *ContextError {
	return &ContextError{
		Context: ctx,
		Err:     err,
	}
}

// Error returns the error message with context
func (e *ContextError) Error() string {
	return fmt.Sprintf("%s: %v", e.Context, e.Err)
}

// Unwrap returns the underlying error
func (e *ContextError) Unwrap() error {
	return e.Err
}

// WithContext wraps an error with additional context information
func WithContext(err error, ctx string) error {
	if err == nil {
		return nil
	}
	return NewContextError(ctx, err)
}

// WithNodeIDContext wraps an error with node ID context
func WithNodeIDContext(err error, nodeID string) error {
	if err == nil {
		return nil
	}
	return WithContext(err, fmt.Sprintf("node %s", nodeID))
}

// WithOperationContext wraps an error with operation context
func WithOperationContext(err error, operation string) error {
	if err == nil {
		return nil
	}
	return WithContext(err, fmt.Sprintf("operation %s failed", operation))
}

// WithErrorTimeout creates a context with timeout cancellation and returns a function to wrap any resulting error
func WithErrorTimeout(ctx context.Context, timeout, operation string) (context.Context, context.CancelFunc, func(error) error) {
	timeoutCtx, cancel := context.WithTimeout(ctx, ParseDuration(timeout))
	
	wrapError := func(err error) error {
		if err == nil {
			return nil
		}
		
		if errors.Is(err, context.DeadlineExceeded) {
			return WithOperationContext(err, fmt.Sprintf("%s (timeout: %s)", operation, timeout))
		}
		
		return WithOperationContext(err, operation)
	}
	
	return timeoutCtx, cancel, wrapError
}

// ExtractNodeError attempts to extract a NodeCommunicationError from an error chain
func ExtractNodeError(err error) (*NodeCommunicationError, bool) {
	var nodeErr *NodeCommunicationError
	if errors.As(err, &nodeErr) {
		return nodeErr, true
	}
	return nil, false
}

// MergeErrors combines multiple errors into a single error message
// Returns nil if all errors are nil
func MergeErrors(errs ...error) error {
	var nonNilErrs []error
	for _, err := range errs {
		if err != nil {
			nonNilErrs = append(nonNilErrs, err)
		}
	}
	
	if len(nonNilErrs) == 0 {
		return nil
	}
	
	if len(nonNilErrs) == 1 {
		return nonNilErrs[0]
	}
	
	messages := make([]string, 0, len(nonNilErrs))
	for _, err := range nonNilErrs {
		messages = append(messages, err.Error())
	}
	
	return errors.New(fmt.Sprintf("multiple errors: %v", messages))
}

// ParseDuration parses a duration string and returns a time.Duration
// It provides a default value if the parsing fails
func ParseDuration(duration string) time.Duration {
	d, err := time.ParseDuration(duration)
	if err != nil {
		// Default to 30 seconds if parsing fails
		return 30 * time.Second
	}
	return d
}

// CreateTimeoutContext creates a context with a timeout and returns a function to wrap timeout errors
// This is useful for operations that need to timeout after a certain duration
func CreateTimeoutContext(ctx context.Context, operation string, duration time.Duration) (context.Context, context.CancelFunc, func(error) error) {
	timeoutCtx, cancel := context.WithTimeout(ctx, duration)
	
	wrapError := func(err error) error {
		if err == nil {
			return nil
		}
		
		if errors.Is(err, context.DeadlineExceeded) {
			return &NodeCommunicationError{
				BasicClusterError: BasicClusterError{
					NodeIdentifier: "",  // To be filled by caller
					ErrorMessage:   fmt.Sprintf("%s operation timed out after %v", operation, duration),
					ErrorSeverity:  TemporaryError,
					ErrorCategory:  TimeoutError,
					WrappedError:   err,
				},
				Operation:     operation,
				Endpoint:      "",
				Retryable:     true,
				RetryCount:    0,
				NextRetryTime: time.Time{},
				Metadata:      make(map[string]string),
			}
		}
		
		return err
	}
	
	return timeoutCtx, cancel, wrapError
}