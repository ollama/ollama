package cerrors

import (
	"fmt"
	"time"
)

// BasicClusterError provides a base implementation of ClusterError
type BasicClusterError struct {
	NodeIdentifier string        `json:"node_id"`
	ErrorMessage   string        `json:"message"`
	ErrorSeverity  ErrorSeverity `json:"severity"`
	ErrorCategory  ErrorCategory `json:"category"`
	WrappedError   error         `json:"-"`
}

// Error implements the error interface
func (e *BasicClusterError) Error() string {
	base := fmt.Sprintf("[%s:%s] Node %s: %s", e.ErrorSeverity, e.ErrorCategory, e.NodeIdentifier, e.ErrorMessage)
	
	if e.WrappedError != nil {
		base += fmt.Sprintf(" - caused by: %v", e.WrappedError)
	}
	
	return base
}

// NodeID returns the node identifier
func (e *BasicClusterError) NodeID() string {
	return e.NodeIdentifier
}

// Message returns the error message
func (e *BasicClusterError) Message() string {
	return e.ErrorMessage
}

// Severity returns the error severity
func (e *BasicClusterError) Severity() ErrorSeverity {
	return e.ErrorSeverity
}

// Category returns the error category
func (e *BasicClusterError) Category() ErrorCategory {
	return e.ErrorCategory
}

// Cause returns the underlying error
func (e *BasicClusterError) Cause() error {
	return e.WrappedError
}

// IsTemporary returns whether the error is temporary
func (e *BasicClusterError) IsTemporary() bool {
	return e.ErrorSeverity == TemporaryError
}

// NewCommunicationError creates an error for node communication issues
func NewCommunicationError(nodeID, message string, severity ErrorSeverity, cause error) ClusterError {
	return &BasicClusterError{
		NodeIdentifier: nodeID,
		ErrorMessage:   message,
		ErrorSeverity:  severity,
		ErrorCategory:  NetworkTemporary,
		WrappedError:   cause,
	}
}

// This function was moved to communication.go

// NewTimeoutError creates a specific timeout error
func NewTimeoutError(nodeID string, operation string, timeout time.Duration, cause error) ClusterError {
	message := fmt.Sprintf("%s operation timed out after %s", operation, timeout)
	return &BasicClusterError{
		NodeIdentifier: nodeID,
		ErrorMessage:   message,
		ErrorSeverity:  TemporaryError,
		ErrorCategory:  TimeoutError,
		WrappedError:   cause,
	}
}

// NewConnectionRefusedError creates a connection refused error
func NewConnectionRefusedError(nodeID string, endpoint string, cause error) ClusterError {
	message := fmt.Sprintf("Connection refused to %s", endpoint)
	return &BasicClusterError{
		NodeIdentifier: nodeID,
		ErrorMessage:   message,
		ErrorSeverity:  TemporaryError,
		ErrorCategory:  ConnectionRefused,
		WrappedError:   cause,
	}
}

// NewNameResolutionError creates a DNS resolution error
func NewNameResolutionError(nodeID string, hostname string, cause error) ClusterError {
	message := fmt.Sprintf("Failed to resolve hostname: %s", hostname)
	return &BasicClusterError{
		NodeIdentifier: nodeID,
		ErrorMessage:   message,
		ErrorSeverity:  TemporaryError,
		ErrorCategory:  NameResolution,
		WrappedError:   cause,
	}
}

// NewNetworkSendError creates a network send error
func NewNetworkSendError(nodeID string, endpoint string, cause error) ClusterError {
	message := fmt.Sprintf("Failed to send data to %s", endpoint)
	return &BasicClusterError{
		NodeIdentifier: nodeID,
		ErrorMessage:   message,
		ErrorSeverity:  TemporaryError, 
		ErrorCategory:  NetworkSendError,
		WrappedError:   cause,
	}
}

// NewNetworkReceiveError creates a network receive error
func NewNetworkReceiveError(nodeID string, endpoint string, cause error) ClusterError {
	message := fmt.Sprintf("Failed to receive data from %s", endpoint)
	return &BasicClusterError{
		NodeIdentifier: nodeID,
		ErrorMessage:   message,
		ErrorSeverity:  TemporaryError,
		ErrorCategory:  NetworkReceiveError,
		WrappedError:   cause,
	}
}