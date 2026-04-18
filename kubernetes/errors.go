package kubernetes

import "fmt"

// ErrorType represents the type of Kubernetes error.
type ErrorType string

const (
	// ErrTypeClusterUnavailable indicates the cluster is not available.
	ErrTypeClusterUnavailable ErrorType = "cluster_unavailable"

	// ErrTypeAuthFailed indicates authentication to the cluster failed.
	ErrTypeAuthFailed ErrorType = "auth_failed"

	// ErrTypeNotFound indicates the requested resource was not found.
	ErrTypeNotFound ErrorType = "not_found"

	// ErrTypeAlreadyExists indicates the resource already exists.
	ErrTypeAlreadyExists ErrorType = "already_exists"

	// ErrTypeInsufficientResources indicates the cluster lacks resources.
	ErrTypeInsufficientResources ErrorType = "insufficient_resources"

	// ErrTypeDeploymentFailed indicates deployment failed.
	ErrTypeDeploymentFailed ErrorType = "deployment_failed"

	// ErrTypeTimeout indicates an operation timed out.
	ErrTypeTimeout ErrorType = "timeout"

	// ErrTypeInvalidConfig indicates invalid configuration.
	ErrTypeInvalidConfig ErrorType = "invalid_config"

	// ErrTypeNetworkError indicates a network error occurred.
	ErrTypeNetworkError ErrorType = "network_error"

	// ErrTypeStorageError indicates a storage-related error.
	ErrTypeStorageError ErrorType = "storage_error"
)

// KubernetesError represents an error that occurred during Kubernetes operations.
type KubernetesError struct {
	Type    ErrorType
	Message string
	Cause   error
	Details map[string]interface{}
}

// Error implements the error interface.
func (e *KubernetesError) Error() string {
	if e.Cause != nil {
		return fmt.Sprintf("kubernetes error (%s): %s: %v", e.Type, e.Message, e.Cause)
	}
	return fmt.Sprintf("kubernetes error (%s): %s", e.Type, e.Message)
}

// NewKubernetesError creates a new KubernetesError.
func NewKubernetesError(errType ErrorType, message string, cause error) *KubernetesError {
	return &KubernetesError{
		Type:    errType,
		Message: message,
		Cause:   cause,
		Details: make(map[string]interface{}),
	}
}

// WithDetails adds details to the error.
func (e *KubernetesError) WithDetails(key string, value interface{}) *KubernetesError {
	e.Details[key] = value
	return e
}

// IsClusterUnavailable returns true if the error is a cluster unavailability error.
func IsClusterUnavailable(err error) bool {
	if ke, ok := err.(*KubernetesError); ok {
		return ke.Type == ErrTypeClusterUnavailable
	}
	return false
}

// IsAuthFailed returns true if the error is an authentication failure.
func IsAuthFailed(err error) bool {
	if ke, ok := err.(*KubernetesError); ok {
		return ke.Type == ErrTypeAuthFailed
	}
	return false
}

// IsNotFound returns true if the error is a not found error.
func IsNotFound(err error) bool {
	if ke, ok := err.(*KubernetesError); ok {
		return ke.Type == ErrTypeNotFound
	}
	return false
}

// IsAlreadyExists returns true if the error is an already exists error.
func IsAlreadyExists(err error) bool {
	if ke, ok := err.(*KubernetesError); ok {
		return ke.Type == ErrTypeAlreadyExists
	}
	return false
}

// IsTimeout returns true if the error is a timeout error.
func IsTimeout(err error) bool {
	if ke, ok := err.(*KubernetesError); ok {
		return ke.Type == ErrTypeTimeout
	}
	return false
}

// IsInsufficientResources returns true if the error is due to insufficient resources.
func IsInsufficientResources(err error) bool {
	if ke, ok := err.(*KubernetesError); ok {
		return ke.Type == ErrTypeInsufficientResources
	}
	return false
}
