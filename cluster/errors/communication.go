package cerrors

import (
	"fmt"
	"time"
)

// NodeCommunicationError represents errors that occur during node-to-node communication
type NodeCommunicationError struct {
	BasicClusterError
	
	// Endpoint contains information about the target endpoint
	Endpoint string
	
	// Operation describes what operation was being performed
	Operation string
	
	// Retryable indicates if the error can be retried
	Retryable bool
	
	// RetryCount tracks how many times this operation has been retried
	RetryCount int
	
	// NextRetryTime is when the next retry should be attempted
	NextRetryTime time.Time
	
	// Metadata contains additional contextual information about the error
	Metadata map[string]string
}

// NewNodeCommunicationError creates a new communication error with basic information
func NewNodeCommunicationError(nodeID, endpoint, operation, message string, severity ErrorSeverity, category ErrorCategory, cause error) *NodeCommunicationError {
	// Determine if the error is retryable based on severity
	retryable := severity == TemporaryError
	
	return &NodeCommunicationError{
		BasicClusterError: BasicClusterError{
			NodeIdentifier: nodeID,
			ErrorMessage:   message,
			ErrorSeverity:  severity,
			ErrorCategory:  category,
			WrappedError:   cause,
		},
		Endpoint:      endpoint,
		Operation:     operation,
		Retryable:     retryable,
		RetryCount:    0,
		NextRetryTime: time.Time{},
		Metadata:      make(map[string]string),
	}
}

// Alternate signature match for existing code
func NewNodeCommunicationError4(nodeID, endpoint string, severity ErrorSeverity, cause error) *NodeCommunicationError {
	return NewNodeCommunicationError(nodeID, endpoint, "", fmt.Sprintf("Communication error with node %s", endpoint), severity, NetworkTemporary, cause)
}

// Error enhances the error message with communication details
func (e *NodeCommunicationError) Error() string {
	base := e.BasicClusterError.Error()
	
	// Add communication-specific details
	details := fmt.Sprintf(" [endpoint: %s, operation: %s", e.Endpoint, e.Operation)
	
	// Add retry information if this is a retryable error
	if e.Retryable {
		if e.RetryCount > 0 {
			details += fmt.Sprintf(", retry: %d", e.RetryCount)
		}
		if !e.NextRetryTime.IsZero() {
			details += fmt.Sprintf(", next retry: %s", time.Until(e.NextRetryTime).Round(time.Millisecond))
		}
	}
	
	// Close the details section
	details += "]"
	
	return base + details
}

// WithRetry adds retry information to the error
func (e *NodeCommunicationError) WithRetry(retryCount int, nextRetry time.Time) *NodeCommunicationError {
	e.RetryCount = retryCount
	e.NextRetryTime = nextRetry
	e.Retryable = true
	return e
}

// WithMetadata adds custom metadata to the error
func (e *NodeCommunicationError) WithMetadata(metadata map[string]string) *NodeCommunicationError {
	for k, v := range metadata {
		e.Metadata[k] = v
	}
	return e
}

// IsRetryable returns whether this error can be retried
func (e *NodeCommunicationError) IsRetryable() bool {
	return e.Retryable
}

// GetEndpoint returns the target endpoint information
func (e *NodeCommunicationError) GetEndpoint() string {
	return e.Endpoint
}

// GetOperation returns the operation being performed
func (e *NodeCommunicationError) GetOperation() string {
	return e.Operation
}

// GetRetryCount returns the current retry count
func (e *NodeCommunicationError) GetRetryCount() int {
	return e.RetryCount
}

// GetNextRetryTime returns when the next retry should be attempted
func (e *NodeCommunicationError) GetNextRetryTime() time.Time {
	return e.NextRetryTime
}

// GetMetadata returns the error's metadata
func (e *NodeCommunicationError) GetMetadata() map[string]string {
	return e.Metadata
}

// ShouldRetryNow checks if it's time to retry this operation
func (e *NodeCommunicationError) ShouldRetryNow() bool {
	if !e.Retryable {
		return false
	}
	
	return time.Now().After(e.NextRetryTime)
}

// DiscoveryError represents errors that occur during node discovery and registration
type DiscoveryError struct {
	BasicClusterError
	
	// Endpoint contains discovery-specific information
	Endpoint string
	
	// Protocol indicates the discovery protocol in use
	Protocol string
	
	// Metadata contains additional contextual information
	Metadata map[string]string
}

// NewDiscoveryError creates a new error related to node discovery
func NewDiscoveryError(nodeID, endpoint, message string, severity ErrorSeverity, category ErrorCategory, cause error) *DiscoveryError {
	return &DiscoveryError{
		BasicClusterError: BasicClusterError{
			NodeIdentifier: nodeID,
			ErrorMessage:   message,
			ErrorSeverity:  severity,
			ErrorCategory:  category,
			WrappedError:   cause,
		},
		Endpoint:  endpoint,
		Protocol:  "cluster-discovery",
		Metadata:  make(map[string]string),
	}
}

// Alternate signature for NewDiscoveryError to match existing code expectations
func NewDiscoveryError4(nodeID, endpoint string, severity ErrorSeverity, cause error) *DiscoveryError {
	// Create a new discovery error with default values for the additional parameters
	return &DiscoveryError{
		BasicClusterError: BasicClusterError{
			NodeIdentifier: nodeID,
			ErrorMessage:   fmt.Sprintf("Discovery error in node %s", nodeID),
			ErrorSeverity:  severity,
			ErrorCategory:  NetworkTemporary,
			WrappedError:   cause,
		},
		Endpoint:  endpoint,
		Protocol:  "cluster-discovery",
		Metadata:  make(map[string]string),
	}
}

// Error enhances the error message with discovery details
func (e *DiscoveryError) Error() string {
	base := e.BasicClusterError.Error()
	return fmt.Sprintf("%s [discovery endpoint: %s, protocol: %s]", base, e.Endpoint, e.Protocol)
}

// WithMetadata adds custom metadata to the error
func (e *DiscoveryError) WithMetadata(metadata map[string]string) *DiscoveryError {
	for k, v := range metadata {
		e.Metadata[k] = v
	}
	return e
}

// ConfigErrorType represents errors related to cluster configuration
type ConfigErrorType struct {
	BasicClusterError
	
	// ConfigKey is the configuration key that caused the error
	ConfigKey string
	
	// ConfigValue is the problematic value
	ConfigValue string
	
	// Metadata contains additional contextual information
	Metadata map[string]string
}

// NewConfigurationError creates a new error related to cluster configuration
func NewConfigurationError(nodeID, configKey, configValue, message string, category ErrorCategory, cause error) *ConfigErrorType {
	return &ConfigErrorType{
		BasicClusterError: BasicClusterError{
			NodeIdentifier: nodeID,
			ErrorMessage:   message,
			ErrorSeverity:  PersistentError, // Configuration errors are always persistent
			ErrorCategory:  category,
			WrappedError:   cause,
		},
		ConfigKey:   configKey,
		ConfigValue: configValue,
		Metadata:    make(map[string]string),
	}
}

// Alternate signature for NewConfigurationError to match existing code expectations
func NewConfigurationError4(nodeID, configKey string, severity ErrorSeverity, cause error) *ConfigErrorType {
	// Create configuration error directly without calling the other function
	return &ConfigErrorType{
		BasicClusterError: BasicClusterError{
			NodeIdentifier: nodeID,
			ErrorMessage:   fmt.Sprintf("Configuration error for key %s", configKey),
			ErrorSeverity:  severity,
			ErrorCategory:  ConfigurationError,
			WrappedError:   cause,
		},
		ConfigKey:   configKey,
		ConfigValue: "",
		Metadata:    make(map[string]string),
	}
}

// Error enhances the error message with configuration details
func (e *ConfigErrorType) Error() string {
	base := e.BasicClusterError.Error()
	return fmt.Sprintf("%s [config key: %s, value: %s]", base, e.ConfigKey, e.ConfigValue)
}

// WithMetadata adds custom metadata to the error
func (e *ConfigErrorType) WithMetadata(metadata map[string]string) *ConfigErrorType {
	for k, v := range metadata {
		e.Metadata[k] = v
	}
	return e
}

// NewDetailedCommunicationError creates a NodeCommunicationError with more context
// This is a version of NewNodeCommunicationError that supports additional metadata
func NewDetailedCommunicationError(nodeID, message string, severity ErrorSeverity, category ErrorCategory, cause error) *NodeCommunicationError {
	return &NodeCommunicationError{
		BasicClusterError: BasicClusterError{
			NodeIdentifier: nodeID,
			ErrorMessage:   message,
			ErrorSeverity:  severity,
			ErrorCategory:  category,
			WrappedError:   cause,
		},
		Endpoint:      "",
		Operation:     "",
		Retryable:     severity == TemporaryError,
		RetryCount:    0,
		NextRetryTime: time.Time{},
		Metadata:      make(map[string]string),
	}
}

// SerializationError represents errors during serialization/deserialization of data
type SerializationError struct {
	BasicClusterError
	
	// DataType contains information about what was being serialized
	DataType string
	
	// Operation describes whether this was serialization or deserialization
	Operation string
	
	// Metadata contains additional contextual information
	Metadata map[string]string
}

// NewSerializationError creates a new error related to data serialization
func NewSerializationError(nodeID, dataType, operation, message string, cause error) *SerializationError {
	return &SerializationError{
		BasicClusterError: BasicClusterError{
			NodeIdentifier: nodeID,
			ErrorMessage:   message,
			ErrorSeverity:  TemporaryError,
			ErrorCategory:  InternalError,
			WrappedError:   cause,
		},
		DataType:  dataType,
		Operation: operation,
		Metadata:  make(map[string]string),
	}
}

// Error enhances the error message with serialization details
func (e *SerializationError) Error() string {
	base := e.BasicClusterError.Error()
	return fmt.Sprintf("%s [data type: %s, operation: %s]", base, e.DataType, e.Operation)
}

// WithMetadata adds custom metadata to the error
func (e *SerializationError) WithMetadata(metadata map[string]string) *SerializationError {
	for k, v := range metadata {
		e.Metadata[k] = v
	}
	return e
}

// Function with 4-argument signature to match discovery.go calls
func NewSerializationError4(nodeID, dataType string, severity ErrorSeverity, cause error) *SerializationError {
	return &SerializationError{
		BasicClusterError: BasicClusterError{
			NodeIdentifier: nodeID,
			ErrorMessage:   fmt.Sprintf("Serialization error for %s", dataType),
			ErrorSeverity:  severity,
			ErrorCategory:  InternalError,
			WrappedError:   cause,
		},
		DataType:  dataType,
		Operation: "serialize",
		Metadata:  make(map[string]string),
	}
}

// ProtocolError represents errors related to cluster protocol
type ProtocolError struct {
	BasicClusterError
	
	// Protocol indicates which protocol encountered an error
	Protocol string
	
	// Operation describes what protocol operation failed
	Operation string
	
	// Metadata contains additional contextual information
	Metadata map[string]string
}

// NewProtocolError creates a new error related to cluster protocols
func NewProtocolError(nodeID, protocol, operation, message string, cause error) *ProtocolError {
	return &ProtocolError{
		BasicClusterError: BasicClusterError{
			NodeIdentifier: nodeID,
			ErrorMessage:   message,
			ErrorSeverity:  TemporaryError,
			ErrorCategory:  ProtocolViolation,
			WrappedError:   cause,
		},
		Protocol:  protocol,
		Operation: operation,
		Metadata:  make(map[string]string),
	}
}

// Error enhances the error message with protocol details
func (e *ProtocolError) Error() string {
	base := e.BasicClusterError.Error()
	return fmt.Sprintf("%s [protocol: %s, operation: %s]", base, e.Protocol, e.Operation)
}

// Function with 4-argument signature to match discovery.go calls
func NewProtocolError4(nodeID, protocol string, severity ErrorSeverity, cause error) *ProtocolError {
	return &ProtocolError{
		BasicClusterError: BasicClusterError{
			NodeIdentifier: nodeID,
			ErrorMessage:   fmt.Sprintf("Protocol error for %s", protocol),
			ErrorSeverity:  severity,
			ErrorCategory:  ProtocolViolation,
			WrappedError:   cause,
		},
		Protocol:  protocol,
		Operation: "protocol",
		Metadata:  make(map[string]string),
	}
}

// ValidationError represents errors related to data validation
type ValidationError struct {
	BasicClusterError
	
	// Field indicates which field failed validation
	Field string
	
	// Value contains the invalid value
	Value string
	
	// Metadata contains additional contextual information
	Metadata map[string]string
}

// NewValidationError creates a new error related to data validation
func NewValidationError(nodeID, field, value, message string, cause error) *ValidationError {
	return &ValidationError{
		BasicClusterError: BasicClusterError{
			NodeIdentifier: nodeID,
			ErrorMessage:   message,
			ErrorSeverity:  PersistentError,
			ErrorCategory:  ValidationFailure,
			WrappedError:   cause,
		},
		Field:    field,
		Value:    value,
		Metadata: make(map[string]string),
	}
}

// Error enhances the error message with validation details
func (e *ValidationError) Error() string {
	base := e.BasicClusterError.Error()
	return fmt.Sprintf("%s [field: %s, value: %s]", base, e.Field, e.Value)
}

// WithMetadata adds custom metadata to the error
func (e *ValidationError) WithMetadata(metadata map[string]string) *ValidationError {
	for k, v := range metadata {
		e.Metadata[k] = v
	}
	return e
}

// Function with 4-argument signature to match discovery.go calls
func NewValidationError4(nodeID, field string, severity ErrorSeverity, cause error) *ValidationError {
	return &ValidationError{
		BasicClusterError: BasicClusterError{
			NodeIdentifier: nodeID,
			ErrorMessage:   fmt.Sprintf("Validation error for field %s", field),
			ErrorSeverity:  severity,
			ErrorCategory:  ValidationFailure,
			WrappedError:   cause,
		},
		Field:    field,
		Value:    "",
		Metadata: make(map[string]string),
	}
}

// Function with 6-argument signature to match discovery.go calls
func NewDetailedCommunicationError6(nodeID, message string, severity ErrorSeverity, category ErrorCategory, cause error, metadata map[string]string) *NodeCommunicationError {
	// Create the basic error
	err := &NodeCommunicationError{
		BasicClusterError: BasicClusterError{
			NodeIdentifier: nodeID,
			ErrorMessage:   message,
			ErrorSeverity:  severity,
			ErrorCategory:  category,
			WrappedError:   cause,
		},
		Endpoint:      "",
		Operation:     "",
		Retryable:     severity == TemporaryError,
		RetryCount:    0,
		NextRetryTime: time.Time{},
		Metadata:      make(map[string]string),
	}
	
	// Add metadata if provided
	if metadata != nil {
		for k, v := range metadata {
			err.Metadata[k] = v
		}
	}
	
	return err
}

// NewConnectionError creates a new error specifically for connection issues
func NewConnectionError(nodeID, message string, severity ErrorSeverity, cause error) *NodeCommunicationError {
	return &NodeCommunicationError{
		BasicClusterError: BasicClusterError{
			NodeIdentifier: nodeID,
			ErrorMessage:   message,
			ErrorSeverity:  severity,
			ErrorCategory:  ConnectionRefused, // Default to connection refused
			WrappedError:   cause,
		},
		Endpoint:      "",
		Operation:     "connection",
		Retryable:     severity == TemporaryError,
		RetryCount:    0,
		NextRetryTime: time.Time{},
		Metadata:      make(map[string]string),
	}
}