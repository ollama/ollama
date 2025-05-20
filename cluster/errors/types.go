package cerrors

// ErrorSeverity indicates how severe an error is
type ErrorSeverity string

const (
	// TemporaryError indicates a transient error that may resolve itself
	TemporaryError ErrorSeverity = "temporary"
	
	// PersistentError indicates a lasting error requiring intervention
	PersistentError ErrorSeverity = "persistent"
	
	// CriticalError indicates a severe error that impacts system stability
	CriticalError ErrorSeverity = "critical"
	
	// UnknownSeverity indicates an error with undetermined severity
	UnknownSeverity ErrorSeverity = "unknown"
)

// ErrorCategory classifies errors by their underlying cause
type ErrorCategory string

const (
	// TimeoutError occurs when an operation takes too long
	TimeoutError ErrorCategory = "timeout"
	
	// ConnectionRefused occurs when a connection attempt is rejected
	ConnectionRefused ErrorCategory = "connection_refused"
	
	// NameResolution occurs when hostname lookup fails
	NameResolution ErrorCategory = "name_resolution"
	
	// NetworkTemporary indicates transient network issues
	NetworkTemporary ErrorCategory = "network_temporary"
	
	// NetworkSendError indicates issues sending data
	NetworkSendError ErrorCategory = "network_send"
	
	// NetworkReceiveError indicates issues receiving data
	NetworkReceiveError ErrorCategory = "network_receive"
	
	// PermissionDenied indicates permission or access issues
	PermissionDenied ErrorCategory = "permission_denied"
	
	// ResourceBusy indicates a resource is already in use
	ResourceBusy ErrorCategory = "resource_busy"
	
	// InternalError indicates an internal system error
	InternalError ErrorCategory = "internal"

	// ProtocolViolation indicates incorrect protocol interaction
	ProtocolViolation ErrorCategory = "protocol_violation"

	// ValidationFailure indicates errors in data/field validation
	ValidationFailure ErrorCategory = "validation_failure"

	// ConfigurationError indicates issues with cluster configuration
	ConfigurationError ErrorCategory = "configuration_error"
	
	// UnknownError is used when the error type cannot be determined
	UnknownError ErrorCategory = "unknown"
)

// ClusterError is the interface all custom errors implement
type ClusterError interface {
	error
	
	// NodeID returns the ID of the node associated with the error
	NodeID() string
	
	// Message returns a human-readable error message
	Message() string
	
	// Severity indicates how severe the error is
	Severity() ErrorSeverity
	
	// Category indicates the error's classification
	Category() ErrorCategory
	
	// Cause returns the underlying error
	Cause() error
	
	// IsTemporary returns true if this is a temporary error
	IsTemporary() bool
}