package cerrors

import (
	"errors"
	"fmt"
	"net"
	"net/url"
	"os"
	"strings"
	"syscall"
	"time"
)

// ErrorCategoryFromError determines the category of an error based on its type and content
func ErrorCategoryFromError(err error) ErrorCategory {
	if err == nil {
		return UnknownError
	}
	
	errStr := err.Error()
	
	// Check for timeout errors
	if strings.Contains(errStr, "timeout") || strings.Contains(errStr, "deadline exceeded") {
		return TimeoutError
	}
	
	// Check for connection refused errors
	if strings.Contains(errStr, "connection refused") {
		return ConnectionRefused
	}
	
	// Check for network unreachable
	if strings.Contains(errStr, "network is unreachable") || strings.Contains(errStr, "no route to host") {
		return NetworkTemporary
	}
	
	// Check for DNS/resolution errors
	if strings.Contains(errStr, "no such host") || strings.Contains(errStr, "cannot resolve") ||
		strings.Contains(errStr, "lookup") {
		return NameResolution
	}
	
	// Check for network errors related to sending
	if strings.Contains(errStr, "write:") || strings.Contains(errStr, "send:") {
		return NetworkSendError
	}
	
	// Check for network errors related to receiving
	if strings.Contains(errStr, "read:") || strings.Contains(errStr, "recv:") {
		return NetworkReceiveError
	}
	
	// Use type assertions for more specific error types
	switch e := err.(type) {
	case *net.OpError:
		return categorizeNetOpError(e)
	case *os.SyscallError:
		return categorizeSyscallError(e)
	case net.Error:
		if e.Timeout() {
			return TimeoutError
		}
		return NetworkTemporary
	}
	
	// Default to unknown error type
	return UnknownError
}

// ErrorSeverityFromError determines if an error is temporary or persistent
func ErrorSeverityFromError(err error) ErrorSeverity {
	if err == nil {
		return UnknownSeverity
	}
	
	// Check if it's one of our custom errors
	if clusterErr, ok := err.(ClusterError); ok {
		return clusterErr.Severity()
	}
	
	// Check if the error is explicitly marked as temporary
	if tempErr, ok := err.(interface{ Temporary() bool }); ok && tempErr.Temporary() {
		return TemporaryError
	}
	
	// Check error string for common temporary error indicators
	errStr := err.Error()
	if strings.Contains(errStr, "temporarily") || strings.Contains(errStr, "timeout") ||
		strings.Contains(errStr, "deadline exceeded") || strings.Contains(errStr, "connection refused") {
		return TemporaryError
	}
	
	// Special handling for DNS errors
	if IsNameResolutionError(err) {
		if temporaryDNSError(err) {
			return TemporaryError
		}
		return PersistentError
	}
	
	// Default to persistent error when uncertain
	return PersistentError
}

// IsTimeoutError checks if an error is related to timeouts
func IsTimeoutError(err error) bool {
	if err == nil {
		return false
	}
	
	// Check if it's a timeout error using the TimeoutError interface
	if netErr, ok := err.(net.Error); ok && netErr.Timeout() {
		return true
	}
	
	// Check URL errors for timeout
	if urlErr, ok := err.(*url.Error); ok && urlErr.Timeout() {
		return true
	}
	
	// Check string content for timeout indicators
	errStr := err.Error()
	return strings.Contains(errStr, "timeout") ||
		strings.Contains(errStr, "deadline exceeded") ||
		strings.Contains(errStr, "timed out")
}

// IsConnectionRefusedError checks if an error is about connection refusal
func IsConnectionRefusedError(err error) bool {
	if err == nil {
		return false
	}
	
	var opErr *net.OpError
	if errors.As(err, &opErr) {
		if opErr.Err != nil {
			if syscallErr, ok := opErr.Err.(*os.SyscallError); ok {
				if syscallErr.Err == syscall.ECONNREFUSED {
					return true
				}
			}
			return opErr.Err.Error() == "connection refused"
		}
	}
	
	// Check string content
	return strings.Contains(err.Error(), "connection refused") ||
		strings.Contains(err.Error(), "ECONNREFUSED")
}

// IsNameResolutionError checks if an error is about DNS/hostname resolution
func IsNameResolutionError(err error) bool {
	if err == nil {
		return false
	}
	
	// Check if it's a DNS error
	if _, ok := err.(*net.DNSError); ok {
		return true
	}
	
	// Check string content
	errStr := err.Error()
	return strings.Contains(errStr, "no such host") ||
		strings.Contains(errStr, "cannot resolve") ||
		strings.Contains(errStr, "lookup") ||
		strings.Contains(errStr, "name resolution failed")
}

// IsNetworkUnreachable checks if an error indicates the network is unreachable
func IsNetworkUnreachable(err error) bool {
	if err == nil {
		return false
	}
	
	var netErr net.Error
	if errors.As(err, &netErr) && netErr.Timeout() {
		return true
	}
	
	var opErr *net.OpError
	if errors.As(err, &opErr) {
		if opErr.Err != nil {
			if syscallErr, ok := opErr.Err.(*os.SyscallError); ok {
				if syscallErr.Err == syscall.ENETUNREACH {
					return true
				}
			}
			return opErr.Err.Error() == "network is unreachable"
		}
	}
	
	// Check string content
	errStr := err.Error()
	return strings.Contains(errStr, "network is unreachable") ||
		strings.Contains(errStr, "no route to host") ||
		strings.Contains(errStr, "network unreachable")
}

// IsConnectionResetError checks if a connection was reset by peer
func IsConnectionResetError(err error) bool {
	if err == nil {
		return false
	}
	
	var opErr *net.OpError
	if errors.As(err, &opErr) {
		if opErr.Err != nil {
			if syscallErr, ok := opErr.Err.(*os.SyscallError); ok {
				if syscallErr.Err == syscall.ECONNRESET {
					return true
				}
			}
			return opErr.Err.Error() == "connection reset by peer"
		}
	}
	
	// Check string content
	return strings.Contains(err.Error(), "connection reset by peer") ||
		strings.Contains(err.Error(), "connection reset") ||
		strings.Contains(err.Error(), "ECONNRESET")
}

// IsTemporaryNetworkError checks if it's a recoverable network error
func IsTemporaryNetworkError(err error) bool {
	if err == nil {
		return false
	}
	
	// Check if error is explicitly marked as temporary
	if netErr, ok := err.(net.Error); ok && netErr.Temporary() {
		return true
	}
	
	// Check other common temporary conditions
	return IsTimeoutError(err) || IsConnectionRefusedError(err) || 
		IsNameResolutionError(err) || IsNetworkUnreachable(err) ||
		IsConnectionResetError(err)
}

// temporaryDNSError checks if a DNS error is temporary (timeout-related)
func temporaryDNSError(err error) bool {
	var dnsErr *net.DNSError
	return errors.As(err, &dnsErr) && dnsErr.IsTimeout
}

// categorizeNetOpError handles network operation errors
func categorizeNetOpError(err *net.OpError) ErrorCategory {
	if err.Timeout() {
		return TimeoutError
	}
	
	if err.Op == "dial" {
		return ConnectionRefused
	}
	
	if err.Op == "read" {
		return NetworkReceiveError
	}
	
	if err.Op == "write" {
		return NetworkSendError
	}
	
	return NetworkTemporary
}

// categorizeSyscallError categorizes OS syscall errors
func categorizeSyscallError(err *os.SyscallError) ErrorCategory {
	// Check for specific syscall errors
	if strings.Contains(err.Error(), "connection refused") {
		return ConnectionRefused
	}
	
	if strings.Contains(err.Error(), "permission denied") {
		return PermissionDenied
	}
	
	return UnknownError
}

// LogError logs a cluster error with standard formatting
func LogError(err ClusterError) {
	timestamp := time.Now().UTC().Format(time.RFC3339)
	fmt.Printf("[%s] ERROR [%s:%s] %s\n", 
		timestamp, 
		err.Severity(), 
		err.Category(),
		err.Error())
}

// LogErrorf logs a cluster error with custom formatting
func LogErrorf(err ClusterError, format string, args ...interface{}) {
	timestamp := time.Now().UTC().Format(time.RFC3339)
	message := fmt.Sprintf(format, args...)
	fmt.Printf("[%s] ERROR [%s:%s] %s - %s\n",
		timestamp,
		err.Severity(),
		err.Category(),
		err.Error(),
		message)
}

// TraceContextWithError adds context information to an error
func TraceContextWithError(ctx interface{}, err error) error {
	if err == nil {
		return nil
	}
	
	// In a real implementation, this would extract trace information from the context
	// and add it to the error, but for now, we'll just return the original error
	return err
}

// IsIOError checks if the error is related to I/O operations
func IsIOError(err error) bool {
	if err == nil {
		return false
	}
	
	return strings.Contains(err.Error(), "i/o error") ||
		strings.Contains(err.Error(), "i/o timeout") ||
		strings.Contains(err.Error(), "EOF") ||
		strings.Contains(err.Error(), "unexpected EOF")
}

// IsBufferError checks if the error is related to buffer operations
func IsBufferError(err error) bool {
	if err == nil {
		return false
	}
	
	return strings.Contains(err.Error(), "buffer") ||
		strings.Contains(err.Error(), "overflow") ||
		strings.Contains(err.Error(), "underflow")
}

// IsPermissionError checks if the error is due to permission issues
func IsPermissionError(err error) bool {
	if err == nil {
		return false
	}
	
	var opErr *net.OpError
	if errors.As(err, &opErr) {
		if opErr.Err != nil {
			if syscallErr, ok := opErr.Err.(*os.SyscallError); ok {
				if syscallErr.Err == syscall.EACCES || syscallErr.Err == syscall.EPERM {
					return true
				}
			}
		}
	}
	
	return strings.Contains(err.Error(), "permission") ||
		strings.Contains(err.Error(), "EACCES") ||
		strings.Contains(err.Error(), "EPERM") ||
		strings.Contains(err.Error(), "access denied")
}

// IsAddressAlreadyInUseError checks if the error is due to address already in use
func IsAddressAlreadyInUseError(err error) bool {
	if err == nil {
		return false
	}
	
	var opErr *net.OpError
	if errors.As(err, &opErr) {
		if opErr.Err != nil {
			if syscallErr, ok := opErr.Err.(*os.SyscallError); ok {
				if syscallErr.Err == syscall.EADDRINUSE {
					return true
				}
			}
		}
	}
	
	return strings.Contains(err.Error(), "address already in use") ||
		strings.Contains(err.Error(), "EADDRINUSE")
}