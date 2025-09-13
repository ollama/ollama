//go:build windows

package windows

import (
	"fmt"
	"net"
	"os"
	"runtime"
	"syscall"
	"unsafe"

	"golang.org/x/sys/windows"
)

const (
	MB_OK              = 0x00000000
	MB_OKCANCEL        = 0x00000001
	MB_ICONERROR       = 0x00000010
	MB_ICONWARNING     = 0x00000030
	MB_ICONINFORMATION = 0x00000040
)

var (
	user32           = windows.NewLazySystemDLL("User32.dll")
	pMessageBox      = user32.NewProc("MessageBoxW")
)

// MessageBox displays a message box with the specified message, title, and style.
// Returns the user's response (IDOK, IDCANCEL, etc.)
func MessageBox(hwnd uintptr, text, caption string, style uint) int {
	textPtr, _ := windows.UTF16PtrFromString(text)
	captionPtr, _ := windows.UTF16PtrFromString(caption)
	
	ret, _, _ := pMessageBox.Call(
		hwnd,
		uintptr(unsafe.Pointer(textPtr)),
		uintptr(unsafe.Pointer(captionPtr)),
		uintptr(style),
	)
	
	return int(ret)
}

// ShowPortInUseDialog displays a message box informing the user that the port is in use
func ShowPortInUseDialog(port string) {
	MessageBox(0, 
		fmt.Sprintf("Ollama could not start because port %s is already in use by another application.\n\n"+
			"Please either:\n"+
			"1. Close the application using port %s and restart Ollama, or\n"+
			"2. Set the OLLAMA_HOST environment variable to use a different port.\n\n"+
			"Example: OLLAMA_HOST=127.0.0.1:11435", port, port),
		"Ollama - Port in Use",
		MB_OK|MB_ICONERROR)
}

// ShowPortError shows a port error dialog or falls back to console output
func ShowPortError(host string) {
	_, port, err := net.SplitHostPort(host)
	if err != nil || port == "" {
		port = "11434" // Default port
	}
	
	// Try to show a message box
	if runtime.GOOS == "windows" {
		ShowPortInUseDialog(port)
	}
	
	// Also output to stderr as fallback
	fmt.Fprintf(os.Stderr, "ERROR: Port %s is already in use.\n", port)
	fmt.Fprintf(os.Stderr, "Please free up the port or set OLLAMA_HOST environment variable to use a different port.\n")
	fmt.Fprintf(os.Stderr, "Example: OLLAMA_HOST=127.0.0.1:11435\n")
}