//go:build darwin

package power

/*
#cgo LDFLAGS: -framework IOKit -framework CoreFoundation
#include <IOKit/pwr_mgt/IOPMLib.h>
#include <CoreFoundation/CoreFoundation.h>

IOPMAssertionID assertionID = kIOPMNullAssertionID;

void PreventSleep(char* reason) {
    if (assertionID != kIOPMNullAssertionID) {
        return;
    }
    CFStringRef reasonParams = CFStringCreateWithCString(kCFAllocatorDefault, reason, kCFStringEncodingUTF8);
    IOPMAssertionCreateWithName(kIOPMAssertionTypeNoIdleSleep, kIOPMAssertionLevelOn, reasonParams, &assertionID);
    CFRelease(reasonParams);
}

void AllowSleep() {
    if (assertionID != kIOPMNullAssertionID) {
        IOPMAssertionRelease(assertionID);
        assertionID = kIOPMNullAssertionID;
    }
}
*/
import "C"
import (
	"log/slog"
	"unsafe"
)

func preventSleep() {
	slog.Debug("asserting system wake lock")
	reason := C.CString("Ollama Inference")
	defer C.free(unsafe.Pointer(reason))
	C.PreventSleep(reason)
}

func allowSleep() {
	slog.Debug("releasing system wake lock")
	C.AllowSleep()
}
