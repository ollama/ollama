//go:build darwin

package server

/*
#cgo LDFLAGS: -framework IOKit -framework CoreFoundation
#include <IOKit/pwr_mgt/IOPMLib.h>
#include <CoreFoundation/CoreFoundation.h>

// createNoIdleSleepAssertion takes a NoIdleSleep assertion (prevents idle
// system sleep, allows display sleep). Returns 1 on success.
static int createNoIdleSleepAssertion(const char *name, IOPMAssertionID *outID) {
	CFStringRef cfName = CFStringCreateWithCString(NULL, name, kCFStringEncodingUTF8);
	IOReturn ret = IOPMAssertionCreateWithName(kIOPMAssertionTypeNoIdleSleep,
		kIOPMAssertionLevelOn, cfName, outID);
	if (cfName) CFRelease(cfName);
	return ret == kIOReturnSuccess;
}

static void releaseAssertion(IOPMAssertionID assertionID) {
	IOPMAssertionRelease(assertionID);
}
*/
import "C"
import "unsafe"

// platformAcquirePowerAssertion implements the darwin side of issue #4072 via
// IOPMAssertion (same mechanism as `caffeinate -i`).
var platformAcquirePowerAssertion = func() (func(), bool) {
	name := C.CString("ollama inference")
	defer C.free(unsafe.Pointer(name))
	var id C.IOPMAssertionID
	if C.createNoIdleSleepAssertion(name, &id) == 0 {
		return nil, false
	}
	return func() { C.releaseAssertion(id) }, true
}
