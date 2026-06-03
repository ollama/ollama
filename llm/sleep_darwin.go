//go:build darwin

package llm

/*
#cgo LDFLAGS: -framework IOKit -framework CoreFoundation
#include <IOKit/pwr_mgt/IOPMLib.h>
#include <CoreFoundation/CoreFoundation.h>

static IOPMAssertionID assertionID = 0;

void darwin_inhibit_sleep() {
    if (assertionID != 0) return;
    CFStringRef reason = CFStringCreateWithCString(NULL, "Ollama is running model inference", kCFStringEncodingUTF8);
    IOPMAssertionCreateWithName(kIOPMAssertionTypeNoDisplaySleep,
                                kIOPMAssertionLevelOn,
                                reason,
                                &assertionID);
    CFRelease(reason);
}

void darwin_uninhibit_sleep() {
    if (assertionID == 0) return;
    IOPMAssertionRelease(assertionID);
    assertionID = 0;
}
*/
import "C"

func init() {
	inhibitSleep = func() {
		C.darwin_inhibit_sleep()
	}
	uninhibitSleep = func() {
		C.darwin_uninhibit_sleep()
	}
}
