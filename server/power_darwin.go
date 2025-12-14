//go:build darwin

package server

/*
#cgo LDFLAGS: -framework IOKit -framework CoreFoundation
#include <IOKit/pwr_mgt/IOPMLib.h>
#include <CoreFoundation/CoreFoundation.h>

static IOPMAssertionID assertionID = 0;

int preventSleep() {
    if (assertionID != 0) {
        return 0; // Already preventing sleep
    }

    CFStringRef reasonForActivity = CFSTR("Ollama is processing requests");
    IOReturn result = IOPMAssertionCreateWithName(
        kIOPMAssertionTypeNoIdleSleep,
        kIOPMAssertionLevelOn,
        reasonForActivity,
        &assertionID
    );

    return (result == kIOReturnSuccess) ? 0 : -1;
}

int allowSleep() {
    if (assertionID == 0) {
        return 0; // Not currently preventing sleep
    }

    IOReturn result = IOPMAssertionRelease(assertionID);
    if (result == kIOReturnSuccess) {
        assertionID = 0;
        return 0;
    }
    return -1;
}
*/
import "C"

import "errors"

func platformPreventSleep() error {
	if C.preventSleep() != 0 {
		return errors.New("failed to create IOPMAssertion")
	}
	return nil
}

func platformAllowSleep() error {
	if C.allowSleep() != 0 {
		return errors.New("failed to release IOPMAssertion")
	}
	return nil
}
