package main

/*
#cgo CFLAGS: -x objective-c
#cgo LDFLAGS: -framework Foundation
#include <Foundation/Foundation.h>
#include <stdlib.h>

static bool isInTrash(const char *path) {
    @autoreleasepool {
        NSURL *url = [[NSURL fileURLWithPath:@(path)] URLByResolvingSymlinksInPath];
        NSURLRelationship relationship;
        return [[NSFileManager defaultManager] getRelationship:&relationship
                                                   ofDirectory:NSTrashDirectory
                                                      inDomain:0
                                                   toItemAtURL:url
                                                         error:nil]
            && relationship == NSURLRelationshipContains;
    }
}
*/
import "C"

import (
	"log/slog"
	"os"
	"unsafe"
)

func isInTrash(path string) bool {
	if path == "" {
		return false
	}
	p := C.CString(path)
	defer C.free(unsafe.Pointer(p))
	return bool(C.isInTrash(p))
}

func exitIfRunningFromTrash() {
	// Use the executable's actual location, not updater.BundlePath, which may
	// resolve to an installed copy. This also covers the background helper and
	// the legacy ShipIt entry point, before either can launch another process.
	executable, err := os.Executable()
	if err == nil && isInTrash(executable) {
		slog.Info("not starting Ollama from Trash", "path", executable)
		os.Exit(0)
	}
}
