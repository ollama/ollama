#import <Cocoa/Cocoa.h>
#import "AppDelegate.h"
#import "app_darwin.h"

void run() {
    @autoreleasepool {
        [NSApplication sharedApplication];
        AppDelegate *appDelegate = [[AppDelegate alloc] init];
        [NSApp setDelegate:appDelegate];
        [NSApp run];
    }
}

// killOtherInstances kills all other instances of the app currently
// running. This way we can ensure that only the most recently started
// instance of Ollama is running
void killOtherInstances() {
    pid_t pid = getpid();
    NSArray *all = [[NSWorkspace sharedWorkspace] runningApplications];
    NSMutableArray *apps = [NSMutableArray array];

    for (NSRunningApplication *app in all) {
        if ([app.bundleIdentifier isEqualToString:[[NSBundle mainBundle] bundleIdentifier]] ||
            [app.bundleIdentifier isEqualToString:@"ai.ollama.ollama"] ||
            [app.bundleIdentifier isEqualToString:@"com.electron.ollama"]) {
            if (app.processIdentifier != pid) {
                [apps addObject:app];
            }
        }
    }

    for (NSRunningApplication *app in apps) {
        kill(app.processIdentifier, SIGTERM);
    }

    for (NSRunningApplication *app in apps) {
        while (!app.terminated) {
            [[NSRunLoop currentRunLoop] runUntilDate:[NSDate dateWithTimeIntervalSinceNow:0.1]];
        }
    }
}