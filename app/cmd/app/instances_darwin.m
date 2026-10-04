#import "app_darwin.h"
#import "../../updater/updater_darwin.h"
#import <Cocoa/Cocoa.h>
#include <errno.h>
#include <libproc.h>
#include <signal.h>
#include <stdlib.h>
#include <unistd.h>

static BOOL isOllamaApplication(NSRunningApplication *app) {
    NSString *bundleId = app.bundleIdentifier;
    if (bundleId == nil || bundleId.length == 0) {
        return NO;
    }
    return [bundleId isEqualToString:[[NSBundle mainBundle] bundleIdentifier]] ||
        [bundleId isEqualToString:@"ai.ollama.ollama"] ||
        [bundleId isEqualToString:@"com.electron.ollama"];
}

bool otherOllamaProcesses(AppProcessIdentity **processes, size_t *count) {
    pid_t myPid = getpid();
    NSArray *apps = [[NSWorkspace sharedWorkspace] runningApplications];
    AppProcessIdentity *result = calloc(apps.count, sizeof(*result));
    if (result == NULL && apps.count > 0) {
        return false;
    }

    size_t resultCount = 0;
    for (NSRunningApplication *app in apps) {
        pid_t pid = app.processIdentifier;
        if (!isOllamaApplication(app) || pid == myPid) {
            continue;
        }
        if (pid <= 0) {
            appLogInfo([NSString stringWithFormat:
                @"skipping app with invalid pid: %@", app.bundleIdentifier]);
            continue;
        }

        // Tie the NSWorkspace match to the kernel process. Re-read the start
        // time after confirming the current app so PID reuse is rejected.
        struct proc_bsdinfo before = {0};
        int size = proc_pidinfo(pid, PROC_PIDTBSDINFO, 0, &before,
                               sizeof(before));
        if (size != sizeof(before)) {
            if (kill(pid, 0) != 0 && errno == ESRCH) {
                continue;
            }
            appLogInfo([NSString stringWithFormat:
                @"unable to inspect ollama instance %d", pid]);
            free(result);
            return false;
        }

        NSRunningApplication *current =
            [NSRunningApplication runningApplicationWithProcessIdentifier:pid];
        if (current == nil || current.isTerminated ||
            !isOllamaApplication(current)) {
            continue;
        }

        struct proc_bsdinfo after = {0};
        size = proc_pidinfo(pid, PROC_PIDTBSDINFO, 0, &after, sizeof(after));
        if (size != sizeof(after)) {
            if (kill(pid, 0) != 0 && errno == ESRCH) {
                continue;
            }
            appLogInfo([NSString stringWithFormat:
                @"unable to confirm ollama instance %d", pid]);
            free(result);
            return false;
        }
        if (before.pbi_start_tvsec != after.pbi_start_tvsec ||
            before.pbi_start_tvusec != after.pbi_start_tvusec) {
            continue;
        }

        result[resultCount++] = (AppProcessIdentity){
            .pid = pid,
            .started_at = (int64_t)after.pbi_start_tvsec * 1000000 +
                after.pbi_start_tvusec,
        };
    }

    *processes = result;
    *count = resultCount;
    return true;
}
