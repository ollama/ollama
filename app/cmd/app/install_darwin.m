#import "app_darwin.h"
#import "../../updater/updater_darwin.h"
#import <Cocoa/Cocoa.h>
#import <Security/Security.h>
#include <sys/wait.h>

extern NSString *SystemWidePath;

// Move the source bundle to the system-wide applications location
// without prompting for additional authorization
static bool moveToApplications(const char *src) {
    NSString *bundlePath = @(src);
    appLogInfo([NSString
        stringWithFormat:
            @"trying move to /Applications without extra authorization"]);
    NSFileManager *fileManager = [NSFileManager defaultManager];

    // Check if the newPath already exists
    if ([fileManager fileExistsAtPath:SystemWidePath]) {
        appLogInfo([NSString stringWithFormat:@"existing install exists"]);
        NSError *removeError = nil;
        [fileManager removeItemAtPath:SystemWidePath error:&removeError];
        if (removeError) {
            appLogInfo([NSString
                stringWithFormat:@"Error removing without authorization %@: %@",
                                 SystemWidePath, removeError]);
            return false;
        }
    }

    // Move can be problematic, so use copy
    NSError *err = nil;
    [fileManager copyItemAtPath:bundlePath toPath:SystemWidePath error:&err];
    if (err) {
        appLogInfo(
            [NSString stringWithFormat:
                          @"unable to copy without authorization %@ to %@: %@",
                          bundlePath, SystemWidePath, err]);
        return false;
    }

    // Best effort attempt to remove old content
    if ([fileManager isDeletableFileAtPath:bundlePath]) {
        err = nil;
        [fileManager trashItemAtURL:[NSURL fileURLWithPath:bundlePath]
                   resultingItemURL:nil
                              error:&err];
        if (err) {
            appLogInfo(
                [NSString stringWithFormat:@"unable to clean up now stale "
                                           @"bundle via file manager %@: %@",
                                           bundlePath, err]);
        }
    } else {
        appLogInfo([NSString stringWithFormat:@"unable to clean up now stale "
                                              @"bundle via file manager %@",
                                              bundlePath]);
    }

    appLogInfo([NSString stringWithFormat:@"app relocated %@ to %@", bundlePath,
                                          SystemWidePath]);
    return true;
}

static AuthorizationRef getSymlinkAuthorization(void) {
    return getAuthorization(@"Ollama is trying to install its command line "
                            @"interface (CLI) tool.",
                            @"symlink");
}

// Prompt the user for authorization and move to the system wide
// location
//
// Note: this flow must not be executed from the old app instance
//       otherwise the malware scanner will trigger on subsequent
//       AuthorizationExecuteWithPrivileges calls as it can not
//       verify the calling app's signature on the filesystem
//       once the files are removed
static bool moveToApplicationsWithAuthorization(const char *src) {
    int pid, status;
    AuthorizationRef authRef = getAppInstallAuthorization();
    if (authRef == NULL) {
        return NO;
    }

    // Remove existing /Applications/Ollama.app (if any)
    //    - We do this via /bin/rm with elevated privileges
    //
    const char *rmTool = "/bin/rm";
    const char *rmArgs[] = {"-rf", [SystemWidePath UTF8String], NULL};

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
    OSStatus err = AuthorizationExecuteWithPrivileges(
        authRef, rmTool, kAuthorizationFlagDefaults, (char *const *)rmArgs,
        NULL);
#pragma clang diagnostic pop

    if (err != errAuthorizationSuccess) {
        appLogInfo([NSString
            stringWithFormat:@"Failed to remove existing %@. err = %d",
                             SystemWidePath, err]);
        AuthorizationFree(authRef, kAuthorizationFlagDestroyRights);
        return NO;
    }

    // wait for the command to finish
    pid = wait(&status);
    if (pid == -1 || !WIFEXITED(status)) {
        appLogInfo([NSString stringWithFormat:@"rm of %@ failed pid=%d exit=%d",
                                              SystemWidePath, pid,
                                              WEXITSTATUS(status)]);
    }
    appLogDebug([NSString
        stringWithFormat:@"finished cleaning up prior %@", SystemWidePath]);

    // Copy bundle to /Applications
    // We can't use mv as we may be denied if we're sandboxed
    const char *cpTool = "/bin/cp";
    const char *cpArgs[] = {"-pR", src, [SystemWidePath UTF8String], NULL};
    appLogDebug([NSString stringWithFormat:@"running authorized cp -pR %s %@",
                                           src, SystemWidePath]);

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
    err = AuthorizationExecuteWithPrivileges(authRef, cpTool,
                                             kAuthorizationFlagDefaults,
                                             (char *const *)cpArgs, NULL);
#pragma clang diagnostic pop

    if (err != errAuthorizationSuccess) {
        appLogInfo(
            [NSString stringWithFormat:@"Failed to copy %s -> %@. err = %d",
                                       src, SystemWidePath, err]);
        AuthorizationFree(authRef, kAuthorizationFlagDestroyRights);
        return NO;
    }

    // Wait for the command to finish
    pid = wait(&status);
    appLogInfo([NSString stringWithFormat:@"cp -pR %s %@ - pid=%d exit=%d", src,
                                          SystemWidePath, pid,
                                          WEXITSTATUS(status)]);

    if (pid == -1 || !WIFEXITED(status) || WEXITSTATUS(status)) {
        AuthorizationFree(authRef, kAuthorizationFlagDestroyRights);
        return NO;
    }

    // Copy worked, now best effort try to clean up the source bundle
    // Try file manager, then authorized rm -rf
    NSFileManager *fileManager = [NSFileManager defaultManager];
    NSString *bundlePath = @(src);
    NSError *removeError = nil;
    err = [fileManager trashItemAtURL:[NSURL fileURLWithPath:bundlePath]
                     resultingItemURL:nil
                                error:&removeError];
    if (removeError) {
        appLogInfo(
            [NSString stringWithFormat:@"unable to clean up now stale "
                                       @"bundle via NSFileManager %@: %@",
                                       bundlePath, removeError]);
        const char *rm2Args[] = {"-rf", src, NULL};
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
        err = AuthorizationExecuteWithPrivileges(authRef, rmTool,
                                                 kAuthorizationFlagDefaults,
                                                 (char *const *)rm2Args, NULL);
#pragma clang diagnostic pop
        if (err != errAuthorizationSuccess) {
            appLogInfo([NSString
                stringWithFormat:@"Failed to remove existing %s. err = %d", src,
                                 err]);
        } else {
            // wait for the command to finish
            pid = wait(&status);
            appLogInfo([NSString stringWithFormat:@"rm of %s pid=%d exit=%d",
                                                  src, pid,
                                                  WEXITSTATUS(status)]);
            if (pid == -1 || !WIFEXITED(status) || WEXITSTATUS(status)) {
                appLogInfo([NSString
                    stringWithFormat:@"rm of %s failed pid=%d exit=%d", src,
                                     pid, WEXITSTATUS(status)]);
            } else {
                appLogDebug([NSString
                    stringWithFormat:@"finished cleaning up %s", src]);
            }
        }
    }
    AuthorizationFree(authRef, kAuthorizationFlagDestroyRights);
    return YES;
}

enum AppMove askToMoveToApplications(void) {
    NSAppleEventDescriptor *evt =
        [[NSAppleEventManager sharedAppleEventManager] currentAppleEvent];
    if (!evt || [evt eventID] != kAEOpenApplication) {
        // This scenario triggers if we were launched from a double click,
        // or the CLI spawns the app via open -a Ollama.app
        appLogDebug([NSString
            stringWithFormat:@"launched from double click or open -a"]);
    }
    NSAppleEventDescriptor *prop =
        [evt paramDescriptorForKeyword:keyAEPropData];
    if (prop && [prop enumCodeValue] == keyAELaunchedAsLogInItem) {
        // For a login session launch, we don't want to prompt for moving if
        // the user opted out
        appLogDebug([NSString stringWithFormat:@"launched from login"]);
        return LoginSession;
    }
    pid_t pid = getpid();
    NSString *bundlePath = [[NSBundle mainBundle] bundlePath];
    appLogInfo(@"asking to move to system wide location");

    NSAlert *alert = [[NSAlert alloc] init];
    [alert setMessageText:@"Move to Applications?"];
    [alert setInformativeText:
               @"Ollama works best when run from the Applications directory."];
    [alert addButtonWithTitle:@"Move to Applications"];
    [alert addButtonWithTitle:@"Don't move"];

    [NSApp activateIgnoringOtherApps:YES];

    if ([alert runModal] != NSAlertFirstButtonReturn) {
        appLogInfo([NSString
            stringWithFormat:@"user rejected moving to /Applications"]);
        return UserDeclinedMove;
    }

    // move to applications
    if (!moveToApplications([bundlePath UTF8String])) {
        if (!moveToApplicationsWithAuthorization([bundlePath UTF8String])) {
            appLogInfo([NSString
                stringWithFormat:@"unable to move with authorization"]);
            return PermissionDenied;
        }
    }

    appLogInfo([NSString
        stringWithFormat:@"Launching %@ from PID=%d", SystemWidePath, pid]);
    NSError *error = nil;
    NSWorkspace *workspace = [NSWorkspace sharedWorkspace];
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
    [workspace launchApplicationAtURL:[NSURL fileURLWithPath:SystemWidePath]
                              options:NSWorkspaceLaunchNewInstance |
                                      NSWorkspaceLaunchDefault
                        configuration:@{}
                                error:&error];
    return MoveCompleted;
}

void launchApp(const char *appPath) {
    pid_t pid = getpid();
    appLogInfo([NSString
        stringWithFormat:@"Launching %@ from PID=%d", @(appPath), pid]);
    NSError *error = nil;
    NSWorkspace *workspace = [NSWorkspace sharedWorkspace];
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
    [workspace launchApplicationAtURL:[NSURL fileURLWithPath:@(appPath)]
                              options:NSWorkspaceLaunchNewInstance |
                                      NSWorkspaceLaunchDefault
                        configuration:@{}
                                error:&error];
}

int installSymlink(const char *cliPath) {
    NSString *linkPath = @"/usr/local/bin/ollama";
    NSString *dirPath = @"/usr/local/bin";
    NSError *error = nil;

    NSFileManager *fileManager = [NSFileManager defaultManager];
    NSString *symlinkPath =
        [fileManager destinationOfSymbolicLinkAtPath:linkPath error:&error];
    NSString *resPath = [NSString stringWithUTF8String:cliPath];

    // if the symlink already exists and points to the right place, don't
    // prompt
    if ([symlinkPath isEqualToString:resPath]) {
        appLogDebug(
            @"symbolic link already exists and points to the right place");
        return 0;
    }

    // Get authorization once for both operations
    AuthorizationRef authRef = getSymlinkAuthorization();
    if (authRef == NULL) {
        return NO;
    }

    // Check if /usr/local/bin directory exists, create it if it doesn't
    BOOL isDirectory;
    if (![fileManager fileExistsAtPath:dirPath isDirectory:&isDirectory] || !isDirectory) {
        appLogInfo(@"/usr/local/bin directory does not exist, creating it");
        
        const char *mkdirTool = "/bin/mkdir";
        const char *mkdirArgs[] = {"-p", [dirPath UTF8String], NULL};
        
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
        OSStatus err = AuthorizationExecuteWithPrivileges(
            authRef, mkdirTool, kAuthorizationFlagDefaults, (char *const *)mkdirArgs,
            NULL);
        if (err != errAuthorizationSuccess) {
            appLogInfo(@"Failed to create /usr/local/bin directory");
            AuthorizationFree(authRef, kAuthorizationFlagDestroyRights);
            return -1;
        }
        
        // Wait for mkdir to complete
        int status;
        wait(&status);
    }

    // Create the symlink using the same authorization
    const char *toolPath = "/bin/ln";
    const char *args[] = {"-s", "-F", [resPath UTF8String],
                          "/usr/local/bin/ollama", NULL};
    FILE *pipe = NULL;

#pragma clang diagnostic ignored "-Wdeprecated-declarations"
    OSStatus err = AuthorizationExecuteWithPrivileges(
        authRef, toolPath, kAuthorizationFlagDefaults, (char *const *)args,
        &pipe);
    if (err != errAuthorizationSuccess) {
        appLogInfo(@"Failed to create symlink");
        AuthorizationFree(authRef, kAuthorizationFlagDestroyRights);
        return -1;
    }

    AuthorizationFree(authRef, kAuthorizationFlagDestroyRights);
    return 0;
}

