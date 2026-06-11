#import "updater_darwin.h"
#import <AppKit/AppKit.h>
#import <Cocoa/Cocoa.h>
#import <CoreServices/CoreServices.h>
#import <Security/Security.h>
#import <ServiceManagement/ServiceManagement.h>
#import <string.h>

void appLogInfo(NSString *msg) {
    NSLog(@"%@", msg);
    goLogInfo([msg UTF8String]);
}

void appLogDebug(NSString *msg) {
    NSLog(@"%@", msg);
    goLogDebug([msg UTF8String]);
}

NSString *SystemWidePath = @"/Applications/Ollama.app";

// TODO - how to detect if the user has admin access?
// Possible APIs to explore:
// - SFAuthorization
// - CSIdentityQueryCreateForCurrentUser + CSIdentityQueryCreateForName(NULL,
// CFSTR("admin"), kCSIdentityQueryStringEquals, kCSIdentityClassGroup,
// CSGetDefaultIdentityAuthority());

// Caller must call AuthorizationFree(authRef, kAuthorizationFlagDestroyRights)
// once finished
// TODO consider a struct response type to capture user cancel scenario from
// other error/failure scenarios
AuthorizationRef getAuthorization(NSString *authorizationPrompt,
                                  NSString *right) {
    appLogInfo([NSString stringWithFormat:@"XXX in getAuthorization"]);
    AuthorizationRef authRef = NULL;
    OSStatus err = AuthorizationCreate(NULL, kAuthorizationEmptyEnvironment,
                                          kAuthorizationFlagDefaults, &authRef);
    if (err != errAuthorizationSuccess) {
        appLogInfo([NSString
            stringWithFormat:
                @"Failed to create authorization reference. Status = %d",
                err]);
        return NULL;
    }
    NSString *bundleIdentifier = [[NSBundle mainBundle] bundleIdentifier];
    NSString *rightNameString =
        [NSString stringWithFormat:@"%@.%@", bundleIdentifier, right];
    const char *rightName = [rightNameString UTF8String];
    appLogInfo([NSString stringWithFormat:@"XXX requesting right %@", rightNameString]);

    OSStatus getRightResult = AuthorizationRightGet(rightName, NULL);
    if (getRightResult == errAuthorizationDenied) {
        // Create or update the right if it doesn't exist
        if (AuthorizationRightSet(
                authRef, rightName,
                (__bridge CFTypeRef _Nonnull)(
                    @(kAuthorizationRuleAuthenticateAsAdmin)),
                (__bridge CFStringRef _Nullable)(authorizationPrompt), NULL,
                NULL) != errAuthorizationSuccess) {
            appLogInfo([NSString
                stringWithFormat:
                    @"Failed to set right for moving to /Applications"]);
            AuthorizationFree(authRef, kAuthorizationFlagDestroyRights);
            return NULL;
        }
    }
    AuthorizationItem rightItem = {
        .name = rightName, .valueLength = 0, .value = NULL, .flags = 0};
    AuthorizationRights rights = {.count = 1, .items = &rightItem};
    AuthorizationFlags flags =
        (AuthorizationFlags)(kAuthorizationFlagExtendRights |
                             kAuthorizationFlagInteractionAllowed);

    err = AuthorizationCopyRights(authRef, &rights, NULL, flags, NULL);
    if (err != errAuthorizationSuccess) {
        if (err == errAuthorizationCanceled) {
            appLogInfo([NSString
                stringWithFormat:@"User cancelled authorization. Status = %d",
                                 err]);
            // TODO bubble up user cancel/reject so we can keep track

        } else {
            appLogInfo([NSString
                stringWithFormat:@"failed to grant authorization. Status = %d",
                                 err]);
        }
        AuthorizationFree(authRef, kAuthorizationFlagDestroyRights);
        return NULL;
    }
    return authRef;
}

AuthorizationRef getAppInstallAuthorization() {
    return getAuthorization(
        @"Ollama needs additional permission to move or update itself as a "
         "system-wide Application",
        @"systemApplication");
}

bool chownWithAuthorization(const char *user) {
    AuthorizationRef authRef = getAppInstallAuthorization();
    if (authRef == NULL) {
        return NO;
    }
    const char *chownTool = "/usr/sbin/chown";
    const char *chownArgs[] = {"-R", user, [SystemWidePath UTF8String], NULL};

    FILE *pipe = NULL;
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
    OSStatus err = AuthorizationExecuteWithPrivileges(
        authRef, chownTool, kAuthorizationFlagDefaults,
        (char *const *)chownArgs, &pipe);
#pragma clang diagnostic pop

    if (err != errAuthorizationSuccess) {
        appLogInfo([NSString
            stringWithFormat:@"Failed to update ownership of %@  Status = %d",
                             SystemWidePath, err]);
        AuthorizationFree(authRef, kAuthorizationFlagDestroyRights);
        return NO;
    }

    // wait for the command to finish
    while (pipe && !feof(pipe)) {
        fgetc(pipe);
    }
    if (pipe) {
        fclose(pipe);
    }
    appLogDebug([NSString stringWithFormat:@"XXX finished chown"]);
    AuthorizationFree(authRef, kAuthorizationFlagDestroyRights);
    return true;
}

bool replaceBundleWithAuthorization(const char *stagedApp, const char *backupApp,
                                    const char *destApp, const char *owner) {
    AuthorizationRef authRef = getAppInstallAuthorization();
    if (authRef == NULL) {
        return NO;
    }

    static const char *successMarker = "__OLLAMA_AUTHORIZED_UPDATE_SUCCESS__";
    static const char *script =
        "set -u\n"
        "exec 2>&1\n"
        "staged=$1\n"
        "backup=$2\n"
        "dest=$3\n"
        "owner=$4\n"
        "success_marker=$5\n"
        "if [ -e \"$backup\" ]; then\n"
        "  echo \"backup already exists: $backup\" >&2\n"
        "  exit 73\n"
        "fi\n"
        "mkdir -p \"$(dirname \"$backup\")\" || exit 1\n"
        "mv \"$dest\" \"$backup\" || exit 1\n"
        "if cp -pR \"$staged\" \"$dest\"; then\n"
        "  chown -R \"$owner\" \"$backup\" >/dev/null 2>&1 || true\n"
        "  printf '%s\\n' \"$success_marker\"\n"
        "  exit 0\n"
        "fi\n"
        "status=$?\n"
        "rm -rf \"$dest\" >/dev/null 2>&1 || true\n"
        "mv \"$backup\" \"$dest\" >/dev/null 2>&1 || true\n"
        "exit \"$status\"\n";

    const char *shellTool = "/bin/sh";
    const char *shellArgs[] = {"-c", script, "ollama-authorized-update",
                               stagedApp, backupApp, destApp, owner,
                               successMarker, NULL};
    appLogInfo([NSString
        stringWithFormat:@"requesting authorization to replace %@ from %@",
                         @(destApp), @(stagedApp)]);

    FILE *pipe = NULL;
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
    OSStatus err = AuthorizationExecuteWithPrivileges(
        authRef, shellTool, kAuthorizationFlagDefaults,
        (char *const *)shellArgs, &pipe);
#pragma clang diagnostic pop

    if (err != errAuthorizationSuccess) {
        appLogInfo([NSString
            stringWithFormat:@"Failed to start authorized update. Status = %d",
                             err]);
        AuthorizationFree(authRef, kAuthorizationFlagDestroyRights);
        return NO;
    }

    BOOL succeeded = NO;
    if (pipe != NULL) {
        char line[4096];
        while (fgets(line, sizeof(line), pipe) != NULL) {
            line[strcspn(line, "\r\n")] = '\0';
            appLogDebug([NSString
                stringWithFormat:@"authorized update output: %s", line]);
            if (strcmp(line, successMarker) == 0) {
                succeeded = YES;
            }
        }
        if (ferror(pipe)) {
            appLogInfo(@"failed to read authorized update output");
            succeeded = NO;
        }
        fclose(pipe);
    } else {
        appLogInfo(@"authorized update did not provide a completion pipe");
    }
    AuthorizationFree(authRef, kAuthorizationFlagDestroyRights);

    if (!succeeded) {
        appLogInfo(@"authorized update failed or did not report success");
        return NO;
    }

    appLogInfo([NSString
        stringWithFormat:@"authorized update replaced %@ successfully",
                         @(destApp)]);
    return YES;
}

// nil if bundle is good, error string otherwise
const char *verifyExtractedBundle(char *path) {
    NSString *p = [NSString stringWithFormat:@"%s", path];

    appLogDebug([NSString stringWithFormat:@"verifyExtractedBundle: %@", p]);
    SecStaticCodeRef staticCode = NULL;

    OSStatus result = SecStaticCodeCreateWithPath(
        CFURLCreateFromFileSystemRepresentation(
            (__bridge CFAllocatorRef)(kCFAllocatorSystemDefault),
            (const UInt8 *)path, strlen(path), kCFStringEncodingMacRoman),
        kSecCSDefaultFlags, &staticCode);

    if (result != noErr) {
        NSString *failureReason =
            CFBridgingRelease(SecCopyErrorMessageString(result, NULL));
        appLogDebug([NSString
            stringWithFormat:@"Failed to get static code for bundle: %@",
                             failureReason]);
        if (staticCode != NULL)
            CFRelease(staticCode);
        return [[NSString
            stringWithFormat:@"Failed to get static code for bundle:  %@",
                             failureReason] UTF8String];
    }

    CFErrorRef validityError = NULL;
    result = SecStaticCodeCheckValidityWithErrors(
        staticCode, kSecCSCheckAllArchitectures, NULL, &validityError);

    if (result != noErr) {
        NSString *failureReason =
            CFBridgingRelease(SecCopyErrorMessageString(result, NULL));
        appLogDebug([NSString
            stringWithFormat:@"Signatures did not verify on bundle: %@",
                             failureReason]);

        // TODO - consider extracting additional details from validityError

        if (validityError != NULL)
            CFRelease(validityError);
        return [[NSString
            stringWithFormat:@"Signatures did not verify on bundle: %@",
                             failureReason] UTF8String];
    }
    appLogDebug([NSString stringWithFormat:@"bundle passed verification"]);
    return NULL;
}
