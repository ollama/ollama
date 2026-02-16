#import "updater_darwin.h"
#import <AppKit/AppKit.h>
#import <Cocoa/Cocoa.h>
#import <CoreServices/CoreServices.h>
#import <Security/Security.h>
#import <ServiceManagement/ServiceManagement.h>

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