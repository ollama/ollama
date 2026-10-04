#import "login_item_darwin.h"
#import "app_darwin.h"
#import "../../updater/updater_darwin.h"
#import <ServiceManagement/ServiceManagement.h>

extern NSString *SystemWidePath;

// Ollama opens at login through the LaunchAgent at
// Ollama.app/Contents/Library/LaunchAgents/com.ollama.ollama.plist.
static NSString *const OLLaunchAgentPlist = @"com.ollama.ollama.plist";

// Set when people turn off opening at login in Settings, so later launches
// don't register the login item again.
static NSString *const OLLoginItemDisabledKey = @"LoginItemDisabled";

static SMAppService *OLLaunchAgentService(void) {
    return [SMAppService agentServiceWithPlistName:OLLaunchAgentPlist];
}

OLLoginItemStatus OLLoginItemGetStatus(void) {
    switch (OLLaunchAgentService().status) {
    case SMAppServiceStatusEnabled:
        return OLLoginItemStatusEnabled;
    case SMAppServiceStatusRequiresApproval:
        return OLLoginItemStatusRequiresApproval;
    case SMAppServiceStatusNotRegistered:
        return OLLoginItemStatusDisabled;
    case SMAppServiceStatusNotFound:
    default:
        return OLLoginItemStatusUnavailable;
    }
}

BOOL OLLoginItemSetEnabled(BOOL enabled, NSError **error) {
    SMAppService *service = OLLaunchAgentService();
    [[NSUserDefaults standardUserDefaults] setBool:!enabled forKey:OLLoginItemDisabledKey];
    if (enabled) {
        if (service.status == SMAppServiceStatusEnabled) {
            return YES;
        }
        appLogInfo(@"registering login item");
        return [service registerAndReturnError:error];
    }
    if (service.status == SMAppServiceStatusNotRegistered ||
        service.status == SMAppServiceStatusNotFound) {
        return YES;
    }
    appLogInfo(@"unregistering login item");
    return [service unregisterAndReturnError:error];
}

void OLLoginItemOpenSystemSettings(void) {
    [SMAppService openSystemSettingsLoginItems];
}

void registerSelfAsLoginItem(bool firstTimeRun) {
    (void)firstTimeRun;
    dispatch_async(dispatch_get_main_queue(), ^{
        if ([[NSUserDefaults standardUserDefaults] boolForKey:OLLoginItemDisabledKey]) {
            appLogInfo(@"login item turned off in Settings, not registering");
            return;
        }
        SMAppService *service = OLLaunchAgentService();
        switch (service.status) {
        case SMAppServiceStatusNotRegistered:
            appLogInfo(@"service not registered, registering now");
            break;
        case SMAppServiceStatusEnabled:
            appLogInfo(@"service is already enabled, no need to register again");
            return;
        case SMAppServiceStatusRequiresApproval:
            // People turned off opening at login in System Settings; leave it.
            appLogInfo(@"service is currently disabled and will not start at login");
            return;
        case SMAppServiceStatusNotFound:
            appLogInfo(@"service not found, registering now");
            break;
        default:
            appLogInfo([NSString stringWithFormat:@"unexpected status: %ld", (long)service.status]);
            break;
        }
        NSError *error = nil;
        if (![service registerAndReturnError:&error]) {
            appLogInfo([NSString stringWithFormat:@"Failed to register %@ as a login item: %@",
                                                  NSBundle.mainBundle.bundleURL, error]);
        }
    });
}

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
// Remove Ollama from the deprecated Login Items list as it now uses a
// LaunchAgent.
void unregisterSelfFromLoginItem(void) {
    dispatch_async(dispatch_get_main_queue(), ^{
        NSString *bundlePrefix = [SystemWidePath stringByDeletingPathExtension];
        LSSharedFileListRef loginItems =
            LSSharedFileListCreate(NULL, kLSSharedFileListSessionLoginItems, NULL);
        if (!loginItems) {
            return;
        }

        UInt32 seed;
        CFArrayRef currentItems = LSSharedFileListCopySnapshot(loginItems, &seed);
        for (id item in (__bridge NSArray *)currentItems) {
            LSSharedFileListItemRef itemRef = (__bridge LSSharedFileListItemRef)item;
            CFURLRef itemURL = NULL;
            if (LSSharedFileListItemResolve(itemRef, 0, &itemURL, NULL) == noErr) {
                NSString *loginPath = CFBridgingRelease(
                    CFURLCopyFileSystemPath(itemURL, kCFURLPOSIXPathStyle));
                // Match the prefix to catch "keep both" copies, such as
                // "/Applications/Ollama 2.app".
                if ([loginPath hasPrefix:bundlePrefix]) {
                    appLogInfo([NSString stringWithFormat:@"removing login item %@", loginPath]);
                    LSSharedFileListItemRemove(loginItems, itemRef);
                }
                if (itemURL) {
                    CFRelease(itemURL);
                }
            } else if (!itemURL) {
                // A login item for a removed app can't be resolved to a path,
                // so match it by name.
                NSString *name = CFBridgingRelease(LSSharedFileListItemCopyDisplayName(itemRef));
                if ([name hasPrefix:@"Ollama"]) {
                    LSSharedFileListItemRemove(loginItems, itemRef);
                    appLogInfo([NSString stringWithFormat:@"removing dangling login item %@", name]);
                }
            }
        }
        if (currentItems) {
            CFRelease(currentItems);
        }
        CFRelease(loginItems);
    });
}
#pragma clang diagnostic pop
