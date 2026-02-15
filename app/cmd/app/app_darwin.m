#import "app_darwin.h"
#import "menu.h"
#import "../../updater/updater_darwin.h"
#import <AppKit/AppKit.h>
#import <Cocoa/Cocoa.h>
#import <CoreServices/CoreServices.h>
#import <Security/Security.h>
#import <ServiceManagement/ServiceManagement.h>
#import <WebKit/WebKit.h>
#import <objc/runtime.h>

extern NSString *SystemWidePath;

@interface AppDelegate () <NSWindowDelegate, WKNavigationDelegate, WKUIDelegate>
@property(strong, nonatomic) NSStatusItem *statusItem;
@property(assign, nonatomic) BOOL updateAvailable;
@property(assign, nonatomic) BOOL systemShutdownInProgress;
@end

@implementation AppDelegate

bool firstTimeRun,startHidden; // Set in run before initialization

- (void)application:(NSApplication *)application openURLs:(NSArray<NSURL *> *)urls {
    for (NSURL *url in urls) {
        if ([url.scheme isEqualToString:@"ollama"]) {
            NSString *path = url.path;

            if (path && ([path isEqualToString:@"/connect"] || [url.host isEqualToString:@"connect"])) {
                // Special case: handle connect by opening browser instead of app
                handleConnectURL();
            } else {
                // Set app to be active and visible
                [NSApp setActivationPolicy:NSApplicationActivationPolicyRegular];
                [NSApp activateIgnoringOtherApps:YES];
            }
            
            break;
        }
    }
}

- (void)applicationDidFinishLaunching:(NSNotification *)aNotification {
    // Register for system shutdown/restart notification so we can allow termination
    [[[NSWorkspace sharedWorkspace] notificationCenter]
        addObserver:self
           selector:@selector(systemWillPowerOff:)
               name:NSWorkspaceWillPowerOffNotification
             object:nil];

    // if we're in development mode, set the app icon
    NSString *bundlePath = [[NSBundle mainBundle] bundlePath];
    if (![bundlePath hasSuffix:@".app"]) {
        NSString *cwdPath =
            [[NSFileManager defaultManager] currentDirectoryPath];
        NSString *iconPath = [cwdPath
            stringByAppendingPathComponent:
                [NSString
                    stringWithFormat:
                        @"darwin/Ollama.app/Contents/Resources/icon.icns"]];
        NSImage *customIcon = [[NSImage alloc] initWithContentsOfFile:iconPath];
        [NSApp setApplicationIconImage:customIcon];
    }

    // Create status item and menu
    NSMenu *menu = [[NSMenu alloc] init];
    NSMenuItem *openMenuItem =
        [[NSMenuItem alloc] initWithTitle:@"Open Ollama"
                                   action:@selector(openUI)
                            keyEquivalent:@""];
    [openMenuItem setTarget:self];
    [menu addItem:openMenuItem];

    [menu addItemWithTitle:@"Settings..."
                    action:@selector(settingsUI)
             keyEquivalent:@","];
    [menu addItem:[NSMenuItem separatorItem]];

    NSMenuItem *updateAvailable =
        [[NSMenuItem alloc] initWithTitle:@"An update is available"
                                   action:nil
                            keyEquivalent:@""];
    [updateAvailable setEnabled:NO];
    [updateAvailable setHidden:YES];
    [menu addItem:updateAvailable];

    NSMenuItem *restartMenuItem =
        [[NSMenuItem alloc] initWithTitle:@"Restart to update"
                                   action:@selector(startUpdate)
                            keyEquivalent:@""];
    [restartMenuItem setTarget:self];
    [restartMenuItem setHidden:YES];
    [menu addItem:restartMenuItem];

    [menu addItem:[NSMenuItem separatorItem]];

    [menu addItemWithTitle:@"Quit Ollama"
                    action:@selector(quit)
             keyEquivalent:@"q"];

    self.statusItem = [[NSStatusBar systemStatusBar]
        statusItemWithLength:NSVariableStatusItemLength];
    [self.statusItem addObserver:self
                      forKeyPath:@"button.effectiveAppearance"
                         options:NSKeyValueObservingOptionNew |
                                 NSKeyValueObservingOptionInitial
                         context:nil];

    self.statusItem.menu = menu;
    [self showIcon];

    // Application menu
    NSString *appName = @"Ollama";

    NSMenu *mainMenu = [[NSMenu alloc] init];
    NSMenuItem *appMenuItem = [[NSMenuItem alloc] initWithTitle:appName
                                                        action:nil
                                                 keyEquivalent:@""];
    NSMenu *appMenu = [[NSMenu alloc] initWithTitle:appName];
    [appMenuItem setSubmenu:appMenu];
    [mainMenu addItem:appMenuItem];

    [appMenu addItemWithTitle:[NSString stringWithFormat:@"About %@", appName]
                       action:@selector(aboutOllama)
                keyEquivalent:@""];
    [appMenu addItem:[NSMenuItem separatorItem]];
    [appMenu addItemWithTitle:@"Settings..."
                    action:@selector(settingsUI)
                keyEquivalent:@","];
    [appMenu addItem:[NSMenuItem separatorItem]];
    [appMenu addItemWithTitle:[NSString stringWithFormat:@"Hide %@", appName]
                       action:@selector(hide:)
                keyEquivalent:@"h"];

    NSMenuItem *hideOthers = [[NSMenuItem alloc] initWithTitle:@"Hide Others" action:@selector(hideOtherApplications:) keyEquivalent:@"h"];
    hideOthers.keyEquivalentModifierMask = NSEventModifierFlagOption | NSEventModifierFlagCommand;
    [appMenu addItem:hideOthers];
    [appMenu addItemWithTitle:@"Show All"
                       action:@selector(unhideAllApplications:)
                keyEquivalent:@""];
    [appMenu addItem:[NSMenuItem separatorItem]];
    [appMenu addItemWithTitle:[NSString stringWithFormat:@"Quit %@", appName]
                       action:@selector(hide)
                keyEquivalent:@"q"];

    NSMenuItem *fileMenuItem = [[NSMenuItem alloc] init];
    NSMenu *fileMenu        = [[NSMenu alloc] initWithTitle:@"File"];

    NSMenuItem *newChatItem = [[NSMenuItem alloc] initWithTitle:@"New Chat"
                                                    action:@selector(newChat)
                                                keyEquivalent:@"n"];
    [newChatItem setTarget:self];
    [fileMenu addItem:newChatItem];
    [fileMenu addItem:[NSMenuItem separatorItem]];

    NSMenuItem *closeItem = [[NSMenuItem alloc] initWithTitle:@"Close Window" action:@selector(hide:) keyEquivalent:@"w"];
    [fileMenu addItem:closeItem];
    [fileMenuItem setSubmenu:fileMenu];
    [mainMenu addItem:fileMenuItem];

    NSMenuItem *editMenuItem = [[NSMenuItem alloc] init];
    NSMenu *editMenu = [[NSMenu alloc] initWithTitle:@"Edit"];

    [editMenu addItemWithTitle:@"Undo"
                        action:@selector(undo:)
                 keyEquivalent:@"z"];
    [editMenu addItemWithTitle:@"Redo"
                        action:@selector(redo:)
                 keyEquivalent:@"Z"];
    [editMenu addItem:[NSMenuItem separatorItem]];
    [editMenu addItemWithTitle:@"Cut"
                        action:@selector(cut:)
                 keyEquivalent:@"x"];
    [editMenu addItemWithTitle:@"Copy"
                        action:@selector(copy:)
                 keyEquivalent:@"c"];
    [editMenu addItemWithTitle:@"Paste"
                        action:@selector(paste:)
                 keyEquivalent:@"v"];
    [editMenu addItemWithTitle:@"Select All"
                        action:@selector(selectAll:)
                 keyEquivalent:@"a"];

    [editMenuItem setSubmenu:editMenu];
    [mainMenu addItem:editMenuItem];

    NSMenuItem *windowMenuItem = [[NSMenuItem alloc] init];
    NSMenu *windowMenu         = [[NSMenu alloc] initWithTitle:@"Window"];
    [windowMenu addItemWithTitle:@"Minimize"
                          action:@selector(performMiniaturize:)
                   keyEquivalent:@"m"];
    [windowMenu addItemWithTitle:@"Zoom"
                          action:@selector(performZoom:)
                   keyEquivalent:@""];
    [windowMenu addItem:[NSMenuItem separatorItem]];
    [windowMenu addItemWithTitle:@"Bring All to Front"
                          action:@selector(arrangeInFront:)
                   keyEquivalent:@""];
    [windowMenuItem setSubmenu:windowMenu];
    [mainMenu addItem:windowMenuItem];
    [NSApp setWindowsMenu:windowMenu];

    NSMenuItem *helpMenuItem = [[NSMenuItem alloc] init];
    NSMenu *helpMenu         = [[NSMenu alloc] initWithTitle:@"Help"];
    [helpMenu addItemWithTitle:[NSString stringWithFormat:@"%@ Help", appName]
                        action:@selector(openHelp:)
                 keyEquivalent:@"?"];
    [helpMenuItem setSubmenu:helpMenu];
    [mainMenu addItem:helpMenuItem];
    [NSApp setHelpMenu:helpMenu];
    [NSApp setMainMenu:mainMenu];

    BOOL hidden = [NSApp isHidden];
    dispatch_async(dispatch_get_main_queue(), ^{
        if (hidden || startHidden) {
            darwinStartHiddenTasks();
        } else {
            if (!startHidden) {
                StartUI("/");
            }
        }
    });
}

- (void)applicationDidBecomeActive:(NSNotification *)notification {
    NSRunningApplication *currentApp = [NSRunningApplication currentApplication];
    if (currentApp.activationPolicy == NSApplicationActivationPolicyAccessory) {
        for (NSWindow *window in [NSApp windows]) {
            if ([window isVisible]) {
                // Switch to regular activation policy since we have a visible window
                [NSApp setActivationPolicy:NSApplicationActivationPolicyRegular];
                return;
            }
        }
        [NSApp hide:nil];
        return;
    }
}

- (BOOL)applicationShouldHandleReopen:(NSApplication *)sender hasVisibleWindows:(BOOL)hasVisibleWindows {
    [self openUI];
    return YES;
}

- (void)showUpdateAvailable {
    self.updateAvailable = YES;
    [self.statusItem.menu.itemArray[3] setHidden:NO];
    [self.statusItem.menu.itemArray[4] setHidden:NO];
    [self showIcon];
}

- (void)aboutOllama {
    [[NSApplication sharedApplication] orderFrontStandardAboutPanel:nil];
    [NSApp activateIgnoringOtherApps:YES];
}

- (void)openHelp:(id)sender {
    NSURL *url = [NSURL URLWithString:@"https://docs.ollama.com/"];
    [[NSWorkspace sharedWorkspace] openURL:url];
}

- (void)settingsUI {
    [self uiRequest:@"/settings"];
}

- (void)openUI {
    ShowUI();
}

- (void)newChat {
    [self uiRequest:@"/c/new"];
}

- (void)uiRequest:(NSString *)path {
    if (path == nil) {
        appLogInfo(@"app UI request for URL is missing");
    }

    appLogInfo([NSString
        stringWithFormat:@"XXX got app UI request for URL: %@", path]);
    StartUI([path UTF8String]);
}

- (void)startUpdate {
    StartUpdate();
    [NSApp activateIgnoringOtherApps:YES];
}

- (void)systemWillPowerOff:(NSNotification *)notification {
    // Set flag so applicationShouldTerminate: knows to allow termination.
    // The system will call applicationShouldTerminate: after posting this notification.
    self.systemShutdownInProgress = YES;
}

- (NSApplicationTerminateReply)applicationShouldTerminate:(NSApplication *)sender {
    // Allow termination if the system is shutting down or restarting
    if (self.systemShutdownInProgress) {
        return NSTerminateNow;
    }
    // Otherwise just hide the app (for Cmd+Q, close button, etc.)
    [NSApp hide:nil];
    [NSApp setActivationPolicy:NSApplicationActivationPolicyAccessory];
    return NSTerminateCancel;
}

- (IBAction)terminate:(id)sender {
    [NSApp hide:nil];
    [NSApp setActivationPolicy:NSApplicationActivationPolicyAccessory];
}

- (BOOL)windowShouldClose:(id)sender {
    [NSApp hide:nil];
    return NO;
}

- (void)showIcon {
    NSAppearance *appearance = self.statusItem.button.effectiveAppearance;
    NSString *appearanceName = (NSString *)(appearance.name);
    NSString *iconName = @"ollama";
    if (self.updateAvailable) {
        iconName = [iconName stringByAppendingString:@"Update"];
    }
    if ([appearanceName containsString:@"Dark"]) {
        iconName = [iconName stringByAppendingString:@"Dark"];
    }

    NSImage *statusImage;
    NSBundle *bundle = [NSBundle mainBundle];
    if (![bundle.bundlePath hasSuffix:@".app"]) {
        NSString *cwdPath =
            [[NSFileManager defaultManager] currentDirectoryPath];
        NSString *bundlePath =
            [cwdPath stringByAppendingPathComponent:
                         [NSString stringWithFormat:@"darwin/Ollama.app"]];
        bundle = [NSBundle bundleWithPath:bundlePath];
    }

    statusImage = [bundle imageForResource:iconName];
    [statusImage setTemplate:YES];
    self.statusItem.button.image = statusImage;
}

- (void)observeValueForKeyPath:(NSString *)keyPath
                      ofObject:(id)object
                        change:(NSDictionary<NSKeyValueChangeKey, id> *)change
                       context:(void *)context {
    [self showIcon];
}

- (void)hide {
    [NSApp hide:nil];
    [NSApp setActivationPolicy:NSApplicationActivationPolicyAccessory];
}

- (void)quit {
    [NSApp stop:self];
    [NSApp postEvent:[NSEvent otherEventWithType:NSEventTypeApplicationDefined
                                        location:NSZeroPoint
                                   modifierFlags:0
                                       timestamp:0
                                    windowNumber:0
                                         context:nil
                                         subtype:0
                                           data1:0
                                           data2:0]
             atStart:YES];
}

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
- (void)registerSelfAsLoginItem:(BOOL)firstTimeRun {
    appLogInfo(@"using v13+ SMAppService for login registration");
    // Maps to the file Ollama.app/Contents/Library/LaunchAgents/com.ollama.ollama.plist
    SMAppService* service = [SMAppService agentServiceWithPlistName:@"com.ollama.ollama.plist"];
    if (!service) {
        appLogInfo(@"SMAppService failed to find service for com.ollama.ollama.plist");
        return;
    }
    SMAppServiceStatus status = [service status];
    switch (status) {
        case SMAppServiceStatusNotRegistered:
            appLogInfo(@"service not registered, registering now");
            break;
        case SMAppServiceStatusEnabled:
            appLogInfo(@"service is already enabled, no need to register again");
            return;
        case SMAppServiceStatusRequiresApproval: 
            // User has disabled our login behavior explicitly so leave it as is
            appLogInfo(@"service is currently disabled and will not start at login");
            return;
        case SMAppServiceStatusNotFound:
            appLogInfo(@"service not found, registering now");
            break;
        default:
            appLogInfo([NSString stringWithFormat:@"unexpected status: %ld", (long)status]);
            break;
    }
    NSError *error = nil;
    if (![service registerAndReturnError:&error]) {
        appLogInfo([NSString stringWithFormat:@"Failed to register %@ as a login item: %@", NSBundle.mainBundle.bundleURL, error]);
        return;
    }
    return;
}

/// Remove ollama from the deprecated Login Items list as we now use LaunchAgents
- (void)unregisterSelfFromLoginItem {
    NSURL *bundleURL = NSBundle.mainBundle.bundleURL;
    NSString *bundlePrefix = [SystemWidePath stringByDeletingPathExtension];

    LSSharedFileListRef loginItems =
        LSSharedFileListCreate(NULL, kLSSharedFileListSessionLoginItems, NULL);
    if (!loginItems) {
        return;
    }

    UInt32 seed;
    CFArrayRef currentItems = LSSharedFileListCopySnapshot(loginItems, &seed);

    for (id item in (__bridge NSArray *)currentItems) {
        CFURLRef itemURL = NULL;
        if (LSSharedFileListItemResolve((LSSharedFileListItemRef)item, 0,
                                        &itemURL, NULL) == noErr) {
            CFStringRef loginPath = CFURLCopyFileSystemPath(itemURL, kCFURLPOSIXPathStyle);
            // Compare the prefix to match against "keep existing" flow, e.g. // "/Applications/Ollama.app" vs "/Applications/Ollama 2.app"
            if (loginPath && [(NSString *)loginPath hasPrefix:bundlePrefix]) {
                appLogInfo([NSString stringWithFormat:@"removing login item %@", loginPath]);
                LSSharedFileListItemRemove(loginItems,
                                           (LSSharedFileListItemRef)item);
            }
            if (itemURL) {
                CFRelease(itemURL);
            }
        } else if (!itemURL) {
            // If the user has removed the App that has a current login item, we can't use
            // LSSharedFileListItemResolve to get the file path, since it doesn't "resolve"
            CFStringRef displayName = LSSharedFileListItemCopyDisplayName((LSSharedFileListItemRef)item);
            if (displayName) {
                NSString *name = (__bridge NSString *)displayName;
                if ([name hasPrefix:@"Ollama"]) {
                    LSSharedFileListItemRemove(loginItems, (LSSharedFileListItemRef)item);
                    appLogInfo([NSString stringWithFormat:@"removing dangling login item %@", displayName]);
                }
                CFRelease(displayName);
            }
        }
    }
    if (currentItems) {
        CFRelease(currentItems);
    }
    CFRelease(loginItems);
}
#pragma clang diagnostic pop

- (void)windowWillEnterFullScreen:(NSNotification *)notification {
    NSWindow *w = notification.object;
    if (w.toolbar != nil) {
        [w.toolbar setVisible:NO];            // hide the (empty) toolbar
    }
}

- (void)windowDidExitFullScreen:(NSNotification *)notification {
    NSWindow *w = notification.object;
    if (w.toolbar != nil) {
        [w.toolbar setVisible:YES];           // show it again
    }
}

- (void)      webView:(WKWebView *)webView
decidePolicyForNavigationAction:(WKNavigationAction *)action
        decisionHandler:(void (^)(WKNavigationActionPolicy))handler
{
    NSURL *url = action.request.URL;
    if (action.navigationType == WKNavigationTypeLinkActivated) {
        NSString *host = [url.host lowercaseString];
        if ([host isEqualToString:@"localhost"] ||
            [host isEqualToString:@"127.0.0.1"]) {
            handler(WKNavigationActionPolicyCancel);
            NSString *path = url.path;
            if (path.length == 0) {
                path = @"/";
            }
            [self uiRequest:path];
            return;
        }

        [[NSWorkspace sharedWorkspace] openURL:url];
        handler(WKNavigationActionPolicyCancel);
        return;
    }
    handler(WKNavigationActionPolicyAllow);
}

- (nullable WKWebView *)webView:(WKWebView *)webView
   createWebViewWithConfiguration:(WKWebViewConfiguration *)configuration
             forNavigationAction:(WKNavigationAction *)action
                  windowFeatures:(WKWindowFeatures *)features
{
    // "Open Link in New Window" (or target="_blank") ends up here.
    NSURL *url = action.request.URL;
    if (url) {
        NSString *host = [url.host lowercaseString];
        if ([host isEqualToString:@"localhost"] ||
            [host isEqualToString:@"127.0.0.1"]) {
            return nil;
        }
        [[NSWorkspace sharedWorkspace] openURL:url];
    }
    return nil;
}

// TODO (jmorganca): the confirm button is always "Confirm"
// it should be customizable in the future
- (void)webView:(WKWebView *)webView
    runJavaScriptConfirmPanelWithMessage:(NSString *)message
    initiatedByFrame:(WKFrameInfo *)frame
    completionHandler:(void (^)(BOOL))completionHandler {

    NSAlert *alert = [[NSAlert alloc] init];
    [alert setMessageText:message];
    [alert addButtonWithTitle:@"Confirm"];
    [alert addButtonWithTitle:@"Cancel"];

    completionHandler([alert runModal] == NSAlertFirstButtonReturn);
}

// HACK (jmorganca): remove the "Copy Link with Highlight" item from the context menu by
// swizzling the WKWebView's willOpenMenu:withEvent: method. In the future we should probably
// subclass the WKWebView and override the context menu items, but this is a quick fix for now.
+ (void)load {
    static dispatch_once_t onceToken;
    dispatch_once(&onceToken, ^{
        [self swizzleWKWebViewContextMenu];
    });
}

+ (void)swizzleWKWebViewContextMenu {
    Class class = [WKWebView class];

    SEL originalSelector = @selector(willOpenMenu:withEvent:);
    SEL swizzledSelector = @selector(ollama_willOpenMenu:withEvent:);

    Method originalMethod = class_getInstanceMethod(class, originalSelector);
    Method swizzledMethod = class_getInstanceMethod(class, swizzledSelector);
    BOOL didAddMethod = class_addMethod(class, originalSelector,
                                       method_getImplementation(swizzledMethod),
                                       method_getTypeEncoding(swizzledMethod));

    if (didAddMethod) {
        class_replaceMethod(class,
                           swizzledSelector,
                           method_getImplementation(originalMethod),
                           method_getTypeEncoding(originalMethod));
    } else {
        method_exchangeImplementations(originalMethod, swizzledMethod);
    }
}

@end

@implementation WKWebView (OllamaContextMenu)
- (void)ollama_willOpenMenu:(NSMenu *)menu withEvent:(NSEvent *)event {
    [self ollama_willOpenMenu:menu withEvent:event];
    NSMutableArray *itemsToRemove = [NSMutableArray array];
    for (NSMenuItem *item in menu.itemArray) {
        if ([item.title containsString:@"Copy Link with Highlight"] ||
            [item.title containsString:@"Open Link in New Window"] ||
            [item.title containsString:@"Services"] ||
            [item.title containsString:@"Download Linked File"] ||
            [item.title containsString:@"Back"] ||
            [item.title containsString:@"Reload"] ||
            [item.title containsString:@"Refresh"] ||
            [item.title containsString:@"Open Link"] ||
            [item.title containsString:@"Copy Link"] ||
            [item.title containsString:@"Share"]) {
            [itemsToRemove addObject:item];
            continue;
        }
    }

    for (NSMenuItem *item in itemsToRemove) {
        [menu removeItem:item];
    }

    int customItemCount = menu_get_item_count();
    if (customItemCount > 0) {
        menuItem* customItems = (menuItem*)menu_get_items();
        if (customItems) {
            NSInteger insertIndex = 0;
            
            for (int i = 0; i < customItemCount; i++) {
                if (customItems[i].separator) {
                    [menu insertItem:[NSMenuItem separatorItem] atIndex:insertIndex++];
                } else if (customItems[i].label) {
                    NSString *label = [NSString stringWithUTF8String:customItems[i].label];
                    NSMenuItem *item = [[NSMenuItem alloc] initWithTitle:label
                                                                  action:@selector(handleCustomMenuItem:)
                                                           keyEquivalent:@""];
                    [item setTarget:self];
                    [item setRepresentedObject:label];
                    [item setEnabled:customItems[i].enabled];
                    [menu insertItem:item atIndex:insertIndex++];
                }
            }
            
            // Add separator after custom items if there are remaining items
            if (insertIndex > 0 && menu.itemArray.count > insertIndex) {
                [menu insertItem:[NSMenuItem separatorItem] atIndex:insertIndex];
            }
        }
    }
}

- (void)handleCustomMenuItem:(NSMenuItem *)sender {
    NSString *label = [sender representedObject];
    if (label) {
        menu_handle_selection((char*)[label UTF8String]);
    }
}

@end

AppDelegate *appDelegate;
void run(bool ftr, bool sh) {
    [NSApplication sharedApplication];
    [NSApp setActivationPolicy:NSApplicationActivationPolicyAccessory];
    appDelegate = [[AppDelegate alloc] init];
    [NSApp setDelegate:appDelegate];
    firstTimeRun = ftr;
    startHidden = sh;
    [NSApp run];
    StopUI();
}

// killOtherInstances kills all other instances of the app currently
// running. This way we can ensure that only the most recently started
// instance of Ollama is running
void killOtherInstances() {
    pid_t myPid = getpid();
    NSArray *apps = [[NSWorkspace sharedWorkspace] runningApplications];

    for (NSRunningApplication *app in apps) {
        NSString *bundleId = app.bundleIdentifier;
        
        // Skip apps without bundle identifiers
        if (!bundleId || [bundleId length] == 0) {
            continue;
        }
        
        if ([bundleId isEqualToString:[[NSBundle mainBundle] bundleIdentifier]] ||
            [bundleId isEqualToString:@"ai.ollama.ollama"] ||
            [bundleId isEqualToString:@"com.electron.ollama"]) {
            
            pid_t pid = app.processIdentifier;
            if (pid != myPid && pid > 0) {
                appLogInfo([NSString stringWithFormat:@"terminating other ollama instance %d", pid]);
                kill(pid, SIGTERM);
            } else if (pid == -1) {
                appLogInfo([NSString stringWithFormat:@"skipping app with invalid pid: %@", bundleId]);
            }
        }
    }
}

// Move the source bundle to the system-wide applications location
// without prompting for additional authorization
bool moveToApplications(const char *src) {
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

AuthorizationRef getSymlinkAuthorization() {
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
bool moveToApplicationsWithAuthorization(const char *src) {
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

enum AppMove askToMoveToApplications() {
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

void updateAvailable() {
    dispatch_async(dispatch_get_main_queue(), ^{
      [appDelegate showUpdateAvailable];
    });
}

void quit() {
    dispatch_async(dispatch_get_main_queue(), ^{
      [appDelegate quit];
    });
}

void uiRequest(char *path) {
    NSString *p = [NSString stringWithFormat:@"%s", path];
    appLogInfo([NSString stringWithFormat:@"XXX UI request for URL: %@", p]);
    dispatch_async(dispatch_get_main_queue(), ^{
      [appDelegate uiRequest:p];
    });
}

void registerSelfAsLoginItem(bool firstTimeRun) {
    dispatch_async(dispatch_get_main_queue(), ^{
      [appDelegate registerSelfAsLoginItem:firstTimeRun];
    });
}

void unregisterSelfFromLoginItem() {
    dispatch_async(dispatch_get_main_queue(), ^{
        [appDelegate unregisterSelfFromLoginItem];
    });
}

static WKWebView *FindWKWebView(NSView *root) {
    if ([root isKindOfClass:[WKWebView class]]) {
        return (WKWebView *)root;
    }
    for (NSView *child in root.subviews) {
        WKWebView *found = FindWKWebView(child);
        if (found) {
            return found;
        }
    }
    return nil;
}

void setWindowDelegate(void* window) {
    NSWindow *w = (__bridge NSWindow *)window;
    [w setDelegate:appDelegate];
    WKWebView *webView = FindWKWebView(w.contentView);
    if (webView) {
        webView.navigationDelegate = appDelegate;
        webView.UIDelegate = appDelegate;
    }
}

void hideWindow(uintptr_t wndPtr) {
    NSWindow *w = (__bridge NSWindow *)wndPtr;
    [NSApp setActivationPolicy:NSApplicationActivationPolicyAccessory];
    [w orderOut:nil];
}

void showWindow(uintptr_t wndPtr) {
    NSWindow *w = (__bridge NSWindow *)wndPtr;
        
    [NSApp setActivationPolicy:NSApplicationActivationPolicyRegular];

    dispatch_async(dispatch_get_main_queue(), ^{
        [NSApp unhide:nil];
        [NSApp activateIgnoringOtherApps:YES];
        [w makeKeyAndOrderFront:nil];
    });
}

void styleWindow(uintptr_t wndPtr) {
    NSWindow *w = (__bridge NSWindow *)wndPtr;
    if (!w) return;

    // Define the desired style mask
    NSWindowStyleMask desiredStyleMask = NSWindowStyleMaskTitled |
                    NSWindowStyleMaskClosable |
                    NSWindowStyleMaskMiniaturizable |
                    NSWindowStyleMaskResizable |
                    NSWindowStyleMaskFullSizeContentView |
                    NSWindowStyleMaskUnifiedTitleAndToolbar;

    if (!(w.styleMask & NSWindowStyleMaskFullScreen)) {
        w.styleMask = desiredStyleMask;
    }

    if (w.toolbar == nil) {
        NSToolbar *tb = [[NSToolbar alloc] initWithIdentifier:@"OllamaToolbar"];
        tb.displayMode            = NSToolbarDisplayModeIconOnly;
        tb.showsBaselineSeparator = NO;
        w.toolbar                 = tb;
    }

    w.titleVisibility = NSWindowTitleHidden;
    w.titlebarAppearsTransparent = YES;
    w.toolbarStyle = NSWindowToolbarStyleUnified;
    w.movableByWindowBackground = NO;
    w.hasShadow = YES;

    NSView *cv = w.contentView;
    cv.wantsLayer = YES;
    CALayer *L = cv.layer;
    L.cornerRadius = 0.0;
    L.masksToBounds = NO;
    L.borderColor = nil;
    L.borderWidth = 0.0;
}

void drag(uintptr_t wndPtr) {
    NSWindow *w = (__bridge NSWindow *)wndPtr;
    if (!w) return;
    NSPoint mouseLoc = [NSEvent mouseLocation];
    NSPoint locInWindow = [w convertPointFromScreen:mouseLoc];

    NSEvent *e = [NSEvent mouseEventWithType:NSEventTypeLeftMouseDown
                                    location:locInWindow
                                modifierFlags:0
                                    timestamp:NSTimeIntervalSince1970
                                windowNumber:[w windowNumber]
                                        context:nil
                                    eventNumber:0
                                    clickCount:1
                                    pressure:1.0];
    [w performWindowDragWithEvent:e];
}

void doubleClick(uintptr_t wndPtr) {
    NSWindow *w = (__bridge NSWindow *)wndPtr;
    if (!w) return;

    // Respect the user's Dock preference
    NSString *action =
        [[NSUserDefaults standardUserDefaults] stringForKey:@"AppleActionOnDoubleClick"];

    if ([action isEqualToString:@"Minimize"]) {
        [w performMiniaturize:nil];
    } else {
        [w performZoom:nil];
    }
}
