#import "app_darwin.h"
#import "resources_darwin.h"
#import "settings_window_darwin.h"
#import "../../updater/updater_darwin.h"
#import <AppKit/AppKit.h>

@interface AppDelegate ()
@property(strong, nonatomic) NSStatusItem *statusItem;
@property(assign, nonatomic) BOOL updateAvailable;
@property(assign, nonatomic) BOOL systemShutdownInProgress;
@property(strong, nonatomic) NSMenuItem *updateAvailableMenuItem;
@property(strong, nonatomic) NSMenuItem *restartMenuItem;
@property(strong, nonatomic) NSMenuItem *updateSeparatorItem;
@property(strong, nonatomic, nullable) OLSettingsWindowController *settingsWindowController;
@property(assign, nonatomic) BOOL quitInProgress;
@property(assign, nonatomic) BOOL systemTerminationReplyPending;
@property(strong, nonatomic, nullable) NSApplication *systemTerminationApplication;
@end

@implementation AppDelegate

- (void)application:(NSApplication *)application openURLs:(NSArray<NSURL *> *)urls {
    (void)application;
    for (NSURL *url in urls) {
        if ([url.scheme isEqualToString:@"ollama"]) {
            handleURLScheme((char *)url.absoluteString.UTF8String);
            break;
        }
    }
}

- (void)applicationDidFinishLaunching:(NSNotification *)aNotification {
    (void)aNotification;
    // Register for system shutdown/restart notification so we can allow termination
    [[[NSWorkspace sharedWorkspace] notificationCenter]
        addObserver:self
           selector:@selector(systemWillPowerOff:)
               name:NSWorkspaceWillPowerOffNotification
             object:nil];

    // During development the app isn't in a bundle, so set its icon.
    if (![[[NSBundle mainBundle] bundlePath] hasSuffix:@".app"]) {
        [NSApp setApplicationIconImage:OLApplicationIcon()];
    }

    [self installStatusItem];
    [self installMainMenu];

    BOOL hidden = [NSApp isHidden];
    dispatch_async(dispatch_get_main_queue(), ^{
        appDidFinishLaunching(hidden);
    });
}

- (void)installStatusItem {
    NSMenu *menu = [[NSMenu alloc] init];
    [menu setAutoenablesItems:NO];

    NSMenuItem *settingsItem = [[NSMenuItem alloc] initWithTitle:@"Settings…"
                                                          action:@selector(showSettings:)
                                                   keyEquivalent:@","];
    [settingsItem setTarget:self];
    [menu addItem:settingsItem];
    [menu addItem:[NSMenuItem separatorItem]];

    self.updateAvailableMenuItem =
        [[NSMenuItem alloc] initWithTitle:@"An update is available"
                                   action:nil
                            keyEquivalent:@""];
    [self.updateAvailableMenuItem setEnabled:NO];
    [self.updateAvailableMenuItem setHidden:YES];
    [menu addItem:self.updateAvailableMenuItem];

    self.restartMenuItem =
        [[NSMenuItem alloc] initWithTitle:@"Restart to update"
                                   action:@selector(startUpdate)
                            keyEquivalent:@""];
    [self.restartMenuItem setTarget:self];
    [self.restartMenuItem setHidden:YES];
    [menu addItem:self.restartMenuItem];

    self.updateSeparatorItem = [NSMenuItem separatorItem];
    [self.updateSeparatorItem setHidden:YES];
    [menu addItem:self.updateSeparatorItem];

    NSMenuItem *quitItem = [[NSMenuItem alloc] initWithTitle:@"Quit Ollama"
                                                      action:@selector(requestQuit)
                                               keyEquivalent:@"q"];
    [quitItem setTarget:self];
    [menu addItem:quitItem];

    self.statusItem = [[NSStatusBar systemStatusBar]
        statusItemWithLength:NSVariableStatusItemLength];
    [self.statusItem addObserver:self
                      forKeyPath:@"button.effectiveAppearance"
                         options:NSKeyValueObservingOptionNew |
                                 NSKeyValueObservingOptionInitial
                         context:nil];

    self.statusItem.menu = menu;
    [self refreshStatusItem];
}

// The main menu appears while the Settings window is in front.
- (void)installMainMenu {
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
    [appMenu addItemWithTitle:@"Settings…"
                       action:@selector(showSettings:)
                keyEquivalent:@","];
    [appMenu addItem:[NSMenuItem separatorItem]];
    [appMenu addItemWithTitle:[NSString stringWithFormat:@"Hide %@", appName]
                       action:@selector(hide:)
                keyEquivalent:@"h"];
    NSMenuItem *hideOthers = [[NSMenuItem alloc] initWithTitle:@"Hide Others"
                                                        action:@selector(hideOtherApplications:)
                                                 keyEquivalent:@"h"];
    hideOthers.keyEquivalentModifierMask = NSEventModifierFlagOption | NSEventModifierFlagCommand;
    [appMenu addItem:hideOthers];
    [appMenu addItemWithTitle:@"Show All"
                       action:@selector(unhideAllApplications:)
                keyEquivalent:@""];
    [appMenu addItem:[NSMenuItem separatorItem]];
    // Ollama keeps running in the menu bar, so Quit only hides its windows.
    [appMenu addItemWithTitle:[NSString stringWithFormat:@"Quit %@", appName]
                       action:@selector(hide)
                keyEquivalent:@"q"];

    NSMenuItem *fileMenuItem = [[NSMenuItem alloc] init];
    NSMenu *fileMenu = [[NSMenu alloc] initWithTitle:@"File"];
    [fileMenu addItemWithTitle:@"Close Window"
                        action:@selector(performClose:)
                 keyEquivalent:@"w"];
    [fileMenuItem setSubmenu:fileMenu];
    [mainMenu addItem:fileMenuItem];

    NSMenuItem *editMenuItem = [[NSMenuItem alloc] init];
    NSMenu *editMenu = [[NSMenu alloc] initWithTitle:@"Edit"];
    [editMenu addItemWithTitle:@"Undo" action:@selector(undo:) keyEquivalent:@"z"];
    [editMenu addItemWithTitle:@"Redo" action:@selector(redo:) keyEquivalent:@"Z"];
    [editMenu addItem:[NSMenuItem separatorItem]];
    [editMenu addItemWithTitle:@"Cut" action:@selector(cut:) keyEquivalent:@"x"];
    [editMenu addItemWithTitle:@"Copy" action:@selector(copy:) keyEquivalent:@"c"];
    [editMenu addItemWithTitle:@"Paste" action:@selector(paste:) keyEquivalent:@"v"];
    [editMenu addItemWithTitle:@"Select All" action:@selector(selectAll:) keyEquivalent:@"a"];
    [editMenuItem setSubmenu:editMenu];
    [mainMenu addItem:editMenuItem];

    NSMenuItem *windowMenuItem = [[NSMenuItem alloc] init];
    NSMenu *windowMenu = [[NSMenu alloc] initWithTitle:@"Window"];
    [windowMenu addItemWithTitle:@"Minimize"
                          action:@selector(performMiniaturize:)
                   keyEquivalent:@"m"];
    [windowMenu addItemWithTitle:@"Zoom" action:@selector(performZoom:) keyEquivalent:@""];
    [windowMenu addItem:[NSMenuItem separatorItem]];
    [windowMenu addItemWithTitle:@"Bring All to Front"
                          action:@selector(arrangeInFront:)
                   keyEquivalent:@""];
    [windowMenuItem setSubmenu:windowMenu];
    [mainMenu addItem:windowMenuItem];
    [NSApp setWindowsMenu:windowMenu];

    NSMenuItem *helpMenuItem = [[NSMenuItem alloc] init];
    NSMenu *helpMenu = [[NSMenu alloc] initWithTitle:@"Help"];
    [helpMenu addItemWithTitle:[NSString stringWithFormat:@"%@ Help", appName]
                        action:@selector(openHelp:)
                 keyEquivalent:@"?"];
    [helpMenuItem setSubmenu:helpMenu];
    [mainMenu addItem:helpMenuItem];
    [NSApp setHelpMenu:helpMenu];
    [NSApp setMainMenu:mainMenu];
}

- (void)applicationDidBecomeActive:(NSNotification *)notification {
    (void)notification;
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
    }
}

- (BOOL)applicationShouldHandleReopen:(NSApplication *)sender hasVisibleWindows:(BOOL)hasVisibleWindows {
    (void)sender;
    (void)hasVisibleWindows;
    [self showSettingsPane:nil];
    return NO;
}

#pragma mark - Settings

- (void)showSettings:(id)sender {
    (void)sender;
    [self.statusItem.menu cancelTracking];
    [self showSettingsPane:nil];
}

- (void)showSettingsPane:(nullable NSString *)pane {
    if (self.settingsWindowController == nil) {
        self.settingsWindowController = [[OLSettingsWindowController alloc] init];
        [[NSNotificationCenter defaultCenter]
            addObserver:self
               selector:@selector(settingsWindowWillClose:)
                   name:NSWindowWillCloseNotification
                 object:self.settingsWindowController.window];
    }
    // Show Ollama in the Dock and app switcher while Settings is open.
    [NSApp setActivationPolicy:NSApplicationActivationPolicyRegular];
    [NSApp unhide:nil];
    [self.settingsWindowController showPane:pane];
    [NSApp activate];
}

- (void)settingsWindowWillClose:(NSNotification *)notification {
    (void)notification;
    [NSApp setActivationPolicy:NSApplicationActivationPolicyAccessory];
}

- (void)showUpdateAvailable {
    self.updateAvailable = YES;
    [self refreshStatusItem];
}

- (void)aboutOllama {
    [NSApp orderFrontStandardAboutPanel:nil];
    [NSApp activate];
}

- (void)openHelp:(id)sender {
    (void)sender;
    [[NSWorkspace sharedWorkspace] openURL:[NSURL URLWithString:@"https://docs.ollama.com/"]];
}

- (void)startUpdate {
    StartUpdate();
    [NSApp activate];
}

- (void)refreshStatusItem {
    [self.updateAvailableMenuItem setHidden:!self.updateAvailable];
    [self.restartMenuItem setHidden:!self.updateAvailable];
    [self.updateSeparatorItem setHidden:!self.updateAvailable];

    NSAppearance *appearance = self.statusItem.button.effectiveAppearance;
    NSString *appearanceName = (NSString *)(appearance.name);
    NSString *iconName = @"ollama";
    if (self.updateAvailable) {
        iconName = [iconName stringByAppendingString:@"Update"];
    }
    if ([appearanceName containsString:@"Dark"]) {
        iconName = [iconName stringByAppendingString:@"Dark"];
    }

    NSImage *statusImage = [OLResourceBundle() imageForResource:iconName];
    if (statusImage) {
        [statusImage setTemplate:YES];
        self.statusItem.button.title = @"";
        self.statusItem.button.image = statusImage;
    } else {
        self.statusItem.button.image = nil;
        self.statusItem.button.title = @"Ollama";
    }
}

- (void)observeValueForKeyPath:(NSString *)keyPath
                      ofObject:(id)object
                        change:(NSDictionary<NSKeyValueChangeKey, id> *)change
                       context:(void *)context {
    (void)keyPath;
    (void)object;
    (void)change;
    (void)context;
    [self refreshStatusItem];
}

#pragma mark - Quitting

- (void)systemWillPowerOff:(NSNotification *)notification {
    (void)notification;
    // Set flag so applicationShouldTerminate: knows to allow termination.
    // The system will call applicationShouldTerminate: after posting this notification.
    self.systemShutdownInProgress = YES;
}

- (NSApplicationTerminateReply)applicationShouldTerminate:(NSApplication *)sender {
    if (self.systemShutdownInProgress) {
        if (!IsClaudeGatewayConfigured()) {
            return NSTerminateNow;
        }
        self.systemTerminationApplication = sender;
        self.systemTerminationReplyPending = YES;
        if (self.quitInProgress) {
            return NSTerminateLater;
        }
        self.quitInProgress = YES;
        dispatch_async(dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^{
            BOOL succeeded = RestoreClaudeGatewayForShutdown();
            if (!succeeded) {
                appLogInfo(@"Unable to restore Claude during system shutdown");
            }
            dispatch_async(dispatch_get_main_queue(), ^{
                [self completeSystemTermination];
            });
        });
        return NSTerminateLater;
    }
    // Otherwise just hide the app (for Cmd+Q, close button, etc.)
    [self hide];
    return NSTerminateCancel;
}

- (void)completeSystemTermination {
    if (!self.systemTerminationReplyPending) {
        return;
    }
    NSApplication *application = self.systemTerminationApplication;
    self.systemTerminationReplyPending = NO;
    self.systemTerminationApplication = nil;
    self.quitInProgress = NO;
    [application replyToApplicationShouldTerminate:YES];
}

- (IBAction)terminate:(id)sender {
    (void)sender;
    [self hide];
}

- (void)hide {
    [NSApp hide:nil];
    [NSApp setActivationPolicy:NSApplicationActivationPolicyAccessory];
}

- (void)requestQuit {
    if (self.quitInProgress) {
        return;
    }
    if (!IsClaudeGatewayConfigured()) {
        [self quit];
        return;
    }

    BOOL restartClaude = IsClaudeDesktopRunning();
    if (restartClaude) {
        NSAlert *alert = [[NSAlert alloc] init];
        [alert setAlertStyle:NSAlertStyleWarning];
        [alert setIcon:OLApplicationIcon()];
        [alert setMessageText:@"Restart Claude before quitting Ollama?"];
        [alert setInformativeText:
            @"Claude must restart before Ollama quits. Any running task will stop."];
        [alert addButtonWithTitle:@"Restart Claude and Quit"];
        [alert addButtonWithTitle:@"Cancel"];
        if ([alert runModal] != NSAlertFirstButtonReturn) {
            return;
        }
    }

    self.quitInProgress = YES;
    dispatch_async(dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^{
        BOOL succeeded = SetClaudeGatewayInstalled(false, restartClaude);
        dispatch_async(dispatch_get_main_queue(), ^{
            if (self.systemTerminationReplyPending) {
                if (!succeeded) {
                    appLogInfo(@"Unable to restore Claude during system shutdown");
                }
                [self completeSystemTermination];
                return;
            }
            if (succeeded) {
                [self quit];
                return;
            }

            self.quitInProgress = NO;
            NSAlert *alert = [[NSAlert alloc] init];
            [alert setAlertStyle:NSAlertStyleWarning];
            [alert setIcon:OLApplicationIcon()];
            [alert setMessageText:@"Unable to quit Ollama"];
            [alert setInformativeText:
                @"Ollama couldn’t update Claude, so it is still running. Check the Ollama log and try again."];
            [alert runModal];
        });
    });
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

@end

static AppDelegate *appDelegate;

void run(void) {
    // Retain update notifications that arrive while AppKit initializes.
    appDelegate = [[AppDelegate alloc] init];
    [NSApplication sharedApplication];
    [NSApp setActivationPolicy:NSApplicationActivationPolicyAccessory];
    [NSApp setDelegate:appDelegate];
    [NSApp run];
}

void showSettings(const char *pane) {
    NSString *identifier = pane != NULL ? @(pane) : @"";
    dispatch_async(dispatch_get_main_queue(), ^{
        [appDelegate showSettingsPane:identifier.length > 0 ? identifier : nil];
    });
}

void updateAvailable(void) {
    dispatch_async(dispatch_get_main_queue(), ^{
        [appDelegate showUpdateAvailable];
        [[NSNotificationCenter defaultCenter]
            postNotificationName:OLSettingsDidChangeNotification
                          object:nil];
    });
}

void quit(void) {
    dispatch_async(dispatch_get_main_queue(), ^{
        [appDelegate quit];
    });
}
