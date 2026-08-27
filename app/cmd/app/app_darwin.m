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
#include <errno.h>
#include <libproc.h>
#include <stdlib.h>
#include <unistd.h>

extern NSString *SystemWidePath;

static NSString *const ClaudeDownloadPageURL = @"https://claude.com/download";
static NSString *const ShowAppsInMenuDefaultsKey = @"ShowAppsInMenu";
static NSBundle *OllamaResourceBundle(void);

static BOOL shouldShowAppsInMenu(void) {
    NSUserDefaults *defaults = [NSUserDefaults standardUserDefaults];
    if ([defaults objectForKey:ShowAppsInMenuDefaultsKey] == nil) {
        return YES;
    }
    return [defaults boolForKey:ShowAppsInMenuDefaultsKey];
}

static NSImage *integrationAppIcon(NSString *appName,
                                   NSString *fallbackSymbolName) {
    NSArray<NSString *> *candidates = @[
        [NSString stringWithFormat:@"/Applications/%@.app", appName],
        [NSHomeDirectory() stringByAppendingPathComponent:
            [NSString stringWithFormat:@"Applications/%@.app", appName]],
    ];
    for (NSString *path in candidates) {
        if (![[NSFileManager defaultManager] fileExistsAtPath:path]) {
            continue;
        }
        NSImage *icon = [[NSWorkspace sharedWorkspace] iconForFile:path];
        if (icon != nil) {
            [icon setTemplate:NO];
            return icon;
        }
    }
    NSBundle *bundle = OllamaResourceBundle();
    NSImage *bundledIcon = [bundle imageForResource:appName.lowercaseString];
    if (bundledIcon != nil) {
        [bundledIcon setTemplate:NO];
        return bundledIcon;
    }
    return [NSImage imageWithSystemSymbolName:fallbackSymbolName
                      accessibilityDescription:nil];
}

@interface MenuSwitch : NSButton
@end

@implementation MenuSwitch

- (instancetype)initWithFrame:(NSRect)frameRect {
    self = [super initWithFrame:frameRect];
    if (self) {
        [self setButtonType:NSButtonTypeToggle];
        [self setBordered:NO];
        [self setTitle:@""];
        [self setFocusRingType:NSFocusRingTypeNone];
        [self setAccessibilityRole:NSAccessibilityCheckBoxRole];
    }
    return self;
}

- (NSSize)intrinsicContentSize {
    return NSMakeSize(30, 18);
}

- (void)setState:(NSControlStateValue)state {
    [super setState:state];
    [self setNeedsDisplay:YES];
}

- (void)setEnabled:(BOOL)enabled {
    [super setEnabled:enabled];
    [self setNeedsDisplay:YES];
}

- (void)drawRect:(NSRect)dirtyRect {
    (void)dirtyRect;
    NSRect track = NSInsetRect(self.bounds, 0, 1);
    NSColor *trackColor = self.state == NSControlStateValueOn
        ? [NSColor systemBlueColor]
        : [NSColor colorWithWhite:0.55 alpha:1];
    if (!self.enabled) {
        trackColor = [trackColor colorWithAlphaComponent:0.45];
    }
    [trackColor setFill];
    [[NSBezierPath bezierPathWithRoundedRect:track
                                     xRadius:NSHeight(track) / 2
                                     yRadius:NSHeight(track) / 2] fill];

    CGFloat knobSize = NSHeight(track) - 4;
    CGFloat knobX = self.state == NSControlStateValueOn
        ? NSMaxX(track) - knobSize - 2
        : NSMinX(track) + 2;
    NSRect knob = NSMakeRect(knobX,
                             NSMidY(track) - knobSize / 2,
                             knobSize,
                             knobSize);
    [[NSColor whiteColor] setFill];
    [[NSBezierPath bezierPathWithOvalInRect:knob] fill];
}

@end

@interface IntegrationMenuRow : NSButton
@property(strong, nonatomic) NSView *controlSurface;
@property(strong, nonatomic) NSButton *openButton;
@property(strong, nonatomic) NSTextField *integrationTitleLabel;
@property(strong, nonatomic) NSTextField *integrationStatusLabel;
@property(strong, nonatomic) MenuSwitch *integrationSwitch;
@property(strong, nonatomic) NSLayoutConstraint *titleWithStatusConstraint;
@property(strong, nonatomic) NSLayoutConstraint *titleCenteredConstraint;
@property(strong, nonatomic) NSTrackingArea *hoverTrackingArea;
@property(assign, nonatomic) BOOL pointerInside;
@property(assign, nonatomic) SEL integrationOpenAction;
@property(copy, nonatomic) NSString *activeStatusText;
@property(copy, nonatomic) NSString *inactiveStatusText;
- (instancetype)initWithTitle:(NSString *)title
                   symbolName:(NSString *)symbolName
                       target:(id)target
                   openAction:(SEL)openAction
                 toggleAction:(SEL)toggleAction;
- (void)setIntegrationActive:(BOOL)active;
- (void)setIntegrationReady:(BOOL)ready;
- (void)setActiveStatusText:(NSString *)status;
- (void)setInactiveStatusText:(NSString *)status;
- (void)resetHover;
@end

@implementation IntegrationMenuRow

- (instancetype)initWithTitle:(NSString *)title
                   symbolName:(NSString *)symbolName
                       target:(id)target
                   openAction:(SEL)openAction
                 toggleAction:(SEL)toggleAction {
    self = [super initWithFrame:NSMakeRect(0, 0, 310, 50)];
    if (self) {
        [self setButtonType:NSButtonTypeMomentaryChange];
        [self setBordered:NO];
        [self setTitle:@""];
        [self setFocusRingType:NSFocusRingTypeNone];
        [self setTarget:target];
        [self setAction:openAction];
        self.integrationOpenAction = openAction;
        [self setAccessibilityLabel:[NSString stringWithFormat:@"Open %@", title]];
        [self setAccessibilityRole:NSAccessibilityButtonRole];

        self.controlSurface = [[NSView alloc] initWithFrame:NSZeroRect];
        [self.controlSurface setWantsLayer:YES];
        self.controlSurface.layer.cornerRadius = 13;
        [self.controlSurface setTranslatesAutoresizingMaskIntoConstraints:NO];

        NSImage *icon = integrationAppIcon(title, symbolName);
        self.openButton = [NSButton buttonWithImage:icon
                                             target:target
                                             action:openAction];
        [self.openButton setBordered:NO];
        [self.openButton setImagePosition:NSImageOnly];
        [self.openButton setImageScaling:NSImageScaleProportionallyDown];
        [self.openButton setFocusRingType:NSFocusRingTypeNone];
        [self.openButton setAccessibilityLabel:[NSString stringWithFormat:@"Open %@", title]];
        [self.openButton setTranslatesAutoresizingMaskIntoConstraints:NO];

        self.integrationTitleLabel = [NSTextField labelWithString:title];
        [self.integrationTitleLabel setFont:[NSFont systemFontOfSize:13 weight:NSFontWeightSemibold]];
        [self.integrationTitleLabel setLineBreakMode:NSLineBreakByTruncatingTail];
        [self.integrationTitleLabel setTranslatesAutoresizingMaskIntoConstraints:NO];

        self.integrationStatusLabel = [NSTextField labelWithString:@""];
        [self.integrationStatusLabel setFont:[NSFont systemFontOfSize:11.5]];
        [self.integrationStatusLabel setLineBreakMode:NSLineBreakByTruncatingTail];
        [self.integrationStatusLabel setTranslatesAutoresizingMaskIntoConstraints:NO];

        self.integrationSwitch = [[MenuSwitch alloc] initWithFrame:NSZeroRect];
        [self.integrationSwitch setTarget:target];
        [self.integrationSwitch setAction:toggleAction];
        [self.integrationSwitch setAccessibilityLabel:[NSString stringWithFormat:@"Use Ollama with %@", title]];
        [self.integrationSwitch setTranslatesAutoresizingMaskIntoConstraints:NO];

        [self addSubview:self.controlSurface];
        [self addSubview:self.openButton];
        [self addSubview:self.integrationTitleLabel];
        [self addSubview:self.integrationStatusLabel];
        [self addSubview:self.integrationSwitch];
        self.titleWithStatusConstraint =
            [self.integrationTitleLabel.bottomAnchor
                constraintEqualToAnchor:self.controlSurface.centerYAnchor
                               constant:-1];
        self.titleCenteredConstraint =
            [self.integrationTitleLabel.centerYAnchor
                constraintEqualToAnchor:self.controlSurface.centerYAnchor];
        [NSLayoutConstraint activateConstraints:@[
            [self.controlSurface.leadingAnchor constraintEqualToAnchor:self.leadingAnchor constant:7],
            [self.controlSurface.trailingAnchor constraintEqualToAnchor:self.trailingAnchor constant:-7],
            [self.controlSurface.topAnchor constraintEqualToAnchor:self.topAnchor constant:3],
            [self.controlSurface.bottomAnchor constraintEqualToAnchor:self.bottomAnchor constant:-3],
            [self.openButton.leadingAnchor constraintEqualToAnchor:self.controlSurface.leadingAnchor constant:9],
            [self.openButton.centerYAnchor constraintEqualToAnchor:self.controlSurface.centerYAnchor],
            [self.openButton.widthAnchor constraintEqualToConstant:34],
            [self.openButton.heightAnchor constraintEqualToConstant:34],
            [self.integrationTitleLabel.leadingAnchor constraintEqualToAnchor:self.openButton.trailingAnchor constant:10],
            self.titleWithStatusConstraint,
            [self.integrationTitleLabel.trailingAnchor constraintLessThanOrEqualToAnchor:self.integrationSwitch.leadingAnchor constant:-12],
            [self.integrationStatusLabel.leadingAnchor constraintEqualToAnchor:self.integrationTitleLabel.leadingAnchor],
            [self.integrationStatusLabel.topAnchor constraintEqualToAnchor:self.controlSurface.centerYAnchor constant:1],
            [self.integrationStatusLabel.trailingAnchor constraintLessThanOrEqualToAnchor:self.integrationSwitch.leadingAnchor constant:-12],
            [self.integrationSwitch.trailingAnchor constraintEqualToAnchor:self.controlSurface.trailingAnchor constant:-12],
            [self.integrationSwitch.centerYAnchor constraintEqualToAnchor:self.controlSurface.centerYAnchor],
        ]];
        [self setIntegrationActive:NO];
    }
    return self;
}

- (NSView *)hitTest:(NSPoint)point {
    if (!NSPointInRect(point, self.bounds)) {
        return nil;
    }
    if (NSPointInRect(point, self.integrationSwitch.frame)) {
        return self.integrationSwitch;
    }
    if (NSPointInRect(point, self.openButton.frame)) {
        return self.openButton;
    }
    return self;
}

- (void)updateTrackingAreas {
    [super updateTrackingAreas];
    if (self.hoverTrackingArea != nil) {
        [self removeTrackingArea:self.hoverTrackingArea];
    }
    self.hoverTrackingArea = [[NSTrackingArea alloc]
        initWithRect:NSZeroRect
             options:NSTrackingMouseEnteredAndExited |
                     NSTrackingActiveAlways |
                     NSTrackingInVisibleRect
               owner:self
            userInfo:nil];
    [self addTrackingArea:self.hoverTrackingArea];
}

- (void)mouseEntered:(NSEvent *)event {
    (void)event;
    self.pointerInside = YES;
    [self updateAppearance];
}

- (void)mouseExited:(NSEvent *)event {
    (void)event;
    self.pointerInside = NO;
    [self updateAppearance];
}

- (void)resetHover {
    self.pointerInside = NO;
    [self updateAppearance];
}

- (void)viewDidChangeEffectiveAppearance {
    [super viewDidChangeEffectiveAppearance];
    [self updateAppearance];
}

- (void)updateAppearance {
    BOOL active = self.integrationSwitch.state == NSControlStateValueOn;
    BOOL hovered = self.pointerInside && self.enabled;
    NSColor *surfaceColor = hovered
        ? [[NSColor labelColor] colorWithAlphaComponent:0.08]
        : [NSColor clearColor];
    self.controlSurface.layer.backgroundColor = surfaceColor.CGColor;
    self.integrationTitleLabel.textColor = [NSColor labelColor];
    NSString *status = active ? (self.activeStatusText ?: @"Using Ollama")
                              : (self.inactiveStatusText ?: @"Use Ollama models");
    BOOL hasStatus = status.length > 0;
    self.titleWithStatusConstraint.active = hasStatus;
    self.titleCenteredConstraint.active = !hasStatus;
    self.integrationStatusLabel.hidden = !hasStatus;
    self.integrationStatusLabel.stringValue = status;
    self.integrationStatusLabel.textColor = [NSColor secondaryLabelColor];
    [self setAccessibilityValue:status];
}

- (void)setIntegrationActive:(BOOL)active {
    self.integrationSwitch.state = active ? NSControlStateValueOn
                                          : NSControlStateValueOff;
    [self setIntegrationReady:active];
    [self updateAppearance];
}

- (void)setIntegrationReady:(BOOL)ready {
    self.action = ready ? self.integrationOpenAction : nil;
    self.openButton.enabled = ready;
    self.openButton.alphaValue = ready ? 1.0 : 0.45;
}

- (void)setActiveStatusText:(NSString *)status {
    _activeStatusText = [status copy];
    [self updateAppearance];
}

- (void)setInactiveStatusText:(NSString *)status {
    _inactiveStatusText = [status copy];
    [self updateAppearance];
}

@end

@interface AppDelegate () <NSWindowDelegate, WKNavigationDelegate, WKUIDelegate, NSMenuDelegate, NSURLSessionDownloadDelegate>
@property(strong, nonatomic) NSStatusItem *statusItem;
@property(assign, nonatomic) BOOL updateAvailable;
@property(assign, nonatomic) BOOL systemShutdownInProgress;
@property(strong, nonatomic) NSMenuItem *updateAvailableMenuItem;
@property(strong, nonatomic) NSMenuItem *restartMenuItem;
@property(assign, nonatomic) BOOL claudeAppEnabled;
@property(assign, nonatomic) BOOL claudeAppReady;
@property(strong, nonatomic) IntegrationMenuRow *claudeAppRow;
@property(strong, nonatomic) NSMenuItem *claudeMenuItem;
@property(strong, nonatomic) NSMenuItem *claudeMenuSeparatorItem;
@property(strong, nonatomic) NSURLSession *claudeDownloadSession;
@property(strong, nonatomic) NSURLSessionDownloadTask *claudeDownloadTask;
@property(strong, nonatomic) NSAlert *claudeDownloadAlert;
@property(strong, nonatomic) NSProgressIndicator *claudeDownloadProgress;
@property(strong, nonatomic) NSURL *claudeDownloadedInstallerURL;
@property(strong, nonatomic) NSError *claudeDownloadError;
@property(assign, nonatomic) BOOL claudeDownloadCancelled;
@property(assign, nonatomic) BOOL claudeDownloadCompleted;
@property(assign, nonatomic) BOOL claudeDownloadModalRunning;
@property(assign, nonatomic) BOOL quitInProgress;
@property(assign, nonatomic) BOOL systemTerminationReplyPending;
@property(strong, nonatomic) NSApplication *systemTerminationApplication;
- (void)openClaudeApp:(id)sender;
- (enum ClaudeInstallResult)downloadClaude;
- (enum ClaudeInstallResult)finishClaudeDownload;
- (void)showClaudeDownloadFailure:(NSError *)error;
- (void)showClaudeInstallFailure:(NSError *)error;
- (void)toggleClaudeAppProxy:(NSButton *)sender;
- (void)refreshClaudeAppState;
- (void)applyShowAppsInMenu:(BOOL)visible;
- (void)requestQuit;
- (void)completeSystemTermination;
@end

@implementation AppDelegate

bool showOnboarding,startHidden; // Set in run before initialization

static NSBundle *OllamaResourceBundle(void) {
    NSBundle *bundle = [NSBundle mainBundle];
    if ([bundle.bundlePath hasSuffix:@".app"]) {
        return bundle;
    }

    NSString *cwdPath = [[NSFileManager defaultManager] currentDirectoryPath];
    NSArray<NSString *> *bundlePaths = @[
        [cwdPath stringByAppendingPathComponent:@"darwin/Ollama.app"],
        [cwdPath stringByAppendingPathComponent:@"app/darwin/Ollama.app"],
    ];
    for (NSString *bundlePath in bundlePaths) {
        if ([[NSFileManager defaultManager] fileExistsAtPath:bundlePath]) {
            return [NSBundle bundleWithPath:bundlePath];
        }
    }

    return bundle;
}

static NSImage *ollamaApplicationIcon(void) {
    NSBundle *bundle = OllamaResourceBundle();
    NSString *iconPath = [bundle pathForResource:@"icon" ofType:@"icns"];
    if (iconPath == nil) {
        return [NSApp applicationIconImage];
    }
    return [[NSImage alloc] initWithContentsOfFile:iconPath];
}

- (void)application:(NSApplication *)application openURLs:(NSArray<NSURL *> *)urls {
    for (NSURL *url in urls) {
        if ([url.scheme isEqualToString:@"ollama"]) {
            NSString *path = url.path;

            if (path && ([path isEqualToString:@"/connect"] || [url.host isEqualToString:@"connect"])) {
                // Special case: handle connect by opening browser instead of app
                handleConnectURL();
            } else {
                [self openUI];
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
        NSImage *customIcon = ollamaApplicationIcon();
        if (customIcon != nil) {
            [NSApp setApplicationIconImage:customIcon];
        }
    }

    // Create status item and menu
    NSMenu *menu = [[NSMenu alloc] init];
    [menu setDelegate:self];
    [menu setAutoenablesItems:NO];

    self.claudeMenuItem = [[NSMenuItem alloc] initWithTitle:@"Claude"
                                                     action:nil
                                              keyEquivalent:@""];
    [self.claudeMenuItem setEnabled:YES];
    self.claudeAppRow = [[IntegrationMenuRow alloc]
        initWithTitle:@"Claude"
           symbolName:@"sparkles"
               target:self
           openAction:@selector(openClaudeApp:)
         toggleAction:@selector(toggleClaudeAppProxy:)];
    [self.claudeMenuItem setView:self.claudeAppRow];
    [menu addItem:self.claudeMenuItem];
    [self refreshClaudeAppState];
    self.claudeMenuSeparatorItem = [NSMenuItem separatorItem];
    [menu addItem:self.claudeMenuSeparatorItem];
    [self applyShowAppsInMenu:shouldShowAppsInMenu()];

    NSMenuItem *appsMenuItem =
        [[NSMenuItem alloc] initWithTitle:@"Open Ollama"
                                   action:@selector(appsUI)
                            keyEquivalent:@""];
    [appsMenuItem setTarget:self];
    [menu addItem:appsMenuItem];

    [menu addItemWithTitle:@"Settings"
                    action:@selector(settingsUI)
             keyEquivalent:@","];
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

    [menu addItem:[NSMenuItem separatorItem]];

    [menu addItemWithTitle:@"Quit Ollama"
                    action:@selector(requestQuit)
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
        } else if (showOnboarding) {
            StartUI("/");
        } else {
            StartUI("/connect");
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
    ShowUI();
    return YES;
}

- (void)menuWillOpen:(NSMenu *)menu {
    if (menu != self.statusItem.menu) {
        return;
    }
    [self refreshClaudeAppState];
}

- (void)refreshClaudeAppState {
    BOOL hasUsed = HasUsedClaudeDesktopIntegration();
    BOOL visible = shouldShowAppsInMenu() && hasUsed;
    [self.claudeMenuItem setHidden:!visible];
    [self.claudeMenuSeparatorItem setHidden:!visible];
    BOOL installed = IsClaudeDesktopInstalled();
    BOOL startFailed = ClaudeGatewayStartFailed();
    BOOL portConflict = startFailed && ClaudeGatewayPortConflict();
    BOOL configured = installed && IsClaudeGatewayConfigured();
    NSString *failureStatus = portConflict
        ? [NSString stringWithFormat:@"Port %d is in use", ClaudeGatewayPort()]
        : (startFailed ? @"Unable to use Ollama" : nil);
    self.claudeAppEnabled = configured;
    self.claudeAppReady = configured && !startFailed;
    [self.claudeAppRow setActiveStatusText:configured ? failureStatus : nil];
    [self.claudeAppRow setInactiveStatusText:configured
        ? nil
        : (!installed ? @"Not installed" : failureStatus)];
    [self.claudeAppRow setIntegrationActive:self.claudeAppEnabled];
    [self.claudeAppRow setIntegrationReady:self.claudeAppReady];
    RefreshClaudeProxyMenu();
}

- (void)applyShowAppsInMenu:(BOOL)visible {
    visible = visible && HasUsedClaudeDesktopIntegration();
    [self.claudeMenuItem setHidden:!visible];
    [self.claudeMenuSeparatorItem setHidden:!visible];
}

- (void)menuDidClose:(NSMenu *)menu {
    if (menu == self.statusItem.menu) {
        [self.claudeAppRow resetHover];
    }
}

- (void)showUpdateAvailable {
    self.updateAvailable = YES;
    [self.updateAvailableMenuItem setHidden:NO];
    [self.restartMenuItem setHidden:NO];
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
    [self uiRequest:@"/"];
}

- (void)openClaudeApp:(id)sender {
    (void)sender;
    if (!self.claudeAppReady) {
        return;
    }
    [self.statusItem.menu cancelTracking];

    NSArray<NSString *> *candidates = @[
        @"/Applications/Claude.app",
        [NSHomeDirectory() stringByAppendingPathComponent:@"Applications/Claude.app"],
    ];
    for (NSString *path in candidates) {
        if (![[NSFileManager defaultManager] fileExistsAtPath:path]) {
            continue;
        }
        NSWorkspaceOpenConfiguration *configuration =
            [NSWorkspaceOpenConfiguration configuration];
        configuration.activates = YES;
        configuration.createsNewApplicationInstance = NO;
        [[NSWorkspace sharedWorkspace]
            openApplicationAtURL:[NSURL fileURLWithPath:path]
                   configuration:configuration
               completionHandler:^(NSRunningApplication *application,
                                   NSError *error) {
                   (void)application;
                   if (error != nil) {
                       appLogInfo([NSString stringWithFormat:
                           @"Unable to open Claude: %@", error]);
                   }
               }];
        return;
    }

    NSAlert *alert = [[NSAlert alloc] init];
    [alert setAlertStyle:NSAlertStyleWarning];
    [alert setMessageText:@"Unable to open Claude"];
    [alert setInformativeText:@"Install Claude in Applications, then try again."];
    [alert runModal];
}

- (void)showClaudeDownloadFailure:(NSError *)error {
    appLogInfo([NSString stringWithFormat:@"Unable to download Claude: %@",
                                          error]);
    NSAlert *alert = [[NSAlert alloc] init];
    [alert setAlertStyle:NSAlertStyleWarning];
    [alert setIcon:ollamaApplicationIcon()];
    [alert setMessageText:@"Claude couldn’t be downloaded"];
    [alert setInformativeText:
        @"Try again or download Claude from its website."];
    [alert addButtonWithTitle:@"Open download page"];
    [alert addButtonWithTitle:@"Cancel"];
    if ([alert runModal] == NSAlertFirstButtonReturn) {
        [[NSWorkspace sharedWorkspace]
            openURL:[NSURL URLWithString:ClaudeDownloadPageURL]];
    }
}

- (void)showClaudeInstallFailure:(NSError *)error {
    appLogInfo([NSString stringWithFormat:@"Unable to install Claude: %@",
                                          error]);
    NSAlert *alert = [[NSAlert alloc] init];
    [alert setAlertStyle:NSAlertStyleWarning];
    [alert setIcon:ollamaApplicationIcon()];
    [alert setMessageText:@"Claude couldn’t be installed"];
    [alert setInformativeText:
        @"Try again or install Claude from its website."];
    [alert addButtonWithTitle:@"Open download page"];
    [alert addButtonWithTitle:@"Cancel"];
    if ([alert runModal] == NSAlertFirstButtonReturn) {
        [[NSWorkspace sharedWorkspace]
            openURL:[NSURL URLWithString:ClaudeDownloadPageURL]];
    }
}

- (enum ClaudeInstallResult)downloadClaude {
    if (self.claudeDownloadTask != nil) {
        return ClaudeInstallCancelled;
    }

    char *downloadAuthorization = NULL;
    char *downloadURL = ClaudeDesktopDownloadRequest(&downloadAuthorization);
    NSString *downloadURLString = downloadURL == NULL
        ? nil
        : [NSString stringWithUTF8String:downloadURL];
    NSString *authorization = downloadAuthorization == NULL
        ? nil
        : [NSString stringWithUTF8String:downloadAuthorization];
    free(downloadURL);
    free(downloadAuthorization);
    NSURL *url = downloadURLString == nil
        ? nil
        : [NSURL URLWithString:downloadURLString];
    if (url == nil || authorization.length == 0) {
        NSError *error = [NSError
            errorWithDomain:@"com.ollama.app"
                       code:3
                   userInfo:@{NSLocalizedDescriptionKey:
                       @"Ollama could not authenticate the download request."}];
        [self showClaudeDownloadFailure:error];
        return ClaudeInstallFailed;
    }

    [self.claudeAppRow setInactiveStatusText:@"Downloading Claude…"];
    [self.claudeAppRow.integrationSwitch setEnabled:NO];

    self.claudeDownloadAlert = [[NSAlert alloc] init];
    [self.claudeDownloadAlert setAlertStyle:NSAlertStyleInformational];
    [self.claudeDownloadAlert setIcon:ollamaApplicationIcon()];
    [self.claudeDownloadAlert setMessageText:@"Downloading Claude"];
    [self.claudeDownloadAlert setInformativeText:
        @"Claude will be installed when the download finishes."];
    [self.claudeDownloadAlert addButtonWithTitle:@"Cancel"];
    self.claudeDownloadProgress = [[NSProgressIndicator alloc]
        initWithFrame:NSMakeRect(0, 0, 260, 12)];
    [self.claudeDownloadProgress setStyle:NSProgressIndicatorStyleBar];
    [self.claudeDownloadProgress setMinValue:0];
    [self.claudeDownloadProgress setMaxValue:100];
    [self.claudeDownloadProgress setIndeterminate:YES];
    [self.claudeDownloadProgress startAnimation:nil];
    [self.claudeDownloadAlert setAccessoryView:self.claudeDownloadProgress];

    self.claudeDownloadCancelled = NO;
    self.claudeDownloadCompleted = NO;
    self.claudeDownloadedInstallerURL = nil;
    self.claudeDownloadError = nil;
    NSURLSessionConfiguration *configuration =
        [NSURLSessionConfiguration ephemeralSessionConfiguration];
    self.claudeDownloadSession = [NSURLSession
        sessionWithConfiguration:configuration
                        delegate:self
                   delegateQueue:[NSOperationQueue mainQueue]];
    NSMutableURLRequest *request = [NSMutableURLRequest requestWithURL:url];
    [request setValue:authorization forHTTPHeaderField:@"Authorization"];
    self.claudeDownloadTask = [self.claudeDownloadSession
        downloadTaskWithRequest:request];

    [NSApp activateIgnoringOtherApps:YES];
    [self.claudeDownloadTask resume];
    self.claudeDownloadModalRunning = YES;
    NSModalResponse response = [self.claudeDownloadAlert runModal];
    self.claudeDownloadModalRunning = NO;
    if (response == NSAlertFirstButtonReturn &&
        !self.claudeDownloadCompleted) {
        self.claudeDownloadCancelled = YES;
        [self.claudeDownloadTask cancel];
        return ClaudeInstallCancelled;
    }
    if (self.claudeDownloadCompleted) {
        return [self finishClaudeDownload];
    }
    return ClaudeInstallCancelled;
}

- (void)URLSession:(NSURLSession *)session
              task:(NSURLSessionTask *)task
willPerformHTTPRedirection:(NSHTTPURLResponse *)response
        newRequest:(NSURLRequest *)request
  completionHandler:(void (^)(NSURLRequest * _Nullable))completionHandler {
    (void)response;
    if (session != self.claudeDownloadSession ||
        task != self.claudeDownloadTask) {
        completionHandler(request);
        return;
    }
    NSMutableURLRequest *redirect = [request mutableCopy];
    [redirect setValue:nil forHTTPHeaderField:@"Authorization"];
    completionHandler(redirect);
}

- (void)URLSession:(NSURLSession *)session
      downloadTask:(NSURLSessionDownloadTask *)downloadTask
      didWriteData:(int64_t)bytesWritten
 totalBytesWritten:(int64_t)totalBytesWritten
totalBytesExpectedToWrite:(int64_t)totalBytesExpectedToWrite {
    (void)session;
    (void)bytesWritten;
    if (downloadTask != self.claudeDownloadTask ||
        totalBytesExpectedToWrite <= 0) {
        return;
    }
    if (self.claudeDownloadProgress.indeterminate) {
        [self.claudeDownloadProgress stopAnimation:nil];
        [self.claudeDownloadProgress setIndeterminate:NO];
    }
    [self.claudeDownloadProgress
        setDoubleValue:(100.0 * totalBytesWritten) /
                       totalBytesExpectedToWrite];
}

- (void)URLSession:(NSURLSession *)session
      downloadTask:(NSURLSessionDownloadTask *)downloadTask
didFinishDownloadingToURL:(NSURL *)location {
    (void)session;
    if (downloadTask != self.claudeDownloadTask ||
        self.claudeDownloadCancelled) {
        return;
    }

    NSError *error = nil;
    NSHTTPURLResponse *response =
        [downloadTask.response isKindOfClass:[NSHTTPURLResponse class]]
            ? (NSHTTPURLResponse *)downloadTask.response
            : nil;
    NSString *host = response.URL.host.lowercaseString;
    BOOL trustedHost = [host isEqualToString:@"claude.ai"] ||
                       [host hasSuffix:@".claude.ai"];
    if (response.statusCode != 200 || !trustedHost ||
        ![response.URL.scheme isEqualToString:@"https"]) {
        error = [NSError
            errorWithDomain:@"com.ollama.app"
                       code:1
                   userInfo:@{NSLocalizedDescriptionKey:
                       @"Claude returned an invalid download response."}];
    }

    if (error == nil) {
        NSDictionary *attributes = [[NSFileManager defaultManager]
            attributesOfItemAtPath:location.path
                             error:&error];
        if (error == nil && [attributes fileSize] < 1024 * 1024) {
            error = [NSError
                errorWithDomain:@"com.ollama.app"
                           code:2
                       userInfo:@{NSLocalizedDescriptionKey:
                           @"Claude returned an incomplete download."}];
        }
    }

    if (error == nil) {
        NSString *fileName = [NSString
            stringWithFormat:@"Claude-%@.zip", [NSUUID UUID].UUIDString];
        NSURL *installerURL = [NSURL fileURLWithPath:
            [NSTemporaryDirectory() stringByAppendingPathComponent:fileName]];
        if ([[NSFileManager defaultManager] moveItemAtURL:location
                                                   toURL:installerURL
                                                   error:&error]) {
            self.claudeDownloadedInstallerURL = installerURL;
        }
    }
    self.claudeDownloadError = error;
}

- (void)URLSession:(NSURLSession *)session
              task:(NSURLSessionTask *)task
didCompleteWithError:(NSError *)error {
    if (session != self.claudeDownloadSession ||
        task != self.claudeDownloadTask) {
        return;
    }
    if (error != nil && !self.claudeDownloadCancelled) {
        self.claudeDownloadError = error;
    }
    self.claudeDownloadCompleted = YES;
    if (self.claudeDownloadModalRunning) {
        [NSApp abortModal];
        return;
    }
    [self finishClaudeDownload];
}

- (enum ClaudeInstallResult)finishClaudeDownload {
    BOOL cancelled = self.claudeDownloadCancelled;

    [self.claudeDownloadProgress stopAnimation:nil];
    [self.claudeDownloadAlert.window orderOut:nil];
    [self.claudeAppRow.integrationSwitch setEnabled:YES];
    [self.claudeAppRow setInactiveStatusText:@"Not installed"];
    [self.claudeDownloadSession finishTasksAndInvalidate];
    self.claudeDownloadTask = nil;
    self.claudeDownloadSession = nil;
    self.claudeDownloadProgress = nil;
    self.claudeDownloadAlert = nil;

    if (cancelled) {
        self.claudeDownloadError = nil;
        self.claudeDownloadedInstallerURL = nil;
        self.claudeDownloadCancelled = NO;
        self.claudeDownloadCompleted = NO;
        return ClaudeInstallCancelled;
    }
    if (self.claudeDownloadError != nil ||
        self.claudeDownloadedInstallerURL == nil) {
        if (self.claudeDownloadError == nil) {
            self.claudeDownloadError = [NSError
                errorWithDomain:@"com.ollama.app"
                           code:3
                       userInfo:@{NSLocalizedDescriptionKey:
                           @"The Claude installer was not downloaded."}];
        }
        [self showClaudeDownloadFailure:self.claudeDownloadError];
        self.claudeDownloadError = nil;
        self.claudeDownloadedInstallerURL = nil;
        self.claudeDownloadCancelled = NO;
        self.claudeDownloadCompleted = NO;
        return ClaudeInstallFailed;
    }
    [self.claudeAppRow setInactiveStatusText:@"Installing Claude…"];
    BOOL installed = InstallClaudeDesktopArchive(
        self.claudeDownloadedInstallerURL.fileSystemRepresentation);
    NSError *removeError = nil;
    [[NSFileManager defaultManager]
        removeItemAtURL:self.claudeDownloadedInstallerURL
                  error:&removeError];
    if (removeError != nil) {
        appLogInfo([NSString stringWithFormat:
            @"Unable to remove downloaded Claude archive: %@", removeError]);
    }
    if (!installed) {
        NSError *installError = [NSError
            errorWithDomain:@"com.ollama.app"
                       code:4
                   userInfo:@{NSLocalizedDescriptionKey:
                       @"Claude could not be installed."}];
        [self showClaudeInstallFailure:installError];
    }
    self.claudeDownloadError = nil;
    self.claudeDownloadedInstallerURL = nil;
    self.claudeDownloadCancelled = NO;
    self.claudeDownloadCompleted = NO;
    [self refreshClaudeAppState];
    return installed ? ClaudeInstallerOpened : ClaudeInstallFailed;
}

- (void)toggleClaudeAppProxy:(NSButton *)sender {
    BOOL enabled = sender.state == NSControlStateValueOn;
    if (enabled && !IsClaudeDesktopInstalled()) {
        [self.claudeAppRow setInactiveStatusText:@"Not installed"];
        [self.claudeAppRow setIntegrationActive:NO];

        NSAlert *installAlert = [[NSAlert alloc] init];
        [installAlert setAlertStyle:NSAlertStyleInformational];
        [installAlert setIcon:ollamaApplicationIcon()];
        [installAlert setMessageText:@"Claude is not installed"];
        [installAlert setInformativeText:
            @"Download Claude to add Ollama models to the Claude app."];
        [installAlert addButtonWithTitle:@"Download Claude"];
        [installAlert addButtonWithTitle:@"Cancel"];
        if ([installAlert runModal] == NSAlertFirstButtonReturn) {
            if ([self downloadClaude] != ClaudeInstallerOpened) {
                return;
            }
        } else {
            return;
        }
    }
    BOOL restartClaude = IsClaudeDesktopRunning();
    if (restartClaude) {
        NSAlert *restartAlert = [[NSAlert alloc] init];
        [restartAlert setAlertStyle:NSAlertStyleWarning];
        [restartAlert setIcon:ollamaApplicationIcon()];
        [restartAlert setMessageText:enabled
            ? @"Restart Claude Desktop to use Ollama?"
            : @"Restart Claude Desktop to remove Ollama?"];
        [restartAlert setInformativeText:enabled
            ? @"Claude Desktop must restart to use Ollama. Any running task will stop."
            : @"Claude Desktop must restart to remove Ollama. Any running task will stop."];
        [restartAlert addButtonWithTitle:@"Restart Claude Desktop"];
        [restartAlert addButtonWithTitle:@"Cancel"];
        if ([restartAlert runModal] != NSAlertFirstButtonReturn) {
            [self refreshClaudeAppState];
            return;
        }
    }

    [sender setEnabled:NO];
    dispatch_async(dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^{
        BOOL succeeded = SetClaudeGatewayInstalled(enabled, restartClaude);
        dispatch_async(dispatch_get_main_queue(), ^{
            [sender setEnabled:YES];
            if (!succeeded) {
                BOOL portConflict = enabled && ClaudeGatewayPortConflict();
                char *rawGatewayError = ClaudeGatewayErrorMessage();
                NSString *gatewayError = rawGatewayError != NULL && rawGatewayError[0] != '\0'
                    ? [NSString stringWithUTF8String:rawGatewayError]
                    : nil;
                free(rawGatewayError);
                [self refreshClaudeAppState];
                NSAlert *alert = [[NSAlert alloc] init];
                [alert setAlertStyle:NSAlertStyleWarning];
                [alert setIcon:ollamaApplicationIcon()];
                [alert setMessageText:portConflict
                    ? [NSString stringWithFormat:@"Port %d is already in use", ClaudeGatewayPort()]
                    : (enabled
                        ? @"Unable to use Ollama with Claude"
                        : @"Unable to remove Ollama from Claude")];
                [alert setInformativeText:portConflict
                    ? [NSString stringWithFormat:@"Change OLLAMA_HOST or quit the app using port %d, then try again.", ClaudeGatewayPort()]
                    : (gatewayError.length > 0
                        ? gatewayError
                        : @"Ollama could not update Claude. Check the Ollama log for details.")];
                [alert runModal];
                return;
            }

            self.claudeAppEnabled = enabled;
            self.claudeAppReady = enabled;
            [self.claudeAppRow setActiveStatusText:nil];
            [self.claudeAppRow setInactiveStatusText:nil];
            [self.claudeAppRow setIntegrationActive:enabled];
            [self.claudeAppRow setIntegrationReady:enabled];
            if (enabled) {
                RefreshClaudeProxyMenu();
                [self openClaudeApp:nil];
            }
        });
    });
}

- (void)appsUI {
    [self uiRequest:@"/connect"];
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
    [NSApp hide:nil];
    [NSApp setActivationPolicy:NSApplicationActivationPolicyAccessory];
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
    statusImage = [OllamaResourceBundle() imageForResource:iconName];
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
    [self showIcon];
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
        [alert setIcon:ollamaApplicationIcon()];
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
            [self refreshClaudeAppState];
            NSAlert *alert = [[NSAlert alloc] init];
            [alert setAlertStyle:NSAlertStyleWarning];
            [alert setIcon:ollamaApplicationIcon()];
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

- (void)webView:(WKWebView *)webView
    runJavaScriptConfirmPanelWithMessage:(NSString *)message
    initiatedByFrame:(WKFrameInfo *)frame
    completionHandler:(void (^)(BOOL))completionHandler {

    NSAlert *alert = [[NSAlert alloc] init];
    [alert setAlertStyle:NSAlertStyleWarning];
    [alert setIcon:ollamaApplicationIcon()];

    if ([message isEqualToString:@"Restart Claude Desktop to use Ollama? Any running task will stop."]) {
        [alert setMessageText:@"Restart Claude Desktop to use Ollama?"];
        [alert setInformativeText:
            @"Claude Desktop must restart to use Ollama. Any running task will stop."];
        [alert addButtonWithTitle:@"Restart Claude Desktop"];
    } else if ([message isEqualToString:@"Restart Claude Desktop to remove Ollama? Any running task will stop."]) {
        [alert setMessageText:@"Restart Claude Desktop to remove Ollama?"];
        [alert setInformativeText:
            @"Claude Desktop must restart to remove Ollama. Any running task will stop."];
        [alert addButtonWithTitle:@"Restart Claude Desktop"];
    } else {
        [alert setMessageText:message];
        [alert addButtonWithTitle:@"Confirm"];
    }
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
void run(bool so, bool sh) {
    [NSApplication sharedApplication];
    [NSApp setActivationPolicy:NSApplicationActivationPolicyAccessory];
    appDelegate = [[AppDelegate alloc] init];
    [NSApp setDelegate:appDelegate];
    showOnboarding = so;
    startHidden = sh;
    [NSApp run];
    StopUI();
}

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

void updateClaudeProxyMenu(unsigned long long routed) {
    void (^updateMenu)(void) = ^{
        NSString *status = routed == 1
            ? @"1 request this session"
            : [NSString stringWithFormat:@"%llu requests this session", routed];
        [appDelegate.claudeAppRow setActiveStatusText:status];
    };
    if ([NSThread isMainThread]) {
        updateMenu();
    } else {
        dispatch_async(dispatch_get_main_queue(), updateMenu);
    }
}

bool ShowAppsInMenu(void) {
    return shouldShowAppsInMenu();
}

void SetShowAppsInMenu(bool visible) {
    [[NSUserDefaults standardUserDefaults]
        setBool:visible
         forKey:ShowAppsInMenuDefaultsKey];
    dispatch_async(dispatch_get_main_queue(), ^{
        [appDelegate applyShowAppsInMenu:visible];
    });
}

enum ClaudeInstallResult installClaudeDesktop(void) {
    __block enum ClaudeInstallResult result = ClaudeInstallFailed;
    void (^install)(void) = ^{
        if (appDelegate != nil) {
            result = [appDelegate downloadClaude];
        }
    };
    if ([NSThread isMainThread]) {
        install();
    } else {
        dispatch_sync(dispatch_get_main_queue(), install);
    }
    return result;
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

void setWindowResizable(uintptr_t wndPtr, bool resizable) {
    NSWindow *w = (__bridge NSWindow *)wndPtr;
    if (!w) return;

    if (resizable) {
        w.styleMask |= NSWindowStyleMaskResizable;
    } else {
        w.styleMask &= ~NSWindowStyleMaskResizable;
    }
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
