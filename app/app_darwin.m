#import <AppKit/AppKit.h>
#import <Cocoa/Cocoa.h>
#import <CoreServices/CoreServices.h>
#import <Security/Security.h>
#import <ServiceManagement/ServiceManagement.h>
#import "app_darwin.h"

@interface AppDelegate ()

@property (strong, nonatomic) NSStatusItem *statusItem;

@end

@implementation AppDelegate

- (void)applicationDidFinishLaunching:(NSNotification *)aNotification {
    // show status menu
    NSMenu *menu = [[NSMenu alloc] init];

    NSMenuItem *aboutMenuItem = [[NSMenuItem alloc] initWithTitle:@"About Ollama" action:@selector(aboutOllama) keyEquivalent:@""];
    [aboutMenuItem setTarget:self];
    [menu addItem:aboutMenuItem];

    // Settings submenu
    NSMenu *settingsMenu = [[NSMenu alloc] initWithTitle:@"Settings"];

    // Submenu items
    NSMenuItem *chooseModelDirectoryItem = [[NSMenuItem alloc] initWithTitle:@"Choose model directory..." action:@selector(chooseModelDirectory) keyEquivalent:@""];
    [chooseModelDirectoryItem setTarget:self];
    [chooseModelDirectoryItem setEnabled:YES];
    [settingsMenu addItem:chooseModelDirectoryItem];

    NSMenuItem *exposeExternallyItem = [[NSMenuItem alloc] initWithTitle:@"Allow external connections" action:@selector(toggleExposeExternally:) keyEquivalent:@""];
    [exposeExternallyItem setTarget:self];
    [exposeExternallyItem setState:NSOffState]; // Set initial state to off
    [exposeExternallyItem setEnabled:YES];
    [settingsMenu addItem:exposeExternallyItem];

    NSMenuItem *allowCrossOriginItem = [[NSMenuItem alloc] initWithTitle:@"Allow browser requests" action:@selector(toggleCrossOrigin:) keyEquivalent:@""];
    [allowCrossOriginItem setTarget:self];
    [allowCrossOriginItem setState:NSOffState]; // Set initial state to off
    [allowCrossOriginItem setEnabled:YES];
    [settingsMenu addItem:allowCrossOriginItem];

    NSMenuItem *settingsMenuItem = [[NSMenuItem alloc] initWithTitle:@"Settings" action:nil keyEquivalent:@""];
    [settingsMenuItem setSubmenu:settingsMenu];
    [menu addItem:settingsMenuItem];

    [menu addItemWithTitle:@"Quit Ollama" action:@selector(quit) keyEquivalent:@"q"];

    self.statusItem = [[NSStatusBar systemStatusBar] statusItemWithLength:NSVariableStatusItemLength];
    [self.statusItem addObserver:self forKeyPath:@"button.effectiveAppearance" options:NSKeyValueObservingOptionNew|NSKeyValueObservingOptionInitial context:nil];

    self.statusItem.menu = menu;
    [self showIcon];
}

- (void)aboutOllama {
    [[NSApplication sharedApplication] orderFrontStandardAboutPanel:nil];
}

- (void)toggleCrossOrigin:(id)sender {
    NSMenuItem *item = (NSMenuItem *)sender;
    if ([item state] == NSOffState) {
        // Do something when cross-origin requests are allowed
        [item setState:NSOnState];
    } else {
        // Do something when cross-origin requests are disallowed
        [item setState:NSOffState];
    }
}

- (void)toggleExposeExternally:(id)sender {
    NSMenuItem *item = (NSMenuItem *)sender;
    if ([item state] == NSOffState) {
        // Do something when Ollama is exposed externally
        [item setState:NSOnState];
    } else {
        // Do something when Ollama is not exposed externally
        [item setState:NSOffState];
    }
}

- (void)chooseModelDirectory {
    NSOpenPanel *openPanel = [NSOpenPanel openPanel];
    [openPanel setCanChooseFiles:NO];
    [openPanel setCanChooseDirectories:YES];
    [openPanel setAllowsMultipleSelection:NO];

    NSInteger result = [openPanel runModal];
    if (result == NSModalResponseOK) {
        NSURL *selectedDirectoryURL = [openPanel URLs].firstObject;
        // Do something with the selected directory URL
    }
}

-(void) showIcon {
    NSAppearance* appearance = self.statusItem.button.effectiveAppearance;
    NSString* appearanceName = (NSString*)(appearance.name);
    NSString* iconName = [[appearanceName lowercaseString] containsString:@"dark"] ? @"iconDark" : @"icon";
    NSImage* statusImage = [NSImage imageNamed:iconName];
    [statusImage setTemplate:YES];
    self.statusItem.button.image = statusImage;
}

-(void)observeValueForKeyPath:(NSString *)keyPath ofObject:(id)object change:(NSDictionary<NSKeyValueChangeKey,id> *)change context:(void *)context {
    [self showIcon];
}

- (void)quit {
    [NSApp stop:nil];
}

@end

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

    NSDate *startTime = [NSDate date];
    for (NSRunningApplication *app in apps) {
        while (!app.terminated) {
            if (-[startTime timeIntervalSinceNow] >= 5) {
                kill(app.processIdentifier, SIGKILL);
                break;
            }

            [[NSRunLoop currentRunLoop] runUntilDate:[NSDate dateWithTimeIntervalSinceNow:0.1]];
        }
    }
}

bool askToMoveToApplications() {
    NSString *bundlePath = [[NSBundle mainBundle] bundlePath];
    if ([bundlePath hasPrefix:@"/Applications"]) {
        return false;
    }

    NSAlert *alert = [[NSAlert alloc] init];
    [alert setMessageText:@"Move to Applications?"];
    [alert setInformativeText:@"Ollama works best when run from the Applications directory."];
    [alert addButtonWithTitle:@"Move to Applications"];
    [alert addButtonWithTitle:@"Don't move"];

    [NSApp activateIgnoringOtherApps:YES];

    if ([alert runModal] != NSAlertFirstButtonReturn) {
        return false;
    }

    // move to applications
    NSString *applicationsPath = @"/Applications";
    NSString *newPath = [applicationsPath stringByAppendingPathComponent:@"Ollama.app"];
    NSFileManager *fileManager = [NSFileManager defaultManager];

    // Check if the newPath already exists
    if ([fileManager fileExistsAtPath:newPath]) {
        NSError *removeError = nil;
        [fileManager removeItemAtPath:newPath error:&removeError];
        if (removeError) {
            NSLog(@"Error removing file at %@: %@", newPath, removeError);
            return false; // or handle the error
        }
    }

    NSError *moveError = nil;
    [fileManager moveItemAtPath:bundlePath toPath:newPath error:&moveError];
    if (moveError) {
        NSLog(@"Error moving file from %@ to %@: %@", bundlePath, newPath, moveError);
        return false;
    }

    NSLog(@"Opening %@", newPath);
    NSError *error = nil;
    NSWorkspace *workspace = [NSWorkspace sharedWorkspace];
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
    [workspace launchApplicationAtURL:[NSURL fileURLWithPath:newPath]
               options:NSWorkspaceLaunchNewInstance | NSWorkspaceLaunchDefault
               configuration:@{}
               error:&error];

    return true;
}

int installSymlink() {
    NSString *linkPath = @"/usr/local/bin/ollama";
    NSError *error = nil;

    NSFileManager *fileManager = [NSFileManager defaultManager];
    NSString *symlinkPath = [fileManager destinationOfSymbolicLinkAtPath:linkPath error:&error];
    NSString *bundlePath = [[NSBundle mainBundle] bundlePath];
    NSString *execPath = [[NSBundle mainBundle] executablePath];
    NSString *resPath = [[NSBundle mainBundle] pathForResource:@"ollama" ofType:nil];

    // if the symlink already exists and points to the right place, don't prompt
    if ([symlinkPath isEqualToString:resPath]) {
        NSLog(@"symbolic link already exists and points to the right place");
        return 0;
    }

    NSString *authorizationPrompt = @"Ollama is trying to install its command line interface (CLI) tool.";

    AuthorizationRef auth = NULL;
    OSStatus createStatus = AuthorizationCreate(NULL, kAuthorizationEmptyEnvironment, kAuthorizationFlagDefaults, &auth);
    if (createStatus != errAuthorizationSuccess) {
        NSLog(@"Error creating authorization");
        return -1;
    }

    NSString * bundleIdentifier = [[NSBundle mainBundle] bundleIdentifier];
    NSString *rightNameString = [NSString stringWithFormat:@"%@.%@", bundleIdentifier, @"auth3"];
    const char *rightName = rightNameString.UTF8String;

    OSStatus getRightResult = AuthorizationRightGet(rightName, NULL);
    if (getRightResult == errAuthorizationDenied) {
        if (AuthorizationRightSet(auth, rightName, (__bridge CFTypeRef _Nonnull)(@(kAuthorizationRuleAuthenticateAsAdmin)), (__bridge CFStringRef _Nullable)(authorizationPrompt), NULL, NULL) != errAuthorizationSuccess) {
            NSLog(@"Failed to set right");
            return -1;
        }
    }

    AuthorizationItem right = { .name = rightName, .valueLength = 0, .value = NULL, .flags = 0 };
    AuthorizationRights rights = { .count = 1, .items = &right };
    AuthorizationFlags flags = (AuthorizationFlags)(kAuthorizationFlagExtendRights | kAuthorizationFlagInteractionAllowed);
    AuthorizationItem iconAuthorizationItem = {.name = kAuthorizationEnvironmentIcon, .valueLength = 0, .value = NULL, .flags = 0};
    AuthorizationEnvironment authorizationEnvironment = {.count = 0, .items = NULL};

    BOOL failedToUseSystemDomain = NO;
    OSStatus copyStatus = AuthorizationCopyRights(auth, &rights, &authorizationEnvironment, flags, NULL);
    if (copyStatus != errAuthorizationSuccess) {
        failedToUseSystemDomain = YES;

        if (copyStatus == errAuthorizationCanceled) {
            NSLog(@"User cancelled authorization");
            return -1;
        } else {
            NSLog(@"Failed copying system domain rights: %d", copyStatus);
            return -1;
        }
    }

    const char *toolPath = "/bin/ln";
    const char *args[] = {"-s", "-F", [resPath UTF8String], "/usr/local/bin/ollama", NULL};
    FILE *pipe = NULL;

#pragma clang diagnostic ignored "-Wdeprecated-declarations"
    OSStatus status = AuthorizationExecuteWithPrivileges(auth, toolPath, kAuthorizationFlagDefaults, (char *const *)args, &pipe);
    if (status != errAuthorizationSuccess) {
        NSLog(@"Failed to create symlink");
        return -1;
    }

    AuthorizationFree(auth, kAuthorizationFlagDestroyRights);
    return 0;
}
