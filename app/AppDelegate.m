#import <CoreServices/CoreServices.h>
#import <AppKit/AppKit.h>
#import <Security/Security.h>
#import "AppDelegate.h"
#import "app_darwin.h"

@interface AppDelegate () <NSToolbarDelegate>

@property (strong, nonatomic) NSStatusItem *statusItem;
@property (strong) NSWindow *settingsWindow;

@end

@implementation AppDelegate

- (void)applicationDidFinishLaunching:(NSNotification *)aNotification {
    // Ask to move to applications directory
    askToMoveToApplications();

    // Once in the desired directory, offer to create a symlink
    // TODO (jmorganca): find a way to provide more context to the
    // user about what this is doing, and ideally use Touch ID.
    // or add an alias in the current shell environment,
    // which wouldn't require any special privileges
    // dispatch_async(dispatch_get_main_queue(), ^{
    //     createSymlinkWithAuthorization();
    // });

    // show status menu
    NSMenu *menu = [[NSMenu alloc] init];
    [menu addItemWithTitle:@"Quit Ollama" action:@selector(quit) keyEquivalent:@"q"];
    self.statusItem = [[NSStatusBar systemStatusBar] statusItemWithLength:NSVariableStatusItemLength];
    [self.statusItem addObserver:self forKeyPath:@"button.effectiveAppearance" options:NSKeyValueObservingOptionNew|NSKeyValueObservingOptionInitial context:nil];

    self.statusItem.menu = menu;
    [self showIcon];
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

- (void)openSettingsWindow {
    if (!self.settingsWindow) {
        // Create the settings window centered on the screen
        self.settingsWindow = [[NSWindow alloc] initWithContentRect:NSMakeRect(0, 0, 420, 460)
                                                         styleMask:(NSWindowStyleMaskTitled | NSClosableWindowMask | NSWindowStyleMaskFullSizeContentView)
                                                           backing:NSBackingStoreBuffered
                                                             defer:NO];
        [self.settingsWindow setTitle:@"Settings"];
        [self.settingsWindow makeKeyAndOrderFront:nil];
        [self.settingsWindow center];

        // Create and configure the toolbar
        NSToolbar *toolbar = [[NSToolbar alloc] initWithIdentifier:@"SettingsToolbar"];
        toolbar.delegate = self;
        // toolbar.showsBaselineSeparator
        toolbar.displayMode = NSToolbarDisplayModeIconAndLabel;
        self.settingsWindow.toolbar = toolbar;
        self.settingsWindow.toolbarStyle = NSWindowToolbarStylePreference;

        // Necessary to make the toolbar display immediately
        [self.settingsWindow makeKeyAndOrderFront:nil];
    } else {
        [self.settingsWindow makeKeyAndOrderFront:nil];
    }
}

- (void)quit {
    [NSApp stop:nil];
}

@end

int askToMoveToApplications() {
    NSString *bundlePath = [[NSBundle mainBundle] bundlePath];
    if ([bundlePath hasPrefix:@"/Applications"]) {
        return 0;
    }

    NSAlert *alert = [[NSAlert alloc] init];
    [alert setMessageText:@"Move to Applications?"];
    [alert setInformativeText:@"Ollama works best when run from the Applications directory."];
    [alert addButtonWithTitle:@"Move to Applications"];
    [alert addButtonWithTitle:@"Don't move"];

    [NSApp activateIgnoringOtherApps:YES];

    if ([alert runModal] != NSAlertFirstButtonReturn) {
        return 0;
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
            return -1; // or handle the error
        }
    }

    NSError *moveError = nil;
    [fileManager moveItemAtPath:bundlePath toPath:newPath error:&moveError];
    if (moveError) {
        NSLog(@"Error moving file from %@ to %@: %@", bundlePath, newPath, moveError);
        return -1; // or handle the error
    }

    NSLog(@"Opening %@", newPath);
    NSError *error = nil;
    NSWorkspace *workspace = [NSWorkspace sharedWorkspace];
    [workspace launchApplicationAtURL:[NSURL fileURLWithPath:newPath]
                                options:NSWorkspaceLaunchNewInstance | NSWorkspaceLaunchDefault
                        configuration:@{}
                                error:&error];
    return 0;
}

int createSymlinkWithAuthorization() {
    NSString *linkPath = @"/usr/local/bin/ollama";
    NSError *error = nil;

    NSFileManager *fileManager = [NSFileManager defaultManager];
    NSString *symlinkPath = [fileManager destinationOfSymbolicLinkAtPath:linkPath error:&error];
    NSString *bundlePath = [[NSBundle mainBundle] bundlePath];
    NSString *execPath = [[NSBundle mainBundle] executablePath];
    NSString *resPath = [[NSBundle mainBundle] pathForResource:@"ollama" ofType:nil];

    // if the symlink already exists and points to the right place, don't prompt
    if ([symlinkPath isEqualToString:resPath]) {
        return 0;
    }

    OSStatus status;
    AuthorizationRef authorizationRef;
    status = AuthorizationCreate(NULL, kAuthorizationEmptyEnvironment, kAuthorizationFlagDefaults, &authorizationRef);
    if (status != errAuthorizationSuccess) {
        NSLog(@"Failed to create authorization");
        return -1;
    }

    const char *toolPath = "/bin/ln";
    const char *args[] = {"-s", "-F", [resPath UTF8String], "/usr/local/bin/ollama", NULL};
    FILE *pipe = NULL;

    status = AuthorizationExecuteWithPrivileges(authorizationRef, toolPath, kAuthorizationFlagDefaults, (char *const *)args, &pipe);
    if (status != errAuthorizationSuccess) {
        NSLog(@"Failed to create symlink");
        return -1;
    }

    AuthorizationFree(authorizationRef, kAuthorizationFlagDestroyRights);

    return 0;
}
