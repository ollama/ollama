#import "settings_pane_darwin.h"
#import "resources_darwin.h"

const CGFloat OLSettingsPaneWidth = 640;
const CGFloat OLSettingsContentWidth = 580;

// The space between a pane's content and the window's edges.
static const CGFloat OLPaneTopMargin = 26;
static const CGFloat OLPaneSideMargin = 30;
static const CGFloat OLPaneBottomMargin = 20;

@interface OLSettingsPane ()
@property(nonatomic, readwrite, copy) NSString *paneIdentifier;
@property(nonatomic, readwrite, copy) NSString *paneTitle;
@property(nonatomic, readwrite, strong) NSImage *paneImage;
@property(nonatomic, readwrite, nullable) NSURL *helpURL;
@end

@implementation OLSettingsPane

- (instancetype)initWithIdentifier:(NSString *)identifier
                             title:(NSString *)title
                             image:(NSImage *)image
                           helpURL:(NSURL *)helpURL {
    self = [super initWithNibName:nil bundle:nil];
    if (self) {
        _paneIdentifier = [identifier copy];
        _paneTitle = [title copy];
        _paneImage = image;
        _helpURL = helpURL;
        self.title = title;
    }
    return self;
}

- (void)loadView {
    NSView *root = [[NSView alloc] initWithFrame:NSMakeRect(0, 0, OLSettingsPaneWidth, 200)];
    root.translatesAutoresizingMaskIntoConstraints = NO;

    NSView *content = [self makeContentView];
    content.translatesAutoresizingMaskIntoConstraints = NO;
    [root addSubview:content];

    NSMutableArray<NSLayoutConstraint *> *constraints = [@[
        [root.widthAnchor constraintEqualToConstant:OLSettingsPaneWidth],
        [content.topAnchor constraintEqualToAnchor:root.topAnchor constant:OLPaneTopMargin],
        [content.leadingAnchor constraintEqualToAnchor:root.leadingAnchor constant:OLPaneSideMargin],
        [content.trailingAnchor constraintEqualToAnchor:root.trailingAnchor constant:-OLPaneSideMargin],
    ] mutableCopy];

    if (self.helpURL != nil) {
        NSButton *help = [[NSButton alloc] initWithFrame:NSZeroRect];
        help.bezelStyle = NSBezelStyleHelpButton;
        help.title = @"";
        help.target = self;
        help.action = @selector(openHelp:);
        help.toolTip = [NSString stringWithFormat:@"%@ Help", self.paneTitle];
        help.translatesAutoresizingMaskIntoConstraints = NO;
        [help setAccessibilityLabel:help.toolTip];
        [root addSubview:help];
        [constraints addObjectsFromArray:@[
            [help.topAnchor constraintEqualToAnchor:content.bottomAnchor constant:12],
            [help.trailingAnchor constraintEqualToAnchor:root.trailingAnchor constant:-OLPaneBottomMargin],
            [help.bottomAnchor constraintEqualToAnchor:root.bottomAnchor constant:-OLPaneBottomMargin],
        ]];
    } else {
        [constraints addObject:[content.bottomAnchor constraintEqualToAnchor:root.bottomAnchor
                                                                     constant:-OLPaneBottomMargin]];
    }
    [NSLayoutConstraint activateConstraints:constraints];
    self.view = root;
}

- (NSView *)makeContentView {
    return [[NSView alloc] initWithFrame:NSZeroRect];
}

- (void)reload {
}

- (NSSize)fittingSize {
    [self.view layoutSubtreeIfNeeded];
    return self.view.fittingSize;
}

- (void)contentDidChange {
    // Panes update their controls while their view loads; the window sizes
    // itself to the pane once it's loaded.
    if (self.viewLoaded && self.contentSizeDidChange != nil) {
        self.contentSizeDidChange(self);
    }
}

- (void)presentErrorWithTitle:(NSString *)title message:(NSString *)message {
    NSAlert *alert = [[NSAlert alloc] init];
    alert.alertStyle = NSAlertStyleWarning;
    alert.icon = OLApplicationIcon();
    alert.messageText = title;
    alert.informativeText = message ?: @"";
    NSWindow *window = self.view.window;
    if (window != nil && window.visible) {
        [alert beginSheetModalForWindow:window completionHandler:nil];
    } else {
        [alert runModal];
    }
}

- (void)openHelp:(id)sender {
    (void)sender;
    [[NSWorkspace sharedWorkspace] openURL:self.helpURL];
}

@end
