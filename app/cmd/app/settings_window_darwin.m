#import "settings_window_darwin.h"
#import "settings_pane_darwin.h"
#import "settings_panes_darwin.h"

// Remembers the pane people viewed last, which Settings reopens to.
static NSString *const OLSelectedPaneKey = @"SettingsSelectedPane";
static NSString *const OLFrameAutosaveName = @"Settings";
static NSToolbarIdentifier const OLToolbarIdentifier = @"SettingsToolbar";

@interface OLSettingsWindowController () <NSToolbarDelegate, NSWindowDelegate>
@property(nonatomic, copy) NSArray<OLSettingsPane *> *panes;
@property(nonatomic, strong, nullable) OLSettingsPane *currentPane;
@property(nonatomic) BOOL positioned;
// reloadWhenKey is set once the window loses focus, so returning to it,
// such as after signing in with a browser, refreshes the current pane.
@property(nonatomic) BOOL reloadWhenKey;
@end

@implementation OLSettingsWindowController

- (instancetype)init {
    // Settings windows have no minimize or zoom buttons to use.
    NSWindow *window = [[NSWindow alloc]
        initWithContentRect:NSMakeRect(0, 0, OLSettingsPaneWidth, 300)
                  styleMask:NSWindowStyleMaskTitled | NSWindowStyleMaskClosable
                    backing:NSBackingStoreBuffered
                      defer:YES];
    self = [super initWithWindow:window];
    if (self) {
        window.releasedWhenClosed = NO;
        window.restorable = NO;
        window.toolbarStyle = NSWindowToolbarStylePreference;
        window.collectionBehavior = NSWindowCollectionBehaviorFullScreenNone;
        window.autorecalculatesKeyViewLoop = YES;
        window.delegate = self;

        __weak OLSettingsWindowController *weakSelf = self;
        _panes = @[
            [[OLGeneralSettingsPane alloc] init],
            [[OLAccountSettingsPane alloc] init],
            [[OLAppsSettingsPane alloc] init],
        ];
        for (OLSettingsPane *pane in _panes) {
            pane.contentSizeDidChange = ^(OLSettingsPane *changed) {
                [weakSelf resizeToFitPane:changed animated:YES];
            };
        }

        // The toolbar switches panes, so it can't be customized or hidden.
        NSToolbar *toolbar = [[NSToolbar alloc] initWithIdentifier:OLToolbarIdentifier];
        toolbar.delegate = self;
        toolbar.allowsUserCustomization = NO;
        toolbar.autosavesConfiguration = NO;
        toolbar.displayMode = NSToolbarDisplayModeIconAndLabel;
        window.toolbar = toolbar;
        [window standardWindowButton:NSWindowZoomButton].enabled = NO;
    }
    return self;
}

- (void)showPane:(NSString *)identifier {
    NSString *saved = [[NSUserDefaults standardUserDefaults] stringForKey:OLSelectedPaneKey];
    OLSettingsPane *pane = [self paneWithIdentifier:identifier]
        ?: [self paneWithIdentifier:saved]
        ?: self.panes.firstObject;

    BOOL visible = self.window.visible;
    [self selectPane:pane animated:visible];
    if (!visible && !self.positioned) {
        self.positioned = YES;
        // Restore where people left the window, keeping the height that fits
        // the pane.
        NSRect fitted = self.window.frame;
        if ([self.window setFrameUsingName:OLFrameAutosaveName]) {
            NSRect restored = self.window.frame;
            fitted.origin = NSMakePoint(NSMinX(restored), NSMaxY(restored) - NSHeight(fitted));
            [self.window setFrame:fitted display:NO];
        } else {
            [self.window center];
        }
        self.window.frameAutosaveName = OLFrameAutosaveName;
    }
    [self showWindow:nil];
}

- (nullable OLSettingsPane *)paneWithIdentifier:(nullable NSString *)identifier {
    for (OLSettingsPane *pane in self.panes) {
        if ([pane.paneIdentifier isEqualToString:identifier]) {
            return pane;
        }
    }
    return nil;
}

- (void)selectPane:(OLSettingsPane *)pane animated:(BOOL)animated {
    if (pane == self.currentPane) {
        [pane reload];
        return;
    }

    [self.currentPane.view removeFromSuperview];
    self.currentPane = pane;
    [[NSUserDefaults standardUserDefaults] setObject:pane.paneIdentifier forKey:OLSelectedPaneKey];
    self.window.toolbar.selectedItemIdentifier = pane.paneIdentifier;
    self.window.title = pane.paneTitle;

    // Resize before showing the new pane, as settings windows do.
    NSView *view = pane.view;
    [pane reload];
    [self resizeToFitPane:pane animated:animated];
    NSView *container = self.window.contentView;
    [container addSubview:view];
    [NSLayoutConstraint activateConstraints:@[
        [view.topAnchor constraintEqualToAnchor:container.topAnchor],
        [view.leadingAnchor constraintEqualToAnchor:container.leadingAnchor],
    ]];
    [self.window makeFirstResponder:nil];
}

- (void)resizeToFitPane:(OLSettingsPane *)pane animated:(BOOL)animated {
    if (pane != self.currentPane) {
        return;
    }
    NSSize size = [pane fittingSize];
    NSWindow *window = self.window;
    NSSize current = window.contentView.frame.size;
    CGFloat deltaHeight = size.height - current.height;
    CGFloat deltaWidth = size.width - current.width;
    if (fabs(deltaHeight) < 0.5 && fabs(deltaWidth) < 0.5) {
        return;
    }
    // Keep the top edge in place so the toolbar doesn't move.
    NSRect frame = window.frame;
    frame.size.height += deltaHeight;
    frame.size.width += deltaWidth;
    frame.origin.y -= deltaHeight;
    [window setFrame:frame display:YES animate:animated && window.visible];
}

- (void)selectToolbarItem:(NSToolbarItem *)item {
    OLSettingsPane *pane = [self paneWithIdentifier:item.itemIdentifier];
    if (pane != nil) {
        [self selectPane:pane animated:YES];
    }
}

#pragma mark - NSWindowDelegate

- (void)windowDidBecomeKey:(NSNotification *)notification {
    (void)notification;
    if (self.reloadWhenKey) {
        self.reloadWhenKey = NO;
        [self.currentPane reload];
    }
}

- (void)windowDidResignKey:(NSNotification *)notification {
    (void)notification;
    self.reloadWhenKey = YES;
}

- (void)windowWillClose:(NSNotification *)notification {
    (void)notification;
    // Reopening the window reloads the pane anyway.
    self.reloadWhenKey = NO;
}

#pragma mark - NSToolbarDelegate

- (NSArray<NSToolbarItemIdentifier> *)paneIdentifiers {
    NSMutableArray<NSToolbarItemIdentifier> *identifiers = [NSMutableArray array];
    for (OLSettingsPane *pane in self.panes) {
        [identifiers addObject:pane.paneIdentifier];
    }
    return identifiers;
}

- (NSArray<NSToolbarItemIdentifier> *)toolbarAllowedItemIdentifiers:(NSToolbar *)toolbar {
    (void)toolbar;
    return [self paneIdentifiers];
}

- (NSArray<NSToolbarItemIdentifier> *)toolbarDefaultItemIdentifiers:(NSToolbar *)toolbar {
    (void)toolbar;
    return [self paneIdentifiers];
}

- (NSArray<NSToolbarItemIdentifier> *)toolbarSelectableItemIdentifiers:(NSToolbar *)toolbar {
    (void)toolbar;
    return [self paneIdentifiers];
}

- (NSToolbarItem *)toolbar:(NSToolbar *)toolbar
        itemForItemIdentifier:(NSToolbarItemIdentifier)identifier
    willBeInsertedIntoToolbar:(BOOL)flag {
    (void)toolbar;
    (void)flag;
    OLSettingsPane *pane = [self paneWithIdentifier:identifier];
    if (pane == nil) {
        return nil;
    }
    NSToolbarItem *item = [[NSToolbarItem alloc] initWithItemIdentifier:identifier];
    item.label = pane.paneTitle;
    item.image = pane.paneImage;
    item.target = self;
    item.action = @selector(selectToolbarItem:);
    return item;
}

@end
