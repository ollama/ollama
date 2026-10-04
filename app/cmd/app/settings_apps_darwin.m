#import "settings_panes_darwin.h"
#import "integrations_darwin.h"

// The Apps pane lists apps the way System Settings lists them: rows in a
// rounded group, each with an icon, a name, a status, and a switch.
static const CGFloat OLGroupCornerRadius = 10;
static const CGFloat OLRowPadding = 10;
static const CGFloat OLRowMinimumHeight = 44;
static const CGFloat OLAppIconSize = 28;

// OLGroupView draws the rounded background behind a group of rows.
@interface OLGroupView : NSView
@end

@implementation OLGroupView

- (void)drawRect:(NSRect)dirtyRect {
    (void)dirtyRect;
    [[NSColor quaternarySystemFillColor] setFill];
    [[NSBezierPath bezierPathWithRoundedRect:self.bounds
                                     xRadius:OLGroupCornerRadius
                                     yRadius:OLGroupCornerRadius] fill];
}

@end

// OLAppRowView shows one app: its icon, its name with a short status
// underneath, and a switch that turns Ollama models on or off in it.
@interface OLAppRowView : NSView
@property(nonatomic, readonly) OLIntegrationApp app;
@property(nonatomic, strong) NSSwitch *toggle;
@property(nonatomic, strong) NSTextField *statusLabel;
@property(nonatomic, strong) NSProgressIndicator *progress;
- (instancetype)initWithApp:(OLIntegrationApp)app target:(id)target action:(SEL)action;
- (void)showState:(nullable OLIntegrationState *)state;
@end

@implementation OLAppRowView

- (instancetype)initWithApp:(OLIntegrationApp)app target:(id)target action:(SEL)action {
    self = [super initWithFrame:NSZeroRect];
    if (self) {
        _app = app;
        NSString *name = app == OLIntegrationAppClaude ? @"Claude" : @"ChatGPT";
        self.translatesAutoresizingMaskIntoConstraints = NO;

        NSImageView *icon = [NSImageView imageViewWithImage:[[OLIntegrations sharedIntegrations] iconForApp:app]];
        icon.translatesAutoresizingMaskIntoConstraints = NO;
        icon.imageScaling = NSImageScaleProportionallyUpOrDown;

        NSTextField *title = [NSTextField labelWithString:name];
        title.font = [NSFont systemFontOfSize:NSFont.systemFontSize];

        _statusLabel = [NSTextField labelWithString:@""];
        _statusLabel.font = [NSFont systemFontOfSize:NSFont.smallSystemFontSize];
        _statusLabel.textColor = [NSColor secondaryLabelColor];
        _statusLabel.lineBreakMode = NSLineBreakByTruncatingTail;

        NSStackView *text = [NSStackView stackViewWithViews:@[title, _statusLabel]];
        text.orientation = NSUserInterfaceLayoutOrientationVertical;
        text.alignment = NSLayoutAttributeLeading;
        text.spacing = 1;
        text.translatesAutoresizingMaskIntoConstraints = NO;

        _progress = [[NSProgressIndicator alloc] initWithFrame:NSZeroRect];
        _progress.style = NSProgressIndicatorStyleSpinning;
        _progress.controlSize = NSControlSizeSmall;
        _progress.displayedWhenStopped = NO;
        _progress.translatesAutoresizingMaskIntoConstraints = NO;

        _toggle = [[NSSwitch alloc] initWithFrame:NSZeroRect];
        _toggle.controlSize = NSControlSizeSmall;
        _toggle.target = target;
        _toggle.action = action;
        _toggle.tag = app;
        _toggle.translatesAutoresizingMaskIntoConstraints = NO;
        [_toggle setAccessibilityLabel:[NSString stringWithFormat:@"Use Ollama models in %@", name]];

        for (NSView *view in @[icon, text, _progress, _toggle]) {
            [self addSubview:view];
        }
        [NSLayoutConstraint activateConstraints:@[
            [self.heightAnchor constraintGreaterThanOrEqualToConstant:OLRowMinimumHeight],
            [icon.leadingAnchor constraintEqualToAnchor:self.leadingAnchor constant:OLRowPadding],
            [icon.centerYAnchor constraintEqualToAnchor:self.centerYAnchor],
            [icon.widthAnchor constraintEqualToConstant:OLAppIconSize],
            [icon.heightAnchor constraintEqualToConstant:OLAppIconSize],
            [text.leadingAnchor constraintEqualToAnchor:icon.trailingAnchor constant:8],
            [text.centerYAnchor constraintEqualToAnchor:self.centerYAnchor],
            [text.topAnchor constraintGreaterThanOrEqualToAnchor:self.topAnchor constant:8],
            [text.trailingAnchor constraintLessThanOrEqualToAnchor:_progress.leadingAnchor constant:-8],
            [_progress.trailingAnchor constraintEqualToAnchor:_toggle.leadingAnchor constant:-8],
            [_progress.centerYAnchor constraintEqualToAnchor:self.centerYAnchor],
            [_toggle.trailingAnchor constraintEqualToAnchor:self.trailingAnchor constant:-OLRowPadding],
            [_toggle.centerYAnchor constraintEqualToAnchor:self.centerYAnchor],
        ]];
        [self showState:nil];
    }
    return self;
}

- (void)showState:(OLIntegrationState *)state {
    self.toggle.state = state.enabled ? NSControlStateValueOn : NSControlStateValueOff;
    self.toggle.enabled = state != nil && !state.busy;
    self.statusLabel.stringValue = state.status ?: @"";
    self.statusLabel.hidden = self.statusLabel.stringValue.length == 0;
    if (state.busy) {
        [self.progress startAnimation:nil];
    } else {
        [self.progress stopAnimation:nil];
    }
}

@end

@interface OLAppsSettingsPane ()
@property(nonatomic, copy) NSArray<OLAppRowView *> *rows;
@end

@implementation OLAppsSettingsPane

- (instancetype)init {
    self = [self initWithIdentifier:@"apps"
                              title:@"Apps"
                              image:[NSImage imageWithSystemSymbolName:@"square.grid.2x2"
                                              accessibilityDescription:@"Apps"]
                            helpURL:[NSURL URLWithString:@"https://docs.ollama.com/integrations"]];
    if (self) {
        [[NSNotificationCenter defaultCenter] addObserver:self
                                                 selector:@selector(integrationsDidChange:)
                                                     name:OLIntegrationsDidChangeNotification
                                                   object:nil];
    }
    return self;
}

- (void)dealloc {
    [[NSNotificationCenter defaultCenter] removeObserver:self];
}

- (NSView *)makeContentView {
    NSStackView *list = [[NSStackView alloc] initWithFrame:NSZeroRect];
    list.orientation = NSUserInterfaceLayoutOrientationVertical;
    list.alignment = NSLayoutAttributeLeading;
    list.spacing = 0;
    list.translatesAutoresizingMaskIntoConstraints = NO;

    NSMutableArray<OLAppRowView *> *rows = [NSMutableArray array];
    for (NSNumber *app in @[@(OLIntegrationAppClaude), @(OLIntegrationAppChatGPT)]) {
        if (rows.count > 0) {
            NSView *separator = [self separator];
            [list addArrangedSubview:separator];
            [separator.widthAnchor constraintEqualToAnchor:list.widthAnchor].active = YES;
        }
        OLAppRowView *row = [[OLAppRowView alloc] initWithApp:app.integerValue
                                                       target:self
                                                       action:@selector(toggleApp:)];
        [list addArrangedSubview:row];
        [row.widthAnchor constraintEqualToAnchor:list.widthAnchor].active = YES;
        [rows addObject:row];
    }
    self.rows = rows;

    OLGroupView *group = [[OLGroupView alloc] initWithFrame:NSZeroRect];
    group.translatesAutoresizingMaskIntoConstraints = NO;
    [group addSubview:list];
    [NSLayoutConstraint activateConstraints:@[
        [list.topAnchor constraintEqualToAnchor:group.topAnchor constant:2],
        [list.bottomAnchor constraintEqualToAnchor:group.bottomAnchor constant:-2],
        [list.leadingAnchor constraintEqualToAnchor:group.leadingAnchor],
        [list.trailingAnchor constraintEqualToAnchor:group.trailingAnchor],
    ]];

    NSTextField *footer = OLDescriptionLabel(
        @"Apps that are on can use the models on this Mac and your cloud models. "
        @"An open app restarts when you turn it on or off.");
    footer.preferredMaxLayoutWidth = OLSettingsContentWidth - 2 * OLRowPadding;

    NSStackView *content = [NSStackView stackViewWithViews:@[group, footer]];
    content.orientation = NSUserInterfaceLayoutOrientationVertical;
    content.alignment = NSLayoutAttributeLeading;
    content.spacing = 6;
    content.edgeInsets = NSEdgeInsetsZero;
    [group.widthAnchor constraintEqualToAnchor:content.widthAnchor].active = YES;
    [footer.leadingAnchor constraintEqualToAnchor:content.leadingAnchor constant:OLRowPadding].active = YES;
    return content;
}

// separator returns a line between rows, inset like System Settings.
- (NSView *)separator {
    NSView *container = [[NSView alloc] initWithFrame:NSZeroRect];
    container.translatesAutoresizingMaskIntoConstraints = NO;
    NSBox *line = [[NSBox alloc] initWithFrame:NSZeroRect];
    line.boxType = NSBoxSeparator;
    line.translatesAutoresizingMaskIntoConstraints = NO;
    [container addSubview:line];
    [NSLayoutConstraint activateConstraints:@[
        [line.leadingAnchor constraintEqualToAnchor:container.leadingAnchor constant:OLRowPadding],
        [line.trailingAnchor constraintEqualToAnchor:container.trailingAnchor constant:-OLRowPadding],
        [line.topAnchor constraintEqualToAnchor:container.topAnchor],
        [line.bottomAnchor constraintEqualToAnchor:container.bottomAnchor],
    ]];
    return container;
}

- (void)reload {
    for (OLAppRowView *row in self.rows) {
        [self reloadRow:row];
    }
}

- (void)reloadRow:(OLAppRowView *)row {
    [[OLIntegrations sharedIntegrations] loadStateForApp:row.app
                                              completion:^(OLIntegrationState *state) {
        [row showState:state];
        [self contentDidChange];
    }];
}

- (void)integrationsDidChange:(NSNotification *)notification {
    if (!self.viewLoaded || self.view.window == nil) {
        return;
    }
    NSNumber *app = notification.userInfo[@"app"];
    for (OLAppRowView *row in self.rows) {
        if (app == nil || row.app == app.integerValue) {
            [self reloadRow:row];
        }
    }
}

- (void)toggleApp:(NSSwitch *)sender {
    [[OLIntegrations sharedIntegrations] setApp:sender.tag
                                        enabled:sender.state == NSControlStateValueOn];
}

@end
