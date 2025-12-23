#import "settings_darwin.h"
#import <Cocoa/Cocoa.h>

// Go callbacks - defined in settings_darwin.go
extern bool getSettingsExpose(void);
extern void setSettingsExpose(bool expose);
extern bool getSettingsBrowser(void);
extern void setSettingsBrowser(bool browser);
extern const char* getSettingsModels(void);
extern void setSettingsModels(const char* path);
extern int getSettingsContextLength(void);
extern void setSettingsContextLength(int length);
extern void restartOllamaServer(void);

// Account callbacks
extern const char* getAccountName(void);
extern const char* getAccountEmail(void);
extern const char* getAccountPlan(void);
extern const char* getAccountAvatarURL(void);
extern void signOutAccount(void);
extern void openConnectUrl(void);
extern void refreshAccountFromAPI(void);
extern void prefetchAccountData(void);

static SettingsWindowController *sharedInstance = nil;
    
@interface SettingsWindowController () <NSToolbarDelegate>
@property (nonatomic, strong) NSToolbar *toolbar;
@property (nonatomic, strong) NSView *generalView;
@property (nonatomic, strong) NSView *modelsView;
@property (nonatomic, strong) NSView *accountView;
@property (nonatomic, strong) NSString *currentTab;
@end

@implementation SettingsWindowController

+ (instancetype)sharedController {
    static dispatch_once_t onceToken;
    dispatch_once(&onceToken, ^{
        sharedInstance = [[SettingsWindowController alloc] init];
    });
    return sharedInstance;
}

- (instancetype)init {
    NSRect frame = NSMakeRect(0, 0, 550, 340);
    NSWindowStyleMask style = NSWindowStyleMaskTitled | 
                              NSWindowStyleMaskClosable |
                              NSWindowStyleMaskUnifiedTitleAndToolbar;
    
    NSWindow *window = [[NSWindow alloc] initWithContentRect:frame
                                                   styleMask:style
                                                     backing:NSBackingStoreBuffered
                                                       defer:NO];
    
    self = [super initWithWindow:window];
    if (self) {
        self.currentTab = @"General";
        [self setupWindow];
        [self setupToolbar];
        [self createGeneralView];
        [self createModelsView];
        [self createAccountView];
        [self showTab:@"General"];
        
        // Prefetch account data in background for faster Account tab loading
        prefetchAccountData();
    }
    return self;
}

- (void)setupWindow {
    NSWindow *window = self.window;
    window.delegate = self;
    [window center];
}

- (void)setupToolbar {
    self.toolbar = [[NSToolbar alloc] initWithIdentifier:@"SettingsToolbar"];
    self.toolbar.delegate = self;
    self.toolbar.displayMode = NSToolbarDisplayModeIconAndLabel;
    self.toolbar.allowsUserCustomization = NO;
    self.toolbar.centeredItemIdentifiers = [NSSet setWithArray:@[@"General", @"Models", @"Account"]];
    
    // Preference style - centered title with tabs below (like Apple Music)
    self.window.toolbarStyle = NSWindowToolbarStylePreference;
    self.window.toolbar = self.toolbar;
    self.toolbar.selectedItemIdentifier = @"General";
}

#pragma mark - NSToolbarDelegate

- (NSArray<NSToolbarItemIdentifier> *)toolbarAllowedItemIdentifiers:(NSToolbar *)toolbar {
    return @[@"General", @"Models", @"Account", NSToolbarFlexibleSpaceItemIdentifier];
}

- (NSArray<NSToolbarItemIdentifier> *)toolbarDefaultItemIdentifiers:(NSToolbar *)toolbar {
    return @[NSToolbarFlexibleSpaceItemIdentifier, @"General", @"Models", @"Account", NSToolbarFlexibleSpaceItemIdentifier];
}

- (NSArray<NSToolbarItemIdentifier> *)toolbarSelectableItemIdentifiers:(NSToolbar *)toolbar {
    return @[@"General", @"Models", @"Account"];
}

- (NSToolbarItem *)toolbar:(NSToolbar *)toolbar itemForItemIdentifier:(NSToolbarItemIdentifier)itemIdentifier willBeInsertedIntoToolbar:(BOOL)flag {
    NSToolbarItem *item = [[NSToolbarItem alloc] initWithItemIdentifier:itemIdentifier];
    
    if ([itemIdentifier isEqualToString:@"General"]) {
        item.label = @"General";
        item.image = [NSImage imageWithSystemSymbolName:@"gearshape" accessibilityDescription:@"General"];
        item.target = self;
        item.action = @selector(toolbarItemClicked:);
    } else if ([itemIdentifier isEqualToString:@"Models"]) {
        item.label = @"Models";
        item.image = [NSImage imageWithSystemSymbolName:@"folder" accessibilityDescription:@"Models"];
        item.target = self;
        item.action = @selector(toolbarItemClicked:);
    } else if ([itemIdentifier isEqualToString:@"Account"]) {
        item.label = @"Account";
        item.image = [NSImage imageWithSystemSymbolName:@"person.circle" accessibilityDescription:@"Account"];
        item.target = self;
        item.action = @selector(toolbarItemClicked:);
    }
    
    return item;
}

- (void)toolbarItemClicked:(NSToolbarItem *)sender {
    [self showTab:sender.itemIdentifier];
}

- (CGFloat)fittingHeightForView:(NSView *)view withWidth:(CGFloat)width {
    // Temporarily set width constraint to calculate fitting height
    NSLayoutConstraint *widthConstraint = [view.widthAnchor constraintEqualToConstant:width];
    widthConstraint.active = YES;
    
    // Force layout and get fitting size
    [view layoutSubtreeIfNeeded];
    CGFloat height = view.fittingSize.height;
    
    widthConstraint.active = NO;
    
    // Enforce minimum height
    return MAX(height, 150);
}

- (void)showTab:(NSString *)tabName {
    if ([self.currentTab isEqualToString:tabName] && self.window.contentView.subviews.count > 0) {
        return;  // Already showing this tab
    }
    
    self.currentTab = tabName;
    self.toolbar.selectedItemIdentifier = tabName;
    self.window.title = tabName;  // Update title to match selected tab
    
    NSView *tabView = nil;
    if ([tabName isEqualToString:@"General"]) {
        tabView = self.generalView;
    } else if ([tabName isEqualToString:@"Models"]) {
        tabView = self.modelsView;
    } else if ([tabName isEqualToString:@"Account"]) {
        tabView = self.accountView;
        [self refreshAccountInfo];
    }
    if (!tabView) return;
    
    // Calculate target height based on content
    CGFloat windowWidth = self.window.frame.size.width;
    CGFloat contentWidth = windowWidth;  // Content uses full width
    CGFloat targetHeight = [self fittingHeightForView:tabView withWidth:contentWidth];
    
    // Calculate new window frame (anchor at top-left)
    NSRect windowFrame = self.window.frame;
    CGFloat titleBarHeight = windowFrame.size.height - self.window.contentView.frame.size.height;
    CGFloat newWindowHeight = targetHeight + titleBarHeight;
    CGFloat deltaHeight = newWindowHeight - windowFrame.size.height;
    
    NSRect newFrame = NSMakeRect(
        windowFrame.origin.x,
        windowFrame.origin.y - deltaHeight,  // Move origin down so top stays fixed
        windowFrame.size.width,
        newWindowHeight
    );
    
    // Remove old content
    for (NSView *subview in self.window.contentView.subviews) {
        [subview removeFromSuperview];
    }
    
    // Instant resize (no animation)
    [self.window setFrame:newFrame display:YES];
    
    // Add new content
    NSView *contentView = self.window.contentView;
    tabView.translatesAutoresizingMaskIntoConstraints = NO;
    [contentView addSubview:tabView];
    [NSLayoutConstraint activateConstraints:@[
        [tabView.topAnchor constraintEqualToAnchor:contentView.topAnchor],
        [tabView.bottomAnchor constraintEqualToAnchor:contentView.bottomAnchor],
        [tabView.leadingAnchor constraintEqualToAnchor:contentView.leadingAnchor],
        [tabView.trailingAnchor constraintEqualToAnchor:contentView.trailingAnchor],
    ]];
}

#pragma mark - Helper: Create a row with label and content

- (NSView *)createRowWithLabel:(NSString *)labelText content:(NSView *)content description:(NSString *)descText {
    return [self createRowWithLabel:labelText content:content description:descText descOffset:18 descTopSpacing:4];
}

- (NSView *)createRowWithLabel:(NSString *)labelText content:(NSView *)content description:(NSString *)descText descOffset:(CGFloat)descOffset descTopSpacing:(CGFloat)descTopSpacing {
    return [self createRowWithLabel:labelText content:content description:descText descOffset:descOffset descTopSpacing:descTopSpacing alignTop:NO];
}

- (NSView *)createRowWithLabel:(NSString *)labelText content:(NSView *)content description:(NSString *)descText descOffset:(CGFloat)descOffset descTopSpacing:(CGFloat)descTopSpacing alignTop:(BOOL)alignTop {
    NSView *row = [[NSView alloc] init];
    row.translatesAutoresizingMaskIntoConstraints = NO;
    
    // Label
    NSTextField *label = [NSTextField labelWithString:labelText];
    label.font = [NSFont systemFontOfSize:13];
    label.alignment = NSTextAlignmentRight;
    label.translatesAutoresizingMaskIntoConstraints = NO;
    [label setContentHuggingPriority:NSLayoutPriorityRequired forOrientation:NSLayoutConstraintOrientationHorizontal];
    [row addSubview:label];
    
    // Content
    content.translatesAutoresizingMaskIntoConstraints = NO;
    [row addSubview:content];
    
    // Constraints - align label baseline with content for proper text alignment, unless alignTop is YES
    NSMutableArray *constraints = [NSMutableArray arrayWithArray:@[
        [label.leadingAnchor constraintEqualToAnchor:row.leadingAnchor],
        [label.widthAnchor constraintEqualToConstant:110],
        
        [content.leadingAnchor constraintEqualToAnchor:label.trailingAnchor constant:12],
        [content.trailingAnchor constraintLessThanOrEqualToAnchor:row.trailingAnchor],
        [content.topAnchor constraintEqualToAnchor:row.topAnchor],
    ]];
    
    if (alignTop) {
        // Align label to top of row (for sliders, complex content)
        [constraints addObject:[label.topAnchor constraintEqualToAnchor:row.topAnchor constant:2]];
    } else {
        // Align baselines (for text, checkboxes)
        [constraints addObject:[label.firstBaselineAnchor constraintEqualToAnchor:content.firstBaselineAnchor]];
    }
    
    [NSLayoutConstraint activateConstraints:constraints];
    
    NSLayoutConstraint *bottomConstraint;
    
    if (descText) {
        NSTextField *desc = [NSTextField wrappingLabelWithString:descText];
        desc.font = [NSFont systemFontOfSize:11];
        desc.textColor = [NSColor secondaryLabelColor];
        desc.selectable = NO;  // Not copyable
        desc.translatesAutoresizingMaskIntoConstraints = NO;
        [row addSubview:desc];
        
        // Calculate actual text offset for checkboxes dynamically
        CGFloat actualOffset = descOffset;
        if ([content isKindOfClass:[NSButton class]]) {
            NSButton *button = (NSButton *)content;
            // Get the title rect to find where text actually starts
            NSRect bounds = button.bounds;
            if (NSIsEmptyRect(bounds)) {
                bounds = NSMakeRect(0, 0, 300, 22);
            }
            NSRect titleRect = [button.cell titleRectForBounds:bounds];
            actualOffset = titleRect.origin.x;
        }
        
        [NSLayoutConstraint activateConstraints:@[
            [desc.leadingAnchor constraintEqualToAnchor:content.leadingAnchor constant:actualOffset],
            [desc.trailingAnchor constraintEqualToAnchor:row.trailingAnchor],
            [desc.topAnchor constraintEqualToAnchor:content.bottomAnchor constant:descTopSpacing],
        ]];
        bottomConstraint = [row.bottomAnchor constraintEqualToAnchor:desc.bottomAnchor constant:8];
    } else {
        bottomConstraint = [row.bottomAnchor constraintEqualToAnchor:content.bottomAnchor];
    }
    
    bottomConstraint.active = YES;
    return row;
}

- (NSBox *)createSeparator {
    NSBox *sep = [[NSBox alloc] init];
    sep.boxType = NSBoxSeparator;
    sep.translatesAutoresizingMaskIntoConstraints = NO;
    return sep;
}

#pragma mark - General Tab

- (void)createGeneralView {
    self.generalView = [[NSView alloc] init];
    
    // Stack view for vertical layout
    NSStackView *stack = [NSStackView stackViewWithViews:@[]];
    stack.orientation = NSUserInterfaceLayoutOrientationVertical;
    stack.alignment = NSLayoutAttributeLeading;
    stack.spacing = 16;
    stack.translatesAutoresizingMaskIntoConstraints = NO;
    [self.generalView addSubview:stack];
    
    // Constraints for stack - include bottom for proper fitting size calculation
    [NSLayoutConstraint activateConstraints:@[
        [stack.topAnchor constraintEqualToAnchor:self.generalView.topAnchor constant:24],
        [stack.leadingAnchor constraintEqualToAnchor:self.generalView.leadingAnchor constant:30],
        [stack.trailingAnchor constraintEqualToAnchor:self.generalView.trailingAnchor constant:-30],
        [stack.bottomAnchor constraintEqualToAnchor:self.generalView.bottomAnchor constant:-24],
    ]];
    
    // === Connectivity Section (Network + Browser combined) ===
    NSStackView *connectivityStack = [NSStackView stackViewWithViews:@[]];
    connectivityStack.orientation = NSUserInterfaceLayoutOrientationVertical;
    connectivityStack.alignment = NSLayoutAttributeLeading;
    connectivityStack.spacing = 16;
    connectivityStack.translatesAutoresizingMaskIntoConstraints = NO;
    
    // Network checkbox with description
    NSStackView *networkStack = [NSStackView stackViewWithViews:@[]];
    networkStack.orientation = NSUserInterfaceLayoutOrientationVertical;
    networkStack.alignment = NSLayoutAttributeLeading;
    networkStack.spacing = 4;
    
    self.exposeCheckbox = [NSButton checkboxWithTitle:@"Allow connections from other computers on this network"
                                               target:self
                                               action:@selector(exposeChanged:)];
    self.exposeCheckbox.font = [NSFont systemFontOfSize:13];
    [networkStack addArrangedSubview:self.exposeCheckbox];
    
    NSTextField *networkDesc = [NSTextField wrappingLabelWithString:@"When enabled, other devices on your network can connect to Ollama."];
    networkDesc.font = [NSFont systemFontOfSize:11];
    networkDesc.textColor = [NSColor secondaryLabelColor];
    networkDesc.selectable = NO;
    // Indent to align with checkbox text
    NSView *networkDescContainer = [[NSView alloc] init];
    networkDescContainer.translatesAutoresizingMaskIntoConstraints = NO;
    networkDesc.translatesAutoresizingMaskIntoConstraints = NO;
    [networkDescContainer addSubview:networkDesc];
    [NSLayoutConstraint activateConstraints:@[
        [networkDesc.leadingAnchor constraintEqualToAnchor:networkDescContainer.leadingAnchor constant:20],
        [networkDesc.trailingAnchor constraintEqualToAnchor:networkDescContainer.trailingAnchor],
        [networkDesc.topAnchor constraintEqualToAnchor:networkDescContainer.topAnchor],
        [networkDesc.bottomAnchor constraintEqualToAnchor:networkDescContainer.bottomAnchor],
    ]];
    [networkStack addArrangedSubview:networkDescContainer];
    
    [connectivityStack addArrangedSubview:networkStack];
    
    // Browser checkbox with description
    NSStackView *browserStack = [NSStackView stackViewWithViews:@[]];
    browserStack.orientation = NSUserInterfaceLayoutOrientationVertical;
    browserStack.alignment = NSLayoutAttributeLeading;
    browserStack.spacing = 4;
    
    self.browserCheckbox = [NSButton checkboxWithTitle:@"Allow browser extensions and web apps"
                                                target:self
                                                action:@selector(browserChanged:)];
    self.browserCheckbox.font = [NSFont systemFontOfSize:13];
    [browserStack addArrangedSubview:self.browserCheckbox];
    
    NSTextField *browserDesc = [NSTextField wrappingLabelWithString:@"Enables CORS so browser-based applications can access Ollama."];
    browserDesc.font = [NSFont systemFontOfSize:11];
    browserDesc.textColor = [NSColor secondaryLabelColor];
    browserDesc.selectable = NO;
    // Indent to align with checkbox text
    NSView *browserDescContainer = [[NSView alloc] init];
    browserDescContainer.translatesAutoresizingMaskIntoConstraints = NO;
    browserDesc.translatesAutoresizingMaskIntoConstraints = NO;
    [browserDescContainer addSubview:browserDesc];
    [NSLayoutConstraint activateConstraints:@[
        [browserDesc.leadingAnchor constraintEqualToAnchor:browserDescContainer.leadingAnchor constant:20],
        [browserDesc.trailingAnchor constraintEqualToAnchor:browserDescContainer.trailingAnchor],
        [browserDesc.topAnchor constraintEqualToAnchor:browserDescContainer.topAnchor],
        [browserDesc.bottomAnchor constraintEqualToAnchor:browserDescContainer.bottomAnchor],
    ]];
    [browserStack addArrangedSubview:browserDescContainer];
    
    [connectivityStack addArrangedSubview:browserStack];
    
    NSView *connectivityRow = [self createRowWithLabel:@"Connectivity:"
                                          content:connectivityStack
                                      description:nil];
    [stack addArrangedSubview:connectivityRow];
    [connectivityRow.widthAnchor constraintEqualToAnchor:stack.widthAnchor].active = YES;
    
    // Separator
    NSBox *sep1 = [self createSeparator];
    [stack addArrangedSubview:sep1];
    [sep1.widthAnchor constraintEqualToAnchor:stack.widthAnchor].active = YES;
    
    // === Context Length Row ===
    // Fixed sizes: 4K, 8K, 16K, 32K, 64K, 128K, 256K (7 stops)
    // Using labeled tick marks like Energy Saver
    
    NSView *sliderContainer = [[NSView alloc] init];
    sliderContainer.translatesAutoresizingMaskIntoConstraints = NO;
    
    // Slider with 7 tick marks (index 0-6)
    self.contextLengthSlider = [[NSSlider alloc] init];
    self.contextLengthSlider.minValue = 0;
    self.contextLengthSlider.maxValue = 6;
    self.contextLengthSlider.numberOfTickMarks = 7;
    self.contextLengthSlider.allowsTickMarkValuesOnly = YES;
    self.contextLengthSlider.tickMarkPosition = NSTickMarkPositionBelow;
    self.contextLengthSlider.target = self;
    self.contextLengthSlider.action = @selector(contextLengthSliderChanged:);
    self.contextLengthSlider.continuous = YES;
    self.contextLengthSlider.translatesAutoresizingMaskIntoConstraints = NO;
    [sliderContainer addSubview:self.contextLengthSlider];
    
    // Labels for tick marks: 4K, 8K, 16K, 32K, 64K, 128K, 256K
    NSArray *labelTexts = @[@"4K", @"8K", @"16K", @"32K", @"64K", @"128K", @"256K"];
    CGFloat sliderWidth = 300;
    
    // NSSlider has ~10px inset on each side for the thumb
    // Calculate tick positions manually
    CGFloat thumbInset = 10.0;
    CGFloat trackWidth = sliderWidth - (2 * thumbInset);
    CGFloat tickSpacing = trackWidth / 6.0;  // 7 ticks = 6 gaps
    
    // Position labels directly under each tick mark
    for (int i = 0; i < labelTexts.count; i++) {
        NSTextField *tickLabel = [NSTextField labelWithString:labelTexts[i]];
        tickLabel.font = [NSFont systemFontOfSize:11];
        tickLabel.textColor = [NSColor labelColor];
        tickLabel.alignment = NSTextAlignmentCenter;
        tickLabel.translatesAutoresizingMaskIntoConstraints = NO;
        [sliderContainer addSubview:tickLabel];
        
        // Calculate position: thumbInset + (index * tickSpacing)
        CGFloat tickCenterX = thumbInset + (i * tickSpacing);
        
        [NSLayoutConstraint activateConstraints:@[
            [tickLabel.centerXAnchor constraintEqualToAnchor:self.contextLengthSlider.leadingAnchor constant:tickCenterX],
            [tickLabel.topAnchor constraintEqualToAnchor:self.contextLengthSlider.bottomAnchor constant:4],
        ]];
    }
    
    // Slider constraints
    [NSLayoutConstraint activateConstraints:@[
        [self.contextLengthSlider.leadingAnchor constraintEqualToAnchor:sliderContainer.leadingAnchor],
        [self.contextLengthSlider.topAnchor constraintEqualToAnchor:sliderContainer.topAnchor],
        [self.contextLengthSlider.widthAnchor constraintEqualToConstant:sliderWidth],
        [sliderContainer.heightAnchor constraintEqualToConstant:42],
        [sliderContainer.widthAnchor constraintEqualToConstant:sliderWidth],
    ]];
    
    NSView *contextRow = [self createRowWithLabel:@"Context Length:"
                                          content:sliderContainer
                                      description:@"Maximum context window size. Larger values use more memory."
                                       descOffset:0
                                   descTopSpacing:8
                                         alignTop:YES];
    [stack addArrangedSubview:contextRow];
    [contextRow.widthAnchor constraintEqualToAnchor:stack.widthAnchor].active = YES;
}

#pragma mark - Models Tab

- (void)createModelsView {
    self.modelsView = [[NSView alloc] init];
    
    NSStackView *stack = [NSStackView stackViewWithViews:@[]];
    stack.orientation = NSUserInterfaceLayoutOrientationVertical;
    stack.alignment = NSLayoutAttributeLeading;
    stack.spacing = 16;
    stack.translatesAutoresizingMaskIntoConstraints = NO;
    [self.modelsView addSubview:stack];
    
    [NSLayoutConstraint activateConstraints:@[
        [stack.topAnchor constraintEqualToAnchor:self.modelsView.topAnchor constant:24],
        [stack.leadingAnchor constraintEqualToAnchor:self.modelsView.leadingAnchor constant:30],
        [stack.trailingAnchor constraintEqualToAnchor:self.modelsView.trailingAnchor constant:-30],
        [stack.bottomAnchor constraintEqualToAnchor:self.modelsView.bottomAnchor constant:-24],
    ]];
    
    // === Path Control Row ===
    self.modelsPathControl = [[NSPathControl alloc] init];
    self.modelsPathControl.pathStyle = NSPathStyleStandard;
    self.modelsPathControl.backgroundColor = [NSColor clearColor];
    self.modelsPathControl.font = [NSFont systemFontOfSize:13];
    self.modelsPathControl.target = self;
    self.modelsPathControl.action = @selector(pathControlClicked:);
    self.modelsPathControl.doubleAction = @selector(pathControlDoubleClicked:);
    self.modelsPathControl.translatesAutoresizingMaskIntoConstraints = NO;
    
    NSView *pathRow = [self createRowWithLabel:@"Directory:" content:self.modelsPathControl description:nil];
    [stack addArrangedSubview:pathRow];
    [pathRow.widthAnchor constraintEqualToAnchor:stack.widthAnchor].active = YES;
    
    // === Buttons Row ===
    NSStackView *buttonStack = [NSStackView stackViewWithViews:@[]];
    buttonStack.orientation = NSUserInterfaceLayoutOrientationHorizontal;
    buttonStack.spacing = 8;
    
    self.modelsPathButton = [NSButton buttonWithTitle:@"Change..." target:self action:@selector(chooseModelsPath:)];
    self.modelsPathButton.bezelStyle = NSBezelStyleRounded;
    
    NSButton *resetButton = [NSButton buttonWithTitle:@"Reset" target:self action:@selector(resetModelsPath:)];
    resetButton.bezelStyle = NSBezelStyleRounded;
    
    [buttonStack addArrangedSubview:self.modelsPathButton];
    [buttonStack addArrangedSubview:resetButton];
    
    NSView *buttonRow = [self createRowWithLabel:@"" content:buttonStack description:nil];
    [stack addArrangedSubview:buttonRow];
    [buttonRow.widthAnchor constraintEqualToAnchor:stack.widthAnchor].active = YES;
    
    // === Description ===
    NSTextField *desc = [NSTextField wrappingLabelWithString:@"This is where Ollama stores downloaded models. Changing this location will not move existing models."];
    desc.font = [NSFont systemFontOfSize:11];
    desc.textColor = [NSColor secondaryLabelColor];
    desc.selectable = NO;
    desc.translatesAutoresizingMaskIntoConstraints = NO;
    
    NSView *descRow = [self createRowWithLabel:@"" content:desc description:nil];
    [stack addArrangedSubview:descRow];
    [descRow.widthAnchor constraintEqualToAnchor:stack.widthAnchor].active = YES;
}

#pragma mark - Account Tab

- (void)createAccountView {
    self.accountView = [[NSView alloc] init];
    
    NSStackView *stack = [NSStackView stackViewWithViews:@[]];
    stack.orientation = NSUserInterfaceLayoutOrientationVertical;
    stack.alignment = NSLayoutAttributeLeading;
    stack.spacing = 20;
    stack.translatesAutoresizingMaskIntoConstraints = NO;
    [self.accountView addSubview:stack];
    
    [NSLayoutConstraint activateConstraints:@[
        [stack.topAnchor constraintEqualToAnchor:self.accountView.topAnchor constant:24],
        [stack.leadingAnchor constraintEqualToAnchor:self.accountView.leadingAnchor constant:30],
        [stack.trailingAnchor constraintEqualToAnchor:self.accountView.trailingAnchor constant:-30],
        [stack.bottomAnchor constraintEqualToAnchor:self.accountView.bottomAnchor constant:-24],
    ]];
    
    // ==========================================
    // SECTION 1: Account (when signed in)
    // ==========================================
    self.signedInContainer = [[NSView alloc] init];
    self.signedInContainer.translatesAutoresizingMaskIntoConstraints = NO;
    
    // Horizontal layout: Avatar | Name+Email | Buttons
    NSStackView *accountRow = [NSStackView stackViewWithViews:@[]];
    accountRow.orientation = NSUserInterfaceLayoutOrientationHorizontal;
    accountRow.alignment = NSLayoutAttributeCenterY;
    accountRow.spacing = 12;
    accountRow.translatesAutoresizingMaskIntoConstraints = NO;
    [self.signedInContainer addSubview:accountRow];
    
    [NSLayoutConstraint activateConstraints:@[
        [accountRow.topAnchor constraintEqualToAnchor:self.signedInContainer.topAnchor],
        [accountRow.leadingAnchor constraintEqualToAnchor:self.signedInContainer.leadingAnchor],
        [accountRow.trailingAnchor constraintEqualToAnchor:self.signedInContainer.trailingAnchor],
        [accountRow.bottomAnchor constraintEqualToAnchor:self.signedInContainer.bottomAnchor],
    ]];
    
    // Avatar circle
    CGFloat avatarSize = 40;
    self.avatarView = [[NSView alloc] init];
    self.avatarView.wantsLayer = YES;
    self.avatarView.layer.backgroundColor = [[NSColor colorWithRed:0.35 green:0.65 blue:0.65 alpha:1.0] CGColor];
    self.avatarView.layer.cornerRadius = avatarSize / 2;
    self.avatarView.layer.masksToBounds = YES;
    self.avatarView.translatesAutoresizingMaskIntoConstraints = NO;
    [NSLayoutConstraint activateConstraints:@[
        [self.avatarView.widthAnchor constraintEqualToConstant:avatarSize],
        [self.avatarView.heightAnchor constraintEqualToConstant:avatarSize],
    ]];
    
    // Initial label in avatar (fallback)
    self.avatarInitialLabel = [NSTextField labelWithString:@"?"];
    self.avatarInitialLabel.font = [NSFont systemFontOfSize:16 weight:NSFontWeightMedium];
    self.avatarInitialLabel.textColor = [NSColor whiteColor];
    self.avatarInitialLabel.alignment = NSTextAlignmentCenter;
    self.avatarInitialLabel.translatesAutoresizingMaskIntoConstraints = NO;
    [self.avatarView addSubview:self.avatarInitialLabel];
    [NSLayoutConstraint activateConstraints:@[
        [self.avatarInitialLabel.centerXAnchor constraintEqualToAnchor:self.avatarView.centerXAnchor],
        [self.avatarInitialLabel.centerYAnchor constraintEqualToAnchor:self.avatarView.centerYAnchor],
    ]];
    
    // Avatar image view (for actual photo)
    self.avatarImageView = [[NSImageView alloc] init];
    self.avatarImageView.imageScaling = NSImageScaleProportionallyUpOrDown;
    self.avatarImageView.translatesAutoresizingMaskIntoConstraints = NO;
    self.avatarImageView.hidden = YES;  // Hidden until image loads
    [self.avatarView addSubview:self.avatarImageView];
    [NSLayoutConstraint activateConstraints:@[
        [self.avatarImageView.leadingAnchor constraintEqualToAnchor:self.avatarView.leadingAnchor],
        [self.avatarImageView.trailingAnchor constraintEqualToAnchor:self.avatarView.trailingAnchor],
        [self.avatarImageView.topAnchor constraintEqualToAnchor:self.avatarView.topAnchor],
        [self.avatarImageView.bottomAnchor constraintEqualToAnchor:self.avatarView.bottomAnchor],
    ]];
    
    [accountRow addArrangedSubview:self.avatarView];
    
    // Name + Email stack
    NSStackView *nameStack = [NSStackView stackViewWithViews:@[]];
    nameStack.orientation = NSUserInterfaceLayoutOrientationVertical;
    nameStack.alignment = NSLayoutAttributeLeading;
    nameStack.spacing = 2;
    
    self.accountNameLabel = [NSTextField labelWithString:@""];
    self.accountNameLabel.font = [NSFont systemFontOfSize:13 weight:NSFontWeightMedium];
    [nameStack addArrangedSubview:self.accountNameLabel];
    
    self.accountEmailLabel = [NSTextField labelWithString:@""];
    self.accountEmailLabel.font = [NSFont systemFontOfSize:12];
    self.accountEmailLabel.textColor = [NSColor secondaryLabelColor];
    [nameStack addArrangedSubview:self.accountEmailLabel];
    
    [accountRow addArrangedSubview:nameStack];
    
    // Flexible spacer to push buttons to right
    NSView *spacer = [[NSView alloc] init];
    spacer.translatesAutoresizingMaskIntoConstraints = NO;
    [spacer setContentHuggingPriority:1 forOrientation:NSLayoutConstraintOrientationHorizontal];
    [accountRow addArrangedSubview:spacer];
    
    // Buttons on right
    NSStackView *buttonStack = [NSStackView stackViewWithViews:@[]];
    buttonStack.orientation = NSUserInterfaceLayoutOrientationHorizontal;
    buttonStack.spacing = 8;
    
    self.manageButton = [NSButton buttonWithTitle:@"Manage" target:self action:@selector(manageClicked:)];
    self.manageButton.bezelStyle = NSBezelStyleRounded;
    self.manageButton.controlSize = NSControlSizeSmall;
    [buttonStack addArrangedSubview:self.manageButton];
    
    self.signOutButton = [NSButton buttonWithTitle:@"Sign Out" target:self action:@selector(signOutClicked:)];
    self.signOutButton.bezelStyle = NSBezelStyleRounded;
    self.signOutButton.controlSize = NSControlSizeSmall;
    [buttonStack addArrangedSubview:self.signOutButton];
    
    [accountRow addArrangedSubview:buttonStack];
    
    [stack addArrangedSubview:self.signedInContainer];
    [self.signedInContainer.widthAnchor constraintEqualToAnchor:stack.widthAnchor].active = YES;
    
    // ==========================================
    // SECTION 2: Plan (when signed in)
    // ==========================================
    self.planContainer = [[NSView alloc] init];
    self.planContainer.translatesAutoresizingMaskIntoConstraints = NO;
    
    NSStackView *planMainStack = [NSStackView stackViewWithViews:@[]];
    planMainStack.orientation = NSUserInterfaceLayoutOrientationVertical;
    planMainStack.alignment = NSLayoutAttributeLeading;
    planMainStack.spacing = 12;
    planMainStack.translatesAutoresizingMaskIntoConstraints = NO;
    [self.planContainer addSubview:planMainStack];
    
    [NSLayoutConstraint activateConstraints:@[
        [planMainStack.topAnchor constraintEqualToAnchor:self.planContainer.topAnchor],
        [planMainStack.leadingAnchor constraintEqualToAnchor:self.planContainer.leadingAnchor],
        [planMainStack.trailingAnchor constraintEqualToAnchor:self.planContainer.trailingAnchor],
        [planMainStack.bottomAnchor constraintEqualToAnchor:self.planContainer.bottomAnchor],
    ]];
    
    // Separator
    NSBox *sep = [self createSeparator];
    [planMainStack addArrangedSubview:sep];
    [sep.widthAnchor constraintEqualToAnchor:planMainStack.widthAnchor].active = YES;
    
    // Plan row: Label | Plan name | Buttons
    NSStackView *planRow = [NSStackView stackViewWithViews:@[]];
    planRow.orientation = NSUserInterfaceLayoutOrientationHorizontal;
    planRow.alignment = NSLayoutAttributeCenterY;
    planRow.spacing = 12;
    
    NSTextField *planLabel = [NSTextField labelWithString:@"Plan:"];
    planLabel.font = [NSFont systemFontOfSize:13];
    planLabel.textColor = [NSColor secondaryLabelColor];
    [planLabel.widthAnchor constraintEqualToConstant:40].active = YES;
    [planRow addArrangedSubview:planLabel];
    
    self.planNameLabel = [NSTextField labelWithString:@"Free"];
    self.planNameLabel.font = [NSFont systemFontOfSize:13 weight:NSFontWeightMedium];
    [planRow addArrangedSubview:self.planNameLabel];
    
    // Flexible spacer
    NSView *planSpacer = [[NSView alloc] init];
    planSpacer.translatesAutoresizingMaskIntoConstraints = NO;
    [planSpacer setContentHuggingPriority:1 forOrientation:NSLayoutConstraintOrientationHorizontal];
    [planRow addArrangedSubview:planSpacer];
    
    // Plan buttons
    NSStackView *planButtonStack = [NSStackView stackViewWithViews:@[]];
    planButtonStack.orientation = NSUserInterfaceLayoutOrientationHorizontal;
    planButtonStack.spacing = 8;
    
    self.upgradeButton = [NSButton buttonWithTitle:@"Upgrade" target:self action:@selector(upgradeClicked:)];
    self.upgradeButton.bezelStyle = NSBezelStyleRounded;
    self.upgradeButton.controlSize = NSControlSizeSmall;
    [planButtonStack addArrangedSubview:self.upgradeButton];
    
    self.viewUsageButton = [NSButton buttonWithTitle:@"View Usage" target:self action:@selector(viewUsageClicked:)];
    self.viewUsageButton.bezelStyle = NSBezelStyleRounded;
    self.viewUsageButton.controlSize = NSControlSizeSmall;
    [planButtonStack addArrangedSubview:self.viewUsageButton];
    
    [planRow addArrangedSubview:planButtonStack];
    [planMainStack addArrangedSubview:planRow];
    [planRow.widthAnchor constraintEqualToAnchor:planMainStack.widthAnchor].active = YES;
    
    [stack addArrangedSubview:self.planContainer];
    [self.planContainer.widthAnchor constraintEqualToAnchor:stack.widthAnchor].active = YES;
    
    // ==========================================
    // SECTION 3: Signed Out Container
    // ==========================================
    self.signedOutContainer = [[NSView alloc] init];
    self.signedOutContainer.translatesAutoresizingMaskIntoConstraints = NO;
    
    NSStackView *signedOutRow = [NSStackView stackViewWithViews:@[]];
    signedOutRow.orientation = NSUserInterfaceLayoutOrientationHorizontal;
    signedOutRow.alignment = NSLayoutAttributeCenterY;
    signedOutRow.spacing = 12;
    signedOutRow.translatesAutoresizingMaskIntoConstraints = NO;
    [self.signedOutContainer addSubview:signedOutRow];
    
    [NSLayoutConstraint activateConstraints:@[
        [signedOutRow.topAnchor constraintEqualToAnchor:self.signedOutContainer.topAnchor],
        [signedOutRow.leadingAnchor constraintEqualToAnchor:self.signedOutContainer.leadingAnchor],
        [signedOutRow.trailingAnchor constraintEqualToAnchor:self.signedOutContainer.trailingAnchor],
        [signedOutRow.bottomAnchor constraintEqualToAnchor:self.signedOutContainer.bottomAnchor],
    ]];
    
    // Signed out icon - same size as avatar (40px)
    NSImageView *personIcon = [NSImageView imageViewWithImage:[NSImage imageWithSystemSymbolName:@"person.circle.fill" accessibilityDescription:@"Account"]];
    personIcon.translatesAutoresizingMaskIntoConstraints = NO;
    [personIcon.widthAnchor constraintEqualToConstant:40].active = YES;
    [personIcon.heightAnchor constraintEqualToConstant:40].active = YES;
    personIcon.contentTintColor = [NSColor tertiaryLabelColor];
    if (@available(macOS 11.0, *)) {
        personIcon.symbolConfiguration = [NSImageSymbolConfiguration configurationWithPointSize:32 weight:NSFontWeightRegular];
    }
    [signedOutRow addArrangedSubview:personIcon];
    
    // Text stack
    NSStackView *signedOutTextStack = [NSStackView stackViewWithViews:@[]];
    signedOutTextStack.orientation = NSUserInterfaceLayoutOrientationVertical;
    signedOutTextStack.alignment = NSLayoutAttributeLeading;
    signedOutTextStack.spacing = 2;
    
    NSTextField *notConnectedLabel = [NSTextField labelWithString:@"Ollama Account"];
    notConnectedLabel.font = [NSFont systemFontOfSize:13 weight:NSFontWeightMedium];
    [signedOutTextStack addArrangedSubview:notConnectedLabel];
    
    NSTextField *notConnectedDesc = [NSTextField labelWithString:@"Sign in to access cloud models"];
    notConnectedDesc.font = [NSFont systemFontOfSize:12];
    notConnectedDesc.textColor = [NSColor secondaryLabelColor];
    [signedOutTextStack addArrangedSubview:notConnectedDesc];
    
    [signedOutRow addArrangedSubview:signedOutTextStack];
    
    // Spacer
    NSView *signedOutSpacer = [[NSView alloc] init];
    signedOutSpacer.translatesAutoresizingMaskIntoConstraints = NO;
    [signedOutSpacer setContentHuggingPriority:1 forOrientation:NSLayoutConstraintOrientationHorizontal];
    [signedOutRow addArrangedSubview:signedOutSpacer];
    
    // Sign in button
    self.signInButton = [NSButton buttonWithTitle:@"Sign In..." target:self action:@selector(signInClicked:)];
    self.signInButton.bezelStyle = NSBezelStyleRounded;
    self.signInButton.controlSize = NSControlSizeSmall;
    [signedOutRow addArrangedSubview:self.signInButton];
    
    [stack addArrangedSubview:self.signedOutContainer];
    [self.signedOutContainer.widthAnchor constraintEqualToAnchor:stack.widthAnchor].active = YES;
}

- (void)updateAccountUI {
    const char* name = getAccountName();
    const char* email = getAccountEmail();
    const char* plan = getAccountPlan();
    const char* avatarURL = getAccountAvatarURL();
    
    BOOL isSignedIn = (name != NULL && strlen(name) > 0);
    
    if (isSignedIn) {
        self.signedInContainer.hidden = NO;
        self.planContainer.hidden = NO;
        self.signedOutContainer.hidden = YES;
        
        NSString *nameStr = [NSString stringWithUTF8String:name];
        self.accountNameLabel.stringValue = nameStr;
        self.accountEmailLabel.stringValue = email ? [NSString stringWithUTF8String:email] : @"";
        
        // Set avatar initial (fallback)
        if (nameStr.length > 0) {
            self.avatarInitialLabel.stringValue = [[nameStr substringToIndex:1] uppercaseString];
        }
        
        // Load avatar image if URL available
        if (avatarURL && strlen(avatarURL) > 0) {
            NSString *urlStr = [NSString stringWithUTF8String:avatarURL];
            [self loadAvatarFromURL:urlStr];
        }
        
        // Update plan display
        NSString *planStr = plan ? [NSString stringWithUTF8String:plan] : @"free";
        if (planStr.length == 0) planStr = @"free";
        
        // Capitalize first letter
        self.planNameLabel.stringValue = [[[planStr substringToIndex:1] uppercaseString] stringByAppendingString:[planStr substringFromIndex:1]];
        
        // Show upgrade button only for free plan
        self.upgradeButton.hidden = ![planStr.lowercaseString isEqualToString:@"free"];
    } else {
        self.signedInContainer.hidden = YES;
        self.planContainer.hidden = YES;
        self.signedOutContainer.hidden = NO;
        self.avatarImageView.hidden = YES;
        self.avatarInitialLabel.hidden = NO;
    }
    
    // Force layout update
    [self.accountView layoutSubtreeIfNeeded];
}

- (void)loadAvatarFromURL:(NSString *)urlString {
    NSURL *url = [NSURL URLWithString:urlString];
    if (!url) return;
    
    dispatch_async(dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0), ^{
        NSData *imageData = [NSData dataWithContentsOfURL:url];
        if (imageData) {
            NSImage *image = [[NSImage alloc] initWithData:imageData];
            if (image) {
                dispatch_async(dispatch_get_main_queue(), ^{
                    self.avatarImageView.image = image;
                    self.avatarImageView.hidden = NO;
                    self.avatarInitialLabel.hidden = YES;
                });
            }
        }
    });
}

- (void)refreshAccountInfo {
    // Immediately show cached data (no lag)
    [self updateAccountUI];
    
    // Refresh from API in background, then update UI on main thread
    dispatch_async(dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0), ^{
        refreshAccountFromAPI();
        dispatch_async(dispatch_get_main_queue(), ^{
            [self updateAccountUI];
        });
    });
}

- (void)signInClicked:(NSButton *)sender {
    openConnectUrl();
}

- (void)signOutClicked:(NSButton *)sender {
    signOutAccount();
    [self updateAccountUI];
    
    // Resize window to fit signed out view
    [self showTab:@"Account"];
}

- (void)upgradeClicked:(NSButton *)sender {
    [[NSWorkspace sharedWorkspace] openURL:[NSURL URLWithString:@"https://ollama.com/upgrade"]];
}

- (void)manageClicked:(NSButton *)sender {
    [[NSWorkspace sharedWorkspace] openURL:[NSURL URLWithString:@"https://ollama.com/settings"]];
}

- (void)viewUsageClicked:(NSButton *)sender {
    [[NSWorkspace sharedWorkspace] openURL:[NSURL URLWithString:@"https://ollama.com/settings"]];
}

- (void)loadSettings {
    self.exposeCheckbox.state = getSettingsExpose() ? NSControlStateValueOn : NSControlStateValueOff;
    self.browserCheckbox.state = getSettingsBrowser() ? NSControlStateValueOn : NSControlStateValueOff;
    
    const char* modelsPath = getSettingsModels();
    if (modelsPath) {
        NSString *path = [NSString stringWithUTF8String:modelsPath];
        self.modelsPathControl.URL = [NSURL fileURLWithPath:path];
    }
    
    int contextLength = getSettingsContextLength();
    if (contextLength < 4096) contextLength = 4096;
    if (contextLength > 262144) contextLength = 262144;
    int sliderIndex = [self sliderIndexFromContextLength:contextLength];
    self.contextLengthSlider.intValue = sliderIndex;
}

- (void)showSettings {
    [self loadSettings];
    [self showTab:@"General"];
    [self.window makeKeyAndOrderFront:nil];
    [NSApp activateIgnoringOtherApps:YES];
}

#pragma mark - Actions

- (void)exposeChanged:(NSButton *)sender {
    setSettingsExpose(sender.state == NSControlStateValueOn);
    restartOllamaServer();
}

- (void)browserChanged:(NSButton *)sender {
    setSettingsBrowser(sender.state == NSControlStateValueOn);
    restartOllamaServer();
}

- (void)pathControlClicked:(NSPathControl *)sender {
    NSPathControlItem *clickedItem = sender.clickedPathItem;
    if (clickedItem && clickedItem.URL) {
        [[NSWorkspace sharedWorkspace] selectFile:clickedItem.URL.path inFileViewerRootedAtPath:@""];
    }
}

- (void)pathControlDoubleClicked:(NSPathControl *)sender {
    if (sender.URL) {
        [[NSWorkspace sharedWorkspace] openURL:sender.URL];
    }
}

- (void)chooseModelsPath:(NSButton *)sender {
    NSOpenPanel *panel = [NSOpenPanel openPanel];
    panel.canChooseFiles = NO;
    panel.canChooseDirectories = YES;
    panel.allowsMultipleSelection = NO;
    panel.canCreateDirectories = YES;
    panel.prompt = @"Choose";
    panel.message = @"Choose a folder to store Ollama models";
    
    if (self.modelsPathControl.URL) {
        panel.directoryURL = self.modelsPathControl.URL;
    }
    
    [panel beginSheetModalForWindow:self.window completionHandler:^(NSModalResponse result) {
        if (result == NSModalResponseOK && panel.URLs.firstObject) {
            NSURL *url = panel.URLs.firstObject;
            self.modelsPathControl.URL = url;
            setSettingsModels([url.path UTF8String]);
            restartOllamaServer();
        }
    }];
}

- (void)resetModelsPath:(NSButton *)sender {
    NSString *defaultPath = [NSHomeDirectory() stringByAppendingPathComponent:@".ollama/models"];
    self.modelsPathControl.URL = [NSURL fileURLWithPath:defaultPath];
    setSettingsModels([defaultPath UTF8String]);
    restartOllamaServer();
}

// Context length values: 4K, 8K, 16K, 32K, 64K, 128K, 256K
static const int kContextLengthValues[] = {4096, 8192, 16384, 32768, 65536, 131072, 262144};
static const int kContextLengthCount = 7;

- (int)contextLengthFromSliderIndex:(int)index {
    if (index < 0) index = 0;
    if (index >= kContextLengthCount) index = kContextLengthCount - 1;
    return kContextLengthValues[index];
}

- (int)sliderIndexFromContextLength:(int)value {
    for (int i = kContextLengthCount - 1; i >= 0; i--) {
        if (value >= kContextLengthValues[i]) return i;
    }
    return 0;
}

- (void)contextLengthSliderChanged:(NSSlider *)sender {
    int index = (int)round(sender.doubleValue);
    int value = [self contextLengthFromSliderIndex:index];
    setSettingsContextLength(value);
}

- (void)windowWillClose:(NSNotification *)notification {}

- (void)windowDidBecomeKey:(NSNotification *)notification {
    // Refresh account info when window becomes active (e.g., after signing in via browser)
    if ([self.currentTab isEqualToString:@"Account"]) {
        // Refresh from API in background, then update UI
        dispatch_async(dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0), ^{
            refreshAccountFromAPI();
            dispatch_async(dispatch_get_main_queue(), ^{
                [self updateAccountUI];
                // Resize window if needed (signed in vs signed out)
                [self showTab:@"Account"];
            });
        });
    }
}

@end

void openNativeSettings(void) {
    dispatch_async(dispatch_get_main_queue(), ^{
        [[SettingsWindowController sharedController] showSettings];
    });
}
