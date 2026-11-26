#import "onboarding_darwin.h"
#import "resources_darwin.h"
#import "settings_client_darwin.h"

// Onboarding matches the welcome flow earlier versions of the app showed in
// their main window, including its light appearance.

// While people sign in with their browser, onboarding checks for the new
// account this often, for up to OLOnboardingSignInTimeout.
static const NSTimeInterval OLOnboardingSignInPollInterval = 1;
static const NSTimeInterval OLOnboardingSignInTimeout = 5 * 60;

static NSString *const OLFirstModelCommand = @"ollama";

typedef NS_ENUM(NSInteger, OLOnboardingStep) {
    OLOnboardingStepWelcome,
    OLOnboardingStepAccount,
    OLOnboardingStepRun,
};

static NSColor *OLColor(uint32_t rgb) {
    return [NSColor colorWithSRGBRed:((rgb >> 16) & 0xff) / 255.0
                               green:((rgb >> 8) & 0xff) / 255.0
                                blue:(rgb & 0xff) / 255.0
                               alpha:1];
}

static NSFont *OLRoundedFont(CGFloat size) {
    NSFont *font = [NSFont systemFontOfSize:size weight:NSFontWeightMedium];
    NSFontDescriptor *rounded =
        [font.fontDescriptor fontDescriptorWithDesign:NSFontDescriptorSystemDesignRounded];
    return (rounded != nil ? [NSFont fontWithDescriptor:rounded size:size] : nil) ?: font;
}

static NSAttributedString *OLText(NSString *text, NSFont *font, NSColor *color, CGFloat lineHeight) {
    NSMutableParagraphStyle *style = [[NSMutableParagraphStyle alloc] init];
    style.alignment = NSTextAlignmentCenter;
    if (lineHeight > 0) {
        style.minimumLineHeight = lineHeight;
        style.maximumLineHeight = lineHeight;
    }
    return [[NSAttributedString alloc] initWithString:text
                                           attributes:@{
                                               NSFontAttributeName : font,
                                               NSForegroundColorAttributeName : color,
                                               NSParagraphStyleAttributeName : style,
                                           }];
}

// OLLabel returns centered text that wraps at width, or a single line when
// width is 0.
static NSTextField *OLLabel(NSString *text, NSFont *font, NSColor *color, CGFloat lineHeight,
                            CGFloat width) {
    NSTextField *label = width > 0 ? [NSTextField wrappingLabelWithString:@""]
                                   : [NSTextField labelWithString:@""];
    label.attributedStringValue = OLText(text, font, color, lineHeight);
    label.selectable = NO;
    if (width > 0) {
        label.preferredMaxLayoutWidth = width;
        [label.widthAnchor constraintLessThanOrEqualToConstant:width].active = YES;
    }
    return label;
}

static NSImageView *OLImageView(NSImage *_Nullable image, CGFloat size) {
    NSImageView *view = [NSImageView imageViewWithImage:image ?: [[NSImage alloc] init]];
    view.imageScaling = NSImageScaleProportionallyUpOrDown;
    view.translatesAutoresizingMaskIntoConstraints = NO;
    [view.widthAnchor constraintEqualToConstant:size].active = YES;
    [view.heightAnchor constraintEqualToConstant:size].active = YES;
    return view;
}

static NSImageView *OLLogo(CGFloat size) {
    NSImage *logo = [[OLResourceBundle() imageForResource:@"logo"] copy];
    logo.template = YES;
    NSImageView *view = OLImageView(logo, size);
    view.contentTintColor = OLColor(0x0a0a0a);
    return view;
}

// OLCapsuleView fills its bounds with a rounded shape, like the box the
// first model's command is shown in.
@interface OLCapsuleView : NSView
@property(nonatomic, strong) NSColor *fillColor;
@end

@implementation OLCapsuleView

- (void)drawRect:(NSRect)dirtyRect {
    (void)dirtyRect;
    CGFloat radius = NSHeight(self.bounds) / 2;
    [self.fillColor setFill];
    [[NSBezierPath bezierPathWithRoundedRect:self.bounds xRadius:radius yRadius:radius] fill];
}

@end

#pragma mark - Buttons

// OLHoverButton tracks the pointer so its look can follow it, and shows the
// pointing hand like links do.
@interface OLHoverButton : NSButton
@property(nonatomic) BOOL hovering;
@property(nonatomic, strong, nullable) NSTrackingArea *hoverArea;
@end

@implementation OLHoverButton

- (void)updateTrackingAreas {
    [super updateTrackingAreas];
    if (self.hoverArea != nil) {
        [self removeTrackingArea:self.hoverArea];
    }
    self.hoverArea = [[NSTrackingArea alloc]
        initWithRect:NSZeroRect
             options:NSTrackingMouseEnteredAndExited | NSTrackingActiveInActiveApp |
                     NSTrackingInVisibleRect
               owner:self
            userInfo:nil];
    [self addTrackingArea:self.hoverArea];
}

- (void)mouseEntered:(NSEvent *)event {
    (void)event;
    self.hovering = YES;
}

- (void)mouseExited:(NSEvent *)event {
    (void)event;
    self.hovering = NO;
}

- (void)setHovering:(BOOL)hovering {
    _hovering = hovering;
    [self hoveringDidChange];
}

- (void)hoveringDidChange {
    self.needsDisplay = YES;
}

- (void)resetCursorRects {
    if (self.enabled) {
        [self addCursorRect:self.bounds cursor:NSCursor.pointingHandCursor];
    }
}

@end

// OLPillCell draws onboarding's main action, a dark, rounded button.
@interface OLPillCell : NSButtonCell
@end

@implementation OLPillCell

- (void)drawBezelWithFrame:(NSRect)frame inView:(NSView *)controlView {
    BOOL hovering = [controlView isKindOfClass:[OLHoverButton class]] &&
                    ((OLHoverButton *)controlView).hovering;
    NSColor *fill = OLColor(self.isHighlighted ? 0x404040 : hovering ? 0x262626 : 0x171717);
    if (!self.enabled) {
        fill = OLColor(0x404040);
    }
    CGFloat radius = NSHeight(frame) / 2;
    [fill setFill];
    [[NSBezierPath bezierPathWithRoundedRect:frame xRadius:radius yRadius:radius] fill];
}

- (NSRect)drawTitle:(NSAttributedString *)title withFrame:(NSRect)frame inView:(NSView *)controlView {
    (void)frame;
    NSAttributedString *text = [[NSAttributedString alloc]
        initWithString:title.string
            attributes:@{
                NSFontAttributeName : [NSFont systemFontOfSize:14],
                NSForegroundColorAttributeName : NSColor.whiteColor,
            }];
    NSRect bounds = controlView.bounds;
    NSSize size = text.size;
    NSRect rect = NSMakeRect(NSMidX(bounds) - size.width / 2, NSMidY(bounds) - size.height / 2,
                             size.width, size.height);
    [text drawInRect:rect];
    return rect;
}

@end

static OLHoverButton *OLPillButton(NSString *title, id target, SEL action) {
    OLHoverButton *button = [[OLHoverButton alloc] init];
    button.cell = [[OLPillCell alloc] init];
    [button setButtonType:NSButtonTypeMomentaryPushIn];
    button.bezelStyle = NSBezelStyleRegularSquare;
    button.bordered = YES;
    button.title = title;
    button.target = target;
    button.action = action;
    button.translatesAutoresizingMaskIntoConstraints = NO;
    [button.widthAnchor constraintEqualToConstant:240].active = YES;
    [button.heightAnchor constraintEqualToConstant:44].active = YES;
    return button;
}

// OLTextButton is plain text that darkens under the pointer, underlined
// when it's a link.
@interface OLTextButton : OLHoverButton
@property(nonatomic) BOOL underlined;
@property(nonatomic) uint32_t color;
@property(nonatomic) uint32_t hoverColor;
@end

@implementation OLTextButton

- (void)hoveringDidChange {
    NSMutableDictionary *attributes = [@{
        NSFontAttributeName : [NSFont systemFontOfSize:14],
        NSForegroundColorAttributeName : OLColor(self.hovering ? self.hoverColor : self.color),
    } mutableCopy];
    if (self.underlined) {
        attributes[NSUnderlineStyleAttributeName] = @(NSUnderlineStyleSingle);
        attributes[NSUnderlineColorAttributeName] = OLColor(0xd4d4d4);
    }
    self.attributedTitle = [[NSAttributedString alloc] initWithString:self.title
                                                           attributes:attributes];
}

@end

static OLTextButton *OLTextLink(NSString *title, BOOL underlined, uint32_t color,
                                uint32_t hoverColor, id target, SEL action) {
    OLTextButton *button = [[OLTextButton alloc] init];
    button.bordered = NO;
    button.title = title;
    button.underlined = underlined;
    button.color = color;
    button.hoverColor = hoverColor;
    button.target = target;
    button.action = action;
    [button hoveringDidChange];
    return button;
}

#pragma mark - Window

@interface OLOnboardingWindowController () <NSWindowDelegate>
@property(nonatomic) OLOnboardingStep step;
@property(nonatomic, strong) NSStackView *stepView;
@property(nonatomic, strong) NSTitlebarAccessoryViewController *signInAccessory;
@property(nonatomic, strong) OLTextButton *titleSignInButton;
@property(nonatomic, strong) OLHoverButton *signUpButton;
@property(nonatomic, strong) NSTextField *errorLabel;
@property(nonatomic, strong) OLTextButton *retryButton;
@property(nonatomic, strong) NSButton *commandCopyButton;
@property(nonatomic, strong, nullable) OLAccount *account;
@property(nonatomic, strong, nullable) NSTimer *signInTimer;
@property(nonatomic, strong, nullable) NSDate *signInStarted;
// signInAttempt changes when people start or stop signing in, so results
// from an earlier attempt are ignored.
@property(nonatomic) NSInteger signInAttempt;
@property(nonatomic) BOOL checkingAccount;
@end

@implementation OLOnboardingWindowController

- (instancetype)init {
    NSWindow *window = [[NSWindow alloc]
        initWithContentRect:NSMakeRect(0, 0, 560, 600)
                  styleMask:NSWindowStyleMaskTitled | NSWindowStyleMaskClosable |
                            NSWindowStyleMaskMiniaturizable |
                            NSWindowStyleMaskFullSizeContentView
                    backing:NSBackingStoreBuffered
                      defer:YES];
    self = [super initWithWindow:window];
    if (self) {
        window.title = @"Welcome to Ollama";
        window.titleVisibility = NSWindowTitleHidden;
        window.titlebarAppearsTransparent = YES;
        window.movableByWindowBackground = YES;
        window.appearance = [NSAppearance appearanceNamed:NSAppearanceNameAqua];
        window.backgroundColor = NSColor.whiteColor;
        window.releasedWhenClosed = NO;
        window.restorable = NO;
        window.collectionBehavior = NSWindowCollectionBehaviorFullScreenNone;
        window.delegate = self;
        [window standardWindowButton:NSWindowZoomButton].enabled = NO;

        self.titleSignInButton = OLTextLink(@"Sign in", NO, 0x737373, 0x171717, self,
                                            @selector(signIn:));
        NSView *accessory = [[NSView alloc] initWithFrame:NSMakeRect(0, 0, 76, 28)];
        self.titleSignInButton.translatesAutoresizingMaskIntoConstraints = NO;
        [accessory addSubview:self.titleSignInButton];
        [NSLayoutConstraint activateConstraints:@[
            [self.titleSignInButton.centerYAnchor constraintEqualToAnchor:accessory.centerYAnchor],
            [self.titleSignInButton.trailingAnchor constraintEqualToAnchor:accessory.trailingAnchor
                                                                  constant:-16],
        ]];
        self.signInAccessory = [[NSTitlebarAccessoryViewController alloc] init];
        self.signInAccessory.view = accessory;
        self.signInAccessory.layoutAttribute = NSLayoutAttributeTrailing;
        [window addTitlebarAccessoryViewController:self.signInAccessory];

        self.errorLabel = OLLabel(@"", [NSFont systemFontOfSize:14], OLColor(0xdc2626), 0, 400);
        self.retryButton = OLTextLink(@"Try again", YES, 0x525252, 0x0a0a0a, self,
                                      @selector(retry:));

        [self showStep:OLOnboardingStepWelcome];
        [window center];

        [[OLSettingsClient sharedClient]
            loadAccountRefreshing:YES
                       completion:^(OLAccount *account, NSString *error) {
                           (void)error;
                           self.account = account;
                           [self updateSignInAccessory];
                       }];
    }
    return self;
}

#pragma mark - Steps

- (void)showStep:(OLOnboardingStep)step {
    self.step = step;
    [self.stepView removeFromSuperview];
    [self.errorLabel removeFromSuperview];
    [self.retryButton removeFromSuperview];
    [self showError:nil retry:NO];

    switch (step) {
    case OLOnboardingStepWelcome:
        self.stepView = [self welcomeView];
        break;
    case OLOnboardingStepAccount:
        self.stepView = [self accountView];
        break;
    case OLOnboardingStepRun:
        self.stepView = [self runView];
        break;
    }
    [self.stepView setCustomSpacing:12 afterView:self.stepView.arrangedSubviews.lastObject];
    [self.stepView addArrangedSubview:self.errorLabel];
    [self.stepView setCustomSpacing:8 afterView:self.errorLabel];
    [self.stepView addArrangedSubview:self.retryButton];

    NSView *content = self.window.contentView;
    self.stepView.translatesAutoresizingMaskIntoConstraints = NO;
    [content addSubview:self.stepView];
    NSLayoutGuide *area = self.window.contentLayoutGuide;
    [NSLayoutConstraint activateConstraints:@[
        [self.stepView.centerXAnchor constraintEqualToAnchor:area.centerXAnchor],
        [self.stepView.centerYAnchor constraintEqualToAnchor:area.centerYAnchor constant:-12],
        [self.stepView.leadingAnchor constraintGreaterThanOrEqualToAnchor:area.leadingAnchor
                                                                 constant:40],
    ]];
    [self updateSignInAccessory];
}

- (NSStackView *)stackWithViews:(NSArray<NSView *> *)views spacings:(NSArray<NSNumber *> *)spacings {
    NSStackView *stack = [NSStackView stackViewWithViews:views];
    stack.orientation = NSUserInterfaceLayoutOrientationVertical;
    stack.alignment = NSLayoutAttributeCenterX;
    stack.spacing = 0;
    for (NSUInteger i = 0; i < spacings.count && i < views.count; i++) {
        [stack setCustomSpacing:spacings[i].doubleValue afterView:views[i]];
    }
    return stack;
}

- (NSStackView *)welcomeView {
    NSImageView *hello = OLImageView([OLResourceBundle() imageForResource:@"hello"], 72);
    NSTextField *title = OLLabel(@"Welcome to Ollama!", OLRoundedFont(24), OLColor(0x0a0a0a), 32, 0);
    NSTextField *subtitle = OLLabel(
        @"Run open models with your coding agents so you can spend less while keeping your data private.",
        [NSFont systemFontOfSize:14], OLColor(0x737373), 24, 380);

    NSArray *features = @[
        @[ @"apple.terminal", @"Connect your apps",
           @"Power your existing coding apps with open models" ],
        @[ @"arrow.left.arrow.right", @"Easily switch models",
           @"Swap between frontier models in one click." ],
        @[ @"checkmark.shield", @"Your data stays yours",
           @"Your prompt data is never logged or trained on." ],
    ];
    NSMutableArray<NSView *> *rows = [NSMutableArray array];
    NSImageSymbolConfiguration *symbol =
        [NSImageSymbolConfiguration configurationWithPointSize:20 weight:NSFontWeightLight];
    for (NSArray<NSString *> *feature in features) {
        NSImage *image = [[NSImage imageWithSystemSymbolName:feature[0] accessibilityDescription:nil]
            imageWithSymbolConfiguration:symbol];
        NSImageView *icon = OLImageView(image, 28);
        icon.imageScaling = NSImageScaleNone;
        icon.contentTintColor = OLColor(0x404040);
        NSTextField *name = OLLabel(feature[1], [NSFont systemFontOfSize:14 weight:NSFontWeightMedium],
                                    OLColor(0x0a0a0a), 20, 0);
        NSTextField *detail = OLLabel(feature[2], [NSFont systemFontOfSize:13], OLColor(0x737373), 20, 0);
        NSStackView *text = [NSStackView stackViewWithViews:@[ name, detail ]];
        text.orientation = NSUserInterfaceLayoutOrientationVertical;
        text.alignment = NSLayoutAttributeLeading;
        text.spacing = 2;
        NSStackView *row = [NSStackView stackViewWithViews:@[ icon, text ]];
        row.alignment = NSLayoutAttributeTop;
        row.spacing = 16;
        [rows addObject:row];
    }
    NSStackView *list = [NSStackView stackViewWithViews:rows];
    list.orientation = NSUserInterfaceLayoutOrientationVertical;
    list.alignment = NSLayoutAttributeLeading;
    list.spacing = 24;

    OLHoverButton *continueButton = OLPillButton(@"Continue", self, @selector(continueFromWelcome:));
    continueButton.keyEquivalent = @"\r";
    return [self stackWithViews:@[ hello, title, subtitle, list, continueButton ]
                       spacings:@[ @8, @16, @32, @32 ]];
}

- (NSStackView *)accountView {
    NSTextField *title = OLLabel(@"Create an account", OLRoundedFont(24), OLColor(0x0a0a0a), 32, 0);
    NSTextField *benefit = OLLabel(@"Create your account for access to faster, larger open models.",
                                   [NSFont systemFontOfSize:14], OLColor(0xa3a3a3), 24, 400);
    NSTextField *privacy = OLLabel(@"Your data is never logged or trained on.",
                                   [NSFont systemFontOfSize:14], OLColor(0xa3a3a3), 24, 400);
    self.signUpButton = OLPillButton(@"Sign up", self, @selector(signUp:));
    self.signUpButton.keyEquivalent = @"\r";
    OLTextButton *local = OLTextLink(@"No thanks, I’ll use Ollama locally", YES, 0x525252, 0x0a0a0a,
                                     self, @selector(useLocally:));
    NSStackView *view = [self stackWithViews:@[ OLLogo(54), title, benefit, privacy, self.signUpButton, local ]
                                    spacings:@[ @28, @12, @4, @28, @16 ]];
    [self updateSignUpButton];
    return view;
}

- (NSStackView *)runView {
    NSTextField *title = OLLabel(@"Run Ollama", OLRoundedFont(22), OLColor(0x0a0a0a), 28, 0);

    OLCapsuleView *command = [[OLCapsuleView alloc] init];
    command.fillColor = OLColor(0xf5f5f5);
    command.translatesAutoresizingMaskIntoConstraints = NO;
    NSTextField *code = [NSTextField labelWithString:OLFirstModelCommand];
    code.font = [NSFont monospacedSystemFontOfSize:14 weight:NSFontWeightRegular];
    code.textColor = OLColor(0x0a0a0a);
    code.selectable = YES;
    code.translatesAutoresizingMaskIntoConstraints = NO;
    self.commandCopyButton = [NSButton buttonWithImage:[self imageForCopied:NO]
                                         target:self
                                         action:@selector(copyCommand:)];
    self.commandCopyButton.bordered = NO;
    self.commandCopyButton.contentTintColor = OLColor(0xa3a3a3);
    self.commandCopyButton.toolTip = @"Copy command to clipboard";
    self.commandCopyButton.translatesAutoresizingMaskIntoConstraints = NO;
    [command addSubview:code];
    [command addSubview:self.commandCopyButton];
    [NSLayoutConstraint activateConstraints:@[
        [command.widthAnchor constraintEqualToConstant:330],
        [command.heightAnchor constraintEqualToConstant:48],
        [code.leadingAnchor constraintEqualToAnchor:command.leadingAnchor constant:16],
        [code.centerYAnchor constraintEqualToAnchor:command.centerYAnchor],
        [self.commandCopyButton.widthAnchor constraintEqualToConstant:32],
        [self.commandCopyButton.heightAnchor constraintEqualToConstant:32],
        [self.commandCopyButton.trailingAnchor constraintEqualToAnchor:command.trailingAnchor constant:-12],
        [self.commandCopyButton.centerYAnchor constraintEqualToAnchor:command.centerYAnchor],
    ]];

    NSTextField *hint = OLLabel(@"Run this command in your terminal to get started.",
                                [NSFont systemFontOfSize:13], OLColor(0xa3a3a3), 20, 320);
    return [self stackWithViews:@[ OLLogo(42), title, command, hint ] spacings:@[ @24, @24, @12 ]];
}

- (void)showError:(nullable NSString *)message retry:(BOOL)retry {
    self.errorLabel.attributedStringValue =
        OLText(message ?: @"", [NSFont systemFontOfSize:14], OLColor(0xdc2626), 20);
    self.errorLabel.hidden = message == nil;
    self.retryButton.hidden = !retry;
}

// The title bar offers Sign in on the account step, for people who already
// have an account.
- (void)updateSignInAccessory {
    self.titleSignInButton.hidden = self.step != OLOnboardingStepAccount ||
                                    self.account.signedIn || self.signInStarted != nil;
}

- (void)updateSignUpButton {
    BOOL waiting = self.signInStarted != nil;
    self.signUpButton.title = waiting ? @"Finish in your browser…" : @"Sign up";
    self.signUpButton.enabled = !waiting;
    [self updateSignInAccessory];
}

#pragma mark - Actions

- (void)continueFromWelcome:(id)sender {
    (void)sender;
    if (self.account.signedIn) {
        [self finish];
    } else {
        [self showStep:OLOnboardingStepAccount];
    }
}

- (void)signUp:(id)sender {
    (void)sender;
    [self startSigningIn:YES];
}

- (void)signIn:(id)sender {
    (void)sender;
    [self startSigningIn:NO];
}

- (void)startSigningIn:(BOOL)signUp {
    [self showError:nil retry:NO];
    if (self.account.signedIn) {
        [self finish];
        return;
    }
    NSInteger attempt = ++self.signInAttempt;
    self.signInStarted = [NSDate date];
    [self updateSignUpButton];
    [[OLSettingsClient sharedClient]
        onboardingSignInURLForSignUp:signUp
                          completion:^(NSURL *url, NSString *error) {
                              if (attempt != self.signInAttempt) {
                                  return;
                              }
                              if (url == nil) {
                                  NSLog(@"failed to start sign in: %@", error);
                                  [self stopSigningIn];
                                  [self showError:@"Unable to start sign in. Please try again."
                                            retry:NO];
                                  return;
                              }
                              [[NSWorkspace sharedWorkspace] openURL:url];
                              __weak OLOnboardingWindowController *weakSelf = self;
                              self.signInTimer = [NSTimer
                                  scheduledTimerWithTimeInterval:OLOnboardingSignInPollInterval
                                                         repeats:YES
                                                           block:^(NSTimer *timer) {
                                                               (void)timer;
                                                               [weakSelf checkSignIn];
                                                           }];
                          }];
}

// checkSignIn looks for the account people sign in to with their browser.
- (void)checkSignIn {
    if (self.signInStarted == nil || self.checkingAccount) {
        return;
    }
    if (-self.signInStarted.timeIntervalSinceNow > OLOnboardingSignInTimeout) {
        [self stopSigningIn];
        [self showError:@"Connection is taking longer than expected. Please try again." retry:NO];
        return;
    }
    NSInteger attempt = self.signInAttempt;
    self.checkingAccount = YES;
    [[OLSettingsClient sharedClient]
        loadAccountRefreshing:YES
                   completion:^(OLAccount *account, NSString *error) {
                       (void)error;
                       self.checkingAccount = NO;
                       if (attempt != self.signInAttempt || !account.signedIn) {
                           return;
                       }
                       self.account = account;
                       [self stopSigningIn];
                       [NSApp activate];
                       [self.window makeKeyAndOrderFront:nil];
                       [self finish];
                   }];
}

- (void)stopSigningIn {
    self.signInAttempt++;
    [self.signInTimer invalidate];
    self.signInTimer = nil;
    self.signInStarted = nil;
    if (self.step == OLOnboardingStepAccount) {
        [self updateSignUpButton];
    }
}

- (void)useLocally:(id)sender {
    (void)sender;
    [self stopSigningIn];
    [self finish];
}

// finish shows how to run a model and saves that onboarding is done.
- (void)finish {
    [self showStep:OLOnboardingStepRun];
    [self saveCompletion];
}

- (void)saveCompletion {
    [self showError:nil retry:NO];
    [[OLSettingsClient sharedClient] completeOnboarding:^(NSString *error) {
        if (error != nil) {
            NSLog(@"failed to save onboarding: %@", error);
            [self showError:@"Unable to save setup. Please try again." retry:YES];
        }
    }];
}

- (void)retry:(id)sender {
    (void)sender;
    [self saveCompletion];
}

- (NSImage *)imageForCopied:(BOOL)copied {
    NSImageSymbolConfiguration *symbol =
        [NSImageSymbolConfiguration configurationWithPointSize:14 weight:NSFontWeightRegular];
    return [[NSImage imageWithSystemSymbolName:copied ? @"checkmark" : @"square.on.square"
                      accessibilityDescription:copied ? @"Copied" : @"Copy command"]
        imageWithSymbolConfiguration:symbol];
}

- (void)copyCommand:(id)sender {
    (void)sender;
    [[NSPasteboard generalPasteboard] clearContents];
    [[NSPasteboard generalPasteboard] setString:OLFirstModelCommand forType:NSPasteboardTypeString];
    self.commandCopyButton.image = [self imageForCopied:YES];
    __weak OLOnboardingWindowController *weakSelf = self;
    dispatch_after(dispatch_time(DISPATCH_TIME_NOW, 2 * NSEC_PER_SEC), dispatch_get_main_queue(), ^{
        weakSelf.commandCopyButton.image = [weakSelf imageForCopied:NO];
    });
}

#pragma mark - NSWindowDelegate

- (void)windowDidBecomeKey:(NSNotification *)notification {
    (void)notification;
    // People come back here after signing in with their browser.
    [self checkSignIn];
}

- (void)windowWillClose:(NSNotification *)notification {
    (void)notification;
    [self stopSigningIn];
}

@end
