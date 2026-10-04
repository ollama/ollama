#import "settings_panes_darwin.h"
#import "settings_client_darwin.h"

static const CGFloat OLAccountLabelWidth = 160;
static const CGFloat OLAvatarSize = 64;

// After people start signing in with their browser, Settings checks for the
// new account this often, for up to OLSignInTimeout.
static const NSTimeInterval OLSignInPollInterval = 3;
static const NSTimeInterval OLSignInTimeout = 5 * 60;

@interface OLAccountSettingsPane ()
@property(nonatomic, strong) NSImageView *avatarView;
@property(nonatomic, strong) NSTextField *nameLabel;
@property(nonatomic, strong) NSTextField *detailLabel;
@property(nonatomic, strong) NSButton *signInButton;
@property(nonatomic, strong) NSButton *manageButton;
@property(nonatomic, strong) NSButton *upgradeButton;
@property(nonatomic, strong) NSButton *signOutButton;
@property(nonatomic, strong) NSProgressIndicator *progress;
@property(nonatomic, strong) NSTextField *statusLabel;
@property(nonatomic, strong) NSStackView *statusRow;
@property(nonatomic, strong) NSButton *cloudCheckbox;
@property(nonatomic, strong) NSTextField *cloudDescription;

@property(nonatomic, strong, nullable) OLAccount *account;
@property(nonatomic, strong, nullable) NSURL *avatarURL;
@property(nonatomic, strong, nullable) NSImage *avatar;
@property(nonatomic, strong, nullable) NSTimer *signInTimer;
@property(nonatomic, strong, nullable) NSDate *signInStarted;
@property(nonatomic) BOOL refreshing;
@end

@implementation OLAccountSettingsPane

- (instancetype)init {
    return [self initWithIdentifier:@"account"
                              title:@"Account"
                              image:[NSImage imageWithSystemSymbolName:@"person.crop.circle"
                                              accessibilityDescription:@"Account"]
                            helpURL:[NSURL URLWithString:@"https://docs.ollama.com/cloud"]];
}

- (void)dealloc {
    [self.signInTimer invalidate];
}

- (NSView *)makeContentView {
    self.avatarView = [NSImageView imageViewWithImage:[self placeholderAvatar]];
    self.avatarView.translatesAutoresizingMaskIntoConstraints = NO;
    self.avatarView.imageScaling = NSImageScaleProportionallyUpOrDown;
    self.avatarView.wantsLayer = YES;
    self.avatarView.layer.cornerRadius = OLAvatarSize / 2;
    self.avatarView.layer.masksToBounds = YES;
    [NSLayoutConstraint activateConstraints:@[
        [self.avatarView.widthAnchor constraintEqualToConstant:OLAvatarSize],
        [self.avatarView.heightAnchor constraintEqualToConstant:OLAvatarSize],
    ]];

    self.nameLabel = [NSTextField labelWithString:@""];
    self.nameLabel.font = [NSFont systemFontOfSize:15 weight:NSFontWeightSemibold];
    self.nameLabel.alignment = NSTextAlignmentCenter;
    self.nameLabel.lineBreakMode = NSLineBreakByTruncatingTail;

    self.detailLabel = [NSTextField labelWithString:@""];
    self.detailLabel.textColor = [NSColor secondaryLabelColor];
    self.detailLabel.alignment = NSTextAlignmentCenter;
    self.detailLabel.lineBreakMode = NSLineBreakByTruncatingTail;

    self.signInButton = OLPushButton(@"Sign In…", self, @selector(signIn:));
    self.manageButton = OLPushButton(@"Manage Account…", self, @selector(manageAccount:));
    self.upgradeButton = OLPushButton(@"Upgrade…", self, @selector(upgrade:));
    self.signOutButton = OLPushButton(@"Sign Out…", self, @selector(signOut:));
    NSStackView *buttons = OLHorizontalStack(@[
        self.signInButton, self.manageButton, self.upgradeButton, self.signOutButton,
    ]);

    self.progress = [[NSProgressIndicator alloc] initWithFrame:NSZeroRect];
    self.progress.style = NSProgressIndicatorStyleSpinning;
    self.progress.controlSize = NSControlSizeSmall;
    self.progress.displayedWhenStopped = NO;
    self.progress.translatesAutoresizingMaskIntoConstraints = NO;
    self.statusLabel = OLDescriptionLabel(@"");
    self.statusLabel.alignment = NSTextAlignmentCenter;
    self.statusLabel.preferredMaxLayoutWidth = OLSettingsContentWidth - 40;
    self.statusRow = [NSStackView stackViewWithViews:@[self.progress, self.statusLabel]];
    self.statusRow.orientation = NSUserInterfaceLayoutOrientationHorizontal;
    self.statusRow.alignment = NSLayoutAttributeCenterY;
    self.statusRow.spacing = 6;

    NSStackView *header = [NSStackView stackViewWithViews:@[
        self.avatarView, self.nameLabel, self.detailLabel, buttons, self.statusRow,
    ]];
    header.orientation = NSUserInterfaceLayoutOrientationVertical;
    header.alignment = NSLayoutAttributeCenterX;
    header.spacing = 4;
    [header setCustomSpacing:10 afterView:self.avatarView];
    [header setCustomSpacing:14 afterView:self.detailLabel];
    [header setCustomSpacing:10 afterView:buttons];

    OLSettingsForm *form = [[OLSettingsForm alloc] initWithWidth:OLSettingsContentWidth
                                                      labelWidth:OLAccountLabelWidth];
    self.cloudCheckbox =
        OLCheckbox(@"Include cloud models in my model list", self, @selector(toggleCloud:));
    [form addRowWithLabel:@"Cloud:" view:self.cloudCheckbox];
    self.cloudDescription = [form addDescription:@""];

    NSBox *separator = [[NSBox alloc] initWithFrame:NSZeroRect];
    separator.boxType = NSBoxSeparator;
    separator.translatesAutoresizingMaskIntoConstraints = NO;

    NSStackView *content = [NSStackView stackViewWithViews:@[header, separator, form.gridView]];
    content.orientation = NSUserInterfaceLayoutOrientationVertical;
    content.alignment = NSLayoutAttributeCenterX;
    content.spacing = 20;
    [separator.widthAnchor constraintEqualToAnchor:content.widthAnchor].active = YES;

    [self showAccount:nil];
    [self showSettings:nil];
    return content;
}

- (void)viewDidDisappear {
    [super viewDidDisappear];
    [self stopWaitingForSignIn];
}

- (void)reload {
    OLSettingsClient *client = [OLSettingsClient sharedClient];
    [client loadSettings:^(OLAppSettings *settings, NSString *error) {
        (void)error;
        [self showSettings:settings];
    }];

    if (self.account == nil) {
        // Show the last known account right away while checking ollama.com.
        [client loadAccountRefreshing:NO completion:^(OLAccount *account, NSString *error) {
            (void)error;
            if (self.account == nil) {
                [self showAccount:account];
            }
        }];
    }
    [self refreshAccount];
}

- (void)refreshAccount {
    if (self.refreshing) {
        return;
    }
    self.refreshing = YES;
    [[OLSettingsClient sharedClient] loadAccountRefreshing:YES
                                                completion:^(OLAccount *account, NSString *error) {
        self.refreshing = NO;
        [self showAccount:account];
        if (account.signedIn) {
            [self stopWaitingForSignIn];
        }
        if (error != nil && account.cached) {
            [self showStatus:@"Couldn’t reach ollama.com. Showing your account from the last check."
                     waiting:NO];
        }
    }];
}

#pragma mark - Showing the account

- (NSImage *)placeholderAvatar {
    NSImageSymbolConfiguration *configuration =
        [NSImageSymbolConfiguration configurationWithPointSize:OLAvatarSize
                                                        weight:NSFontWeightRegular];
    configuration = [configuration configurationByApplyingConfiguration:
        [NSImageSymbolConfiguration configurationWithHierarchicalColor:[NSColor secondaryLabelColor]]];
    return [[NSImage imageWithSystemSymbolName:@"person.crop.circle.fill"
                      accessibilityDescription:@"Account"]
        imageWithSymbolConfiguration:configuration];
}

- (void)showAccount:(nullable OLAccount *)account {
    self.account = account;
    BOOL waiting = self.signInTimer != nil;
    BOOL signedIn = account.signedIn;

    if (account == nil) {
        self.nameLabel.stringValue = @"";
        self.detailLabel.stringValue = @"";
    } else if (signedIn) {
        self.nameLabel.stringValue = account.name;
        NSMutableArray<NSString *> *details = [NSMutableArray array];
        if (account.email.length > 0) {
            [details addObject:account.email];
        }
        if (account.plan.length > 0) {
            [details addObject:[NSString stringWithFormat:@"%@ plan", account.plan.capitalizedString]];
        }
        self.detailLabel.stringValue = [details componentsJoinedByString:@" · "];
    } else {
        self.nameLabel.stringValue = @"Not signed in";
        self.detailLabel.stringValue = @"Sign in to use cloud models and web search.";
    }

    self.signInButton.hidden = account == nil || signedIn;
    self.signInButton.enabled = !waiting;
    self.manageButton.hidden = !signedIn || account.manageURL == nil;
    self.upgradeButton.hidden = !signedIn || account.upgradeURL == nil ||
        ![account.plan.lowercaseString isEqualToString:@"free"];
    self.signOutButton.hidden = !signedIn;

    [self showStatus:waiting ? @"Finish signing in with your browser." : nil waiting:waiting];
    [self showAvatarForAccount:account];
    [self contentDidChange];
}

// showStatus:waiting: shows a note under the account, with a spinner while
// waiting for people to sign in.
- (void)showStatus:(nullable NSString *)status waiting:(BOOL)waiting {
    if (waiting) {
        [self.progress startAnimation:nil];
    } else {
        [self.progress stopAnimation:nil];
    }
    self.statusLabel.stringValue = status ?: @"";
    self.statusRow.hidden = status.length == 0;
    [self contentDidChange];
}

- (void)showAvatarForAccount:(nullable OLAccount *)account {
    NSURL *url = account.signedIn ? account.avatarURL : nil;
    if (url == nil) {
        if (!account.signedIn || !account.cached) {
            self.avatarURL = nil;
            self.avatar = nil;
        }
        self.avatarView.image = self.avatar ?: [self placeholderAvatar];
        return;
    }
    if ([url isEqual:self.avatarURL] && self.avatar != nil) {
        self.avatarView.image = self.avatar;
        return;
    }
    self.avatarURL = url;
    NSURLSessionDataTask *task = [[NSURLSession sharedSession]
          dataTaskWithURL:url
        completionHandler:^(NSData *data, NSURLResponse *response, NSError *error) {
            (void)response;
            NSImage *image = error == nil && data != nil ? [[NSImage alloc] initWithData:data] : nil;
            dispatch_async(dispatch_get_main_queue(), ^{
                if (image == nil || ![url isEqual:self.avatarURL]) {
                    return;
                }
                self.avatar = image;
                self.avatarView.image = image;
            });
        }];
    [task resume];
}

- (void)showSettings:(nullable OLAppSettings *)settings {
    self.cloudCheckbox.state = settings.cloudEnabled ? NSControlStateValueOn : NSControlStateValueOff;
    self.cloudCheckbox.enabled = settings != nil && !settings.cloudDisabledByEnv;
    self.cloudDescription.stringValue = settings.cloudDisabledByEnv
        ? @"Turned off by the OLLAMA_NO_CLOUD environment variable."
        : @"Cloud models run on ollama.com. Turning this off also turns off web search, "
          @"so Ollama only uses this Mac.";
    [self contentDidChange];
}

#pragma mark - Actions

- (void)signIn:(id)sender {
    (void)sender;
    [[OLSettingsClient sharedClient] signInURL:^(NSURL *url, NSString *error) {
        if (url == nil) {
            [self presentErrorWithTitle:@"Ollama couldn’t start signing in" message:error];
            return;
        }
        [[NSWorkspace sharedWorkspace] openURL:url];
        [self waitForSignIn];
    }];
}

- (void)waitForSignIn {
    [self.signInTimer invalidate];
    self.signInStarted = [NSDate date];
    __weak OLAccountSettingsPane *weakSelf = self;
    self.signInTimer = [NSTimer scheduledTimerWithTimeInterval:OLSignInPollInterval
                                                       repeats:YES
                                                         block:^(NSTimer *timer) {
        OLAccountSettingsPane *pane = weakSelf;
        if (pane == nil || -[pane.signInStarted timeIntervalSinceNow] > OLSignInTimeout) {
            [timer invalidate];
            [pane stopWaitingForSignIn];
            return;
        }
        [pane refreshAccount];
    }];
    [self showAccount:self.account];
}

- (void)stopWaitingForSignIn {
    if (self.signInTimer == nil) {
        return;
    }
    [self.signInTimer invalidate];
    self.signInTimer = nil;
    self.signInStarted = nil;
    [self showAccount:self.account];
}

- (void)manageAccount:(id)sender {
    (void)sender;
    if (self.account.manageURL != nil) {
        [[NSWorkspace sharedWorkspace] openURL:self.account.manageURL];
    }
}

- (void)upgrade:(id)sender {
    (void)sender;
    if (self.account.upgradeURL != nil) {
        [[NSWorkspace sharedWorkspace] openURL:self.account.upgradeURL];
    }
}

- (void)signOut:(id)sender {
    (void)sender;
    NSAlert *alert = [[NSAlert alloc] init];
    alert.messageText = @"Sign out of ollama.com?";
    alert.informativeText = @"This Mac won’t be able to use cloud models or web search until you sign in again.";
    [alert addButtonWithTitle:@"Sign Out"];
    [alert addButtonWithTitle:@"Cancel"];
    alert.buttons.firstObject.hasDestructiveAction = YES;
    [alert beginSheetModalForWindow:self.view.window
                  completionHandler:^(NSModalResponse response) {
                      if (response != NSAlertFirstButtonReturn) {
                          return;
                      }
                      self.signOutButton.enabled = NO;
                      [[OLSettingsClient sharedClient] signOut:^(OLAccount *account, NSString *error) {
                          self.signOutButton.enabled = YES;
                          [self showAccount:account];
                          if (error != nil) {
                              [self presentErrorWithTitle:@"Ollama couldn’t sign out" message:error];
                          }
                      }];
                  }];
}

- (void)toggleCloud:(NSButton *)sender {
    BOOL enabled = sender.state == NSControlStateValueOn;
    sender.enabled = NO;
    [[OLSettingsClient sharedClient] setCloudEnabled:enabled
                                          completion:^(OLAppSettings *settings, NSString *error) {
        [self showSettings:settings];
        if (error != nil) {
            [self presentErrorWithTitle:enabled ? @"Ollama couldn’t include cloud models"
                                                : @"Ollama couldn’t remove cloud models"
                                message:error];
        }
    }];
}

@end
