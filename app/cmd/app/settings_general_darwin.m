#import "settings_panes_darwin.h"
#import "login_item_darwin.h"
#import "settings_client_darwin.h"

static const CGFloat OLGeneralLabelWidth = 160;

// OLContextLengthTitle formats a context length in tokens, such as "32K".
static NSString *OLContextLengthTitle(NSInteger length) {
    return [NSString stringWithFormat:@"%ldK", (long)(length / 1024)];
}

@interface OLGeneralSettingsPane ()
@property(nonatomic, strong) OLSettingsForm *form;
@property(nonatomic, strong) NSButton *loginItemCheckbox;
@property(nonatomic, strong) NSButton *loginItemSettingsButton;
@property(nonatomic, strong) NSTextField *loginItemDescription;
@property(nonatomic, strong) NSButton *autoUpdateCheckbox;
@property(nonatomic, strong) NSButton *installUpdateButton;
@property(nonatomic, strong) NSTextField *updateDescription;
@property(nonatomic, strong) NSPathControl *modelsPathControl;
@property(nonatomic, strong) NSButton *changeModelsButton;
@property(nonatomic, strong) NSButton *resetModelsButton;
@property(nonatomic, strong) NSPopUpButton *contextLengthPopUp;
@property(nonatomic, strong) NSButton *exposeCheckbox;
@property(nonatomic, strong) NSButton *exportButton;
@property(nonatomic, strong) NSProgressIndicator *exportProgress;
@property(nonatomic, strong) NSTextField *exportDescription;
@property(nonatomic, strong) NSGridRow *exportRow;
@property(nonatomic, strong, nullable) OLAppSettings *settings;
// contextLengthChecks counts checks for the server's automatic context
// length while it starts.
@property(nonatomic) NSInteger contextLengthChecks;
@end

@implementation OLGeneralSettingsPane

- (instancetype)init {
    self = [self initWithIdentifier:@"general"
                              title:@"General"
                              image:[NSImage imageWithSystemSymbolName:@"gearshape"
                                              accessibilityDescription:@"General"]
                            helpURL:[NSURL URLWithString:@"https://docs.ollama.com/faq"]];
    if (self) {
        [[NSNotificationCenter defaultCenter] addObserver:self
                                                 selector:@selector(settingsDidChange:)
                                                     name:OLSettingsDidChangeNotification
                                                   object:nil];
    }
    return self;
}

- (void)dealloc {
    [[NSNotificationCenter defaultCenter] removeObserver:self];
}

- (NSView *)makeContentView {
    OLSettingsForm *form = [[OLSettingsForm alloc] initWithWidth:OLSettingsContentWidth
                                                      labelWidth:OLGeneralLabelWidth];
    self.form = form;

    self.loginItemCheckbox = OLCheckbox(@"Open Ollama at login", self, @selector(toggleLoginItem:));
    self.loginItemSettingsButton =
        OLPushButton(@"Open Login Items…", self, @selector(openLoginItemsSettings:));
    [form addRowWithLabel:@"Startup:"
                     view:OLControlWithAccessory(self.loginItemCheckbox, self.loginItemSettingsButton)];
    self.loginItemDescription = [form addDescription:@""];

    self.autoUpdateCheckbox =
        OLCheckbox(@"Automatically download updates", self, @selector(toggleAutoUpdate:));
    self.installUpdateButton = OLPushButton(@"Restart to Update", self, @selector(installUpdate:));
    [form addRowWithLabel:@"Updates:"
                     view:OLControlWithAccessory(self.autoUpdateCheckbox, self.installUpdateButton)];
    self.updateDescription = [form addDescription:@""];

    [form addGroupSpacing];
    // The folder in use, with Change and Reset buttons, as Music shows its
    // media folder. Double-click a folder to show it in Finder.
    self.modelsPathControl = [[NSPathControl alloc] initWithFrame:NSZeroRect];
    self.modelsPathControl.pathStyle = NSPathStyleStandard;
    self.modelsPathControl.editable = NO;
    self.modelsPathControl.backgroundColor = [NSColor clearColor];
    self.modelsPathControl.target = self;
    self.modelsPathControl.doubleAction = @selector(revealModelsLocation:);
    self.modelsPathControl.translatesAutoresizingMaskIntoConstraints = NO;
    [self.modelsPathControl setContentCompressionResistancePriority:NSLayoutPriorityDefaultLow
                                                     forOrientation:NSLayoutConstraintOrientationHorizontal];
    [self.modelsPathControl.widthAnchor constraintLessThanOrEqualToConstant:form.controlWidth].active = YES;
    [form addRowWithLabel:@"Model location:" view:self.modelsPathControl];
    self.changeModelsButton = OLPushButton(@"Change…", self, @selector(changeModelsLocation:));
    self.resetModelsButton = OLPushButton(@"Reset", self, @selector(resetModelsLocation:));
    [form addRowWithLabel:nil view:OLHorizontalStack(@[self.changeModelsButton, self.resetModelsButton])];

    self.contextLengthPopUp = [[NSPopUpButton alloc] initWithFrame:NSZeroRect pullsDown:NO];
    self.contextLengthPopUp.translatesAutoresizingMaskIntoConstraints = NO;
    self.contextLengthPopUp.target = self;
    self.contextLengthPopUp.action = @selector(chooseContextLength:);
    [self.contextLengthPopUp.widthAnchor constraintGreaterThanOrEqualToConstant:160].active = YES;
    [form addRowWithLabel:@"Context length:" view:self.contextLengthPopUp];
    [form addDescription:@"How much of a conversation local models can use at once. "
                         @"Longer context uses more memory."];

    [form addGroupSpacing];
    self.exposeCheckbox =
        OLCheckbox(@"Allow connections from other devices", self, @selector(toggleExpose:));
    [form addRowWithLabel:@"Network:" view:self.exposeCheckbox];
    [form addDescription:@"Other devices on your network can connect to Ollama on this Mac."];

    // Earlier versions of the app saved chats. Offer them as files.
    [form addGroupSpacing];
    self.exportButton = OLPushButton(@"Export Chats…", self, @selector(exportChats:));
    self.exportProgress = [[NSProgressIndicator alloc] initWithFrame:NSZeroRect];
    self.exportProgress.style = NSProgressIndicatorStyleSpinning;
    self.exportProgress.controlSize = NSControlSizeSmall;
    self.exportProgress.displayedWhenStopped = NO;
    self.exportProgress.translatesAutoresizingMaskIntoConstraints = NO;
    self.exportRow = [form addRowWithLabel:@"Chat history:"
                                      view:OLHorizontalStack(@[self.exportButton, self.exportProgress])];
    self.exportDescription = [form addDescription:@""];

    [self showSettings:nil];
    [self showLoginItemStatus:OLLoginItemStatusUnavailable];
    return form.gridView;
}

- (void)reload {
    self.contextLengthChecks = 0;
    [self reloadSettings];
    dispatch_async(dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^{
        OLLoginItemStatus status = OLLoginItemGetStatus();
        dispatch_async(dispatch_get_main_queue(), ^{
            [self showLoginItemStatus:status];
        });
    });
}

- (void)reloadSettings {
    [[OLSettingsClient sharedClient] loadSettings:^(OLAppSettings *settings, NSString *error) {
        if (settings != nil) {
            [self showSettings:settings];
        }
        if (error != nil) {
            [self presentErrorWithTitle:@"Ollama couldn’t read its settings" message:error];
        }
    }];
}

// checkContextLengthLater reloads settings until the server, which picks the
// automatic context length when it starts, reports its choice.
- (void)checkContextLengthLater {
    if (self.contextLengthChecks >= 10) {
        return;
    }
    self.contextLengthChecks++;
    dispatch_after(dispatch_time(DISPATCH_TIME_NOW, 2 * NSEC_PER_SEC), dispatch_get_main_queue(), ^{
        if (self.view.window.visible) {
            [self reloadSettings];
        }
    });
}

- (void)settingsDidChange:(NSNotification *)notification {
    (void)notification;
    if (self.viewLoaded && self.view.window != nil) {
        [self reload];
    }
}

#pragma mark - Showing settings

- (void)showSettings:(nullable OLAppSettings *)settings {
    self.settings = settings;
    BOOL loaded = settings != nil;

    self.autoUpdateCheckbox.enabled = loaded;
    self.autoUpdateCheckbox.state = settings.autoUpdate ? NSControlStateValueOn : NSControlStateValueOff;
    self.installUpdateButton.hidden = !settings.updateReady;
    self.updateDescription.stringValue = settings.updateReady
        ? @"A new version of Ollama is ready to install."
        : (settings.version.length > 0
              ? [NSString stringWithFormat:@"You’re using Ollama %@.", settings.version]
              : @"");
    [self setRowHidden:self.updateDescription.stringValue.length == 0 forView:self.updateDescription];

    [self showModelsLocation:settings];
    [self showContextLength:settings];
    if (loaded && settings.defaultContextLength == 0) {
        [self checkContextLengthLater];
    }

    self.exposeCheckbox.enabled = loaded;
    self.exposeCheckbox.state = settings.expose ? NSControlStateValueOn : NSControlStateValueOff;

    NSInteger chats = settings.chatCount;
    self.exportDescription.stringValue = chats == 1
        ? @"Save your chat from earlier versions of Ollama as a Markdown file."
        : [NSString stringWithFormat:
              @"Save your %ld chats from earlier versions of Ollama as Markdown files.", (long)chats];
    self.exportRow.hidden = chats == 0;
    [self setRowHidden:chats == 0 forView:self.exportDescription];
    [self contentDidChange];
}

- (void)showModelsLocation:(nullable OLAppSettings *)settings {
    self.modelsPathControl.URL = settings.modelsPath.length > 0
        ? [NSURL fileURLWithPath:settings.modelsPath isDirectory:YES]
        : nil;
    self.modelsPathControl.toolTip = settings.modelsPath;
    self.changeModelsButton.enabled = settings != nil;
    // Reset goes back to the default folder, so it only applies to others.
    self.resetModelsButton.enabled = settings != nil && !settings.modelsPathIsDefault;
    self.resetModelsButton.toolTip = settings.defaultModelsPath.length > 0
        ? [NSString stringWithFormat:@"Use %@", settings.defaultModelsPath.stringByAbbreviatingWithTildeInPath]
        : nil;
}

- (void)showContextLength:(nullable OLAppSettings *)settings {
    NSPopUpButton *popUp = self.contextLengthPopUp;
    [popUp removeAllItems];
    popUp.enabled = settings != nil;
    if (settings == nil) {
        [popUp addItemWithTitle:@"Loading…"];
        return;
    }

    NSArray<NSNumber *> *lengths = settings.contextLengths.count > 0 ? settings.contextLengths : @[@0];
    for (NSNumber *length in lengths) {
        NSInteger value = length.integerValue;
        NSString *title = OLContextLengthTitle(value);
        if (value == 0) {
            title = settings.defaultContextLength > 0
                ? [NSString stringWithFormat:@"Automatic (%@)",
                                             OLContextLengthTitle(settings.defaultContextLength)]
                : @"Automatic";
        }
        NSMenuItem *item = [[NSMenuItem alloc] initWithTitle:title action:nil keyEquivalent:@""];
        item.tag = value;
        [popUp.menu addItem:item];
        if (value == 0 && lengths.count > 1) {
            [popUp.menu addItem:[NSMenuItem separatorItem]];
        }
    }
    if (![popUp selectItemWithTag:settings.contextLength]) {
        // Keep a length chosen outside of Settings visible.
        NSMenuItem *item = [[NSMenuItem alloc] initWithTitle:OLContextLengthTitle(settings.contextLength)
                                                      action:nil
                                               keyEquivalent:@""];
        item.tag = settings.contextLength;
        [popUp.menu addItem:item];
        [popUp selectItem:item];
    }
}

- (void)showLoginItemStatus:(OLLoginItemStatus)status {
    self.loginItemCheckbox.state = status == OLLoginItemStatusEnabled ? NSControlStateValueOn
                                                                      : NSControlStateValueOff;
    self.loginItemCheckbox.enabled = status != OLLoginItemStatusUnavailable;
    self.loginItemSettingsButton.hidden = status != OLLoginItemStatusRequiresApproval;

    NSString *description = @"";
    if (status == OLLoginItemStatusRequiresApproval) {
        description = @"Opening at login is turned off in System Settings.";
    } else if (status == OLLoginItemStatusUnavailable) {
        description = @"Available when Ollama is in the Applications folder.";
    }
    self.loginItemDescription.stringValue = description;
    [self setRowHidden:description.length == 0 forView:self.loginItemDescription];
    [self contentDidChange];
}

- (void)setRowHidden:(BOOL)hidden forView:(NSView *)view {
    NSGridCell *cell = [self.form.gridView cellForView:view];
    cell.row.hidden = hidden;
}

#pragma mark - Actions

- (void)applySettings:(void (^)(OLSettingsCompletion completion))change
              control:(NSControl *)control
         failureTitle:(NSString *)failureTitle {
    control.enabled = NO;
    // Changes that restart the server pick a new automatic context length.
    self.contextLengthChecks = 0;
    change(^(OLAppSettings *settings, NSString *error) {
        control.enabled = YES;
        if (settings != nil) {
            [self showSettings:settings];
        }
        if (error != nil) {
            [self presentErrorWithTitle:failureTitle message:error];
        }
    });
}

- (void)toggleLoginItem:(NSButton *)sender {
    BOOL enabled = sender.state == NSControlStateValueOn;
    sender.enabled = NO;
    dispatch_async(dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^{
        NSError *error = nil;
        BOOL succeeded = OLLoginItemSetEnabled(enabled, &error);
        OLLoginItemStatus status = OLLoginItemGetStatus();
        dispatch_async(dispatch_get_main_queue(), ^{
            [self showLoginItemStatus:status];
            if (!succeeded) {
                [self presentErrorWithTitle:enabled ? @"Ollama couldn’t open at login"
                                                    : @"Ollama couldn’t stop opening at login"
                                    message:error.localizedDescription];
            } else if (enabled && status == OLLoginItemStatusRequiresApproval) {
                OLLoginItemOpenSystemSettings();
            }
        });
    });
}

- (void)openLoginItemsSettings:(id)sender {
    (void)sender;
    OLLoginItemOpenSystemSettings();
}

- (void)toggleAutoUpdate:(NSButton *)sender {
    BOOL enabled = sender.state == NSControlStateValueOn;
    [self applySettings:^(OLSettingsCompletion completion) {
        [[OLSettingsClient sharedClient] setAutoUpdate:enabled completion:completion];
    }
                control:sender
           failureTitle:@"Ollama couldn’t change automatic updates"];
}

- (void)installUpdate:(id)sender {
    (void)sender;
    [[OLSettingsClient sharedClient] installUpdate];
}

- (void)revealModelsLocation:(NSPathControl *)sender {
    NSURL *url = sender.clickedPathItem.URL ?: sender.URL;
    if (url == nil) {
        return;
    }
    if ([[NSFileManager defaultManager] fileExistsAtPath:url.path]) {
        [[NSWorkspace sharedWorkspace] activateFileViewerSelectingURLs:@[url]];
    } else {
        // Ollama creates the default folder when it first downloads a model.
        [self presentErrorWithTitle:@"The model folder doesn’t exist yet"
                            message:@"Ollama creates it when you download your first model."];
    }
}

- (void)resetModelsLocation:(id)sender {
    (void)sender;
    [self setModelsPath:nil];
}

- (void)changeModelsLocation:(id)sender {
    (void)sender;
    NSOpenPanel *panel = [NSOpenPanel openPanel];
    panel.canChooseDirectories = YES;
    panel.canChooseFiles = NO;
    panel.canCreateDirectories = YES;
    panel.allowsMultipleSelection = NO;
    panel.prompt = @"Choose";
    panel.message = @"Choose where Ollama stores models. Models you’ve already downloaded "
                    @"stay in their current location.";
    if (self.settings.modelsPath.length > 0) {
        panel.directoryURL = [NSURL fileURLWithPath:self.settings.modelsPath isDirectory:YES];
    }
    [panel beginSheetModalForWindow:self.view.window
                  completionHandler:^(NSModalResponse result) {
                      if (result == NSModalResponseOK && panel.URL != nil) {
                          [self setModelsPath:panel.URL.path];
                      }
                  }];
}

- (void)setModelsPath:(nullable NSString *)path {
    [self applySettings:^(OLSettingsCompletion completion) {
        [[OLSettingsClient sharedClient] setModelsPath:path completion:completion];
    }
                control:self.changeModelsButton
           failureTitle:@"Ollama couldn’t use that folder for models"];
}

- (void)chooseContextLength:(NSPopUpButton *)sender {
    NSInteger length = sender.selectedTag;
    if (length == self.settings.contextLength) {
        return;
    }
    [self applySettings:^(OLSettingsCompletion completion) {
        [[OLSettingsClient sharedClient] setContextLength:length completion:completion];
    }
                control:sender
           failureTitle:@"Ollama couldn’t change the context length"];
}

- (void)exportChats:(id)sender {
    (void)sender;
    NSSavePanel *panel = [NSSavePanel savePanel];
    panel.title = @"Export Chats";
    panel.prompt = @"Export";
    panel.message = @"Ollama saves each chat as a Markdown file in this folder.";
    panel.nameFieldLabel = @"Folder Name:";
    panel.nameFieldStringValue = @"Ollama Chats";
    panel.canCreateDirectories = YES;
    panel.directoryURL = [[NSFileManager defaultManager] URLsForDirectory:NSDocumentDirectory
                                                                inDomains:NSUserDomainMask].firstObject;
    [panel beginSheetModalForWindow:self.view.window
                  completionHandler:^(NSModalResponse result) {
                      if (result == NSModalResponseOK && panel.URL != nil) {
                          [self exportChatsToURL:panel.URL];
                      }
                  }];
}

- (void)exportChatsToURL:(NSURL *)url {
    self.exportButton.enabled = NO;
    [self.exportProgress startAnimation:nil];
    [[OLSettingsClient sharedClient] exportChatsToURL:url
                                           completion:^(NSInteger exported, NSString *error) {
        self.exportButton.enabled = YES;
        [self.exportProgress stopAnimation:nil];
        if (exported > 0) {
            [[NSWorkspace sharedWorkspace] openURL:url];
        }
        if (error != nil) {
            [self presentErrorWithTitle:exported > 0 ? @"Some chats couldn’t be exported"
                                                     : @"Ollama couldn’t export your chats"
                                message:error];
        }
    }];
}

- (void)toggleExpose:(NSButton *)sender {
    BOOL enabled = sender.state == NSControlStateValueOn;
    [self applySettings:^(OLSettingsCompletion completion) {
        [[OLSettingsClient sharedClient] setExpose:enabled completion:completion];
    }
                control:sender
           failureTitle:@"Ollama couldn’t change network access"];
}

@end
