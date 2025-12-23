#import <Cocoa/Cocoa.h>

@interface SettingsWindowController : NSWindowController <NSWindowDelegate>

// General tab
@property(nonatomic, strong) NSButton *exposeCheckbox;
@property(nonatomic, strong) NSButton *browserCheckbox;
@property(nonatomic, strong) NSSlider *contextLengthSlider;

// Models tab
@property(nonatomic, strong) NSPathControl *modelsPathControl;
@property(nonatomic, strong) NSButton *modelsPathButton;

// Account tab
@property(nonatomic, strong) NSView *avatarView;
@property(nonatomic, strong) NSTextField *avatarInitialLabel;
@property(nonatomic, strong) NSImageView *avatarImageView;
@property(nonatomic, strong) NSTextField *accountNameLabel;
@property(nonatomic, strong) NSTextField *accountEmailLabel;
@property(nonatomic, strong) NSButton *manageButton;
@property(nonatomic, strong) NSButton *signOutButton;
@property(nonatomic, strong) NSButton *signInButton;
@property(nonatomic, strong) NSView *signedInContainer;
@property(nonatomic, strong) NSView *signedOutContainer;

// Plan section
@property(nonatomic, strong) NSView *planContainer;
@property(nonatomic, strong) NSTextField *planNameLabel;
@property(nonatomic, strong) NSButton *upgradeButton;
@property(nonatomic, strong) NSButton *viewUsageButton;

+ (instancetype)sharedController;
- (void)showSettings;

@end

// Go callbacks for settings
void openNativeSettings(void);
