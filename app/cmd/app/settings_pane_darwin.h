#import <Cocoa/Cocoa.h>
#import "settings_form_darwin.h"

NS_ASSUME_NONNULL_BEGIN

// OLSettingsPaneWidth is the width of every pane in the settings window.
extern const CGFloat OLSettingsPaneWidth;

// OLSettingsContentWidth is the width available to a pane's content inside
// the window's margins.
extern const CGFloat OLSettingsContentWidth;

// OLSettingsPane is one pane of the settings window. Subclasses build their
// controls in makeContentView and refresh them in reload.
@interface OLSettingsPane : NSViewController
@property(nonatomic, readonly, copy) NSString *paneIdentifier;
@property(nonatomic, readonly, copy) NSString *paneTitle;
@property(nonatomic, readonly, strong) NSImage *paneImage;
@property(nonatomic, readonly, nullable) NSURL *helpURL;
// contentSizeDidChange is called after the pane's content changes size.
@property(nonatomic, copy, nullable) void (^contentSizeDidChange)(OLSettingsPane *pane);

- (instancetype)initWithIdentifier:(NSString *)identifier
                             title:(NSString *)title
                             image:(NSImage *)image
                           helpURL:(nullable NSURL *)helpURL NS_DESIGNATED_INITIALIZER;
- (instancetype)initWithNibName:(nullable NSNibName)nibNameOrNil
                         bundle:(nullable NSBundle *)nibBundleOrNil NS_UNAVAILABLE;
- (instancetype)initWithCoder:(NSCoder *)coder NS_UNAVAILABLE;

// makeContentView returns the pane's controls. It's called once.
- (NSView *)makeContentView;

// reload refreshes the pane from current settings. The window calls it each
// time the pane appears and when the window becomes key.
- (void)reload;

// fittingSize is the size of the pane's view for its current content.
- (NSSize)fittingSize;

// contentDidChange resizes the window to fit the pane's content.
- (void)contentDidChange;

// presentErrorWithTitle:message: explains a failure in a sheet on the
// settings window.
- (void)presentErrorWithTitle:(NSString *)title message:(nullable NSString *)message;
@end

NS_ASSUME_NONNULL_END
