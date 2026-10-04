#import <Cocoa/Cocoa.h>
#import "settings_client_darwin.h"

NS_ASSUME_NONNULL_BEGIN

// OLSettingsWindowController shows Ollama's settings in a window with a
// toolbar of panes, following the macOS conventions for settings windows.
@interface OLSettingsWindowController : NSWindowController
- (instancetype)init;
// showPane: shows the window at the pane with identifier, or at the pane
// people viewed last when identifier is nil or unknown.
- (void)showPane:(nullable NSString *)identifier;
@end

NS_ASSUME_NONNULL_END
