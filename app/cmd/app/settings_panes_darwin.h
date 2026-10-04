#import "settings_pane_darwin.h"

NS_ASSUME_NONNULL_BEGIN

// OLGeneralSettingsPane holds app and server settings: opening at login,
// updates, where models are stored, context length, and network access.
@interface OLGeneralSettingsPane : OLSettingsPane
- (instancetype)init;
@end

// OLAccountSettingsPane shows the ollama.com account this device is signed
// in to, and whether cloud models and web search are on.
@interface OLAccountSettingsPane : OLSettingsPane
- (instancetype)init;
@end

// OLAppsSettingsPane turns Ollama models on and off in other apps.
@interface OLAppsSettingsPane : OLSettingsPane
- (instancetype)init;
@end

NS_ASSUME_NONNULL_END
