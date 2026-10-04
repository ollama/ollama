#import <Cocoa/Cocoa.h>

NS_ASSUME_NONNULL_BEGIN

// OLLoginItemStatus describes whether Ollama opens when people log in.
typedef NS_ENUM(NSInteger, OLLoginItemStatus) {
    // The login item isn't available, such as in a development build.
    OLLoginItemStatusUnavailable,
    OLLoginItemStatusEnabled,
    OLLoginItemStatusDisabled,
    // The login item was turned off in System Settings, which must turn it
    // back on.
    OLLoginItemStatusRequiresApproval,
};

OLLoginItemStatus OLLoginItemGetStatus(void);

// OLLoginItemSetEnabled turns opening at login on or off and remembers the
// choice so later launches respect it. It may block briefly, so call it off
// the main thread.
BOOL OLLoginItemSetEnabled(BOOL enabled, NSError **error);

// OLLoginItemOpenSystemSettings opens Login Items in System Settings.
void OLLoginItemOpenSystemSettings(void);

NS_ASSUME_NONNULL_END
