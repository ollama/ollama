#import <Cocoa/Cocoa.h>

NS_ASSUME_NONNULL_BEGIN

// OLResourceBundle returns the bundle that holds Ollama's images. During
// development the app runs outside a bundle, so this finds the bundle in the
// source tree.
NSBundle *OLResourceBundle(void);

// OLApplicationIcon returns Ollama's app icon for alerts and windows.
NSImage *OLApplicationIcon(void);

NS_ASSUME_NONNULL_END
