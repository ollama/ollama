#import <Cocoa/Cocoa.h>

NS_ASSUME_NONNULL_BEGIN

// OLOnboardingWindowController welcomes people the first time they open
// Ollama: it offers an ollama.com account, then shows how to run a model.
@interface OLOnboardingWindowController : NSWindowController
- (instancetype)init;
@end

NS_ASSUME_NONNULL_END
