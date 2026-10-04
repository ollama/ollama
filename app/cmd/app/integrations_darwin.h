#import <Cocoa/Cocoa.h>

NS_ASSUME_NONNULL_BEGIN

// OLIntegrationApp identifies a desktop app that can use Ollama models.
typedef NS_ENUM(NSInteger, OLIntegrationApp) {
    OLIntegrationAppClaude,
    OLIntegrationAppChatGPT,
};

// Posted on the main thread whenever an app's state may have changed, such
// as after it is turned on or off, installed, or receives requests.
extern NSNotificationName const OLIntegrationsDidChangeNotification;

// OLIntegrationState is a snapshot of how an app is set up to use Ollama.
@interface OLIntegrationState : NSObject
@property(nonatomic, readonly) OLIntegrationApp app;
@property(nonatomic, readonly, copy) NSString *name;
@property(nonatomic, readonly) BOOL installed;
// enabled is set when the app is configured to use Ollama models.
@property(nonatomic, readonly) BOOL enabled;
// ready is set when the app is enabled and Ollama is serving it.
@property(nonatomic, readonly) BOOL ready;
// busy is set while Ollama downloads, installs, or reconfigures the app.
@property(nonatomic, readonly) BOOL busy;
// status describes the app's state in a few words, or is empty.
@property(nonatomic, readonly, copy) NSString *status;
@end

// OLIntegrations turns Ollama models on and off in other apps. Use it on the
// main thread; work that can block runs in the background.
@interface OLIntegrations : NSObject
+ (instancetype)sharedIntegrations;
- (void)loadStateForApp:(OLIntegrationApp)app
             completion:(void (^)(OLIntegrationState *state))completion;
// setApp:enabled: offers to install a missing app and asks before restarting
// a running one. It reports failures in an alert.
- (void)setApp:(OLIntegrationApp)app enabled:(BOOL)enabled;
- (void)openApp:(OLIntegrationApp)app;
- (nullable NSImage *)iconForApp:(OLIntegrationApp)app;
// requestCountDidChange: updates the status after the app sends requests.
- (void)requestCountDidChange:(OLIntegrationApp)app;
@end

NS_ASSUME_NONNULL_END
