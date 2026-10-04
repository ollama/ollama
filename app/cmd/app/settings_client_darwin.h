#import <Cocoa/Cocoa.h>

NS_ASSUME_NONNULL_BEGIN

// Posted on the main thread when settings change outside the settings
// window, such as when an update finishes downloading.
extern NSNotificationName const OLSettingsDidChangeNotification;

// OLAppSettings holds the app and server settings shown in Settings.
@interface OLAppSettings : NSObject
@property(nonatomic, readonly, copy) NSString *version;
@property(nonatomic, readonly) BOOL updateReady;
@property(nonatomic, readonly) BOOL autoUpdate;
@property(nonatomic, readonly) BOOL expose;
@property(nonatomic, readonly, copy) NSString *modelsPath;
@property(nonatomic, readonly, copy) NSString *defaultModelsPath;
@property(nonatomic, readonly) BOOL modelsPathIsDefault;
// contextLength is 0 when the server picks one from available memory, and
// defaultContextLength is that choice, or 0 when it isn't known yet.
@property(nonatomic, readonly) NSInteger contextLength;
@property(nonatomic, readonly) NSInteger defaultContextLength;
@property(nonatomic, readonly, copy) NSArray<NSNumber *> *contextLengths;
@property(nonatomic, readonly) BOOL cloudEnabled;
// cloudDisabledByEnv is set when OLLAMA_NO_CLOUD turns cloud off.
@property(nonatomic, readonly) BOOL cloudDisabledByEnv;
// chatCount is how many chats earlier versions of the app saved.
@property(nonatomic, readonly) NSInteger chatCount;
@end

// OLAccount is the ollama.com account this device is signed in to.
@interface OLAccount : NSObject
@property(nonatomic, readonly) BOOL signedIn;
@property(nonatomic, readonly, copy) NSString *name;
@property(nonatomic, readonly, copy) NSString *email;
@property(nonatomic, readonly, copy) NSString *plan;
@property(nonatomic, readonly, nullable) NSURL *avatarURL;
// cached is set when ollama.com couldn't be reached and the account comes
// from the last successful check.
@property(nonatomic, readonly) BOOL cached;
@property(nonatomic, readonly, nullable) NSURL *manageURL;
@property(nonatomic, readonly, nullable) NSURL *upgradeURL;
@end

typedef void (^OLSettingsCompletion)(OLAppSettings *_Nullable settings,
                                     NSString *_Nullable error);
typedef void (^OLAccountCompletion)(OLAccount *account, NSString *_Nullable error);

// OLSettingsClient reads and changes settings through the Go app. Calls may
// wait on the disk, the network, or a server restart, so they run in the
// background and call completion on the main thread.
@interface OLSettingsClient : NSObject
+ (instancetype)sharedClient;

- (void)loadSettings:(OLSettingsCompletion)completion;
- (void)setAutoUpdate:(BOOL)enabled completion:(OLSettingsCompletion)completion;
- (void)setExpose:(BOOL)enabled completion:(OLSettingsCompletion)completion;
// setModelsPath: with nil restores the default location.
- (void)setModelsPath:(nullable NSString *)path completion:(OLSettingsCompletion)completion;
- (void)setContextLength:(NSInteger)length completion:(OLSettingsCompletion)completion;
- (void)setCloudEnabled:(BOOL)enabled completion:(OLSettingsCompletion)completion;
- (void)installUpdate;
// exportChatsToURL: saves every chat as a Markdown file in the folder at url.
- (void)exportChatsToURL:(NSURL *)url
              completion:(void (^)(NSInteger exported, NSString *_Nullable error))completion;

// loadAccountRefreshing: returns the cached account, or checks ollama.com
// when refresh is set.
- (void)loadAccountRefreshing:(BOOL)refresh completion:(OLAccountCompletion)completion;
- (void)signInURL:(void (^)(NSURL *_Nullable url, NSString *_Nullable error))completion;
- (void)signOut:(OLAccountCompletion)completion;
@end

NS_ASSUME_NONNULL_END
