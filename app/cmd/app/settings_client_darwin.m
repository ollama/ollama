#import "settings_client_darwin.h"
#include "_cgo_export.h"

NSNotificationName const OLSettingsDidChangeNotification = @"OLSettingsDidChangeNotification";

static NSString *OLString(id value) {
    return [value isKindOfClass:[NSString class]] ? value : @"";
}

static BOOL OLBool(id value) {
    return [value respondsToSelector:@selector(boolValue)] ? [value boolValue] : NO;
}

static NSInteger OLInteger(id value) {
    return [value respondsToSelector:@selector(integerValue)] ? [value integerValue] : 0;
}

static NSURL *_Nullable OLURL(id value) {
    NSString *string = OLString(value);
    return string.length > 0 ? [NSURL URLWithString:string] : nil;
}

// OLTakeJSON decodes and frees a JSON object returned by Go.
static NSDictionary *OLTakeJSON(char *json) {
    if (json == NULL) {
        return @{@"error": @"Ollama couldn't read its settings."};
    }
    NSData *data = [NSData dataWithBytesNoCopy:json length:strlen(json) freeWhenDone:YES];
    id object = [NSJSONSerialization JSONObjectWithData:data options:0 error:nil];
    if (![object isKindOfClass:[NSDictionary class]]) {
        return @{@"error": @"Ollama couldn't read its settings."};
    }
    return object;
}

static NSString *_Nullable OLError(NSDictionary *result) {
    NSString *error = OLString(result[@"error"]);
    return error.length > 0 ? error : nil;
}

@interface OLAppSettings ()
- (instancetype)initWithJSON:(NSDictionary *)json;
@end

@implementation OLAppSettings

- (instancetype)initWithJSON:(NSDictionary *)json {
    self = [super init];
    if (self) {
        _version = [OLString(json[@"version"]) copy];
        _updateReady = OLBool(json[@"updateReady"]);
        _autoUpdate = OLBool(json[@"autoUpdate"]);
        _expose = OLBool(json[@"expose"]);
        _modelsPath = [OLString(json[@"modelsPath"]) copy];
        _defaultModelsPath = [OLString(json[@"defaultModelsPath"]) copy];
        _modelsPathIsDefault = OLBool(json[@"modelsPathIsDefault"]);
        _contextLength = OLInteger(json[@"contextLength"]);
        _defaultContextLength = OLInteger(json[@"defaultContextLength"]);
        NSMutableArray<NSNumber *> *lengths = [NSMutableArray array];
        id rawLengths = json[@"contextLengths"];
        if ([rawLengths isKindOfClass:[NSArray class]]) {
            for (id length in rawLengths) {
                if ([length isKindOfClass:[NSNumber class]]) {
                    [lengths addObject:length];
                }
            }
        }
        _contextLengths = [lengths copy];
        _cloudEnabled = OLBool(json[@"cloudEnabled"]);
        _cloudDisabledByEnv = OLBool(json[@"cloudDisabledByEnv"]);
        _chatCount = OLInteger(json[@"chatCount"]);
    }
    return self;
}

@end

@interface OLAccount ()
- (instancetype)initWithJSON:(NSDictionary *)json;
@end

@implementation OLAccount

- (instancetype)initWithJSON:(NSDictionary *)json {
    self = [super init];
    if (self) {
        NSDictionary *account = [json[@"account"] isKindOfClass:[NSDictionary class]]
            ? json[@"account"]
            : @{};
        _signedIn = OLBool(account[@"signedIn"]);
        _name = [OLString(account[@"name"]) copy];
        _email = [OLString(account[@"email"]) copy];
        _plan = [OLString(account[@"plan"]) copy];
        _avatarURL = OLURL(account[@"avatarURL"]);
        _cached = OLBool(account[@"cached"]);
        _manageURL = OLURL(json[@"manageURL"]);
        _upgradeURL = OLURL(json[@"upgradeURL"]);
    }
    return self;
}

@end

@interface OLSettingsClient ()
// Settings changes are applied in order on settingsQueue. Account checks use
// their own queue so a slow network doesn't hold up settings.
@property(nonatomic, strong) dispatch_queue_t settingsQueue;
@property(nonatomic, strong) dispatch_queue_t accountQueue;
@property(nonatomic, strong) dispatch_queue_t exportQueue;
@end

@implementation OLSettingsClient

+ (instancetype)sharedClient {
    static OLSettingsClient *client;
    static dispatch_once_t once;
    dispatch_once(&once, ^{
        client = [[OLSettingsClient alloc] init];
    });
    return client;
}

- (instancetype)init {
    self = [super init];
    if (self) {
        dispatch_queue_attr_t attributes = dispatch_queue_attr_make_with_qos_class(
            DISPATCH_QUEUE_SERIAL, QOS_CLASS_USER_INITIATED, 0);
        _settingsQueue = dispatch_queue_create("com.ollama.settings", attributes);
        _accountQueue = dispatch_queue_create("com.ollama.settings.account", attributes);
        _exportQueue = dispatch_queue_create("com.ollama.settings.export", attributes);
    }
    return self;
}

- (void)onQueue:(dispatch_queue_t)queue
           call:(char *(^)(void))call
     completion:(void (^)(NSDictionary *result))completion {
    dispatch_async(queue, ^{
        NSDictionary *result = OLTakeJSON(call());
        dispatch_async(dispatch_get_main_queue(), ^{
            completion(result);
        });
    });
}

- (void)callSettings:(char *(^)(void))call completion:(OLSettingsCompletion)completion {
    [self onQueue:self.settingsQueue call:call completion:^(NSDictionary *result) {
        NSDictionary *settings = result[@"settings"];
        completion([settings isKindOfClass:[NSDictionary class]]
                       ? [[OLAppSettings alloc] initWithJSON:settings]
                       : nil,
                   OLError(result));
    }];
}

- (void)loadSettings:(OLSettingsCompletion)completion {
    [self callSettings:^char * { return SettingsGet(); } completion:completion];
}

- (void)setAutoUpdate:(BOOL)enabled completion:(OLSettingsCompletion)completion {
    [self callSettings:^char * { return SettingsSetAutoUpdate(enabled); } completion:completion];
}

- (void)setExpose:(BOOL)enabled completion:(OLSettingsCompletion)completion {
    [self callSettings:^char * { return SettingsSetExpose(enabled); } completion:completion];
}

- (void)setModelsPath:(NSString *)path completion:(OLSettingsCompletion)completion {
    NSString *value = [path copy] ?: @"";
    [self callSettings:^char * {
        return SettingsSetModelsPath((char *)value.fileSystemRepresentation);
    }
            completion:completion];
}

- (void)setContextLength:(NSInteger)length completion:(OLSettingsCompletion)completion {
    [self callSettings:^char * { return SettingsSetContextLength((int)length); }
            completion:completion];
}

- (void)setCloudEnabled:(BOOL)enabled completion:(OLSettingsCompletion)completion {
    [self callSettings:^char * { return SettingsSetCloudEnabled(enabled); } completion:completion];
}

- (void)installUpdate {
    // Installing may ask for authorization and replaces this app, so it runs
    // on the main thread like the menu's Restart to Update.
    StartUpdate();
}

- (void)exportChatsToURL:(NSURL *)url
              completion:(void (^)(NSInteger exported, NSString *_Nullable error))completion {
    NSString *path = [url.path copy];
    [self onQueue:self.exportQueue
             call:^char * { return ChatsExport((char *)path.fileSystemRepresentation); }
       completion:^(NSDictionary *result) {
           completion(OLInteger(result[@"exported"]), OLError(result));
       }];
}

- (void)loadAccountRefreshing:(BOOL)refresh completion:(OLAccountCompletion)completion {
    [self onQueue:self.accountQueue
             call:^char * { return AccountGet(refresh); }
       completion:^(NSDictionary *result) {
           completion([[OLAccount alloc] initWithJSON:result], OLError(result));
       }];
}

- (void)signInURL:(void (^)(NSURL *_Nullable url, NSString *_Nullable error))completion {
    [self onQueue:self.accountQueue
             call:^char * { return AccountSignInURL(); }
       completion:^(NSDictionary *result) {
           completion(OLURL(result[@"url"]), OLError(result));
       }];
}

- (void)signOut:(OLAccountCompletion)completion {
    [self onQueue:self.accountQueue
             call:^char * { return AccountSignOut(); }
       completion:^(NSDictionary *result) {
           completion([[OLAccount alloc] initWithJSON:result], OLError(result));
       }];
}

@end
