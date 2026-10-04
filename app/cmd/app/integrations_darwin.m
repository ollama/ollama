#import "integrations_darwin.h"
#import "app_darwin.h"
#import "resources_darwin.h"
#import "../../updater/updater_darwin.h"

NSNotificationName const OLIntegrationsDidChangeNotification =
    @"OLIntegrationsDidChangeNotification";

static NSString *const ClaudeDownloadPageURL = @"https://claude.com/download";
static NSString *const ChatGPTDownloadPageURL = @"https://chatgpt.com/download";
static NSString *const ChatGPTDiskImageURL =
    @"https://persistent.oaistatic.com/codex-app-prod/Codex.dmg";

static NSString *OLAppName(OLIntegrationApp app) {
    return app == OLIntegrationAppClaude ? @"Claude" : @"ChatGPT";
}

// OLRequestCountStatus describes how many requests an app made, or is empty
// before it makes any.
static NSString *OLRequestCountStatus(unsigned long long requests) {
    if (requests == 0) {
        return @"";
    }
    return requests == 1
        ? @"1 request this session"
        : [NSString stringWithFormat:@"%llu requests this session", requests];
}

@interface OLIntegrationState ()
@property(nonatomic, readwrite) OLIntegrationApp app;
@property(nonatomic, readwrite, copy) NSString *name;
@property(nonatomic, readwrite) BOOL installed;
@property(nonatomic, readwrite) BOOL enabled;
@property(nonatomic, readwrite) BOOL ready;
@property(nonatomic, readwrite) BOOL busy;
@property(nonatomic, readwrite, copy) NSString *status;
@end

@implementation OLIntegrationState
@end

@interface OLIntegrations () <NSURLSessionDownloadDelegate>
// progress holds a status such as "Downloading Claude…" for each app that
// is busy, keyed by OLIntegrationApp.
@property(nonatomic, strong) NSMutableDictionary<NSNumber *, NSString *> *progress;
@property(nonatomic, strong) dispatch_queue_t stateQueue;

@property(nonatomic) OLIntegrationApp downloadApp;
@property(nonatomic, strong, nullable) NSURLSession *downloadSession;
@property(nonatomic, strong, nullable) NSURLSessionDownloadTask *downloadTask;
@property(nonatomic, strong, nullable) NSAlert *downloadAlert;
@property(nonatomic, strong, nullable) NSProgressIndicator *downloadProgress;
@property(nonatomic, strong, nullable) NSURL *downloadedInstallerURL;
@property(nonatomic, strong, nullable) NSError *downloadError;
@property(nonatomic) BOOL downloadCancelled;
@property(nonatomic) BOOL downloadCompleted;
@property(nonatomic) BOOL downloadModalRunning;
@end

@implementation OLIntegrations

+ (instancetype)sharedIntegrations {
    static OLIntegrations *integrations;
    static dispatch_once_t once;
    dispatch_once(&once, ^{
        integrations = [[OLIntegrations alloc] init];
    });
    return integrations;
}

- (instancetype)init {
    self = [super init];
    if (self) {
        _progress = [NSMutableDictionary dictionary];
        _stateQueue = dispatch_queue_create("com.ollama.integrations",
                                            DISPATCH_QUEUE_SERIAL);
    }
    return self;
}

#pragma mark - State

- (void)loadStateForApp:(OLIntegrationApp)app
             completion:(void (^)(OLIntegrationState *state))completion {
    NSString *progress = self.progress[@(app)];
    dispatch_async(self.stateQueue, ^{
        OLIntegrationState *state = app == OLIntegrationAppClaude
            ? [OLIntegrations claudeState]
            : [OLIntegrations chatGPTState];
        dispatch_async(dispatch_get_main_queue(), ^{
            NSString *current = self.progress[@(app)] ?: progress;
            if (current != nil) {
                state.busy = YES;
                state.status = current;
            }
            completion(state);
        });
    });
}

// These read state from Go and may run on any thread.
+ (OLIntegrationState *)claudeState {
    OLIntegrationState *state = [[OLIntegrationState alloc] init];
    state.app = OLIntegrationAppClaude;
    state.name = OLAppName(OLIntegrationAppClaude);
    state.installed = IsClaudeDesktopInstalled();
    BOOL startFailed = ClaudeGatewayStartFailed();
    BOOL portConflict = startFailed && ClaudeGatewayPortConflict();
    state.enabled = state.installed && IsClaudeGatewayConfigured();
    state.ready = state.enabled && !startFailed;

    NSString *failure = portConflict
        ? [NSString stringWithFormat:@"Port %d is in use", ClaudeGatewayPort()]
        : (startFailed ? @"Unable to use Ollama" : nil);
    long long requests = ClaudeGatewayRequestCount();
    if (state.enabled) {
        state.status = failure ?: (requests >= 0
            ? OLRequestCountStatus((unsigned long long)requests)
            : @"");
    } else {
        state.status = !state.installed ? @"Not installed" : (failure ?: @"");
    }
    return state;
}

+ (OLIntegrationState *)chatGPTState {
    OLIntegrationState *state = [[OLIntegrationState alloc] init];
    state.app = OLIntegrationAppChatGPT;
    state.name = OLAppName(OLIntegrationAppChatGPT);
    state.installed = IsCodexDesktopInstalled();
    state.enabled = IsCodexDesktopConnected();
    state.ready = state.installed && state.enabled;
    if (state.enabled) {
        state.status = OLRequestCountStatus(CodexDesktopRequestCount());
    } else {
        state.status = state.installed ? @"" : @"Not installed";
    }
    return state;
}

- (void)setProgress:(nullable NSString *)progress forApp:(OLIntegrationApp)app {
    self.progress[@(app)] = progress;
    [self postChangeForApp:app];
}

- (void)postChangeForApp:(OLIntegrationApp)app {
    [[NSNotificationCenter defaultCenter]
        postNotificationName:OLIntegrationsDidChangeNotification
                      object:self
                    userInfo:@{@"app": @(app)}];
}

- (void)requestCountDidChange:(OLIntegrationApp)app {
    [self postChangeForApp:app];
}

#pragma mark - Turning apps on and off

- (void)setApp:(OLIntegrationApp)app enabled:(BOOL)enabled {
    if (self.progress[@(app)] != nil) {
        return;
    }

    BOOL installed = app == OLIntegrationAppClaude ? IsClaudeDesktopInstalled()
                                                   : IsCodexDesktopInstalled();
    if (enabled && !installed) {
        if (![self confirmDownloadOfApp:app] ||
            ![self downloadApp:app]) {
            [self postChangeForApp:app];
            return;
        }
    }

    BOOL restart = app == OLIntegrationAppClaude ? IsClaudeDesktopRunning()
                                                 : IsCodexDesktopRunning();
    if (restart && ![self confirmRestartOfApp:app enabling:enabled]) {
        [self postChangeForApp:app];
        return;
    }

    [self setProgress:enabled ? @"Connecting…" : @"Disconnecting…" forApp:app];
    dispatch_async(dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^{
        BOOL succeeded = app == OLIntegrationAppClaude
            ? SetClaudeGatewayInstalled(enabled, restart)
            : SetCodexDesktopConnected(enabled, restart);
        dispatch_async(dispatch_get_main_queue(), ^{
            [self setProgress:nil forApp:app];
            if (!succeeded) {
                [self showFailureForApp:app enabling:enabled];
                return;
            }
            if (enabled && app == OLIntegrationAppClaude) {
                [self openApp:app];
            }
        });
    });
}

- (BOOL)confirmDownloadOfApp:(OLIntegrationApp)app {
    NSString *name = OLAppName(app);
    NSAlert *alert = [[NSAlert alloc] init];
    [alert setAlertStyle:NSAlertStyleInformational];
    [alert setIcon:OLApplicationIcon()];
    [alert setMessageText:[NSString stringWithFormat:@"%@ is not installed", name]];
    [alert setInformativeText:[NSString stringWithFormat:
        @"Download %@ to add Ollama models to the %@ app.", name, name]];
    [alert addButtonWithTitle:[NSString stringWithFormat:@"Download %@", name]];
    [alert addButtonWithTitle:@"Cancel"];
    return [alert runModal] == NSAlertFirstButtonReturn;
}

- (BOOL)confirmRestartOfApp:(OLIntegrationApp)app enabling:(BOOL)enabled {
    NSAlert *alert = [[NSAlert alloc] init];
    [alert setAlertStyle:NSAlertStyleWarning];
    [alert setIcon:OLApplicationIcon()];
    if (app == OLIntegrationAppClaude) {
        [alert setMessageText:enabled
            ? @"Restart Claude Desktop to use Ollama?"
            : @"Restart Claude Desktop to remove Ollama?"];
        [alert setInformativeText:enabled
            ? @"Claude Desktop must restart to use Ollama. Any running task will stop."
            : @"Claude Desktop must restart to remove Ollama. Any running task will stop."];
        [alert addButtonWithTitle:@"Restart Claude Desktop"];
    } else {
        [alert setMessageText:enabled
            ? @"Restart ChatGPT to add Ollama models?"
            : @"Restart ChatGPT to remove Ollama models?"];
        [alert setInformativeText:enabled
            ? @"ChatGPT must restart to add Ollama models. Any running task will stop."
            : @"ChatGPT must restart to remove Ollama models. Any running task will stop."];
        [alert addButtonWithTitle:@"Restart ChatGPT"];
    }
    [alert addButtonWithTitle:@"Cancel"];
    return [alert runModal] == NSAlertFirstButtonReturn;
}

- (void)showFailureForApp:(OLIntegrationApp)app enabling:(BOOL)enabled {
    NSAlert *alert = [[NSAlert alloc] init];
    [alert setAlertStyle:NSAlertStyleWarning];
    [alert setIcon:OLApplicationIcon()];
    if (app == OLIntegrationAppChatGPT) {
        [alert setMessageText:enabled
            ? @"Unable to add Ollama models to ChatGPT"
            : @"Unable to remove Ollama models from ChatGPT"];
        [alert setInformativeText:
            @"ChatGPT could not complete the model update. Check the Ollama log for details, then try again."];
        [alert runModal];
        return;
    }

    BOOL portConflict = enabled && ClaudeGatewayPortConflict();
    char *rawError = ClaudeGatewayErrorMessage();
    NSString *gatewayError = rawError != NULL && rawError[0] != '\0'
        ? [NSString stringWithUTF8String:rawError]
        : nil;
    free(rawError);
    [alert setMessageText:portConflict
        ? [NSString stringWithFormat:@"Port %d is already in use", ClaudeGatewayPort()]
        : (enabled ? @"Unable to use Ollama with Claude"
                   : @"Unable to remove Ollama from Claude")];
    [alert setInformativeText:portConflict
        ? [NSString stringWithFormat:
              @"Change OLLAMA_HOST or quit the app using port %d, then try again.",
              ClaudeGatewayPort()]
        : (gatewayError.length > 0
              ? gatewayError
              : @"Ollama could not update Claude. Check the Ollama log for details.")];
    [alert runModal];
}

#pragma mark - Opening apps

- (NSArray<NSString *> *)applicationPathsForApp:(OLIntegrationApp)app {
    NSArray<NSString *> *names = app == OLIntegrationAppClaude
        ? @[@"Claude"]
        : @[@"ChatGPT", @"Codex"];
    NSMutableArray<NSString *> *paths = [NSMutableArray array];
    for (NSString *name in names) {
        NSString *bundle = [name stringByAppendingPathExtension:@"app"];
        [paths addObject:[@"/Applications" stringByAppendingPathComponent:bundle]];
        [paths addObject:[[NSHomeDirectory() stringByAppendingPathComponent:@"Applications"]
                             stringByAppendingPathComponent:bundle]];
    }
    return paths;
}

- (nullable NSString *)installedPathForApp:(OLIntegrationApp)app {
    for (NSString *path in [self applicationPathsForApp:app]) {
        if ([[NSFileManager defaultManager] fileExistsAtPath:path]) {
            return path;
        }
    }
    return nil;
}

- (void)openApp:(OLIntegrationApp)app {
    NSString *name = OLAppName(app);
    NSString *path = [self installedPathForApp:app];
    if (path == nil) {
        NSAlert *alert = [[NSAlert alloc] init];
        [alert setAlertStyle:NSAlertStyleWarning];
        [alert setMessageText:[NSString stringWithFormat:@"Unable to open %@", name]];
        [alert setInformativeText:[NSString stringWithFormat:
            @"Install %@ in Applications, then try again.", name]];
        [alert runModal];
        return;
    }

    NSWorkspaceOpenConfiguration *configuration =
        [NSWorkspaceOpenConfiguration configuration];
    configuration.activates = YES;
    configuration.createsNewApplicationInstance = NO;
    [[NSWorkspace sharedWorkspace]
        openApplicationAtURL:[NSURL fileURLWithPath:path]
               configuration:configuration
           completionHandler:^(NSRunningApplication *application, NSError *error) {
               (void)application;
               if (error != nil) {
                   appLogInfo([NSString stringWithFormat:@"Unable to open %@: %@",
                                                         name, error]);
               }
           }];
}

- (nullable NSImage *)iconForApp:(OLIntegrationApp)app {
    NSString *path = [self installedPathForApp:app];
    if (path != nil) {
        NSImage *icon = [[NSWorkspace sharedWorkspace] iconForFile:path];
        if (icon != nil) {
            return icon;
        }
    }
    NSString *name = OLAppName(app);
    NSImage *bundled = [OLResourceBundle() imageForResource:name.lowercaseString];
    if (bundled != nil) {
        bundled = [bundled copy];
        [bundled setTemplate:app == OLIntegrationAppChatGPT];
        return bundled;
    }
    return [NSImage imageWithSystemSymbolName:app == OLIntegrationAppClaude
                                                  ? @"sparkles"
                                                  : @"bubble.left.and.bubble.right"
                     accessibilityDescription:name];
}

#pragma mark - Downloading apps

- (void)showDownloadFailureForApp:(OLIntegrationApp)app
                            error:(NSError *)error
                       installing:(BOOL)installing {
    NSString *name = OLAppName(app);
    appLogInfo([NSString stringWithFormat:@"Unable to %@ %@: %@",
                                          installing ? @"install" : @"download",
                                          name, error]);
    NSAlert *alert = [[NSAlert alloc] init];
    [alert setAlertStyle:NSAlertStyleWarning];
    [alert setIcon:OLApplicationIcon()];
    [alert setMessageText:[NSString stringWithFormat:installing
        ? @"%@ couldn’t be installed"
        : @"%@ couldn’t be downloaded", name]];
    [alert setInformativeText:[NSString stringWithFormat:installing
        ? @"Try again or install %@ from its website."
        : @"Try again or download %@ from its website.", name]];
    [alert addButtonWithTitle:@"Open download page"];
    [alert addButtonWithTitle:@"Cancel"];
    if ([alert runModal] == NSAlertFirstButtonReturn) {
        NSString *page = app == OLIntegrationAppClaude ? ClaudeDownloadPageURL
                                                       : ChatGPTDownloadPageURL;
        [[NSWorkspace sharedWorkspace] openURL:[NSURL URLWithString:page]];
    }
}

// downloadApp: downloads and installs app while showing progress, and
// returns whether it was installed.
- (BOOL)downloadApp:(OLIntegrationApp)app {
    if (self.downloadTask != nil) {
        return NO;
    }

    BOOL chatGPT = app == OLIntegrationAppChatGPT;
    NSString *name = OLAppName(app);
    NSString *downloadURLString = ChatGPTDiskImageURL;
    NSString *authorization = nil;
    if (!chatGPT) {
        char *rawAuthorization = NULL;
        char *rawURL = ClaudeDesktopDownloadRequest(&rawAuthorization);
        downloadURLString = rawURL == NULL ? nil : [NSString stringWithUTF8String:rawURL];
        authorization = rawAuthorization == NULL
            ? nil
            : [NSString stringWithUTF8String:rawAuthorization];
        free(rawURL);
        free(rawAuthorization);
    }
    NSURL *url = downloadURLString == nil ? nil : [NSURL URLWithString:downloadURLString];
    if (url == nil || (!chatGPT && authorization.length == 0)) {
        NSError *error = [NSError
            errorWithDomain:@"com.ollama.app"
                       code:3
                   userInfo:@{NSLocalizedDescriptionKey: chatGPT
                       ? @"Ollama could not prepare the ChatGPT download."
                       : @"Ollama could not authenticate the download request."}];
        [self showDownloadFailureForApp:app error:error installing:NO];
        return NO;
    }

    [self setProgress:[NSString stringWithFormat:@"Downloading %@…", name] forApp:app];

    self.downloadAlert = [[NSAlert alloc] init];
    [self.downloadAlert setAlertStyle:NSAlertStyleInformational];
    [self.downloadAlert setIcon:OLApplicationIcon()];
    [self.downloadAlert setMessageText:[NSString stringWithFormat:@"Downloading %@", name]];
    [self.downloadAlert setInformativeText:[NSString stringWithFormat:
        @"%@ will be installed when the download finishes.", name]];
    [self.downloadAlert addButtonWithTitle:@"Cancel"];
    self.downloadProgress = [[NSProgressIndicator alloc]
        initWithFrame:NSMakeRect(0, 0, 260, 12)];
    [self.downloadProgress setStyle:NSProgressIndicatorStyleBar];
    [self.downloadProgress setMinValue:0];
    [self.downloadProgress setMaxValue:100];
    [self.downloadProgress setIndeterminate:YES];
    [self.downloadProgress startAnimation:nil];
    [self.downloadAlert setAccessoryView:self.downloadProgress];

    self.downloadApp = app;
    self.downloadCancelled = NO;
    self.downloadCompleted = NO;
    self.downloadedInstallerURL = nil;
    self.downloadError = nil;
    self.downloadSession = [NSURLSession
        sessionWithConfiguration:[NSURLSessionConfiguration ephemeralSessionConfiguration]
                        delegate:self
                   delegateQueue:[NSOperationQueue mainQueue]];
    NSMutableURLRequest *request = [NSMutableURLRequest requestWithURL:url];
    if (authorization.length > 0) {
        [request setValue:authorization forHTTPHeaderField:@"Authorization"];
    }
    self.downloadTask = [self.downloadSession downloadTaskWithRequest:request];

    [NSApp activateIgnoringOtherApps:YES];
    [self.downloadTask resume];
    self.downloadModalRunning = YES;
    NSModalResponse response = [self.downloadAlert runModal];
    self.downloadModalRunning = NO;
    if (response == NSAlertFirstButtonReturn && !self.downloadCompleted) {
        self.downloadCancelled = YES;
        [self.downloadTask cancel];
        return NO;
    }
    if (self.downloadCompleted) {
        return [self finishDownload];
    }
    return NO;
}

- (void)URLSession:(NSURLSession *)session
                          task:(NSURLSessionTask *)task
    willPerformHTTPRedirection:(NSHTTPURLResponse *)response
                    newRequest:(NSURLRequest *)request
             completionHandler:(void (^)(NSURLRequest *_Nullable))completionHandler {
    (void)response;
    if (session != self.downloadSession || task != self.downloadTask) {
        completionHandler(request);
        return;
    }
    // The download link is signed for Ollama; don't pass that to the CDN.
    NSMutableURLRequest *redirect = [request mutableCopy];
    [redirect setValue:nil forHTTPHeaderField:@"Authorization"];
    completionHandler(redirect);
}

- (void)URLSession:(NSURLSession *)session
                 downloadTask:(NSURLSessionDownloadTask *)downloadTask
                 didWriteData:(int64_t)bytesWritten
            totalBytesWritten:(int64_t)totalBytesWritten
    totalBytesExpectedToWrite:(int64_t)totalBytesExpectedToWrite {
    (void)session;
    (void)bytesWritten;
    if (downloadTask != self.downloadTask || totalBytesExpectedToWrite <= 0) {
        return;
    }
    if (self.downloadProgress.indeterminate) {
        [self.downloadProgress stopAnimation:nil];
        [self.downloadProgress setIndeterminate:NO];
    }
    [self.downloadProgress
        setDoubleValue:(100.0 * totalBytesWritten) / totalBytesExpectedToWrite];
}

- (void)URLSession:(NSURLSession *)session
                 downloadTask:(NSURLSessionDownloadTask *)downloadTask
    didFinishDownloadingToURL:(NSURL *)location {
    (void)session;
    if (downloadTask != self.downloadTask || self.downloadCancelled) {
        return;
    }

    NSError *error = nil;
    NSHTTPURLResponse *response =
        [downloadTask.response isKindOfClass:[NSHTTPURLResponse class]]
            ? (NSHTTPURLResponse *)downloadTask.response
            : nil;
    NSString *host = response.URL.host.lowercaseString;
    BOOL chatGPT = self.downloadApp == OLIntegrationAppChatGPT;
    BOOL trustedHost = chatGPT
        ? [host isEqualToString:@"persistent.oaistatic.com"]
        : ([host isEqualToString:@"claude.ai"] || [host hasSuffix:@".claude.ai"]);
    if (response.statusCode != 200 || !trustedHost ||
        ![response.URL.scheme isEqualToString:@"https"]) {
        error = [NSError
            errorWithDomain:@"com.ollama.app"
                       code:1
                   userInfo:@{NSLocalizedDescriptionKey: chatGPT
                       ? @"ChatGPT returned an invalid download response."
                       : @"Claude returned an invalid download response."}];
    }

    if (error == nil) {
        NSDictionary *attributes = [[NSFileManager defaultManager]
            attributesOfItemAtPath:location.path
                             error:&error];
        if (error == nil && [attributes fileSize] < 1024 * 1024) {
            error = [NSError
                errorWithDomain:@"com.ollama.app"
                           code:2
                       userInfo:@{NSLocalizedDescriptionKey: chatGPT
                           ? @"ChatGPT returned an incomplete download."
                           : @"Claude returned an incomplete download."}];
        }
    }

    if (error == nil) {
        NSString *fileName = [NSString
            stringWithFormat:chatGPT ? @"ChatGPT-%@.dmg" : @"Claude-%@.zip",
                             [NSUUID UUID].UUIDString];
        NSURL *installerURL = [NSURL fileURLWithPath:
            [NSTemporaryDirectory() stringByAppendingPathComponent:fileName]];
        if ([[NSFileManager defaultManager] moveItemAtURL:location
                                                   toURL:installerURL
                                                   error:&error]) {
            self.downloadedInstallerURL = installerURL;
        }
    }
    self.downloadError = error;
}

- (void)URLSession:(NSURLSession *)session
                    task:(NSURLSessionTask *)task
    didCompleteWithError:(NSError *)error {
    if (session != self.downloadSession || task != self.downloadTask) {
        return;
    }
    if (error != nil && !self.downloadCancelled) {
        self.downloadError = error;
    }
    self.downloadCompleted = YES;
    if (self.downloadModalRunning) {
        [NSApp abortModal];
        return;
    }
    [self finishDownload];
}

- (BOOL)finishDownload {
    OLIntegrationApp app = self.downloadApp;
    BOOL chatGPT = app == OLIntegrationAppChatGPT;
    NSString *name = OLAppName(app);
    BOOL cancelled = self.downloadCancelled;
    NSURL *installerURL = self.downloadedInstallerURL;
    NSError *downloadError = self.downloadError;

    [self.downloadProgress stopAnimation:nil];
    [self.downloadAlert.window orderOut:nil];
    [self.downloadSession finishTasksAndInvalidate];
    self.downloadTask = nil;
    self.downloadSession = nil;
    self.downloadProgress = nil;
    self.downloadAlert = nil;
    self.downloadError = nil;
    self.downloadedInstallerURL = nil;
    self.downloadCancelled = NO;
    self.downloadCompleted = NO;

    if (cancelled) {
        [self setProgress:nil forApp:app];
        return NO;
    }
    if (downloadError != nil || installerURL == nil) {
        [self setProgress:nil forApp:app];
        NSError *error = downloadError ?: [NSError
            errorWithDomain:@"com.ollama.app"
                       code:3
                   userInfo:@{NSLocalizedDescriptionKey: chatGPT
                       ? @"The ChatGPT installer was not downloaded."
                       : @"The Claude installer was not downloaded."}];
        [self showDownloadFailureForApp:app error:error installing:NO];
        return NO;
    }

    [self setProgress:[NSString stringWithFormat:@"Installing %@…", name] forApp:app];
    BOOL installed = NO;
    if (chatGPT) {
        installed = [self installChatGPTFromDiskImage:installerURL];
    } else {
        installed = InstallClaudeDesktopArchive(installerURL.fileSystemRepresentation);
    }

    NSError *removeError = nil;
    if (![[NSFileManager defaultManager] removeItemAtURL:installerURL error:&removeError]) {
        appLogInfo([NSString stringWithFormat:@"Unable to remove downloaded %@ installer: %@",
                                              name, removeError]);
    }
    [self setProgress:nil forApp:app];
    if (!installed) {
        NSError *error = [NSError
            errorWithDomain:@"com.ollama.app"
                       code:4
                   userInfo:@{NSLocalizedDescriptionKey:
                       [NSString stringWithFormat:@"%@ could not be installed.", name]}];
        [self showDownloadFailureForApp:app error:error installing:YES];
    }
    return installed;
}

// installChatGPTFromDiskImage: verifies and copies ChatGPT in the background
// while an alert shows progress.
- (BOOL)installChatGPTFromDiskImage:(NSURL *)imageURL {
    NSAlert *alert = [[NSAlert alloc] init];
    [alert setAlertStyle:NSAlertStyleInformational];
    [alert setIcon:OLApplicationIcon()];
    [alert setMessageText:@"Installing ChatGPT"];
    [alert setInformativeText:@"Ollama is verifying and copying the ChatGPT app."];
    NSButton *installingButton = [alert addButtonWithTitle:@"Installing…"];
    [installingButton setEnabled:NO];
    NSProgressIndicator *progress = [[NSProgressIndicator alloc]
        initWithFrame:NSMakeRect(0, 0, 260, 12)];
    [progress setStyle:NSProgressIndicatorStyleBar];
    [progress setIndeterminate:YES];
    [progress startAnimation:nil];
    [alert setAccessoryView:progress];

    NSString *imagePath = [imageURL.path copy];
    __block BOOL installed = NO;
    dispatch_async(dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^{
        installed = InstallCodexDesktopDiskImage(imagePath.fileSystemRepresentation);
        dispatch_async(dispatch_get_main_queue(), ^{
            [NSApp abortModal];
        });
    });
    [NSApp activateIgnoringOtherApps:YES];
    [alert runModal];
    [progress stopAnimation:nil];
    [alert.window orderOut:nil];
    return installed;
}

@end

// claudeRequestCountChanged is called from Go when Claude sends requests
// through the gateway.
void claudeRequestCountChanged(void) {
    dispatch_async(dispatch_get_main_queue(), ^{
        [[OLIntegrations sharedIntegrations]
            requestCountDidChange:OLIntegrationAppClaude];
    });
}
