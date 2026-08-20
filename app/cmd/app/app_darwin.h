#import <Cocoa/Cocoa.h>
#import <Security/Security.h>

@interface AppDelegate : NSObject <NSApplicationDelegate>
- (void)applicationDidFinishLaunching:(NSNotification *)aNotification;
@end

enum AppMove
{
    CannotMove,
    UserDeclinedMove,
    MoveCompleted,
    AlreadyMoved,
    LoginSession,
    PermissionDenied,
    MoveError,
};

void run(bool showOnboarding, bool startHidden);
void killOtherInstances();
bool otherOllamaInstanceRunning(void);
enum AppMove askToMoveToApplications();
int createSymlinkWithAuthorization();
int installSymlink(const char *cliPath);
extern void Restart();
// extern void Quit();
void StartUI(const char *path);
void ShowUI();
void StopUI();
void StartUpdate();
void darwinStartHiddenTasks();
void launchApp(const char *appPath);
void updateAvailable();
void quit();
void uiRequest(char *path);
void registerSelfAsLoginItem(bool firstTimeRun);
void unregisterSelfFromLoginItem();
void setWindowDelegate(void *window);
void showWindow(uintptr_t wndPtr);
void hideWindow(uintptr_t wndPtr);
void styleWindow(uintptr_t wndPtr);
void setWindowResizable(uintptr_t wndPtr, bool resizable);
void drag(uintptr_t wndPtr);
void doubleClick(uintptr_t wndPtr);
void handleConnectURL();
bool SetClaudeGatewayInstalled(bool installed, bool restartClaude);
bool RestoreClaudeGatewayForShutdown(void);
bool IsClaudeGatewayConfigured(void);
bool IsClaudeDesktopInstalled(void);
bool IsClaudeDesktopRunning(void);
bool ClaudeGatewayStartFailed(void);
bool ClaudeGatewayPortConflict(void);
int ClaudeGatewayPort(void);
void RefreshClaudeProxyMenu(void);
void updateClaudeProxyMenu(unsigned long long routed);
