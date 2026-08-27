#import <Cocoa/Cocoa.h>
#import <Security/Security.h>
#include <stddef.h>
#include <stdint.h>

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
typedef struct {
    int pid;
    int64_t started_at;
} AppProcessIdentity;
bool otherOllamaProcesses(AppProcessIdentity **processes, size_t *count);
enum AppMove askToMoveToApplications();
int createSymlinkWithAuthorization();
int installSymlink(const char *cliPath);
extern void Restart();
// extern void Quit();
void StartUI(const char *path);
void ShowUI();
bool IsOnboardingActive(void);
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
bool HasUsedClaudeDesktopIntegration(void);
bool RestoreClaudeGatewayForShutdown(void);
bool IsClaudeGatewayConfigured(void);
bool IsClaudeDesktopInstalled(void);
bool IsClaudeDesktopRunning(void);
bool ClaudeGatewayStartFailed(void);
bool ClaudeGatewayPortConflict(void);
char *ClaudeGatewayErrorMessage(void);
int ClaudeGatewayPort(void);
void RefreshClaudeProxyMenu(void);
void updateClaudeProxyMenu(unsigned long long routed);
bool ShowAppsInMenu(void);
void SetShowAppsInMenu(bool visible);
enum ClaudeInstallResult
{
    ClaudeInstallCancelled,
    ClaudeInstallerOpened,
    ClaudeInstallFailed,
};
enum ClaudeInstallResult installClaudeDesktop(void);
char *ClaudeDesktopDownloadRequest(char **authorization);
bool InstallClaudeDesktopArchive(const char *archivePath);
