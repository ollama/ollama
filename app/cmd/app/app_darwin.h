#pragma once

#import <Cocoa/Cocoa.h>
#import <Security/Security.h>
#include <stdbool.h>
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

void run(void);
typedef struct {
    int pid;
    int64_t started_at;
} AppProcessIdentity;
bool otherOllamaProcesses(AppProcessIdentity **processes, size_t *count);
enum AppMove askToMoveToApplications(void);
int installSymlink(const char *cliPath);
void showSettings(const char *pane);
void StartUpdate(void);
void appDidFinishLaunching(bool hidden);
void handleURLScheme(char *url);
void launchApp(const char *appPath);
void updateAvailable(void);
void quit(void);
void registerSelfAsLoginItem(bool firstTimeRun);
void unregisterSelfFromLoginItem(void);
bool SetClaudeGatewayInstalled(bool installed, bool restartClaude);
bool RestoreClaudeGatewayForShutdown(void);
bool IsClaudeGatewayConfigured(void);
bool IsClaudeDesktopInstalled(void);
bool IsClaudeDesktopRunning(void);
bool IsCodexDesktopInstalled(void);
bool IsCodexDesktopConnected(void);
bool IsCodexDesktopRunning(void);
unsigned long long CodexDesktopRequestCount(void);
bool SetCodexDesktopConnected(bool connected, bool restartConfirmed);
bool ClaudeGatewayStartFailed(void);
bool ClaudeGatewayPortConflict(void);
char *ClaudeGatewayErrorMessage(void);
int ClaudeGatewayPort(void);
long long ClaudeGatewayRequestCount(void);
void claudeRequestCountChanged(void);
char *ClaudeDesktopDownloadRequest(char **authorization);
bool InstallClaudeDesktopArchive(const char *archivePath);
bool InstallCodexDesktopDiskImage(const char *imagePath);
