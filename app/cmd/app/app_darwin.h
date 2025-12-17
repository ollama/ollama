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

void run(bool firstTimeRun, bool startHidden);
void killOtherInstances();
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
void clearUpdateAvailable();
void quit();
void uiRequest(char *path);
void registerSelfAsLoginItem(bool firstTimeRun);
void unregisterSelfFromLoginItem();
void setWindowDelegate(void *window);
void showWindow(uintptr_t wndPtr);
void hideWindow(uintptr_t wndPtr);
void styleWindow(uintptr_t wndPtr);
void drag(uintptr_t wndPtr);
void doubleClick(uintptr_t wndPtr);
void handleConnectURL();
