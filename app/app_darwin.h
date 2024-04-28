#import <Cocoa/Cocoa.h>

@interface AppDelegate : NSObject <NSApplicationDelegate>
- (void)applicationDidFinishLaunching:(NSNotification *)aNotification;
@end

void run();
void killOtherInstances();
bool askToMoveToApplications();
int createSymlinkWithAuthorization();
int installSymlink();
extern void Restart();
extern void Quit();
