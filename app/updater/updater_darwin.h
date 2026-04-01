#import <Cocoa/Cocoa.h>

// TODO make these macros so we can extract line numbers from the native code
void appLogInfo(NSString *msg);
void appLogDebug(NSString *msg);
void goLogInfo(const char *msg); 
void goLogDebug(const char *msg); 


AuthorizationRef getAuthorization(NSString *authorizationPrompt,
                                  NSString *right);

AuthorizationRef getAppInstallAuthorization();

const char* verifyExtractedBundle(char *path);
bool chownWithAuthorization(const char *user);