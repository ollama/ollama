#import "resources_darwin.h"

NSBundle *OLResourceBundle(void) {
    NSBundle *bundle = [NSBundle mainBundle];
    if ([bundle.bundlePath hasSuffix:@".app"]) {
        return bundle;
    }

    NSString *cwdPath = [[NSFileManager defaultManager] currentDirectoryPath];
    NSArray<NSString *> *bundlePaths = @[
        [cwdPath stringByAppendingPathComponent:@"darwin/Ollama.app"],
        [cwdPath stringByAppendingPathComponent:@"app/darwin/Ollama.app"],
    ];
    for (NSString *bundlePath in bundlePaths) {
        if ([[NSFileManager defaultManager] fileExistsAtPath:bundlePath]) {
            return [NSBundle bundleWithPath:bundlePath] ?: bundle;
        }
    }

    return bundle;
}

NSImage *OLApplicationIcon(void) {
    NSString *iconPath = [OLResourceBundle() pathForResource:@"icon" ofType:@"icns"];
    NSImage *icon = iconPath != nil ? [[NSImage alloc] initWithContentsOfFile:iconPath] : nil;
    return icon ?: [NSApp applicationIconImage];
}
