#import <Cocoa/Cocoa.h>
#include "dlg.h"
#include <string.h>
#include <sys/syslimits.h>

// Import UniformTypeIdentifiers for macOS 11+
#if __MAC_OS_X_VERSION_MAX_ALLOWED >= 110000
#import <UniformTypeIdentifiers/UniformTypeIdentifiers.h>
#endif

void* NSStr(void* buf, int len) {
	return (void*)[[NSString alloc] initWithBytes:buf length:len encoding:NSUTF8StringEncoding];
}

void checkActivationPolicy() {
	NSApplicationActivationPolicy policy = [NSApp activationPolicy];
	// prohibited NSApp will not show the panel at all.
	// It probably means that this is not run in a GUI app, that would set the policy on its own,
	// but in a terminal app - setting it to accessory will allow dialogs to show
	if (policy == NSApplicationActivationPolicyProhibited) {
		[NSApp setActivationPolicy:NSApplicationActivationPolicyAccessory];
	}
}

void NSRelease(void* obj) {
	[(NSObject*)obj release];
}

@interface AlertDlg : NSObject {
	AlertDlgParams* params;
	DlgResult result;
}
+ (AlertDlg*)init:(AlertDlgParams*)params;
- (DlgResult)run;
@end

DlgResult alertDlg(AlertDlgParams* params) {
	return [[AlertDlg init:params] run];
}

@implementation AlertDlg
+ (AlertDlg*)init:(AlertDlgParams*)params {
	AlertDlg* d = [AlertDlg alloc];
	d->params = params;
	return d;
}

- (DlgResult)run {
	if(![NSThread isMainThread]) {
		[self performSelectorOnMainThread:@selector(run) withObject:nil waitUntilDone:YES];
		return self->result;
	}
	NSAlert* alert = [[NSAlert alloc] init];
	if(self->params->title != nil) {
		[[alert window] setTitle:[[NSString alloc] initWithUTF8String:self->params->title]];
	}
	[alert setMessageText:[[NSString alloc] initWithUTF8String:self->params->msg]];
	switch (self->params->style) {
	case MSG_YESNO:
		[alert addButtonWithTitle:@"Yes"];
		[alert addButtonWithTitle:@"No"];
		break;
	case MSG_ERROR:
		[alert setIcon:[NSImage imageNamed:NSImageNameCaution]];
		[alert addButtonWithTitle:@"OK"];
		break;
	case MSG_INFO:
		[alert setIcon:[NSImage imageNamed:NSImageNameInfo]];
		[alert addButtonWithTitle:@"OK"];
		break;
	}

	checkActivationPolicy();

	self->result = [alert runModal] == NSAlertFirstButtonReturn ? DLG_OK : DLG_CANCEL;
	return self->result;
}
@end

@interface FileDlg : NSObject {
	FileDlgParams* params;
	DlgResult result;
}
+ (FileDlg*)init:(FileDlgParams*)params;
- (DlgResult)run;
@end

DlgResult fileDlg(FileDlgParams* params) {
	return [[FileDlg init:params] run];
}

@implementation FileDlg
+ (FileDlg*)init:(FileDlgParams*)params {
	FileDlg* d = [FileDlg alloc];
	d->params = params;
	return d;
}

- (DlgResult)run {
	if(![NSThread isMainThread]) {
		[self performSelectorOnMainThread:@selector(run) withObject:nil waitUntilDone:YES];
	} else if(self->params->mode == SAVEDLG) {
		self->result = [self save];
	} else {
		self->result = [self load];
	}
	return self->result;
}

- (NSInteger)runPanel:(NSSavePanel*)panel {
	[panel setFloatingPanel:YES];
	[panel setShowsHiddenFiles:self->params->showHidden ? YES : NO];
	[panel setCanCreateDirectories:YES];
	if(self->params->title != nil) {
		[panel setTitle:[[NSString alloc] initWithUTF8String:self->params->title]];
	}
	// Use modern allowedContentTypes API for better file type support (especially video files)
	if(self->params->numext > 0) {
		NSMutableArray *utTypes = [NSMutableArray arrayWithCapacity:self->params->numext];
		NSString** exts = (NSString**)self->params->exts;
		for(int i = 0; i < self->params->numext; i++) {
			UTType *type = [UTType typeWithFilenameExtension:exts[i]];
			if(type) {
				[utTypes addObject:type];
			}
		}
		if([utTypes count] > 0) {
			[panel setAllowedContentTypes:utTypes];
		}
	}
	if(self->params->relaxext) {
		[panel setAllowsOtherFileTypes:YES];
	}
	if(self->params->startDir) {
		[panel setDirectoryURL:[NSURL URLWithString:[[NSString alloc] initWithUTF8String:self->params->startDir]]];
	}
	if(self->params->filename != nil) {
		[panel setNameFieldStringValue:[[NSString alloc] initWithUTF8String:self->params->filename]];
	}

	checkActivationPolicy();

	return [panel runModal];
}

- (DlgResult)save {
	NSSavePanel* panel = [NSSavePanel savePanel];
	if(![self runPanel:panel]) {
		return DLG_CANCEL;
	} else if(![[panel URL] getFileSystemRepresentation:self->params->buf maxLength:self->params->nbuf]) {
		return DLG_URLFAIL;
	}
	return DLG_OK;
}

- (DlgResult)load {
	NSOpenPanel* panel = [NSOpenPanel openPanel];
	if(self->params->mode == DIRDLG) {
		[panel setCanChooseDirectories:YES];
		[panel setCanChooseFiles:NO];
	}
	
	if(self->params->allowMultiple) {
		[panel setAllowsMultipleSelection:YES];
	}
	
	if(![self runPanel:panel]) {
		return DLG_CANCEL;
	}
	
	NSArray* urls = [panel URLs];
	if([urls count] == 0) {
		return DLG_CANCEL;
	}
	
	if(self->params->allowMultiple) {
		// For multiple files, we need to return all paths separated by null bytes
		char* bufPtr = self->params->buf;
		int remainingBuf = self->params->nbuf;
		
		// Calculate total required buffer size first
		int totalSize = 0;
		for(NSURL* url in urls) {
			char tempBuf[PATH_MAX];
			if(![url getFileSystemRepresentation:tempBuf maxLength:PATH_MAX]) {
				return DLG_URLFAIL;
			}
			totalSize += strlen(tempBuf) + 1; // +1 for null terminator
		}
		totalSize += 1; // Final null terminator

		if(totalSize > self->params->nbuf) {
			// Not enough buffer space
			return DLG_URLFAIL;
		}

		// Now actually copy the paths (we know we have space)
		bufPtr = self->params->buf;
		for(NSURL* url in urls) {
			char tempBuf[PATH_MAX];
			[url getFileSystemRepresentation:tempBuf maxLength:PATH_MAX];
			int pathLen = strlen(tempBuf);
			strcpy(bufPtr, tempBuf);
			bufPtr += pathLen + 1;
		}
		*bufPtr = '\0'; // Final null terminator
	} else {
		// Single file/directory selection - write path to buffer
		NSURL* url = [urls firstObject];
		if(![url getFileSystemRepresentation:self->params->buf maxLength:self->params->nbuf]) {
			return DLG_URLFAIL;
		}
	}
	
	return DLG_OK;
}

@end
