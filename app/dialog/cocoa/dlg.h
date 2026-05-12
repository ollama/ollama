#include <objc/NSObjCRuntime.h>

typedef enum {
	MSG_YESNO,
	MSG_ERROR,
	MSG_INFO,
} AlertStyle;

typedef struct {
	char* msg;
	char* title;
	AlertStyle style;
} AlertDlgParams;

#define LOADDLG 0
#define SAVEDLG 1
#define DIRDLG 2 // browse for directory

typedef struct {
	int mode; /* which dialog style to invoke (see earlier defines) */
	char* buf; /* buffer to store selected file */
	int nbuf; /* number of bytes allocated at buf */
	char* title; /* title for dialog box (can be nil) */
	void** exts; /* list of valid extensions (elements actual type is NSString*) */
	int numext; /* number of items in exts */
	int relaxext; /* allow other extensions? */
	char* startDir; /* directory to start in (can be nil) */
	char* filename; /* default filename for dialog box (can be nil) */
	int showHidden; /* show hidden files? */
	int allowMultiple; /* allow multiple file selection? */
} FileDlgParams;

typedef enum {
	DLG_OK,
	DLG_CANCEL,
	DLG_URLFAIL,
} DlgResult;

DlgResult alertDlg(AlertDlgParams*);
DlgResult fileDlg(FileDlgParams*);

void* NSStr(void* buf, int len);
void NSRelease(void* obj);
