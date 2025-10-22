package dialog

// #cgo pkg-config: gtk+-3.0
// #cgo LDFLAGS: -lX11
// #include <X11/Xlib.h>
// #include <gtk/gtk.h>
// #include <stdlib.h>
// static GtkWidget* msgdlg(GtkWindow *parent, GtkDialogFlags flags, GtkMessageType type, GtkButtonsType buttons, char *msg) {
// 	return gtk_message_dialog_new(parent, flags, type, buttons, "%s", msg);
// }
// static GtkWidget* filedlg(char *title, GtkWindow *parent, GtkFileChooserAction action, char* acceptText) {
// 	return gtk_file_chooser_dialog_new(title, parent, action, "Cancel", GTK_RESPONSE_CANCEL, acceptText, GTK_RESPONSE_ACCEPT, NULL);
// }
import "C"
import "unsafe"

var initSuccess bool

func init() {
	C.XInitThreads()
	initSuccess = (C.gtk_init_check(nil, nil) == C.TRUE)
}

func checkStatus() {
	if !initSuccess {
		panic("gtk initialisation failed; presumably no X server is available")
	}
}

func closeDialog(dlg *C.GtkWidget) {
	C.gtk_widget_destroy(dlg)
	/* The Destroy call itself isn't enough to remove the dialog from the screen; apparently
	** that happens once the GTK main loop processes some further events. But if we're
	** in a non-GTK app the main loop isn't running, so we empty the event queue before
	** returning from the dialog functions.
	** Not sure how this interacts with an actual GTK app... */
	for C.gtk_events_pending() != 0 {
		C.gtk_main_iteration()
	}
}

func runMsgDlg(defaultTitle string, flags C.GtkDialogFlags, msgtype C.GtkMessageType, buttons C.GtkButtonsType, b *MsgBuilder) C.gint {
	checkStatus()
	cmsg := C.CString(b.Msg)
	defer C.free(unsafe.Pointer(cmsg))
	dlg := C.msgdlg(nil, flags, msgtype, buttons, cmsg)
	ctitle := C.CString(firstOf(b.Dlg.Title, defaultTitle))
	defer C.free(unsafe.Pointer(ctitle))
	C.gtk_window_set_title((*C.GtkWindow)(unsafe.Pointer(dlg)), ctitle)
	defer closeDialog(dlg)
	return C.gtk_dialog_run((*C.GtkDialog)(unsafe.Pointer(dlg)))
}

func (b *MsgBuilder) yesNo() bool {
	return runMsgDlg("Confirm?", 0, C.GTK_MESSAGE_QUESTION, C.GTK_BUTTONS_YES_NO, b) == C.GTK_RESPONSE_YES
}

func (b *MsgBuilder) info() {
	runMsgDlg("Information", 0, C.GTK_MESSAGE_INFO, C.GTK_BUTTONS_OK, b)
}

func (b *MsgBuilder) error() {
	runMsgDlg("Error", 0, C.GTK_MESSAGE_ERROR, C.GTK_BUTTONS_OK, b)
}

func (b *FileBuilder) load() (string, error) {
	return chooseFile("Open File", "Open", C.GTK_FILE_CHOOSER_ACTION_OPEN, b)
}

func (b *FileBuilder) save() (string, error) {
	f, err := chooseFile("Save File", "Save", C.GTK_FILE_CHOOSER_ACTION_SAVE, b)
	if err != nil {
		return "", err
	}
	return f, nil
}

func chooseFile(title string, buttonText string, action C.GtkFileChooserAction, b *FileBuilder) (string, error) {
	checkStatus()
	ctitle := C.CString(title)
	defer C.free(unsafe.Pointer(ctitle))
	cbuttonText := C.CString(buttonText)
	defer C.free(unsafe.Pointer(cbuttonText))
	dlg := C.filedlg(ctitle, nil, action, cbuttonText)
	fdlg := (*C.GtkFileChooser)(unsafe.Pointer(dlg))

	for _, filt := range b.Filters {
		filter := C.gtk_file_filter_new()
		cdesc := C.CString(filt.Desc)
		defer C.free(unsafe.Pointer(cdesc))
		C.gtk_file_filter_set_name(filter, cdesc)

		for _, ext := range filt.Extensions {
			cpattern := C.CString("*." + ext)
			defer C.free(unsafe.Pointer(cpattern))
			C.gtk_file_filter_add_pattern(filter, cpattern)
		}
		C.gtk_file_chooser_add_filter(fdlg, filter)
	}
	if b.StartDir != "" {
		cdir := C.CString(b.StartDir)
		defer C.free(unsafe.Pointer(cdir))
		C.gtk_file_chooser_set_current_folder(fdlg, cdir)
	}
	if b.StartFile != "" {
		cfile := C.CString(b.StartFile)
		defer C.free(unsafe.Pointer(cfile))
		C.gtk_file_chooser_set_current_name(fdlg, cfile)
	}
	if b.ShowHiddenFiles {
		C.gtk_file_chooser_set_show_hidden(fdlg, C.TRUE)
	}
	C.gtk_file_chooser_set_do_overwrite_confirmation(fdlg, C.TRUE)
	r := C.gtk_dialog_run((*C.GtkDialog)(unsafe.Pointer(dlg)))
	defer closeDialog(dlg)
	if r == C.GTK_RESPONSE_ACCEPT {
		return C.GoString(C.gtk_file_chooser_get_filename(fdlg)), nil
	}
	return "", ErrCancelled
}

func (b *DirectoryBuilder) browse() (string, error) {
	return chooseFile("Open Folder", "Open", C.GTK_FILE_CHOOSER_ACTION_SELECT_FOLDER, &FileBuilder{Dlg: b.Dlg, ShowHiddenFiles: b.ShowHiddenFiles})
}
