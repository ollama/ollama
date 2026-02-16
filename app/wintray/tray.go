//go:build windows

package wintray

import (
	"crypto/md5"
	"encoding/hex"
	"fmt"
	"log/slog"
	"os"
	"path/filepath"
	"sort"
	"sync"
	"syscall"
	"unsafe"

	"github.com/ollama/ollama/app/assets"
	"golang.org/x/sys/windows"
)

const (
	UpdateIconName = "tray_upgrade.ico"
	IconName       = "tray.ico"
	ClassName      = "OllamaClass"
)

func NewTray(app AppCallbacks) (TrayCallbacks, error) {
	updateIcon, err := assets.GetIcon(UpdateIconName)
	if err != nil {
		return nil, fmt.Errorf("failed to load icon %s: %w", UpdateIconName, err)
	}
	icon, err := assets.GetIcon(IconName)
	if err != nil {
		return nil, fmt.Errorf("failed to load icon %s: %w", IconName, err)
	}

	return InitTray(icon, updateIcon, app)
}

type TrayCallbacks interface {
	Quit()
	TrayRun()
	UpdateAvailable(ver string) error
	GetIconHandle() windows.Handle
}

type AppCallbacks interface {
	UIRun(path string)
	UIShow()
	UITerminate()
	UIRunning() bool
	Quit()
	DoUpdate()
}

type URLSchemeHandler interface {
	HandleURLScheme(urlScheme string)
}

// Helpful sources: https://github.com/golang/exp/blob/master/shiny/driver/internal/win32

// Contains information about loaded resources
type winTray struct {
	instance,
	icon,
	defaultIcon,
	cursor,
	window windows.Handle

	loadedImages   map[string]windows.Handle
	muLoadedImages sync.RWMutex

	// menus keeps track of the submenus keyed by the menu item ID, plus 0
	// which corresponds to the main popup menu.
	menus    map[uint32]windows.Handle
	muMenus  sync.RWMutex
	menuOf   map[uint32]windows.Handle
	muMenuOf sync.RWMutex
	// menuItemIcons maintains the bitmap of each menu item (if applies). It's
	// needed to show the icon correctly when showing a previously hidden menu
	// item again.
	// menuItemIcons   map[uint32]windows.Handle
	// muMenuItemIcons sync.RWMutex
	visibleItems   map[uint32][]uint32
	muVisibleItems sync.RWMutex

	nid   *notifyIconData
	muNID sync.RWMutex
	wcex  *wndClassEx

	wmSystrayMessage,
	wmTaskbarCreated uint32

	pendingUpdate  bool
	updateNotified bool // Only pop up the notification once - TODO consider daily nag?
	normalIcon     []byte
	updateIcon     []byte

	// TODO clean up exit handling
	quitting bool

	app AppCallbacks
}

var wt winTray

func InitTray(icon, updateIcon []byte, app AppCallbacks) (*winTray, error) {
	wt.normalIcon = icon
	wt.updateIcon = updateIcon
	wt.app = app
	if err := wt.initInstance(); err != nil {
		return nil, fmt.Errorf("Unable to init instance: %w\n", err)
	}

	if err := wt.createMenu(); err != nil {
		return nil, fmt.Errorf("Unable to create menu: %w\n", err)
	}

	iconFilePath, err := iconBytesToFilePath(wt.normalIcon)
	if err != nil {
		return nil, fmt.Errorf("Unable to write icon data to temp file: %w", err)
	}
	if err := wt.setIcon(iconFilePath); err != nil {
		return nil, fmt.Errorf("Unable to set icon: %w", err)
	}

	h, err := wt.loadIconFrom(iconFilePath)
	if err != nil {
		return nil, fmt.Errorf("Unable to set default icon: %w", err)
	}
	wt.defaultIcon = h

	return &wt, wt.initMenus()
}

func (t *winTray) initInstance() error {
	const (
		windowName = ""
	)

	t.wmSystrayMessage = WM_USER + 1
	t.visibleItems = make(map[uint32][]uint32)
	t.menus = make(map[uint32]windows.Handle)
	t.menuOf = make(map[uint32]windows.Handle)

	t.loadedImages = make(map[string]windows.Handle)

	taskbarEventNamePtr, _ := windows.UTF16PtrFromString("TaskbarCreated")
	// https://msdn.microsoft.com/en-us/library/windows/desktop/ms644947
	res, _, err := pRegisterWindowMessage.Call(
		uintptr(unsafe.Pointer(taskbarEventNamePtr)),
	)
	if res == 0 { // success 0xc000-0xfff
		return fmt.Errorf("failed to register window: %w", err)
	}
	t.wmTaskbarCreated = uint32(res)

	instanceHandle, _, err := pGetModuleHandle.Call(0)
	if instanceHandle == 0 {
		return err
	}
	t.instance = windows.Handle(instanceHandle)

	// https://msdn.microsoft.com/en-us/library/windows/desktop/ms648072(v=vs.85).aspx
	iconHandle, _, err := pLoadIcon.Call(0, uintptr(IDI_APPLICATION))
	if iconHandle == 0 {
		return err
	}
	t.icon = windows.Handle(iconHandle)

	// https://msdn.microsoft.com/en-us/library/windows/desktop/ms648391(v=vs.85).aspx
	cursorHandle, _, err := pLoadCursor.Call(0, uintptr(IDC_ARROW))
	if cursorHandle == 0 {
		return err
	}
	t.cursor = windows.Handle(cursorHandle)

	classNamePtr, err := windows.UTF16PtrFromString(ClassName)
	if err != nil {
		return err
	}

	windowNamePtr, err := windows.UTF16PtrFromString(windowName)
	if err != nil {
		return err
	}

	t.wcex = &wndClassEx{
		Style:      CS_HREDRAW | CS_VREDRAW,
		WndProc:    windows.NewCallback(t.wndProc),
		Instance:   t.instance,
		Icon:       t.icon,
		Cursor:     t.cursor,
		Background: windows.Handle(6), // (COLOR_WINDOW + 1)
		ClassName:  classNamePtr,
		IconSm:     t.icon,
	}
	if err := t.wcex.register(); err != nil {
		return err
	}

	windowHandle, _, err := pCreateWindowEx.Call(
		uintptr(0),
		uintptr(unsafe.Pointer(classNamePtr)),
		uintptr(unsafe.Pointer(windowNamePtr)),
		uintptr(WS_OVERLAPPEDWINDOW),
		uintptr(CW_USEDEFAULT),
		uintptr(CW_USEDEFAULT),
		uintptr(CW_USEDEFAULT),
		uintptr(CW_USEDEFAULT),
		uintptr(0),
		uintptr(0),
		uintptr(t.instance),
		uintptr(0),
	)
	if windowHandle == 0 {
		return err
	}
	t.window = windows.Handle(windowHandle)

	pShowWindow.Call(uintptr(t.window), uintptr(SW_HIDE)) //nolint:errcheck

	boolRet, _, err := pUpdateWindow.Call(uintptr(t.window))
	if boolRet == 0 {
		slog.Error(fmt.Sprintf("failed to update window: %s", err))
	}

	t.muNID.Lock()
	defer t.muNID.Unlock()
	t.nid = &notifyIconData{
		Wnd:             t.window,
		ID:              100,
		Flags:           NIF_MESSAGE,
		CallbackMessage: t.wmSystrayMessage,
	}
	t.nid.Size = uint32(unsafe.Sizeof(*t.nid))

	return t.nid.add()
}

func (t *winTray) createMenu() error {
	menuHandle, _, err := pCreatePopupMenu.Call()
	if menuHandle == 0 {
		return err
	}
	t.menus[0] = windows.Handle(menuHandle)

	// https://msdn.microsoft.com/en-us/library/windows/desktop/ms647575(v=vs.85).aspx
	mi := struct {
		Size, Mask, Style, Max uint32
		Background             windows.Handle
		ContextHelpID          uint32
		MenuData               uintptr
	}{
		Mask: MIM_APPLYTOSUBMENUS,
	}
	mi.Size = uint32(unsafe.Sizeof(mi))

	res, _, err := pSetMenuInfo.Call(
		uintptr(t.menus[0]),
		uintptr(unsafe.Pointer(&mi)),
	)
	if res == 0 {
		return err
	}
	return nil
}

// Contains information about a menu item.
// https://msdn.microsoft.com/en-us/library/windows/desktop/ms647578(v=vs.85).aspx
type menuItemInfo struct {
	Size, Mask, Type, State     uint32
	ID                          uint32
	SubMenu, Checked, Unchecked windows.Handle
	ItemData                    uintptr
	TypeData                    *uint16
	Cch                         uint32
	BMPItem                     windows.Handle
}

func (t *winTray) addOrUpdateMenuItem(menuItemId uint32, parentId uint32, title string, disabled bool) error {
	titlePtr, err := windows.UTF16PtrFromString(title)
	if err != nil {
		return err
	}

	mi := menuItemInfo{
		Mask:     MIIM_FTYPE | MIIM_STRING | MIIM_ID | MIIM_STATE,
		Type:     MFT_STRING,
		ID:       menuItemId,
		TypeData: titlePtr,
		Cch:      uint32(len(title)),
	}
	mi.Size = uint32(unsafe.Sizeof(mi))
	if disabled {
		mi.State |= MFS_DISABLED
	}

	var res uintptr
	t.muMenus.RLock()
	menu := t.menus[parentId]
	t.muMenus.RUnlock()
	if t.getVisibleItemIndex(parentId, menuItemId) != -1 {
		// We set the menu item info based on the menuID
		boolRet, _, err := pSetMenuItemInfo.Call(
			uintptr(menu),
			uintptr(menuItemId),
			0,
			uintptr(unsafe.Pointer(&mi)),
		)
		if boolRet == 0 {
			return fmt.Errorf("failed to set menu item: %w", err)
		}
	}

	if res == 0 {
		// Menu item does not already exist, create it
		t.muMenus.RLock()
		submenu, exists := t.menus[menuItemId]
		t.muMenus.RUnlock()
		if exists {
			mi.Mask |= MIIM_SUBMENU
			mi.SubMenu = submenu
		}
		t.addToVisibleItems(parentId, menuItemId)
		position := t.getVisibleItemIndex(parentId, menuItemId)
		res, _, err = pInsertMenuItem.Call(
			uintptr(menu),
			uintptr(position),
			1,
			uintptr(unsafe.Pointer(&mi)),
		)
		if res == 0 {
			t.delFromVisibleItems(parentId, menuItemId)
			return err
		}
		t.muMenuOf.Lock()
		t.menuOf[menuItemId] = menu
		t.muMenuOf.Unlock()
	}

	return nil
}

func (t *winTray) addSeparatorMenuItem(menuItemId, parentId uint32) error {
	mi := menuItemInfo{
		Mask: MIIM_FTYPE | MIIM_ID | MIIM_STATE,
		Type: MFT_SEPARATOR,
		ID:   menuItemId,
	}

	mi.Size = uint32(unsafe.Sizeof(mi))

	t.addToVisibleItems(parentId, menuItemId)
	position := t.getVisibleItemIndex(parentId, menuItemId)
	t.muMenus.RLock()
	menu := uintptr(t.menus[parentId])
	t.muMenus.RUnlock()
	res, _, err := pInsertMenuItem.Call(
		menu,
		uintptr(position),
		1,
		uintptr(unsafe.Pointer(&mi)),
	)
	if res == 0 {
		return err
	}

	return nil
}

// func (t *winTray) hideMenuItem(menuItemId, parentId uint32) error {
// 	const ERROR_SUCCESS syscall.Errno = 0

// 	t.muMenus.RLock()
// 	menu := uintptr(t.menus[parentId])
// 	t.muMenus.RUnlock()
// 	res, _, err := pRemoveMenu.Call(
// 		menu,
// 		uintptr(menuItemId),
// 		MF_BYCOMMAND,
// 	)
// 	if res == 0 && err.(syscall.Errno) != ERROR_SUCCESS {
// 		return err
// 	}
// 	t.delFromVisibleItems(parentId, menuItemId)

// 	return nil
// }

func (t *winTray) showMenu() error {
	p := point{}
	boolRet, _, err := pGetCursorPos.Call(uintptr(unsafe.Pointer(&p)))
	if boolRet == 0 {
		return err
	}
	boolRet, _, err = pSetForegroundWindow.Call(uintptr(t.window))
	if boolRet == 0 {
		slog.Warn(fmt.Sprintf("failed to bring menu to foreground: %s", err))
	}

	boolRet, _, err = pTrackPopupMenu.Call(
		uintptr(t.menus[0]),
		TPM_BOTTOMALIGN|TPM_LEFTALIGN|TPM_RIGHTBUTTON,
		uintptr(p.X),
		uintptr(p.Y),
		0,
		uintptr(t.window),
		0,
	)
	if boolRet == 0 {
		return err
	}

	return nil
}

func (t *winTray) delFromVisibleItems(parent, val uint32) {
	t.muVisibleItems.Lock()
	defer t.muVisibleItems.Unlock()
	visibleItems := t.visibleItems[parent]
	for i, itemval := range visibleItems {
		if val == itemval {
			t.visibleItems[parent] = append(visibleItems[:i], visibleItems[i+1:]...)
			break
		}
	}
}

func (t *winTray) addToVisibleItems(parent, val uint32) {
	t.muVisibleItems.Lock()
	defer t.muVisibleItems.Unlock()
	if visibleItems, exists := t.visibleItems[parent]; !exists {
		t.visibleItems[parent] = []uint32{val}
	} else {
		newvisible := append(visibleItems, val)
		sort.Slice(newvisible, func(i, j int) bool { return newvisible[i] < newvisible[j] })
		t.visibleItems[parent] = newvisible
	}
}

func (t *winTray) getVisibleItemIndex(parent, val uint32) int {
	t.muVisibleItems.RLock()
	defer t.muVisibleItems.RUnlock()
	for i, itemval := range t.visibleItems[parent] {
		if val == itemval {
			return i
		}
	}
	return -1
}

func iconBytesToFilePath(iconBytes []byte) (string, error) {
	bh := md5.Sum(iconBytes)
	dataHash := hex.EncodeToString(bh[:])
	iconFilePath := filepath.Join(os.TempDir(), "ollama_temp_icon_"+dataHash)

	if _, err := os.Stat(iconFilePath); os.IsNotExist(err) {
		if err := os.WriteFile(iconFilePath, iconBytes, 0o644); err != nil {
			return "", err
		}
	}
	return iconFilePath, nil
}

// Loads an image from file and shows it in tray.
// Shell_NotifyIcon: https://msdn.microsoft.com/en-us/library/windows/desktop/bb762159(v=vs.85).aspx
func (t *winTray) setIcon(src string) error {
	h, err := t.loadIconFrom(src)
	if err != nil {
		return err
	}

	t.muNID.Lock()
	defer t.muNID.Unlock()
	t.nid.Icon = h
	t.nid.Flags |= NIF_ICON | NIF_TIP
	if toolTipUTF16, err := syscall.UTF16FromString("Ollama"); err == nil {
		copy(t.nid.Tip[:], toolTipUTF16)
	} else {
		return err
	}
	t.nid.Size = uint32(unsafe.Sizeof(*t.nid))

	return t.nid.modify()
}

// Loads an image from file to be shown in tray or menu item.
// LoadImage: https://msdn.microsoft.com/en-us/library/windows/desktop/ms648045(v=vs.85).aspx
func (t *winTray) loadIconFrom(src string) (windows.Handle, error) {
	// Save and reuse handles of loaded images
	t.muLoadedImages.RLock()
	h, ok := t.loadedImages[src]
	t.muLoadedImages.RUnlock()
	if !ok {
		srcPtr, err := windows.UTF16PtrFromString(src)
		if err != nil {
			return 0, err
		}
		res, _, err := pLoadImage.Call(
			0,
			uintptr(unsafe.Pointer(srcPtr)),
			IMAGE_ICON,
			0,
			0,
			LR_LOADFROMFILE|LR_DEFAULTSIZE,
		)
		if res == 0 {
			return 0, err
		}
		h = windows.Handle(res)
		t.muLoadedImages.Lock()
		t.loadedImages[src] = h
		t.muLoadedImages.Unlock()
	}
	return h, nil
}

func (t *winTray) GetIconHandle() windows.Handle {
	return t.defaultIcon
}

func (t *winTray) DisplayFirstUseNotification() error {
	t.muNID.Lock()
	defer t.muNID.Unlock()
	copy(t.nid.InfoTitle[:], windows.StringToUTF16(firstTimeTitle))
	copy(t.nid.Info[:], windows.StringToUTF16(firstTimeMessage))
	t.nid.Flags |= NIF_INFO
	t.nid.Size = uint32(unsafe.Sizeof(*wt.nid))

	return t.nid.modify()
}
