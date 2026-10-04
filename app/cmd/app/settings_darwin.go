//go:build darwin

package main

// #include <stdbool.h>
// #include <stdlib.h>
import "C"

import (
	"context"
	"encoding/json"
	"errors"
	"log/slog"
	"time"
)

// The settings window calls these exports from background dispatch queues,
// never the main thread, because they can wait on the network, the disk, or a
// server restart. Each returns a JSON object allocated with malloc that the
// caller releases with free.

// settingsTimeout bounds calls that contact ollama.com or the local server.
const settingsTimeout = 15 * time.Second

type settingsResult struct {
	Settings settingsState `json:"settings"`
	Error    string        `json:"error,omitempty"`
}

type accountResult struct {
	Account    accountState `json:"account"`
	ManageURL  string       `json:"manageURL"`
	UpgradeURL string       `json:"upgradeURL"`
	Error      string       `json:"error,omitempty"`
}

type exportResult struct {
	Exported int    `json:"exported"`
	Error    string `json:"error,omitempty"`
}

type signInResult struct {
	URL   string `json:"url,omitempty"`
	Error string `json:"error,omitempty"`
}

//export SettingsGet
func SettingsGet() *C.char {
	return settingsStateResult(nil)
}

//export SettingsSetAutoUpdate
func SettingsSetAutoUpdate(enabled C.bool) *C.char {
	return settingsStateResult(settings.SetAutoUpdate(bool(enabled)))
}

//export SettingsSetExpose
func SettingsSetExpose(enabled C.bool) *C.char {
	return settingsStateResult(settings.SetExpose(bool(enabled)))
}

//export SettingsSetModelsPath
func SettingsSetModelsPath(path *C.char) *C.char {
	return settingsStateResult(settings.SetModelsPath(C.GoString(path)))
}

//export SettingsSetContextLength
func SettingsSetContextLength(length C.int) *C.char {
	return settingsStateResult(settings.SetContextLength(int(length)))
}

//export SettingsSetCloudEnabled
func SettingsSetCloudEnabled(enabled C.bool) *C.char {
	return settingsStateResult(settings.SetCloudEnabled(bool(enabled)))
}

//export AccountGet
func AccountGet(refresh C.bool) *C.char {
	if !refresh {
		return accountStateResult(settings.CachedAccount(), nil)
	}
	ctx, cancel := context.WithTimeout(context.Background(), settingsTimeout)
	defer cancel()
	return accountStateResult(settings.Account(ctx))
}

//export AccountSignInURL
func AccountSignInURL() *C.char {
	u, err := settings.SignInURL()
	if err != nil {
		return jsonCString(signInResult{Error: err.Error()})
	}
	return jsonCString(signInResult{URL: u})
}

//export AccountSignOut
func AccountSignOut() *C.char {
	ctx, cancel := context.WithTimeout(context.Background(), settingsTimeout)
	defer cancel()
	if err := settings.SignOut(ctx); err != nil {
		return accountStateResult(settings.CachedAccount(), err)
	}
	return accountStateResult(settings.Account(ctx))
}

//export ChatsExport
func ChatsExport(dir *C.char) *C.char {
	exported, err := settings.ExportChats(C.GoString(dir))
	result := exportResult{Exported: exported}
	if err != nil {
		result.Error = err.Error()
	}
	return jsonCString(result)
}

func settingsStateResult(err error) *C.char {
	ctx, cancel := context.WithTimeout(context.Background(), settingsTimeout)
	defer cancel()
	state, stateErr := settings.State(ctx)
	result := settingsResult{Settings: state}
	if err := errors.Join(err, stateErr); err != nil {
		result.Error = err.Error()
	}
	return jsonCString(result)
}

func accountStateResult(account accountState, err error) *C.char {
	result := accountResult{
		Account:    account,
		ManageURL:  ollamaDotCom + "/settings",
		UpgradeURL: ollamaDotCom + "/upgrade",
	}
	if err != nil {
		result.Error = err.Error()
	}
	return jsonCString(result)
}

func jsonCString(v any) *C.char {
	data, err := json.Marshal(v)
	if err != nil {
		slog.Error("failed to encode settings response", "error", err)
		data = []byte(`{"error":"Ollama couldn't read its settings."}`)
	}
	return C.CString(string(data))
}
