//go:build windows

package wintray

import "testing"

type lifecycleApp struct {
	running  bool
	runPath  string
	showCall bool
}

func (a *lifecycleApp) UIRun(path string) { a.runPath = path }
func (a *lifecycleApp) UIShow()           { a.showCall = true }
func (a *lifecycleApp) UITerminate()      {}
func (a *lifecycleApp) UIRunning() bool   { return a.running }
func (a *lifecycleApp) Quit()             {}
func (a *lifecycleApp) DoUpdate()         {}

func TestFocusUICreatesOrShowsWindow(t *testing.T) {
	tests := []struct {
		name     string
		running  bool
		wantRun  string
		wantShow bool
	}{
		{name: "creates window when tray only", wantRun: "/"},
		{name: "shows existing window", running: true, wantShow: true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			app := &lifecycleApp{running: tt.running}
			focusUI(app)
			if app.runPath != tt.wantRun {
				t.Errorf("run path = %q, want %q", app.runPath, tt.wantRun)
			}
			if app.showCall != tt.wantShow {
				t.Errorf("show called = %v, want %v", app.showCall, tt.wantShow)
			}
		})
	}
}
