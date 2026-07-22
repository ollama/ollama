//go:build !windows

package server

type sleepInhibitor struct{}

func (si *sleepInhibitor) PreventSleep() {}
func (si *sleepInhibitor) AllowSleep()   {}
func (si *sleepInhibitor) Close()        {}
