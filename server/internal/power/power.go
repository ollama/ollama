package power

// PreventSleep asserts a system-wide lock to prevent the system from sleeping.
func PreventSleep() {
	preventSleep()
}

// AllowSleep releases the sleep prevention lock.
func AllowSleep() {
	allowSleep()
}
