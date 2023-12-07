package readline

func (i *Instance) handleCharCtrlZ(fd int, termios *Termios) (string, error) {
	// not supported
	return "", nil
}
