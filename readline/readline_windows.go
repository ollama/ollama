package readline

func (i *Instance) handleCharCtrlZ(fd int, state *State) (string, error) {
	// not supported
	return "", nil
}
