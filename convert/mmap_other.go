//go:build !unix

package convert

type mmapRegion struct {
	data []byte
}

func mmapOpen(_ string) (*mmapRegion, error) {
	return nil, nil
}

func (m *mmapRegion) Close() error {
	return nil
}
