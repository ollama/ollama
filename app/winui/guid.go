package winui

import (
	"crypto/sha1"
	"encoding/binary"
	"encoding/hex"
	"fmt"
	"strings"
)

// GUID is a COM interface or class identifier, laid out like the Windows
// GUID struct.
type GUID struct {
	Data1 uint32
	Data2 uint16
	Data3 uint16
	Data4 [8]byte
}

// guid parses an identifier such as "4e0d09f5-4358-512c-a987-503b52848e95".
func guid(s string) GUID {
	b, err := hex.DecodeString(strings.ReplaceAll(s, "-", ""))
	if err != nil || len(b) != 16 {
		panic(fmt.Sprintf("winui: invalid GUID %q", s))
	}
	var g GUID
	g.Data1 = binary.BigEndian.Uint32(b[0:4])
	g.Data2 = binary.BigEndian.Uint16(b[4:6])
	g.Data3 = binary.BigEndian.Uint16(b[6:8])
	copy(g.Data4[:], b[8:])
	return g
}

func (g GUID) String() string {
	return fmt.Sprintf("%08x-%04x-%04x-%x-%x", g.Data1, g.Data2, g.Data3, g.Data4[:2], g.Data4[2:])
}

// pinterfaceNamespace seeds the identifiers of generic WinRT interface
// instantiations.
var pinterfaceNamespace = []byte{0x11, 0xf4, 0x7a, 0xd5, 0x7b, 0x73, 0x42, 0xc0, 0xab, 0xae, 0x87, 0x8b, 0x1e, 0x16, 0xad, 0xee}

// parameterizedIID returns the identifier of a generic interface
// instantiation from its WinRT type signature, such as
// "pinterface({61c17706-2d65-11e0-9ae8-d48564015472};i4)" for
// IReference<Int32>.
func parameterizedIID(signature string) GUID {
	h := sha1.New()
	h.Write(pinterfaceNamespace)
	h.Write([]byte(signature))
	sum := h.Sum(nil)
	var g GUID
	g.Data1 = binary.BigEndian.Uint32(sum[0:4])
	g.Data2 = binary.BigEndian.Uint16(sum[4:6])
	g.Data3 = binary.BigEndian.Uint16(sum[6:8])&0x0fff | 0x5000
	copy(g.Data4[:], sum[8:16])
	g.Data4[0] = g.Data4[0]&0x3f | 0x80
	return g
}
