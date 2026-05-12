//go:build arm64

package arm

// #cgo CXXFLAGS: -std=c++17
// #cgo CPPFLAGS: -I${SRCDIR}/../.. -I${SRCDIR}/../../.. -I${SRCDIR}/../../../../include -DHWCAP2_SVE2="2"
import "C"
