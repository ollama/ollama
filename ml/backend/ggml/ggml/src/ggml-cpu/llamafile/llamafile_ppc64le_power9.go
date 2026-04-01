// +build ppc64le.power9

package llamafile

// #cgo CXXFLAGS: -std=c++17 -mcpu=power9
// #cgo CPPFLAGS: -I${SRCDIR}/.. -I${SRCDIR}/../.. -I${SRCDIR}/../../../include
import "C"
