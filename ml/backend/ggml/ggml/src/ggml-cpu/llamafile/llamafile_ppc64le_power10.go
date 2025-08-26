// +build ppc64le.power10

package llamafile

// #cgo CXXFLAGS: -std=c++17 -mcpu=power10
// #cgo CPPFLAGS: -I${SRCDIR}/.. -I${SRCDIR}/../.. -I${SRCDIR}/../../../include
import "C"
