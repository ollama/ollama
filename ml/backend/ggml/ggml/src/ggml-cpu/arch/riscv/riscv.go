//go:build riscv64

package riscv

// #cgo CXXFLAGS: -std=c++17
// #cgo CPPFLAGS: -I${SRCDIR}/../.. -I${SRCDIR}/../../.. -I${SRCDIR}/../../../../include
import "C"
