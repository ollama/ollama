package cpu

// #cgo CXXFLAGS: -std=c++11
// #cgo CPPFLAGS: -I${SRCDIR}/.. -I${SRCDIR}/../include -I${SRCDIR}/amx
// #cgo linux CPPFLAGS: -D_GNU_SOURCE
// #cgo amd64,avx CPPFLAGS: -mavx
// #cgo amd64,avx2 CPPFLAGS: -mavx2 -mfma
// #cgo amd64,f16c CPPFLAGS: -mf16c
// #include "ggml-cpu.h"
// #include "ggml-backend.h"
import "C"
