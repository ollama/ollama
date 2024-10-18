# Common definitions for the various Makefiles
# No rules are defined here so this is safe to include at the beginning of other makefiles

OS := $(shell uname -s)
ARCH ?= $(subst aarch64,arm64,$(subst x86_64,amd64,$(shell uname -m)))
ifneq (,$(findstring MINGW,$(OS))$(findstring MSYS,$(OS)))
	OS := windows
	ARCH := $(shell systeminfo 2>/dev/null | grep "System Type" | grep ARM64 > /dev/null && echo "arm64" || echo "amd64" )
else ifeq ($(OS),Linux)
	OS := linux
else ifeq ($(OS),Darwin)
	OS := darwin
endif
comma:= ,
empty:=
space:= $(empty) $(empty)
uc = $(subst a,A,$(subst b,B,$(subst c,C,$(subst d,D,$(subst e,E,$(subst f,F,$(subst g,G,$(subst h,H,$(subst i,I,$(subst j,J,$(subst k,K,$(subst l,L,$(subst m,M,$(subst n,N,$(subst o,O,$(subst p,P,$(subst q,Q,$(subst r,R,$(subst s,S,$(subst t,T,$(subst u,U,$(subst v,V,$(subst w,W,$(subst x,X,$(subst y,Y,$(subst z,Z,$1))))))))))))))))))))))))))

export CGO_CFLAGS_ALLOW = -mfma|-mf16c
export CGO_CXXFLAGS_ALLOW = -mfma|-mf16c
export HIP_PLATFORM = amd
export CGO_ENABLED=1

SRC_DIR := $(dir $(patsubst %/,%,$(dir $(abspath $(lastword $(MAKEFILE_LIST))))))
BUILD_DIR = $(SRC_DIR)build/$(OS)-$(ARCH)
DIST_BASE = $(abspath $(SRC_DIR)/../dist/$(OS)-$(ARCH))
DIST_LIB_DIR = $(DIST_BASE)/lib/ollama
RUNNERS_DIST_DIR = $(DIST_LIB_DIR)/runners
RUNNERS_PAYLOAD_DIR = $(abspath $(SRC_DIR)/../build/$(OS)/$(ARCH))
RUNNERS_BUILD_DIR = $(BUILD_DIR)/runners
DEFAULT_RUNNER := $(if $(and $(filter darwin,$(OS)),$(filter arm64,$(ARCH))),metal,cpu)
GZIP:=$(shell command -v pigz 2>/dev/null || echo "gzip")
ifneq ($(OS),windows)
	CCACHE:=$(shell command -v ccache 2>/dev/null || echo "")
endif
VERSION?=$(shell git describe --tags --first-parent --abbrev=7 --long --dirty --always | sed -e "s/^v//g")

# Conditionally enable ccache for cgo builds too
ifneq ($(CCACHE),)
	CC=$(CCACHE) gcc
	CXX=$(CCACHE) g++
	export CC
	export CXX
endif


# Override in environment space separated to tune GPU runner CPU vector flags
ifeq ($(ARCH),amd64)
	GPU_RUNNER_CPU_FLAGS ?= avx
endif

ifeq ($(OS),windows)
	CP := cp
	SRC_DIR := $(shell cygpath -m -s "$(SRC_DIR)")
	OBJ_EXT := obj
	SHARED_EXT := dll
	EXE_EXT := .exe
	SHARED_PREFIX := 
	CPU_FLAG_PREFIX := /arch:
else ifeq ($(OS),linux)
	CP := cp -af
	OBJ_EXT := o
	SHARED_EXT := so
	SHARED_PREFIX := lib
	CPU_FLAG_PREFIX := -m
else
	OBJ_EXT := o
	SHARED_EXT := so
	CPU_FLAG_PREFIX := -m
	CP := cp -af
endif

