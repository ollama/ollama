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

BUILD_DIR = ./llama/build/$(OS)-$(ARCH)
DIST_BASE = ./dist/$(OS)-$(ARCH)

ifeq ($(OS),windows)
	# Absolute paths with cygpath to convert to 8.3 without spaces
	PWD="$(shell pwd)"
	DIST_OLLAMA_EXE=$(DIST_BASE)/ollama$(EXE_EXT)
else
	CCACHE:=$(shell command -v ccache 2>/dev/null || echo "")
	DIST_OLLAMA_EXE=$(DIST_BASE)/bin/ollama$(EXE_EXT)
endif
DIST_LIB_DIR = $(DIST_BASE)/lib/ollama
RUNNERS_DIST_DIR = $(DIST_LIB_DIR)/runners
RUNNERS_BUILD_DIR = $(BUILD_DIR)/runners
VERSION?=$(shell git describe --tags --first-parent --abbrev=7 --long --dirty --always | sed -e "s/^v//g")

# Conditionally enable ccache for cgo builds too
ifneq ($(CCACHE),)
	CC?=$(CCACHE) gcc
	CXX?=$(CCACHE) g++
	export CC
	export CXX
endif


# Override in environment to tune CPU vector flags
ifeq ($(ARCH),amd64)
ifeq ($(origin CUSTOM_CPU_FLAGS),undefined)
	GPU_RUNNER_CPU_FLAGS=avx
	GPU_RUNNER_EXTRA_VARIANT=_avx
else
	GPU_RUNNER_CPU_FLAGS=$(subst $(comma),$(space),$(CUSTOM_CPU_FLAGS))
endif
endif

ifeq ($(OS),windows)
	CP := cp
	OBJ_EXT := obj
	SHARED_EXT := dll
	EXE_EXT := .exe
	SHARED_PREFIX := 
	CPU_FLAG_PREFIX := /arch:
ifneq ($(HIP_PATH),)
	# If HIP_PATH has spaces, hipcc trips over them when subprocessing
	HIP_PATH := $(shell cygpath -m -s "$(patsubst %\,%,$(HIP_PATH))")
	export HIP_PATH
endif
else ifeq ($(OS),linux)
	CP := cp -df
	OBJ_EXT := o
	SHARED_EXT := so
	SHARED_PREFIX := lib
	CPU_FLAG_PREFIX := -m
else
	OBJ_EXT := o
	SHARED_EXT := so
	CPU_FLAG_PREFIX := -m
	CP := cp -df
endif

COMMON_SRCS := \
	$(wildcard ./llama/*.c) \
	$(wildcard ./llama/*.cpp)
COMMON_HDRS := \
	$(wildcard ./llama/*.h) \
	$(wildcard ./llama/*.hpp)

OLLAMA_EXE=./ollama$(EXE_EXT)