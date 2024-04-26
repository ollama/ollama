package server

import (
	"os"
	"strconv"
	"unsafe"
)

type Int interface {
	int | int8 | int16 | int32 | int64
}

func parseInt[T Int](v string, fallback T) T {
	i, err := strconv.ParseInt(v, 10, int(unsafe.Sizeof(fallback)))
	if err != nil {
		return fallback
	}
	return T(i)
}

func getEnvInt[T Int](key string, fallback T) T {
	v, ok := os.LookupEnv(key)
	if !ok {
		return fallback
	}
	return parseInt(v, fallback)
}
