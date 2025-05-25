package bfloat16

import (
	"crypto/rand"
	"reflect"
	"testing"
)

func randomBytes(n int) []byte {
	out := make([]byte, n)
	if _, err := rand.Read(out); err != nil {
		panic(err)
	}
	return out
}

func TestEncodeDecode(t *testing.T) {
	b := randomBytes(1024)
	bf16 := Decode(b)
	out := Encode(bf16)
	if !reflect.DeepEqual(b, out) {
		t.Fatalf("%+v != %+v", b, out)
	}
}

func TestEncodeDecodeFloat32(t *testing.T) {
	b := randomBytes(1024)
	bf16 := DecodeFloat32(b)
	out := EncodeFloat32(bf16)
	if !reflect.DeepEqual(b, out) {
		t.Fatalf("%+v != %+v", b, out)
	}
}

func TestBasicFloat32(t *testing.T) {
	var in float32 = 1.0
	out := ToFloat32(FromFloat32(in))
	if !reflect.DeepEqual(in, out) {
		t.Fatalf("%+v != %+v", in, out)
	}
}

func TestComplexFloat32(t *testing.T) {
	var in float32 = 123456789123456789.123456789
	var want float32 = 123286039799267328.0
	out := ToFloat32(FromFloat32(in))
	if in == out {
		t.Fatalf("no loss of precision")
	}
	if out != want {
		t.Fatalf("%.16f != %.16f", want, out)
	}
}
