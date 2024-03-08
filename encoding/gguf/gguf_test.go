package gguf

import (
	"errors"
	"io"
	"strings"
	"testing"

	"kr.dev/diff"
)

func TestStat(t *testing.T) {
	cases := []struct {
		name     string
		data     string
		wantInfo Info
		wantErr  error
	}{
		{
			name:    "empty",
			wantErr: ErrBadMagic,
		},
		{
			name:    "bad magic",
			data:    "\xBB\xAA\xDD\x00",
			wantErr: ErrBadMagic,
		},
		{
			name: "bad version",
			data: string(magicBytes) +
				"\x02\x00\x00\x00" + // version
				"",
			wantErr: ErrUnsupportedVersion,
		},
		{
			name: "valid general.file_type",
			data: string(magicBytes) + // magic
				"\x03\x00\x00\x00" + // version
				"\x00\x00\x00\x00\x00\x00\x00\x00" + // numTensors
				"\x01\x00\x00\x00\x00\x00\x00\x00" + // numMetaValues

				// general.file_type key
				"\x11\x00\x00\x00\x00\x00\x00\x00" + // key length
				"general.file_type" + // key
				"\x04\x00\x00\x00" + // type (uint32)
				"\x01\x00\x00\x00\x00\x00\x00\x00" + // uint32 value
				"",
			wantInfo: Info{
				Version:  3,
				FileType: 1,
			},
		},
	}

	for _, tt := range cases {
		t.Run(tt.name, func(t *testing.T) {
			info, err := StatReader(strings.NewReader(tt.data))
			if tt.wantErr != nil {
				if !errors.Is(err, tt.wantErr) {
					t.Fatalf("err = %v; want %q", err, tt.wantErr)
				}
				return
			}
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			diff.Test(t, t.Errorf, info, tt.wantInfo)
		})
	}
}

func TestReadInfo(t *testing.T) {
	cases := []struct {
		name string
		data string

		wantMeta      []MetaEntry
		wantTensor    []TensorInfo
		wantReadErr   error
		wantMetaErr   error
		wantTensorErr error
		wantInfo      Info
	}{
		{
			name:        "empty",
			wantReadErr: io.ErrUnexpectedEOF,
		},
		{
			name:        "bad magic",
			data:        "\xBB\xAA\xDD\x00",
			wantReadErr: ErrBadMagic,
		},
		{
			name: "bad version",
			data: string(magicBytes) +
				"\x02\x00\x00\x00" + // version
				"",
			wantReadErr: ErrUnsupportedVersion,
		},
		{
			name: "no metadata or tensors",
			data: string(magicBytes) + // magic
				"\x03\x00\x00\x00" + // version
				"\x00\x00\x00\x00\x00\x00\x00\x00" + // numMetaValues
				"\x00\x00\x00\x00\x00\x00\x00\x00" + // numTensors
				"",
			wantReadErr: nil,
		},
		{
			name: "good metadata",
			data: string(magicBytes) + // magic
				"\x03\x00\x00\x00" + // version
				"\x00\x00\x00\x00\x00\x00\x00\x00" + // numTensors
				"\x01\x00\x00\x00\x00\x00\x00\x00" + // numMetaValues
				"\x01\x00\x00\x00\x00\x00\x00\x00" + // key length
				"K" + // key
				"\x08\x00\x00\x00" + // type (string)
				"\x02\x00\x00\x00\x00\x00\x00\x00" + // string length
				"VV" + // string value
				"",
			wantMeta: []MetaEntry{
				{Key: "K", Type: ValueTypeString, Values: []MetaValue{{Type: ValueTypeString, Value: []byte("VV")}}},
			},
		},
		{
			name: "good metadata with multiple values",
			data: string(magicBytes) + // magic
				"\x03\x00\x00\x00" + // version
				"\x00\x00\x00\x00\x00\x00\x00\x00" + // numTensors
				"\x02\x00\x00\x00\x00\x00\x00\x00" + // numMetaValues

				// MetaEntry 1
				"\x01\x00\x00\x00\x00\x00\x00\x00" + // key length
				"x" + // key
				"\x08\x00\x00\x00" + // type (string)
				"\x02\x00\x00\x00\x00\x00\x00\x00" + // string length
				"XX" + // string value

				// MetaEntry 2
				"\x01\x00\x00\x00\x00\x00\x00\x00" + // key length
				"y" + // key
				"\x04\x00\x00\x00" + // type (uint32)
				"\x99\x88\x77\x66" + // uint32 value
				"",
			wantMeta: []MetaEntry{
				{Key: "x", Type: ValueTypeString, Values: []MetaValue{{
					Type:  ValueTypeString,
					Value: []byte("XX"),
				}}},
				{Key: "y", Type: ValueTypeUint32, Values: []MetaValue{{
					Type:  ValueTypeUint32,
					Value: []byte{0x99, 0x88, 0x77, 0x66},
				}}},
			},
		},
		{
			name: "negative string length in meta key",
			data: string(magicBytes) + // magic
				"\x03\x00\x00\x00" + // version
				"\x00\x00\x00\x00\x00\x00\x00\x00" + // numTensors
				"\x01\x00\x00\x00\x00\x00\x00\x00" + // numMetaValues
				"\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF" + // key length
				"K" + // key
				"\x08\x00\x00\x00" + // type (string)
				"\x02\x00\x00\x00\x00\x00\x00\x00" + // string length
				"VV" + // string value
				"",
			wantMetaErr: ErrMangled,
		},

		// Tensor tests
		{
			name: "good tensor",
			data: string(magicBytes) + // magic
				"\x03\x00\x00\x00" + // version
				"\x01\x00\x00\x00\x00\x00\x00\x00" + // numTensors
				"\x00\x00\x00\x00\x00\x00\x00\x00" + // numMetaValues

				// Tensor 1
				"\x01\x00\x00\x00\x00\x00\x00\x00" + // name length
				"t" +

				// dimensions
				"\x01\x00\x00\x00" + // dimensions length
				"\x01\x00\x00\x00\x00\x00\x00\x00" + // dimension[0]

				"\x03\x00\x00\x00" + // type (i8)
				"\x00\x01\x00\x00\x00\x00\x00\x00" + // offset
				"",
			wantTensor: []TensorInfo{
				{
					Name:       "t",
					Dimensions: []uint64{1},
					Type:       TypeQ4_1,
					Offset:     256,
					Size:       256,
				},
			},
		},
		{
			name: "too many dimensions",
			data: string(magicBytes) + // magic
				"\x03\x00\x00\x00" + // version
				"\x01\x00\x00\x00\x00\x00\x00\x00" + // numTensors
				"\x00\x00\x00\x00\x00\x00\x00\x00" + // numMetaValues

				// Tensor 1
				"\x01\x00\x00\x00\x00\x00\x00\x00" + // name length
				"t" +

				"\x00\x00\x00\x01" + // dimensions length
				"",
			wantTensorErr: ErrMangled,
		},
		{
			name: "size computed",
			data: string(magicBytes) + // magic
				"\x03\x00\x00\x00" + // version
				"\x02\x00\x00\x00\x00\x00\x00\x00" + // numTensors
				"\x00\x00\x00\x00\x00\x00\x00\x00" + // numMetaValues

				// Tensor 1
				"\x01\x00\x00\x00\x00\x00\x00\x00" + // name length
				"t" +
				"\x01\x00\x00\x00" + // dimensions length
				"\x01\x00\x00\x00\x00\x00\x00\x00" + // dimension[0]
				"\x03\x00\x00\x00" + // type (i8)
				"\x00\x01\x00\x00\x00\x00\x00\x00" + // offset

				// Tensor 2
				"\x01\x00\x00\x00\x00\x00\x00\x00" + // name length
				"t" +
				"\x01\x00\x00\x00" + // dimensions length
				"\x01\x00\x00\x00\x00\x00\x00\x00" + // dimension[0]
				"\x03\x00\x00\x00" + // type (i8)
				"\x00\x03\x00\x00\x00\x00\x00\x00" + // offset
				"",
			wantTensor: []TensorInfo{
				{
					Name:       "t",
					Dimensions: []uint64{1},
					Type:       TypeQ4_1,
					Offset:     256,
					Size:       256,
				},
				{
					Name:       "t",
					Dimensions: []uint64{1},
					Type:       TypeQ4_1,
					Offset:     768,
					Size:       512,
				},
			},
		},
	}

	for _, tt := range cases {
		t.Run(tt.name, func(t *testing.T) {
			f, err := ReadFile(strings.NewReader(tt.data))
			if err != nil {
				if !errors.Is(err, tt.wantReadErr) {
					t.Fatalf("unexpected ReadFile error: %v", err)
				}
				return
			}

			var got []MetaEntry
			for meta, err := range f.Metadata {
				if !errors.Is(err, tt.wantMetaErr) {
					t.Fatalf("err = %v; want %v", err, ErrMangled)
				}
				if err != nil {
					return
				}
				got = append(got, meta)
			}
			diff.Test(t, t.Errorf, got, tt.wantMeta)

			var gotT []TensorInfo
			for tinfo, err := range f.Tensors {
				if !errors.Is(err, tt.wantTensorErr) {
					t.Fatalf("err = %v; want %v", err, tt.wantTensorErr)
				}
				if err != nil {
					return
				}
				gotT = append(gotT, tinfo)
			}
			diff.Test(t, t.Errorf, gotT, tt.wantTensor)
		})
	}
}

func FuzzReadInfo(f *testing.F) {
	f.Add(string(magicBytes))
	f.Add(string(magicBytes) +
		"\x03\x00\x00\x00" + // version
		"\x00\x00\x00\x00\x00\x00\x00\x00" + // numMetaValues
		"\x00\x00\x00\x00\x00\x00\x00\x00" + // numTensors
		"")
	f.Add(string(magicBytes) +
		"\x03\x00\x00\x00" + // version
		"\x01\x00\x00\x00\x00\x00\x00\x00" + // numMetaValues
		"\x01\x00\x00\x00\x00\x00\x00\x00" + // numTensors
		"\x01\x00\x00\x00\x00\x00\x00\x00" + // key length
		"K" + // key
		"\x08\x00\x00\x00" + // type (string)
		"\x02\x00\x00\x00\x00\x00\x00\x00" + // string length
		"VV" + // string value
		"\x01\x00\x00\x00\x00\x00\x00\x00" + // name length
		"t" +
		"\x01\x00\x00\x00" + // dimensions length
		"\x01\x00\x00\x00\x00\x00\x00\x00" + // dimension[0]
		"\x03\x00\x00\x00" + // type (i8)
		"\x05\x00\x00\x00\x00\x00\x00\x00" + // offset
		"")

	f.Fuzz(func(t *testing.T, data string) {
		gf, err := ReadFile(strings.NewReader(data))
		if err != nil {
			t.Logf("ReadFile error: %v", err)
			t.Skip()
		}
		for _, err := range gf.Metadata {
			if err != nil {
				t.Logf("metadata error: %v", err)
				t.Skip()
			}
		}
		for tinfo, err := range gf.Tensors {
			if err != nil {
				t.Logf("tensor error: %v", err)
				t.Skip()
			}
			if tinfo.Offset <= 0 {
				t.Logf("invalid tensor offset: %+v", t)
				t.Skip()
			}
			if tinfo.Size <= 0 {
				t.Logf("invalid tensor size: %+v", t)
				t.Skip()
			}
		}
	})
}
