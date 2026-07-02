package gguf

import (
	"bufio"
	"encoding/binary"
	"fmt"
	"io"
	"os"
	"strings"
)

const (
	metadataMagicLE = 0x46554747
	metadataMagicBE = 0x47475546
)

type Metadata struct {
	Architecture    string
	FileType        int
	HasFileType     bool
	ContextLength   int
	EmbeddingLength int
	HasAudio        bool
	HasEmbedding    bool
	HasVision       bool
}

// ScanMetadata reads only early GGUF key/value metadata and stops before large
// tokenizer arrays when possible.
func ScanMetadata(path string) (Metadata, error) {
	f, err := os.Open(path)
	if err != nil {
		return Metadata{}, err
	}
	defer f.Close()

	r := bufio.NewReaderSize(f, 32<<10)
	var magic uint32
	if err := binary.Read(r, binary.LittleEndian, &magic); err != nil {
		return Metadata{}, err
	}

	var byteOrder binary.ByteOrder = binary.LittleEndian
	switch magic {
	case metadataMagicLE:
	case metadataMagicBE:
		byteOrder = binary.BigEndian
	default:
		return Metadata{}, fmt.Errorf("invalid file magic")
	}

	var version uint32
	if err := binary.Read(r, byteOrder, &version); err != nil {
		return Metadata{}, err
	}

	var numKV uint64
	switch version {
	case 1:
		var header struct {
			NumTensor uint32
			NumKV     uint32
		}
		if err := binary.Read(r, byteOrder, &header); err != nil {
			return Metadata{}, err
		}
		numKV = uint64(header.NumKV)
	default:
		var header struct {
			NumTensor uint64
			NumKV     uint64
		}
		if err := binary.Read(r, byteOrder, &header); err != nil {
			return Metadata{}, err
		}
		numKV = header.NumKV
	}

	var info Metadata
	for range numKV {
		key, err := readMetadataString(r, byteOrder, version)
		if err != nil {
			return Metadata{}, err
		}

		var valueType uint32
		if err := binary.Read(r, byteOrder, &valueType); err != nil {
			return Metadata{}, err
		}

		if key == "general.architecture" {
			value, err := readMetadataStringValue(r, byteOrder, version, valueType)
			if err != nil {
				return Metadata{}, err
			}
			info.Architecture = value
			continue
		}

		if key == "general.file_type" {
			value, err := readMetadataIntValue(r, byteOrder, version, valueType)
			if err != nil {
				return Metadata{}, err
			}
			info.FileType = value
			info.HasFileType = true
			continue
		}

		if info.Architecture != "" && strings.HasPrefix(key, "tokenizer.") {
			break
		}

		if info.Architecture != "" && strings.HasPrefix(key, info.Architecture+".") {
			switch strings.TrimPrefix(key, info.Architecture+".") {
			case "pooling_type":
				info.HasEmbedding = true
			case "vision.block_count":
				info.HasVision = true
			case "audio.block_count":
				info.HasAudio = true
			case "context_length":
				value, err := readMetadataIntValue(r, byteOrder, version, valueType)
				if err != nil {
					return Metadata{}, err
				}
				info.ContextLength = value
				continue
			case "embedding_length":
				value, err := readMetadataIntValue(r, byteOrder, version, valueType)
				if err != nil {
					return Metadata{}, err
				}
				info.EmbeddingLength = value
				continue
			}
		}

		if err := skipMetadataValue(r, byteOrder, version, valueType); err != nil {
			return Metadata{}, err
		}
	}

	return info, nil
}

func readMetadataStringValue(r io.Reader, byteOrder binary.ByteOrder, version uint32, valueType uint32) (string, error) {
	if valueType != typeString {
		if err := skipMetadataValue(r, byteOrder, version, valueType); err != nil {
			return "", err
		}
		return "", fmt.Errorf("unexpected gguf string type %d", valueType)
	}
	return readMetadataString(r, byteOrder, version)
}

func readMetadataIntValue(r io.Reader, byteOrder binary.ByteOrder, version uint32, valueType uint32) (int, error) {
	switch valueType {
	case typeUint8:
		var value uint8
		if err := binary.Read(r, byteOrder, &value); err != nil {
			return 0, err
		}
		return int(value), nil
	case typeInt8:
		var value int8
		if err := binary.Read(r, byteOrder, &value); err != nil {
			return 0, err
		}
		return int(value), nil
	case typeUint16:
		var value uint16
		if err := binary.Read(r, byteOrder, &value); err != nil {
			return 0, err
		}
		return int(value), nil
	case typeInt16:
		var value int16
		if err := binary.Read(r, byteOrder, &value); err != nil {
			return 0, err
		}
		return int(value), nil
	case typeUint32:
		var value uint32
		if err := binary.Read(r, byteOrder, &value); err != nil {
			return 0, err
		}
		return int(value), nil
	case typeInt32:
		var value int32
		if err := binary.Read(r, byteOrder, &value); err != nil {
			return 0, err
		}
		return int(value), nil
	case typeUint64:
		var value uint64
		if err := binary.Read(r, byteOrder, &value); err != nil {
			return 0, err
		}
		return int(value), nil
	case typeInt64:
		var value int64
		if err := binary.Read(r, byteOrder, &value); err != nil {
			return 0, err
		}
		return int(value), nil
	default:
		if err := skipMetadataValue(r, byteOrder, version, valueType); err != nil {
			return 0, err
		}
		return 0, fmt.Errorf("unexpected gguf integer type %d", valueType)
	}
}

func skipMetadataValue(r io.Reader, byteOrder binary.ByteOrder, version uint32, valueType uint32) error {
	switch valueType {
	case typeUint8, typeInt8, typeBool:
		return discardMetadataBytes(r, 1)
	case typeUint16, typeInt16:
		return discardMetadataBytes(r, 2)
	case typeUint32, typeInt32, typeFloat32:
		return discardMetadataBytes(r, 4)
	case typeUint64, typeInt64, typeFloat64:
		return discardMetadataBytes(r, 8)
	case typeString:
		return skipMetadataString(r, byteOrder, version)
	case typeArray:
		var arrayType uint32
		if err := binary.Read(r, byteOrder, &arrayType); err != nil {
			return err
		}
		var count uint64
		if err := binary.Read(r, byteOrder, &count); err != nil {
			return err
		}
		return skipMetadataArray(r, byteOrder, version, arrayType, count)
	default:
		return fmt.Errorf("unsupported gguf value type %d", valueType)
	}
}

func skipMetadataArray(r io.Reader, byteOrder binary.ByteOrder, version uint32, arrayType uint32, count uint64) error {
	var size uint64
	switch arrayType {
	case typeUint8, typeInt8, typeBool:
		size = 1
	case typeUint16, typeInt16:
		size = 2
	case typeUint32, typeInt32, typeFloat32:
		size = 4
	case typeUint64, typeInt64, typeFloat64:
		size = 8
	case typeString:
		for range count {
			if err := skipMetadataString(r, byteOrder, version); err != nil {
				return err
			}
		}
		return nil
	default:
		return fmt.Errorf("unsupported gguf array type %d", arrayType)
	}
	return discardMetadataBytes(r, int64(count*size))
}

func readMetadataString(r io.Reader, byteOrder binary.ByteOrder, version uint32) (string, error) {
	var length uint64
	if err := binary.Read(r, byteOrder, &length); err != nil {
		return "", err
	}

	if length == 0 {
		return "", nil
	}

	bts := make([]byte, length)
	if _, err := io.ReadFull(r, bts); err != nil {
		return "", err
	}
	if version == 1 && bts[len(bts)-1] == 0 {
		bts = bts[:len(bts)-1]
	}
	return string(bts), nil
}

func skipMetadataString(r io.Reader, byteOrder binary.ByteOrder, _ uint32) error {
	var length uint64
	if err := binary.Read(r, byteOrder, &length); err != nil {
		return err
	}
	return discardMetadataBytes(r, int64(length))
}

func discardMetadataBytes(r io.Reader, n int64) error {
	if n <= 0 {
		return nil
	}
	_, err := io.CopyN(io.Discard, r, n)
	return err
}
