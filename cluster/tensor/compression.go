package tensor

import (
	"bytes"
	"compress/gzip"
	"compress/lzw"
	"compress/zlib"
	"fmt"
	"io"
	"sync"
)

// CompressionType represents the algorithm used for tensor compression
type CompressionType uint8

const (
	// CompressionNone indicates no compression
	CompressionNone CompressionType = iota
	
	// CompressionGzip uses gzip compression
	CompressionGzip
	
	// CompressionZlib uses zlib compression
	CompressionZlib
	
	// CompressionLZW uses LZW compression
	CompressionLZW
	
	// CompressionFP16 compresses 32-bit floats to 16-bit (lossy)
	CompressionFP16
	
	// CompressionQuantized uses quantization (lossy)
	CompressionQuantized
)

// CompressionLevel defines how aggressively to compress
type CompressionLevel int

const (
	// CompressLevelNone applies no compression
	CompressLevelNone CompressionLevel = 0
	
	// CompressLevelFast prioritizes speed over compression ratio
	CompressLevelFast CompressionLevel = 1
	
	// CompressLevelDefault provides a balance of speed and compression
	CompressLevelDefault CompressionLevel = 5
	
	// CompressLevelMax maximizes compression ratio at the cost of speed
	CompressLevelMax CompressionLevel = 9
)

// CompressionOptions configures tensor compression
type CompressionOptions struct {
	// Type is the compression algorithm to use
	Type CompressionType
	
	// Level controls the compression-speed tradeoff
	Level CompressionLevel
	
	// ReuseBuffers enables buffer reuse for better performance
	ReuseBuffers bool
}

// DefaultCompressionOptions provides sensible defaults
var DefaultCompressionOptions = CompressionOptions{
	Type:         CompressionGzip,
	Level:        CompressLevelDefault,
	ReuseBuffers: true,
}

// Compressor handles tensor compression and decompression
type Compressor struct {
	// options stores the current compression settings
	options CompressionOptions
	
	// bufferPool maintains reusable buffers to reduce allocations
	bufferPool sync.Pool
}
// NewCompressor creates a new tensor compressor
func NewCompressor(options CompressionOptions) *Compressor {
	compressor := &Compressor{
		options: options,
	}
	
	if options.ReuseBuffers {
		compressor.bufferPool = sync.Pool{
			New: func() interface{} {
				return new(bytes.Buffer)
			},
		}
	}
	
	return compressor
}

// Compress compresses tensor data using the configured algorithm
func (c *Compressor) Compress(data []byte) ([]byte, error) {
	if c.options.Type == CompressionNone || len(data) == 0 {
		return data, nil
	}
	
	var buf *bytes.Buffer
	if c.options.ReuseBuffers {
		buf = c.bufferPool.Get().(*bytes.Buffer)
		buf.Reset()
		defer c.bufferPool.Put(buf)
	} else {
		buf = &bytes.Buffer{}
	}
	
	var writer io.WriteCloser
	var err error
	
	switch c.options.Type {
	case CompressionGzip:
		writer, err = gzip.NewWriterLevel(buf, int(c.options.Level))
		if err != nil {
			return nil, fmt.Errorf("gzip compression error: %w", err)
		}
	case CompressionZlib:
		writer, err = zlib.NewWriterLevel(buf, int(c.options.Level))
		if err != nil {
			return nil, fmt.Errorf("zlib compression error: %w", err)
		}
	case CompressionLZW:
		writer = lzw.NewWriter(buf, lzw.MSB, 8)
	case CompressionFP16:
		return c.compressFP16(data)
	case CompressionQuantized:
		return c.compressQuantized(data)
	default:
		return nil, fmt.Errorf("unsupported compression type: %d", c.options.Type)
	}
	
	// Write data and close
	if _, err := writer.Write(data); err != nil {
		writer.Close()
		return nil, fmt.Errorf("compression write error: %w", err)
	}
	
	if err := writer.Close(); err != nil {
		return nil, fmt.Errorf("compression close error: %w", err)
	}
	
	return buf.Bytes(), nil
}

// Decompress decompresses tensor data
func (c *Compressor) Decompress(data []byte, compressionType CompressionType) ([]byte, error) {
	if compressionType == CompressionNone || len(data) == 0 {
		return data, nil
	}
	
	var reader io.ReadCloser
	var err error
	
	buf := bytes.NewReader(data)
	
	switch compressionType {
	case CompressionGzip:
		reader, err = gzip.NewReader(buf)
		if err != nil {
			return nil, fmt.Errorf("gzip decompression error: %w", err)
		}
	case CompressionZlib:
		reader, err = zlib.NewReader(buf)
		if err != nil {
			return nil, fmt.Errorf("zlib decompression error: %w", err)
		}
	case CompressionLZW:
		reader = lzw.NewReader(buf, lzw.MSB, 8)
	case CompressionFP16:
		return c.decompressFP16(data)
	case CompressionQuantized:
		return c.decompressQuantized(data)
	default:
		return nil, fmt.Errorf("unsupported compression type: %d", compressionType)
	}
	
	// Read decompressed data and close
	var result bytes.Buffer
	if _, err := io.Copy(&result, reader); err != nil {
		reader.Close()
		return nil, fmt.Errorf("decompression read error: %w", err)
	}
	
	if err := reader.Close(); err != nil {
		return nil, fmt.Errorf("decompression close error: %w", err)
	}
	
	return result.Bytes(), nil
}

// compressFP16 converts float32 tensors to float16 (half precision)
func (c *Compressor) compressFP16(data []byte) ([]byte, error) {
	// Simple implementation that assumes data is a slice of float32
	// In a real implementation, this would use proper float16 conversion
	if len(data)%4 != 0 {
		return nil, fmt.Errorf("data length (%d) not divisible by 4, not a valid float32 tensor", len(data))
	}
	
	// Calculate output size (half the input size since float16 is half the size of float32)
	outputSize := len(data) / 2
	result := make([]byte, outputSize)
	
	// Example implementation (not accurate float16 conversion)
	// In reality, would use a proper IEEE 754 half-precision implementation
	for i := 0; i < len(data); i += 4 {
		// Extract float32 bits (assuming little endian)
		// Not used in this simplified implementation
		// uint32(data[i]) | uint32(data[i+1])<<8 | uint32(data[i+2])<<16 | uint32(data[i+3])<<24
		
		// Convert to float16 (simplified)
		// Real implementation would properly handle sign, exponent, and mantissa
		outIdx := i / 2
		result[outIdx] = data[i]
		if outIdx+1 < len(result) {
			result[outIdx+1] = data[i+1]
		}
	}
	
	return result, nil
}
// decompressFP16 converts float16 tensors back to float32
func (c *Compressor) decompressFP16(data []byte) ([]byte, error) {
	// Simple implementation that assumes data is a slice of float16
	if len(data)%2 != 0 {
		return nil, fmt.Errorf("data length (%d) not divisible by 2, not a valid float16 tensor", len(data))
	}
	
	// Calculate output size (double the input size since float32 is twice the size of float16)
	outputSize := len(data) * 2
	result := make([]byte, outputSize)
	
	// Example implementation (not accurate float16 conversion)
	// In reality, would use a proper IEEE 754 half-precision implementation
	for i := 0; i < len(data); i += 2 {
		// Get float16 bits
		inIdx := i
		
		// Convert to float32 (simplified)
		// Real implementation would properly handle sign, exponent, and mantissa
		outIdx := i * 2
		result[outIdx] = data[inIdx]
		if inIdx+1 < len(data) {
			result[outIdx+1] = data[inIdx+1]
		}
		result[outIdx+2] = 0 // Zero high bits for simplicity
		result[outIdx+3] = 0 // In a real impl, these would be computed from float16
	}
	
	return result, nil
}

// compressQuantized uses quantization to compress floating point tensors
func (c *Compressor) compressQuantized(data []byte) ([]byte, error) {
	// Simple implementation that assumes data is a slice of float32
	// In a real implementation, this would be a proper quantization algorithm
	// such as 8-bit or 4-bit integer quantization

	if len(data)%4 != 0 {
		return nil, fmt.Errorf("data length (%d) not divisible by 4, not a valid float32 tensor", len(data))
	}
	
	// For simplicity, we'll implement an 8-bit quantization (4x compression)
	outputSize := len(data) / 4
	result := make([]byte, outputSize + 8) // Extra 8 bytes for scale and zero point
	
	// Example quantization (simplified)
	// 1. Find min/max values
	// 2. Compute scale factor and zero point
	// 3. Quantize values to 8-bit integers
	
	// Dummy implementation for illustration
	// In reality, we'd calculate proper min/max by iterating over the floats
	// These values are just for reference but not used in the simplified implementation
	// float32(0.01)  // Arbitrary scale factor
	// byte(128)      // Middle of uint8 range
	
	// Store scale factor and zero point at the beginning
	// (simplified - a real implementation would use a proper header)
	copy(result[:4], []byte{0, 0, 0, 0}) // Placeholder for scale
	copy(result[4:8], []byte{128, 0, 0, 0}) // Placeholder for zero point (128 is middle of uint8 range)
	
	// Perform quantization
	for i := 0; i < len(data); i += 4 {
		if i/4 + 8 < len(result) {
			// Arbitrary quantization value for illustration
			result[i/4 + 8] = data[i]
		}
	}
	
	fmt.Printf("Compressed %d bytes to %d bytes using quantization\n", len(data), len(result))
	
	return result, nil
}

// decompressQuantized reverses the quantization process
func (c *Compressor) decompressQuantized(data []byte) ([]byte, error) {
	if len(data) < 8 {
		return nil, fmt.Errorf("invalid quantized data (too short)")
	}
	
	// Extract scale and zero point
	// (simplified - a real implementation would parse a proper header)
	// Not used in this simplified implementation
	// data[4]
	
	// Calculate output size
	outputSize := (len(data) - 8) * 4
	result := make([]byte, outputSize)
	
	// Perform dequantization
	for i := 8; i < len(data); i++ {
		outIdx := (i - 8) * 4
		
		// Simple dequantization (just copy bytes for illustration)
		// In a real implementation, we'd compute actual float32 values
		if outIdx < len(result) {
			result[outIdx] = data[i]
			if outIdx+1 < len(result) {
				result[outIdx+1] = 0
			}
			if outIdx+2 < len(result) {
				result[outIdx+2] = 0
			}
			if outIdx+3 < len(result) {
				result[outIdx+3] = 0
			}
		}
	}
	
	return result, nil
}

// CompressionStats contains statistics about compression
type CompressionStats struct {
	OriginalSize      uint64
	CompressedSize    uint64
	CompressionRatio  float64
	Algorithm         CompressionType
	CompressionTimeMs int64
}

// EstimateCompressionRatio estimates the compression ratio for a given tensor
func (c *Compressor) EstimateCompressionRatio(sampleData []byte) (float64, error) {
	if len(sampleData) == 0 {
		return 1.0, nil
	}
	
	compressed, err := c.Compress(sampleData)
	if err != nil {
		return 0, err
	}
	
	if len(compressed) == 0 {
		return 1.0, nil
	}
	
	return float64(len(sampleData)) / float64(len(compressed)), nil
}

// SetOptions updates the compressor options
func (c *Compressor) SetOptions(options CompressionOptions) {
	c.options = options
}

// GetOptions returns the current compressor options
func (c *Compressor) GetOptions() CompressionOptions {
	return c.options
}