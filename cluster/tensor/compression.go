package tensor

import (
	"bytes"
	"compress/flate"
	"fmt"
	"io"
	"io/ioutil"
	"math"
	"sync"
	"time"

	"github.com/pierrec/lz4/v4"
)

// CompressionConfig defines settings for tensor compression
type CompressionConfig struct {
	// DefaultType is the default compression algorithm to use
	DefaultType CompressionType

	// MinSizeForCompression is the minimum tensor size to apply compression
	MinSizeForCompression uint64

	// AdaptiveThreshold enables automatic selection of compression algorithm
	AdaptiveThreshold bool

	// LZ4Level controls the LZ4 compression level (0-9)
	LZ4Level int

	// DeflateLevel controls the Deflate compression level (1-9)
	DeflateLevel int

	// CompressionCache enables caching of compression results
	CompressionCache bool

	// MaxCacheSize is the maximum number of entries in the compression cache
	MaxCacheSize int
}

// Compressor provides methods for compressing and decompressing tensor data
type Compressor struct {
	// config holds the compression configuration
	config CompressionConfig

	// cache stores recently compressed/decompressed data
	cache map[string][]byte

	// cacheMutex protects the cache map
	cacheMutex sync.RWMutex
}

// DefaultCompressionConfig returns a default compression configuration
func DefaultCompressionConfig() CompressionConfig {
	return CompressionConfig{
		DefaultType:           CompressionLZ4,
		MinSizeForCompression: 4 * 1024, // 4KB
		AdaptiveThreshold:     true,
		LZ4Level:              6, // Medium compression
		DeflateLevel:          6, // Medium compression
		CompressionCache:      true,
		MaxCacheSize:          1000,
	}
}

// NewCompressor creates a new tensor compressor with the given configuration
func NewCompressor(config CompressionConfig) *Compressor {
	return &Compressor{
		config: config,
		cache:  make(map[string][]byte),
	}
}

// Compress compresses the data using the specified compression type
func (c *Compressor) Compress(data []byte, compressionType CompressionType) ([]byte, error) {
	// If data is too small, skip compression
	if uint64(len(data)) < c.config.MinSizeForCompression {
		return data, nil
	}

	// Handle each compression type
	switch compressionType {
	case CompressionNone:
		return data, nil

	case CompressionLZ4:
		return c.compressLZ4(data)

	case CompressionDeflate:
		return c.compressDeflate(data)

	default:
		return nil, fmt.Errorf("unsupported compression type: %d", compressionType)
	}
}

// Decompress decompresses the data using the specified compression type
func (c *Compressor) Decompress(compressed []byte, compressionType CompressionType, originalSize uint64) ([]byte, error) {
	// If no compression was applied, return the data as-is
	if compressionType == CompressionNone {
		return compressed, nil
	}

	// Handle each compression type
	switch compressionType {
	case CompressionLZ4:
		return c.decompressLZ4(compressed, originalSize)

	case CompressionDeflate:
		return c.decompressDeflate(compressed, originalSize)

	default:
		return nil, fmt.Errorf("unsupported compression type: %d", compressionType)
	}
}

// compressLZ4 compresses data using LZ4 algorithm
func (c *Compressor) compressLZ4(data []byte) ([]byte, error) {
	var buf bytes.Buffer
	writer := lz4.NewWriter(&buf)

	// Set compression level
	writer.Apply(lz4.CompressionLevelOption(lz4.CompressionLevel(c.config.LZ4Level)))

	// Write the compressed data
	if _, err := writer.Write(data); err != nil {
		return nil, fmt.Errorf("lz4 compression write error: %w", err)
	}

	// Close the writer to flush any remaining data
	if err := writer.Close(); err != nil {
		return nil, fmt.Errorf("lz4 compression close error: %w", err)
	}

	return buf.Bytes(), nil
}

// decompressLZ4 decompresses LZ4-compressed data
func (c *Compressor) decompressLZ4(compressed []byte, originalSize uint64) ([]byte, error) {
	reader := lz4.NewReader(bytes.NewReader(compressed))
	
	// If we know the original size, preallocate the buffer
	var decompressed []byte
	if originalSize > 0 {
		decompressed = make([]byte, originalSize)
		_, err := io.ReadFull(reader, decompressed)
		if err != nil && err != io.EOF {
			return nil, fmt.Errorf("lz4 decompression read error: %w", err)
		}
	} else {
		// Otherwise, read until EOF
		var err error
		decompressed, err = ioutil.ReadAll(reader)
		if err != nil {
			return nil, fmt.Errorf("lz4 decompression read error: %w", err)
		}
	}

	return decompressed, nil
}

// compressDeflate compresses data using Deflate algorithm
func (c *Compressor) compressDeflate(data []byte) ([]byte, error) {
	var buf bytes.Buffer
	writer, err := flate.NewWriter(&buf, c.config.DeflateLevel)
	if err != nil {
		return nil, fmt.Errorf("deflate compression initialization error: %w", err)
	}

	// Write the compressed data
	if _, err := writer.Write(data); err != nil {
		return nil, fmt.Errorf("deflate compression write error: %w", err)
	}

	// Close the writer to flush any remaining data
	if err := writer.Close(); err != nil {
		return nil, fmt.Errorf("deflate compression close error: %w", err)
	}

	return buf.Bytes(), nil
}

// decompressDeflate decompresses Deflate-compressed data
func (c *Compressor) decompressDeflate(compressed []byte, originalSize uint64) ([]byte, error) {
	reader := flate.NewReader(bytes.NewReader(compressed))
	defer reader.Close()

	// If we know the original size, preallocate the buffer
	var decompressed []byte
	if originalSize > 0 {
		decompressed = make([]byte, originalSize)
		_, err := io.ReadFull(reader, decompressed)
		if err != nil && err != io.EOF {
			return nil, fmt.Errorf("deflate decompression read error: %w", err)
		}
	} else {
		// Otherwise, read until EOF
		var err error
		decompressed, err = ioutil.ReadAll(reader)
		if err != nil {
			return nil, fmt.Errorf("deflate decompression read error: %w", err)
		}
	}

	return decompressed, nil
}

// SelectBestCompression analyzes the data and selects the best compression algorithm
func (c *Compressor) SelectBestCompression(data []byte) CompressionType {
	// If data is too small, don't compress
	if uint64(len(data)) < c.config.MinSizeForCompression {
		return CompressionNone
	}

	// If adaptive threshold is disabled, use the default
	if !c.config.AdaptiveThreshold {
		return c.config.DefaultType
	}

	// Sample the data to determine compressibility
	sampleSize := 4 * 1024 // 4KB sample
	if len(data) < sampleSize {
		sampleSize = len(data)
	}

	// Take a sample from the beginning of the data
	sample := data[:sampleSize]

	// Check entropy of the sample
	entropy := calculateEntropy(sample)

	// High entropy (>7.0) indicates less compressible data
	// Medium entropy (5.0-7.0) indicates moderately compressible data
	// Low entropy (<5.0) indicates highly compressible data
	if entropy > 7.0 {
		// For high entropy data, LZ4 is usually more efficient
		return CompressionLZ4
	} else if entropy < 5.0 {
		// For low entropy data, Deflate often achieves better compression
		return CompressionDeflate
	} else {
		// For medium entropy, use the default
		return c.config.DefaultType
	}
}

// calculateEntropy calculates Shannon entropy of the data
func calculateEntropy(data []byte) float64 {
	if len(data) == 0 {
		return 0.0
	}

	// Count frequency of each byte value
	counts := make(map[byte]int)
	for _, b := range data {
		counts[b]++
	}

	// Calculate entropy
	entropy := 0.0
	size := float64(len(data))
	for _, count := range counts {
		p := float64(count) / size
		entropy -= p * math.Log2(p)
	}

	return entropy
}

// CompressionRatio returns the ratio of compressed size to original size
func (c *Compressor) CompressionRatio(original, compressed []byte) float64 {
	if len(original) == 0 {
		return 1.0
	}
	return float64(len(compressed)) / float64(len(original))
}

// IsBeneficial determines if compression is beneficial for the given data
func (c *Compressor) IsBeneficial(original, compressed []byte) bool {
	// If compression actually increased the size, it's not beneficial
	return len(compressed) < len(original)
}

// CompressWithStats compresses data and returns compression statistics
func (c *Compressor) CompressWithStats(data []byte, compressionType CompressionType) ([]byte, map[string]interface{}, error) {
	// Start with the requested compression type
	effectiveType := compressionType
	
	// If auto-select is requested, determine the best type
	if compressionType == CompressionType(255) { // Special value for auto-select
		effectiveType = c.SelectBestCompression(data)
	}
	
	// Compress the data
	startTime := time.Now()
	compressed, err := c.Compress(data, effectiveType)
	if err != nil {
		return nil, nil, err
	}
	duration := time.Since(startTime)
	
	// Check if compression is beneficial
	isBeneficial := c.IsBeneficial(data, compressed)
	
	// If not beneficial, use uncompressed data
	if !isBeneficial {
		effectiveType = CompressionNone
		compressed = data
	}
	
	// Prepare statistics
	ratio := c.CompressionRatio(data, compressed)
	stats := map[string]interface{}{
		"original_size":     len(data),
		"compressed_size":   len(compressed),
		"compression_type":  effectiveType,
		"compression_ratio": ratio,
		"space_saving":      1.0 - ratio,
		"beneficial":        isBeneficial,
		"duration_ms":       duration.Milliseconds(),
	}
	
	return compressed, stats, nil
}

// ClearCache clears the compression cache
func (c *Compressor) ClearCache() {
	c.cacheMutex.Lock()
	defer c.cacheMutex.Unlock()
	
	c.cache = make(map[string][]byte)
}

// GetCacheStats returns statistics about the compression cache
func (c *Compressor) GetCacheStats() map[string]interface{} {
	c.cacheMutex.RLock()
	defer c.cacheMutex.RUnlock()
	
	totalSize := 0
	for _, data := range c.cache {
		totalSize += len(data)
	}
	
	return map[string]interface{}{
		"entries":     len(c.cache),
		"total_bytes": totalSize,
		"enabled":     c.config.CompressionCache,
		"max_entries": c.config.MaxCacheSize,
	}
}