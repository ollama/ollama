package tensor

import (
	"encoding/binary"
	"fmt"
	"io"
	"net"
	"sync"
	"time"
)

// MessageType defines the type of tensor communication message
type MessageType uint8

const (
	// TypeTensorRequest requests a tensor from another node
	TypeTensorRequest MessageType = iota + 1
	
	// TypeTensorResponse sends a requested tensor
	TypeTensorResponse
	
	// TypeTensorSync synchronizes a tensor across nodes
	TypeTensorSync
	
	// TypeTensorAck acknowledges receipt of a tensor
	TypeTensorAck
	
	// TypeStateRequest requests model state from another node
	TypeStateRequest
	
	// TypeStateResponse sends requested state information
	TypeStateResponse
	
	// TypeError indicates an error occurred
	TypeError
)

// Header is the common header for all tensor protocol messages
type Header struct {
	// Type is the message type
	Type MessageType
	
	// MessageID is a unique identifier for this message
	MessageID uint64
	
	// ModelID identifies which model this tensor belongs to
	ModelID string
	
	// PartitionID identifies the partition
	PartitionID string
	
	// TensorID identifies the specific tensor
	TensorID string
	
	// Timestamp when the message was created
	Timestamp time.Time
	
	// BodyLength is the length of the message body in bytes
	BodyLength uint64
}

// Protocol manages tensor communication between nodes
type Protocol struct {
	// conn is the network connection
	conn net.Conn
	
	// sendMu protects concurrent sends on the connection
	sendMu sync.Mutex
	
	// recvMu protects concurrent receives on the connection
	recvMu sync.Mutex
	
	// nextMessageID is used to generate unique message IDs
	nextMessageID uint64
	
	// idMu protects nextMessageID
	idMu sync.Mutex
}

// ProtocolError represents errors in tensor protocol communication
type ProtocolError struct {
	Code int
	Msg  string
}

func (e ProtocolError) Error() string {
	return fmt.Sprintf("tensor protocol error (code %d): %s", e.Code, e.Msg)
}

// NewProtocol creates a new tensor protocol handler
func NewProtocol(conn net.Conn) *Protocol {
	return &Protocol{
		conn:         conn,
		nextMessageID: 1,
	}
}

// generateMessageID creates a unique message ID
func (p *Protocol) generateMessageID() uint64 {
	p.idMu.Lock()
	defer p.idMu.Unlock()
	
	id := p.nextMessageID
	p.nextMessageID++
	return id
}

// SendTensorRequest sends a request for a specific tensor
func (p *Protocol) SendTensorRequest(modelID, partitionID, tensorID string) error {
	header := Header{
		Type:        TypeTensorRequest,
		MessageID:   p.generateMessageID(),
		ModelID:     modelID,
		PartitionID: partitionID,
		TensorID:    tensorID,
		Timestamp:   time.Now(),
		BodyLength:  0, // No body for requests
	}
	
	return p.sendMessage(header, nil)
}

// SendTensorResponse sends a tensor in response to a request
func (p *Protocol) SendTensorResponse(requestID uint64, modelID, partitionID, tensorID string, data []byte) error {
	header := Header{
		Type:        TypeTensorResponse,
		MessageID:   requestID, // Use original request ID
		ModelID:     modelID,
		PartitionID: partitionID,
		TensorID:    tensorID,
		Timestamp:   time.Now(),
		BodyLength:  uint64(len(data)),
	}
	
	return p.sendMessage(header, data)
}

// SendTensorSync broadcasts a tensor update to other nodes
func (p *Protocol) SendTensorSync(modelID, partitionID, tensorID string, data []byte) error {
	header := Header{
		Type:        TypeTensorSync,
		MessageID:   p.generateMessageID(),
		ModelID:     modelID,
		PartitionID: partitionID,
		TensorID:    tensorID,
		Timestamp:   time.Now(),
		BodyLength:  uint64(len(data)),
	}
	
	return p.sendMessage(header, data)
}

// SendAck sends an acknowledgment for a received message
func (p *Protocol) SendAck(requestID uint64, modelID, partitionID, tensorID string) error {
	header := Header{
		Type:        TypeTensorAck,
		MessageID:   requestID, // Use original request ID
		ModelID:     modelID,
		PartitionID: partitionID,
		TensorID:    tensorID,
		Timestamp:   time.Now(),
		BodyLength:  0, // No body for acks
	}
	
	return p.sendMessage(header, nil)
}
// sendMessage sends a message with header and optional body
func (p *Protocol) sendMessage(header Header, data []byte) error {
	p.sendMu.Lock()
	defer p.sendMu.Unlock()
	
	// Serialize header
	headerBytes, err := encodeHeader(header)
	if err != nil {
		return fmt.Errorf("error encoding header: %w", err)
	}
	
	// Write header
	if _, err := p.conn.Write(headerBytes); err != nil {
		return fmt.Errorf("error sending header: %w", err)
	}
	
	// Write body if present
	if data != nil && len(data) > 0 {
		if _, err := p.conn.Write(data); err != nil {
			return fmt.Errorf("error sending body: %w", err)
		}
	}
	
	return nil
}

// ReceiveMessage reads and parses the next tensor protocol message
func (p *Protocol) ReceiveMessage() (Header, []byte, error) {
	p.recvMu.Lock()
	defer p.recvMu.Unlock()
	
	// Read header
	header, err := p.readHeader()
	if err != nil {
		return Header{}, nil, err
	}
	
	// Read body if present
	var data []byte
	if header.BodyLength > 0 {
		data = make([]byte, header.BodyLength)
		if _, err := io.ReadFull(p.conn, data); err != nil {
			return header, nil, fmt.Errorf("error reading message body: %w", err)
		}
	}
	
	return header, data, nil
}

// readHeader reads and decodes the message header
func (p *Protocol) readHeader() (Header, error) {
	// First read the fixed-size part of the header to get lengths
	fixedHeaderSize := 16 // Type (1) + MessageID (8) + BodyLength (8)
	fixedHeader := make([]byte, fixedHeaderSize)
	
	if _, err := io.ReadFull(p.conn, fixedHeader); err != nil {
		return Header{}, fmt.Errorf("error reading header: %w", err)
	}
	
	// Parse the fixed header
	msgType := MessageType(fixedHeader[0])
	messageID := binary.BigEndian.Uint64(fixedHeader[1:9])
	bodyLength := binary.BigEndian.Uint64(fixedHeader[9:17])
	
	// Now read the variable-length strings and timestamp
	varHeader, err := p.readVarHeader()
	if err != nil {
		return Header{}, err
	}
	
	// Combine into full header
	return Header{
		Type:        msgType,
		MessageID:   messageID,
		ModelID:     varHeader.ModelID,
		PartitionID: varHeader.PartitionID,
		TensorID:    varHeader.TensorID,
		Timestamp:   varHeader.Timestamp,
		BodyLength:  bodyLength,
	}, nil
}

// varHeader holds the variable-length parts of the header
type varHeader struct {
	ModelID     string
	PartitionID string
	TensorID    string
	Timestamp   time.Time
}

// readVarHeader reads the variable-length parts of the header
func (p *Protocol) readVarHeader() (varHeader, error) {
	var result varHeader
	
	// Read ModelID (length + string)
	modelIDLen, err := readUint16(p.conn)
	if err != nil {
		return result, fmt.Errorf("error reading ModelID length: %w", err)
	}
	
	if modelIDLen > 0 {
		modelIDBytes := make([]byte, modelIDLen)
		if _, err := io.ReadFull(p.conn, modelIDBytes); err != nil {
			return result, fmt.Errorf("error reading ModelID: %w", err)
		}
		result.ModelID = string(modelIDBytes)
	}
	
	// Read PartitionID (length + string)
	partitionIDLen, err := readUint16(p.conn)
	if err != nil {
		return result, fmt.Errorf("error reading PartitionID length: %w", err)
	}
	
	if partitionIDLen > 0 {
		partitionIDBytes := make([]byte, partitionIDLen)
		if _, err := io.ReadFull(p.conn, partitionIDBytes); err != nil {
			return result, fmt.Errorf("error reading PartitionID: %w", err)
		}
		result.PartitionID = string(partitionIDBytes)
	}
	
	// Read TensorID (length + string)
	tensorIDLen, err := readUint16(p.conn)
	if err != nil {
		return result, fmt.Errorf("error reading TensorID length: %w", err)
	}
	
	if tensorIDLen > 0 {
		tensorIDBytes := make([]byte, tensorIDLen)
		if _, err := io.ReadFull(p.conn, tensorIDBytes); err != nil {
			return result, fmt.Errorf("error reading TensorID: %w", err)
		}
		result.TensorID = string(tensorIDBytes)
	}
	
	// Read timestamp (8 bytes, nanoseconds since Unix epoch)
	timestampBytes := make([]byte, 8)
	if _, err := io.ReadFull(p.conn, timestampBytes); err != nil {
		return result, fmt.Errorf("error reading timestamp: %w", err)
	}
	
	nanos := binary.BigEndian.Uint64(timestampBytes)
	result.Timestamp = time.Unix(0, int64(nanos))
	
	return result, nil
}

// encodeHeader serializes a header to bytes
func encodeHeader(header Header) ([]byte, error) {
	// Calculate size needed
	// Fixed part: Type (1) + MessageID (8) + BodyLength (8)
	// Variable part: Each string has length (2) + content
	// Plus timestamp (8)
	size := 17
	size += 2 + len(header.ModelID)
	size += 2 + len(header.PartitionID)
	size += 2 + len(header.TensorID)
	size += 8 // Timestamp
	
	buf := make([]byte, size)
	offset := 0
	
	// Write fixed part
	buf[offset] = byte(header.Type)
	offset++
	
	binary.BigEndian.PutUint64(buf[offset:offset+8], header.MessageID)
	offset += 8
	
	binary.BigEndian.PutUint64(buf[offset:offset+8], header.BodyLength)
	offset += 8
	
	// Write ModelID
	binary.BigEndian.PutUint16(buf[offset:offset+2], uint16(len(header.ModelID)))
	offset += 2
	copy(buf[offset:], header.ModelID)
	offset += len(header.ModelID)
	
	// Write PartitionID
	binary.BigEndian.PutUint16(buf[offset:offset+2], uint16(len(header.PartitionID)))
	offset += 2
	copy(buf[offset:], header.PartitionID)
	offset += len(header.PartitionID)
	
	// Write TensorID
	binary.BigEndian.PutUint16(buf[offset:offset+2], uint16(len(header.TensorID)))
	offset += 2
	copy(buf[offset:], header.TensorID)
	offset += len(header.TensorID)
	
	// Write timestamp
	binary.BigEndian.PutUint64(buf[offset:offset+8], uint64(header.Timestamp.UnixNano()))
	
	return buf, nil
}

// Helper function to read a uint16 from a connection
func readUint16(r io.Reader) (uint16, error) {
	buf := make([]byte, 2)
	if _, err := io.ReadFull(r, buf); err != nil {
		return 0, err
	}
	return binary.BigEndian.Uint16(buf), nil
}

// Close closes the underlying connection
func (p *Protocol) Close() error {
	if p.conn != nil {
		return p.conn.Close()
	}
	return nil
}

// SendError sends an error message
func (p *Protocol) SendError(requestID uint64, code int, errorMsg string) error {
	header := Header{
		Type:        TypeError,
		MessageID:   requestID, // Use original request ID
		Timestamp:   time.Now(),
		BodyLength:  uint64(len(errorMsg)),
	}
	
	return p.sendMessage(header, []byte(errorMsg))
}