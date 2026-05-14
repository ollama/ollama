package discover

// RPC device discovery — adapted from PR ollama/ollama#10844 by @gkpln3.
//
// The main Ollama scheduler enumerates GPUs via subprocess bootstrap, which
// never sees devices registered by ggml_backend_rpc_add_server() in the runner.
// Instead of trying to plumb CGo into discover, we speak the ggml-rpc wire
// protocol directly over TCP and synthesize ml.DeviceInfo entries that look
// just like local GPUs to the scheduler.

import (
	"encoding/binary"
	"errors"
	"fmt"
	"log/slog"
	"net"
	"strconv"
	"strings"
	"time"

	"github.com/ollama/ollama/ml"
)

const (
	rpcCmdGetDeviceMemory byte = 11
	rpcCmdHello           byte = 14
	rpcDialTimeout             = 5 * time.Second

	// Wire-protocol sizes (ggml-rpc v3.x). Must stay in sync with
	// ml/backend/ggml/ggml/src/ggml-rpc/ggml-rpc.cpp.
	rpcConnCapsSize       = 24 // RPC_CONN_CAPS_SIZE
	rpcHelloReqSize       = rpcConnCapsSize
	rpcHelloRspSize       = 4 + rpcConnCapsSize // major+minor+patch+padding+conn_caps
	rpcGetDevMemReqSize   = 4                   // uint32_t device
	rpcGetDevMemRspSize   = 16                  // free + total (u64 each)
)

// rpcServerMemory queries free and total memory of a remote ggml-rpc server.
// Returns (0, 0) if the server is unreachable or speaks a different protocol.
func rpcServerMemory(endpoint string) (free, total uint64, err error) {
	conn, err := net.DialTimeout("tcp", endpoint, rpcDialTimeout)
	if err != nil {
		return 0, 0, fmt.Errorf("dial: %w", err)
	}
	defer conn.Close()

	// Handshake: RPC_CMD_HELLO. v3.x expects 24 bytes of conn_caps (zeros are
	// fine for a "no special features" client) and replies with 28 bytes:
	// major + minor + patch + padding + 24 bytes of server conn_caps.
	helloReq := make([]byte, rpcHelloReqSize)
	if err := rpcCall(conn, rpcCmdHello, helloReq, rpcHelloRspSize); err != nil {
		return 0, 0, fmt.Errorf("hello: %w", err)
	}
	helloRsp := make([]byte, rpcHelloRspSize)
	if _, err := conn.Read(helloRsp); err != nil {
		return 0, 0, fmt.Errorf("read hello: %w", err)
	}
	slog.Debug("rpc server version", "endpoint", endpoint,
		"version", fmt.Sprintf("%d.%d.%d", helloRsp[0], helloRsp[1], helloRsp[2]))

	// Query device 0 memory. Request is 4 bytes (uint32 device index, LE),
	// response is 16 bytes (free u64 + total u64).
	memReq := make([]byte, rpcGetDevMemReqSize) // device = 0
	if err := rpcCall(conn, rpcCmdGetDeviceMemory, memReq, rpcGetDevMemRspSize); err != nil {
		return 0, 0, fmt.Errorf("get_device_memory: %w", err)
	}
	memRsp := make([]byte, rpcGetDevMemRspSize)
	if _, err := conn.Read(memRsp); err != nil {
		return 0, 0, fmt.Errorf("read memory: %w", err)
	}
	return binary.LittleEndian.Uint64(memRsp[:8]),
		binary.LittleEndian.Uint64(memRsp[8:]),
		nil
}

// rpcCall sends a command byte + payload, then reads & validates the response
// size header (8 bytes LE), expecting expectedReplySize bytes.
func rpcCall(conn net.Conn, cmd byte, payload []byte, expectedReplySize uint64) error {
	conn.SetDeadline(time.Now().Add(rpcDialTimeout))
	if _, err := conn.Write([]byte{cmd}); err != nil {
		return err
	}
	var sizeBuf [8]byte
	binary.LittleEndian.PutUint64(sizeBuf[:], uint64(len(payload)))
	if _, err := conn.Write(sizeBuf[:]); err != nil {
		return err
	}
	if len(payload) > 0 {
		if _, err := conn.Write(payload); err != nil {
			return err
		}
	}
	conn.SetDeadline(time.Now().Add(rpcDialTimeout))
	var replyBuf [8]byte
	if _, err := conn.Read(replyBuf[:]); err != nil {
		return err
	}
	if got := binary.LittleEndian.Uint64(replyBuf[:]); got != expectedReplySize {
		return fmt.Errorf("unexpected reply size %d (want %d)", got, expectedReplySize)
	}
	return nil
}

// discoverRPCDevices probes each endpoint in OLLAMA_RPC_SERVERS and returns
// synthetic DeviceInfo entries the scheduler can route layers to. The IDs are
// constructed to match what ggml-rpc names its devices on the runner side
// (Library="RPC", ID="RPC0[host:port]") so DeviceID lookups in the load
// handler resolve correctly.
func discoverRPCDevices(endpoints []string) []ml.DeviceInfo {
	var devices []ml.DeviceInfo
	for _, ep := range endpoints {
		ep = strings.TrimSpace(ep)
		if ep == "" {
			continue
		}
		host, port, err := net.SplitHostPort(ep)
		if err != nil {
			slog.Warn("invalid RPC endpoint, expected host:port", "endpoint", ep, "err", err)
			continue
		}
		if _, err := strconv.ParseUint(port, 10, 16); err != nil {
			slog.Warn("invalid RPC endpoint port", "endpoint", ep)
			continue
		}

		free, total, err := rpcServerMemory(ep)
		if err != nil || total == 0 {
			slog.Warn("RPC server unreachable", "endpoint", ep, "err", err)
			continue
		}

		devices = append(devices, ml.DeviceInfo{
			DeviceID: ml.DeviceID{
				ID:      fmt.Sprintf("RPC0[%s]", ep),
				Library: "RPC",
			},
			Name:        "RPC",
			Description: fmt.Sprintf("remote ggml-rpc server at %s", ep),
			TotalMemory: total,
			FreeMemory:  free,
		})
		slog.Info("discovered RPC device", "endpoint", ep, "host", host, "free", free, "total", total)
	}
	return devices
}

// errNoRPCServers is returned from RPCDevices when OLLAMA_RPC_SERVERS is unset
// or contains only empty entries.
var errNoRPCServers = errors.New("no RPC servers configured")
