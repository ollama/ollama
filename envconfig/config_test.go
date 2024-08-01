package envconfig

import (
	"crypto/rand"
	"crypto/rsa"
	"crypto/tls"
	"crypto/x509"
	"crypto/x509/pkix"
	"encoding/pem"
	"fmt"
	"math"
	"math/big"
	"net"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	"github.com/stretchr/testify/assert"
)

func TestHost(t *testing.T) {
	cases := map[string]struct {
		value  string
		expect string
	}{
		"empty":               {"", "http://127.0.0.1:11434"},
		"only address":        {"1.2.3.4", "http://1.2.3.4:11434"},
		"only port":           {":1234", "http://:1234"},
		"address and port":    {"1.2.3.4:1234", "http://1.2.3.4:1234"},
		"hostname":            {"example.com", "http://example.com:11434"},
		"hostname and port":   {"example.com:1234", "http://example.com:1234"},
		"zero port":           {":0", "http://:0"},
		"too large port":      {":66000", "http://:11434"},
		"too small port":      {":-1", "http://:11434"},
		"ipv6 localhost":      {"[::1]", "http://[::1]:11434"},
		"ipv6 world open":     {"[::]", "http://[::]:11434"},
		"ipv6 no brackets":    {"::1", "http://[::1]:11434"},
		"ipv6 + port":         {"[::1]:1337", "http://[::1]:1337"},
		"extra space":         {" 1.2.3.4 ", "http://1.2.3.4:11434"},
		"extra quotes":        {"\"1.2.3.4\"", "http://1.2.3.4:11434"},
		"extra space+quotes":  {" \" 1.2.3.4 \" ", "http://1.2.3.4:11434"},
		"extra single quotes": {"'1.2.3.4'", "http://1.2.3.4:11434"},
		"http":                {"http://1.2.3.4", "http://1.2.3.4:80"},
		"http port":           {"http://1.2.3.4:4321", "http://1.2.3.4:4321"},
		"https":               {"https://1.2.3.4", "https://1.2.3.4:443"},
		"https port":          {"https://1.2.3.4:4321", "https://1.2.3.4:4321"},
		"proxy path":          {"https://example.com/ollama", "https://example.com:443/ollama"},
	}

	for name, tt := range cases {
		t.Run(name, func(t *testing.T) {
			t.Setenv("OLLAMA_HOST", tt.value)
			if host := Host(); host.String() != tt.expect {
				t.Errorf("%s: expected %s, got %s", name, tt.expect, host.String())
			}
		})
	}
}

func TestOrigins(t *testing.T) {
	cases := []struct {
		value  string
		expect []string
	}{
		{"", []string{
			"http://localhost",
			"https://localhost",
			"http://localhost:*",
			"https://localhost:*",
			"http://127.0.0.1",
			"https://127.0.0.1",
			"http://127.0.0.1:*",
			"https://127.0.0.1:*",
			"http://0.0.0.0",
			"https://0.0.0.0",
			"http://0.0.0.0:*",
			"https://0.0.0.0:*",
			"app://*",
			"file://*",
			"tauri://*",
		}},
		{"http://10.0.0.1", []string{
			"http://10.0.0.1",
			"http://localhost",
			"https://localhost",
			"http://localhost:*",
			"https://localhost:*",
			"http://127.0.0.1",
			"https://127.0.0.1",
			"http://127.0.0.1:*",
			"https://127.0.0.1:*",
			"http://0.0.0.0",
			"https://0.0.0.0",
			"http://0.0.0.0:*",
			"https://0.0.0.0:*",
			"app://*",
			"file://*",
			"tauri://*",
		}},
		{"http://172.16.0.1,https://192.168.0.1", []string{
			"http://172.16.0.1",
			"https://192.168.0.1",
			"http://localhost",
			"https://localhost",
			"http://localhost:*",
			"https://localhost:*",
			"http://127.0.0.1",
			"https://127.0.0.1",
			"http://127.0.0.1:*",
			"https://127.0.0.1:*",
			"http://0.0.0.0",
			"https://0.0.0.0",
			"http://0.0.0.0:*",
			"https://0.0.0.0:*",
			"app://*",
			"file://*",
			"tauri://*",
		}},
		{"http://totally.safe,http://definitely.legit", []string{
			"http://totally.safe",
			"http://definitely.legit",
			"http://localhost",
			"https://localhost",
			"http://localhost:*",
			"https://localhost:*",
			"http://127.0.0.1",
			"https://127.0.0.1",
			"http://127.0.0.1:*",
			"https://127.0.0.1:*",
			"http://0.0.0.0",
			"https://0.0.0.0",
			"http://0.0.0.0:*",
			"https://0.0.0.0:*",
			"app://*",
			"file://*",
			"tauri://*",
		}},
	}
	for _, tt := range cases {
		t.Run(tt.value, func(t *testing.T) {
			t.Setenv("OLLAMA_ORIGINS", tt.value)

			if diff := cmp.Diff(Origins(), tt.expect); diff != "" {
				t.Errorf("%s: mismatch (-want +got):\n%s", tt.value, diff)
			}
		})
	}
}

func TestBool(t *testing.T) {
	cases := map[string]bool{
		"":      false,
		"true":  true,
		"false": false,
		"1":     true,
		"0":     false,
		// invalid values
		"random":    true,
		"something": true,
	}

	for k, v := range cases {
		t.Run(k, func(t *testing.T) {
			t.Setenv("OLLAMA_BOOL", k)
			if b := Bool("OLLAMA_BOOL")(); b != v {
				t.Errorf("%s: expected %t, got %t", k, v, b)
			}
		})
	}
}

func TestUint(t *testing.T) {
	cases := map[string]uint{
		"0":    0,
		"1":    1,
		"1337": 1337,
		// default values
		"":       11434,
		"-1":     11434,
		"0o10":   11434,
		"0x10":   11434,
		"string": 11434,
	}

	for k, v := range cases {
		t.Run(k, func(t *testing.T) {
			t.Setenv("OLLAMA_UINT", k)
			if i := Uint("OLLAMA_UINT", 11434)(); i != v {
				t.Errorf("%s: expected %d, got %d", k, v, i)
			}
		})
	}
}

func TestKeepAlive(t *testing.T) {
	cases := map[string]time.Duration{
		"":       5 * time.Minute,
		"1s":     time.Second,
		"1m":     time.Minute,
		"1h":     time.Hour,
		"5m0s":   5 * time.Minute,
		"1h2m3s": 1*time.Hour + 2*time.Minute + 3*time.Second,
		"0":      time.Duration(0),
		"60":     60 * time.Second,
		"120":    2 * time.Minute,
		"3600":   time.Hour,
		"-0":     time.Duration(0),
		"-1":     time.Duration(math.MaxInt64),
		"-1m":    time.Duration(math.MaxInt64),
		// invalid values
		" ":   5 * time.Minute,
		"???": 5 * time.Minute,
		"1d":  5 * time.Minute,
		"1y":  5 * time.Minute,
		"1w":  5 * time.Minute,
	}

	for tt, expect := range cases {
		t.Run(tt, func(t *testing.T) {
			t.Setenv("OLLAMA_KEEP_ALIVE", tt)
			if actual := KeepAlive(); actual != expect {
				t.Errorf("%s: expected %s, got %s", tt, expect, actual)
			}
		})
	}
}

func TestLoadTimeout(t *testing.T) {
	defaultTimeout := 5 * time.Minute
	cases := map[string]time.Duration{
		"":       defaultTimeout,
		"1s":     time.Second,
		"1m":     time.Minute,
		"1h":     time.Hour,
		"5m0s":   defaultTimeout,
		"1h2m3s": 1*time.Hour + 2*time.Minute + 3*time.Second,
		"0":      time.Duration(math.MaxInt64),
		"60":     60 * time.Second,
		"120":    2 * time.Minute,
		"3600":   time.Hour,
		"-0":     time.Duration(math.MaxInt64),
		"-1":     time.Duration(math.MaxInt64),
		"-1m":    time.Duration(math.MaxInt64),
		// invalid values
		" ":   defaultTimeout,
		"???": defaultTimeout,
		"1d":  defaultTimeout,
		"1y":  defaultTimeout,
		"1w":  defaultTimeout,
	}

	for tt, expect := range cases {
		t.Run(tt, func(t *testing.T) {
			t.Setenv("OLLAMA_LOAD_TIMEOUT", tt)
			if actual := LoadTimeout(); actual != expect {
				t.Errorf("%s: expected %s, got %s", tt, expect, actual)
			}
		})
	}
}

func TestVar(t *testing.T) {
	cases := map[string]string{
		"value":       "value",
		" value ":     "value",
		" 'value' ":   "value",
		` "value" `:   "value",
		" ' value ' ": " value ",
		` " value " `: " value ",
	}

	for k, v := range cases {
		t.Run(k, func(t *testing.T) {
			t.Setenv("OLLAMA_VAR", k)
			if s := Var("OLLAMA_VAR"); s != v {
				t.Errorf("%s: expected %q, got %q", k, v, s)
			}
		})
	}
}

// Helper struct for testing TLS
type TLSTestData struct {
	ServerKey  string
	ServerCert string
	ServerCA   string
	ClientKey  string
	ClientCert string
	ClientCA   string
}

func genKeyCertCA(dir string, label string) (string, string, string) {
	// Generate the CA key/cert
	caKey, _ := rsa.GenerateKey(rand.Reader, 2048)
	serialNumberLimit := new(big.Int).Lsh(big.NewInt(1), 128)
	caSerialNumber, _ := rand.Int(rand.Reader, serialNumberLimit)
	ca := &x509.Certificate{
		SerialNumber: caSerialNumber,
		Subject: pkix.Name{
			Organization: []string{fmt.Sprintf("Root Ollama %s", label)},
		},
		NotBefore: time.Now(),
		NotAfter:  time.Now().Add(time.Hour * 24),

		KeyUsage:              x509.KeyUsageCertSign | x509.KeyUsageDigitalSignature,
		ExtKeyUsage:           []x509.ExtKeyUsage{x509.ExtKeyUsageClientAuth, x509.ExtKeyUsageServerAuth},
		BasicConstraintsValid: true,
		IsCA:                  true,
	}
	caCertBytes, _ := x509.CreateCertificate(rand.Reader, ca, ca, &caKey.PublicKey, caKey)

	// Generate the derived key/cert
	derivedKey, _ := rsa.GenerateKey(rand.Reader, 2048)
	derivedSerialNumber, _ := rand.Int(rand.Reader, serialNumberLimit)
	cert := &x509.Certificate{
		SerialNumber: derivedSerialNumber,
		Subject: pkix.Name{
			CommonName:   fmt.Sprintf("foo.bar.%s", label),
			Organization: []string{fmt.Sprintf("Derived Ollama %s", label)},
		},
		NotBefore: time.Now(),
		NotAfter:  time.Now().Add(time.Hour * 24),

		KeyUsage:              x509.KeyUsageDigitalSignature | x509.KeyUsageKeyEncipherment,
		ExtKeyUsage:           []x509.ExtKeyUsage{x509.ExtKeyUsageClientAuth, x509.ExtKeyUsageServerAuth},
		BasicConstraintsValid: true,
		IsCA:                  false,
		DNSNames:              []string{"localhost"},
		IPAddresses:           []net.IP{net.IPv4(127, 0, 0, 1), net.IPv6loopback},
	}
	derivedCertBytes, _ := x509.CreateCertificate(rand.Reader, cert, ca, &derivedKey.PublicKey, caKey)

	// Create the PEM serialized versions and return them
	caPEM := pem.EncodeToMemory(&pem.Block{Type: "CERTIFICATE", Bytes: caCertBytes})
	derivedKeyPEM := pem.EncodeToMemory(&pem.Block{
		Type:  "RSA PRIVATE KEY",
		Bytes: x509.MarshalPKCS1PrivateKey(derivedKey),
	})
	derivedCertPEM := pem.EncodeToMemory(&pem.Block{Type: "CERTIFICATE", Bytes: derivedCertBytes})

	// Write them all to the output dir
	keyFile := filepath.Join(dir, fmt.Sprintf("%s.key.pem", label))
	certFile := filepath.Join(dir, fmt.Sprintf("%s.cert.pem", label))
	caFile := filepath.Join(dir, fmt.Sprintf("%s.ca.pem", label))
	os.WriteFile(keyFile, derivedKeyPEM, 0644)
	os.WriteFile(certFile, derivedCertPEM, 0644)
	os.WriteFile(caFile, caPEM, 0644)

	// Return the file paths
	return keyFile, certFile, caFile
}

func NewTLSTestData(dir string) TLSTestData {
	serverKey, serverCert, serverCA := genKeyCertCA(dir, "server")
	clientKey, clientCert, clientCA := genKeyCertCA(dir, "client")
	return TLSTestData{
		ServerCA:   serverCA,
		ServerKey:  serverKey,
		ServerCert: serverCert,
		ClientCA:   clientCA,
		ClientKey:  clientKey,
		ClientCert: clientCert,
	}
}

// Test that config is parsed correctly if all TLS config is given
func TestTlsConfigFromEnvironment(t *testing.T) {
	tlsTestData := NewTLSTestData(t.TempDir())
	t.Setenv("OLLAMA_HOST", "https://localhost:12345")
	t.Setenv("OLLAMA_TLS_SERVER_KEY", tlsTestData.ServerKey)
	t.Setenv("OLLAMA_TLS_SERVER_CERT", tlsTestData.ServerCert)
	t.Setenv("OLLAMA_TLS_SERVER_CA", tlsTestData.ServerCA)
	t.Setenv("OLLAMA_TLS_CLIENT_KEY", tlsTestData.ClientKey)
	t.Setenv("OLLAMA_TLS_CLIENT_CERT", tlsTestData.ClientCert)
	t.Setenv("OLLAMA_TLS_CLIENT_CA", tlsTestData.ClientCA)
	assert.NotNil(t, ServerTlsConfig())
	assert.NotNil(t, ClientTlsConfig())
}

// Test that server TLS config is parsed correctly if only server config is given
func TestTlsConfigServerOnly(t *testing.T) {
	tlsTestData := NewTLSTestData(t.TempDir())
	t.Setenv("OLLAMA_HOST", "https://localhost:12345")
	t.Setenv("OLLAMA_TLS_SERVER_KEY", tlsTestData.ServerKey)
	t.Setenv("OLLAMA_TLS_SERVER_CERT", tlsTestData.ServerCert)
	t.Setenv("OLLAMA_TLS_CLIENT_CA", tlsTestData.ClientCA)
	assert.NotNil(t, ServerTlsConfig())
	// NOTE: The client TLS config will be configured to use system certs in
	//   this case
}

// Test that client TLS config is parsed correctly if only client config is given
func TestTlsConfigClientOnly(t *testing.T) {
	tlsTestData := NewTLSTestData(t.TempDir())
	t.Setenv("OLLAMA_HOST", "https://localhost:12345")
	t.Setenv("OLLAMA_TLS_CLIENT_KEY", tlsTestData.ClientKey)
	t.Setenv("OLLAMA_TLS_CLIENT_CERT", tlsTestData.ClientCert)
	t.Setenv("OLLAMA_TLS_SERVER_CA", tlsTestData.ServerCA)
	assert.Nil(t, ServerTlsConfig())
	assert.NotNil(t, ClientTlsConfig())
}

// Test that no TLS config is parsed, even if env vars set, when scheme is http
func TestTlsConfigHTTPOnlyScheme(t *testing.T) {
	tlsTestData := NewTLSTestData(t.TempDir())
	t.Setenv("OLLAMA_HOST", "http://localhost:12345")
	t.Setenv("OLLAMA_TLS_SERVER_KEY", tlsTestData.ServerKey)
	t.Setenv("OLLAMA_TLS_SERVER_CERT", tlsTestData.ServerCert)
	t.Setenv("OLLAMA_TLS_SERVER_CA", tlsTestData.ServerCA)
	t.Setenv("OLLAMA_TLS_CLIENT_KEY", tlsTestData.ClientKey)
	t.Setenv("OLLAMA_TLS_CLIENT_CERT", tlsTestData.ClientCert)
	t.Setenv("OLLAMA_TLS_CLIENT_CA", tlsTestData.ClientCA)
	assert.Nil(t, ServerTlsConfig())
	assert.Nil(t, ClientTlsConfig())
}

// Test non-mutual TLS config
func TestTlsConfigNonMutual(t *testing.T) {
	tlsTestData := NewTLSTestData(t.TempDir())
	t.Setenv("OLLAMA_HOST", "https://localhost:12345")
	t.Setenv("OLLAMA_TLS_SERVER_KEY", tlsTestData.ServerKey)
	t.Setenv("OLLAMA_TLS_SERVER_CERT", tlsTestData.ServerCert)
	t.Setenv("OLLAMA_TLS_SERVER_CA", tlsTestData.ServerCA)
	assert.NotNil(t, ServerTlsConfig())
	assert.Nil(t, ServerTlsConfig().ClientCAs)
	assert.Equal(t, ServerTlsConfig().ClientAuth, tls.NoClientCert)
	assert.NotNil(t, ClientTlsConfig())
	assert.Nil(t, ClientTlsConfig().Certificates)
}
