package configtest

import (
	"crypto/rand"
	"crypto/rsa"
	"crypto/x509"
	"crypto/x509/pkix"
	"encoding/pem"
	"fmt"
	"math/big"
	"net"
	"os"
	"path/filepath"
	"time"
)

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
