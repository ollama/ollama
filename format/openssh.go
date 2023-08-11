// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Code originally from https://go-review.googlesource.com/c/crypto/+/218620

// TODO: replace with upstream once the above change is merged and released.

package format

import (
	"crypto"
	"crypto/ecdsa"
	"crypto/ed25519"
	"crypto/elliptic"
	"crypto/rand"
	"crypto/rsa"
	"encoding/binary"
	"encoding/pem"
	"fmt"
	"math/big"

	"golang.org/x/crypto/ssh"
)

const privateKeyAuthMagic = "openssh-key-v1\x00"

type openSSHEncryptedPrivateKey struct {
	CipherName string
	KDFName    string
	KDFOptions string
	KeysCount  uint32
	PubKey     []byte
	KeyBlocks  []byte
}

type openSSHPrivateKey struct {
	Check1  uint32
	Check2  uint32
	Keytype string
	Rest    []byte `ssh:"rest"`
}

type openSSHRSAPrivateKey struct {
	N       *big.Int
	E       *big.Int
	D       *big.Int
	Iqmp    *big.Int
	P       *big.Int
	Q       *big.Int
	Comment string
	Pad     []byte `ssh:"rest"`
}

type openSSHECDSAPrivateKey struct {
	Curve   string
	Pub     []byte
	D       *big.Int
	Comment string
	Pad     []byte `ssh:"rest"`
}

type openSSHEd25519PrivateKey struct {
	Pub     []byte
	Priv    []byte
	Comment string
	Pad     []byte `ssh:"rest"`
}

func OpenSSHPrivateKey(key crypto.PrivateKey, comment string) (*pem.Block, error) {
	var check uint32
	if err := binary.Read(rand.Reader, binary.BigEndian, &check); err != nil {
		return nil, err
	}

	var pk1 openSSHPrivateKey
	pk1.Check1 = check
	pk1.Check2 = check

	var w openSSHEncryptedPrivateKey
	w.KeysCount = 1

	if k, ok := key.(*ed25519.PrivateKey); ok {
		key = *k
	}

	switch k := key.(type) {
	case *rsa.PrivateKey:
		e := new(big.Int).SetInt64(int64(k.E))

		key := openSSHRSAPrivateKey{
			N:       k.N,
			E:       e,
			D:       k.D,
			Iqmp:    k.Precomputed.Qinv,
			P:       k.Primes[0],
			Q:       k.Primes[1],
			Comment: comment,
		}

		pk1.Keytype = ssh.KeyAlgoRSA
		pk1.Rest = ssh.Marshal(key)

		w.PubKey = ssh.Marshal(struct {
			KeyType string
			E       *big.Int
			N       *big.Int
		}{
			ssh.KeyAlgoRSA, e, k.N,
		})
	case *ecdsa.PrivateKey:
		var curve, keytype string
		switch name := k.Curve.Params().Name; name {
		case "P-256":
			curve = "nistp256"
			keytype = ssh.KeyAlgoECDSA256
		case "P-384":
			curve = "nistp384"
			keytype = ssh.KeyAlgoECDSA384
		case "P-521":
			curve = "nistp521"
			keytype = ssh.KeyAlgoECDSA521
		default:
			return nil, fmt.Errorf("ssh: unknown curve %q", name)
		}

		pub := elliptic.Marshal(k.Curve, k.X, k.Y)

		key := openSSHECDSAPrivateKey{
			Curve:   curve,
			Pub:     pub,
			D:       k.D,
			Comment: comment,
		}

		pk1.Keytype = keytype
		pk1.Rest = ssh.Marshal(key)

		w.PubKey = ssh.Marshal(struct {
			KeyType string
			Curve   string
			Pub     []byte
		}{
			keytype, curve, pub,
		})
	case ed25519.PrivateKey:
		pub, priv := k[32:], k
		key := openSSHEd25519PrivateKey{
			Pub:     pub,
			Priv:    priv,
			Comment: comment,
		}

		pk1.Keytype = ssh.KeyAlgoED25519
		pk1.Rest = ssh.Marshal(key)

		w.PubKey = ssh.Marshal(struct {
			KeyType string
			Pub     []byte
		}{
			ssh.KeyAlgoED25519, pub,
		})
	default:
		return nil, fmt.Errorf("ssh: unknown key type %T", k)
	}

	w.KeyBlocks = openSSHPadding(ssh.Marshal(pk1), 8)

	w.CipherName, w.KDFName, w.KDFOptions = "none", "none", ""

	return &pem.Block{
		Type:  "OPENSSH PRIVATE KEY",
		Bytes: append([]byte(privateKeyAuthMagic), ssh.Marshal(w)...),
	}, nil
}

func openSSHPadding(block []byte, blocksize int) []byte {
	for i, j := 0, len(block); (j+i)%blocksize != 0; i++ {
		block = append(block, byte(i+1))
	}

	return block
}
