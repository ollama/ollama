//go:build mlx

package tokenizer

import (
	"strconv"
	"strings"
)

// Decode converts token IDs back to text
func (t *Tokenizer) Decode(ids []int32) string {
	var sb strings.Builder

	for _, id := range ids {
		if int(id) >= len(t.vocab.Values) {
			continue
		}

		token := t.vocab.Values[id]

		switch t.typ {
		case TokenizerSentencePiece:
			// SentencePiece style: replace ▁ with space, decode byte tokens
			token = strings.ReplaceAll(token, "▁", " ")
			// Handle byte fallback tokens like <0x0D>
			if len(token) == 6 && token[0] == '<' && token[1] == '0' && token[2] == 'x' && token[5] == '>' {
				if v, err := strconv.ParseUint(token[3:5], 16, 8); err == nil {
					sb.WriteByte(byte(v))
					continue
				}
			}
			sb.WriteString(token)
		default:
			// GPT-2 BPE style: decode byte-level encoding
			for _, r := range token {
				switch {
				case r == 0x0100:
					// Mirror GGML tokenizer behavior for NULL byte.
					// 0x00 is omitted during decode.
					continue
				case r == 0x0143:
					r = 0x00ad
				case r > 0x0100 && r <= 0x0120:
					r = r - 0x0100
				case r > 0x0120 && r <= 0x0142:
					r = r - 0x00a2
				}

				// Write as byte, not UTF-8 encoded rune
				sb.WriteByte(byte(r))
			}
		}
	}

	return sb.String()
}
