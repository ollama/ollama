package model

const (
	TOKEN_TYPE_NORMAL = iota + 1
	TOKEN_TYPE_UNKNOWN
	TOKEN_TYPE_CONTROL
	TOKEN_TYPE_USER_DEFINED
	TOKEN_TYPE_UNUSED
	TOKEN_TYPE_BYTE
)

type TextProcessor interface {
	Encode(s string, addSpecial bool) ([]int32, error)
	Decode([]int32) (string, error)
	Is(int32, Special) bool
	Vocabulary() *Vocabulary
}
