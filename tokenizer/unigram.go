package tokenizer

import (
	"bytes"
	"errors"
	"fmt"
	"math"
	"slices"
	"strings"
	"unicode/utf8"
	"unsafe"
)

const (
	unigramEscapedSpace             = "\xE2\x96\x81"
	unigramUnknownTokenScorePenalty = 10.0
	xcdaArrayNodeSize               = 4 // uint32 size in bytes
)

type naiveTrie struct {
	children map[byte]*naiveTrie
	hasValue bool
	value    int32
}

func (n *naiveTrie) Insert(key string, value int32) {
	n.insert([]byte(key), value)
}

func (n *naiveTrie) insert(key []byte, value int32) {
	if len(key) == 0 {
		n.hasValue = true
		n.value = value
		return
	}

	if n.children == nil {
		n.children = make(map[byte]*naiveTrie)
	}

	firstByte := key[0]
	child, exists := n.children[firstByte]
	if !exists {
		child = &naiveTrie{}
		n.children[firstByte] = child
	}
	child.insert(key[1:], value)
}

func (n *naiveTrie) GetLongestPrefix(key string) int {
	return n.getLongestPrefix([]byte(key), 0)
}

func (n *naiveTrie) getLongestPrefix(key []byte, offset int) int {
	if offset >= len(key) || n.children == nil {
		return offset
	}

	if child, ok := n.children[key[offset]]; ok {
		return child.getLongestPrefix(key, offset+1)
	}
	return offset
}

func (n *naiveTrie) Traverse(c byte) *naiveTrie {
	if n.children == nil {
		return nil
	}
	return n.children[c]
}

type xcdaArrayView struct {
	xcdaArray []uint32
}

func (v *xcdaArrayView) getNode(index uint32) (uint32, error) {
	if int(index) >= len(v.xcdaArray) {
		return 0, fmt.Errorf("index %d out of array bounds (len=%d)", index, len(v.xcdaArray))
	}
	return v.xcdaArray[index], nil
}

func (v *xcdaArrayView) GetBase(index uint32) (uint32, error) {
	packedNode, err := v.getNode(index)
	if err != nil {
		return 0, err
	}
	shift := (packedNode & (1 << 9)) >> 6
	return (packedNode >> 10) << shift, nil
}

func (v *xcdaArrayView) GetLCheck(index uint32) (uint32, error) {
	packedNode, err := v.getNode(index)
	if err != nil {
		return 0, err
	}
	return packedNode & ((1 << 31) | 0xff), nil
}

func (v *xcdaArrayView) GetLeaf(index uint32) (bool, error) {
	packedNode, err := v.getNode(index)
	if err != nil {
		return false, err
	}
	return ((packedNode >> 8) & 1) == 1, nil
}

func (v *xcdaArrayView) GetValue(index uint32) (uint32, error) {
	packedNode, err := v.getNode(index)
	if err != nil {
		return 0, err
	}
	return packedNode & ((1 << 31) - 1), nil
}

type bestTokenization struct {
	tokenId     int32
	inputOffset int
	scoreSum    float64
}

// Unigram Tokenizer
type Unigram struct {
	xcdaView                xcdaArrayView
	vocab                   *Vocabulary
	prefixReplacements      []uint8
	minScore                float32
	maxScore                float32
	unknownTokenScore       float32
	tokenMatcher            naiveTrie
	userDefinedTokenMatcher naiveTrie
	specialEosId            int32
	specialUnkId            int32
	specialPadId            int32
}

func NewUnigram(vocab *Vocabulary, precompiledCharsMap []uint8) (*Unigram, error) {
	if len(precompiledCharsMap) < 4 {
		return nil, errors.New("unigram: invalid precompiled chars map (too short)")
	}

	t5 := &Unigram{
		vocab:        vocab,
		minScore:     math.MaxFloat32,
		maxScore:     -math.MaxFloat32,
		specialPadId: 0,
		specialUnkId: 2,
		specialEosId: 1,
	}

	if err := t5.parsePrecompiledCharsMap(precompiledCharsMap); err != nil {
		return nil, err
	}

	t5.buildTokenMatchers()

	t5.unknownTokenScore = t5.minScore - unigramUnknownTokenScorePenalty
	return t5, nil
}

func (u *Unigram) parsePrecompiledCharsMap(precompiledCharsMap []uint8) error {
	tmp := unsafe.Slice((*uint32)(unsafe.Pointer(&precompiledCharsMap[0])), len(precompiledCharsMap)/xcdaArrayNodeSize)

	xcdaBlobSize := int(tmp[0])
	offset := 4

	if xcdaBlobSize+offset > len(precompiledCharsMap) {
		return errors.New("index out of array bounds in precompiled charsmap")
	}

	u.xcdaView.xcdaArray = unsafe.Slice(
		(*uint32)(unsafe.Pointer(&precompiledCharsMap[offset])),
		xcdaBlobSize/xcdaArrayNodeSize,
	)
	offset += xcdaBlobSize

	u.prefixReplacements = precompiledCharsMap[offset:]
	return nil
}

func (u *Unigram) buildTokenMatchers() {
	for id, tokenType := range u.vocab.Types {
		if tokenType == TOKEN_TYPE_NORMAL {
			score := u.vocab.Scores[id]
			if score < u.minScore {
				u.minScore = score
			}
			if score > u.maxScore {
				u.maxScore = score
			}
		}

		if tokenType == TOKEN_TYPE_NORMAL || tokenType == TOKEN_TYPE_USER_DEFINED || tokenType == TOKEN_TYPE_UNUSED {
			u.tokenMatcher.Insert(u.vocab.Values[id], int32(id))
		}

		if tokenType == TOKEN_TYPE_USER_DEFINED {
			u.userDefinedTokenMatcher.Insert(u.vocab.Values[id], int32(id))
		}
	}
}

func (u *Unigram) normalizePrefix(input string) (string, int, error) {
	if input == "" {
		return "", 0, nil
	}

	if prefixLen := u.userDefinedTokenMatcher.GetLongestPrefix(input); prefixLen > 0 {
		return input[:prefixLen], prefixLen, nil
	}

	longestPrefixLength, replacement, err := u.matchPrefixXCDA(input)
	if err != nil {
		return "", 0, err
	}

	if longestPrefixLength > 0 {
		return replacement, longestPrefixLength, nil
	}

	return u.getFallbackChar(input)
}

func (u *Unigram) matchPrefixXCDA(input string) (int, string, error) {
	if len(u.xcdaView.xcdaArray) == 0 {
		return 0, "", nil
	}

	nodeIndex, err := u.xcdaView.GetBase(0)
	if err != nil {
		return 0, "", err
	}

	var longestPrefixLength int
	var longestPrefixOffset uint32

	for prefixOffset := 0; prefixOffset < len(input); prefixOffset++ {
		c := uint32(input[prefixOffset])
		if c == 0 {
			break
		}

		nodeIndex ^= c

		lcheck, err := u.xcdaView.GetLCheck(nodeIndex)
		if err != nil {
			return 0, "", err
		}
		if lcheck != c {
			break
		}

		isLeaf, err := u.xcdaView.GetLeaf(nodeIndex)
		if err != nil {
			return 0, "", err
		}

		base, err := u.xcdaView.GetBase(nodeIndex)
		if err != nil {
			return 0, "", err
		}
		nodeIndex ^= base

		if isLeaf {
			longestPrefixLength = prefixOffset + 1
			longestPrefixOffset, err = u.xcdaView.GetValue(nodeIndex)
			if err != nil {
				return 0, "", err
			}
		}
	}

	if longestPrefixLength > 0 {
		replacement, err := u.getReplacementString(longestPrefixOffset)
		if err != nil {
			return 0, "", err
		}
		return longestPrefixLength, replacement, nil
	}

	return 0, "", nil
}

func (u *Unigram) getReplacementString(offset uint32) (string, error) {
	if int(offset) >= len(u.prefixReplacements) {
		return "", errors.New("index out of array bounds in precompiled charsmap")
	}

	prefixReplacement := u.prefixReplacements[offset:]
	index := bytes.IndexByte(prefixReplacement, 0)
	if index < 0 {
		return "", errors.New("unexpected string bound index in precompiled charsmap")
	}

	prefixReplacement = prefixReplacement[:index]
	return unsafe.String(unsafe.SliceData(prefixReplacement), len(prefixReplacement)), nil
}

func (u *Unigram) getFallbackChar(input string) (string, int, error) {
	if len(input) > 0 {
		r, size := utf8.DecodeRuneInString(input)
		if r != utf8.RuneError {
			return string(r), size, nil
		}
	}
	return "\xEF\xBF\xBD", 1, nil
}

func (u *Unigram) normalize(input string) (string, error) {
	var normalized strings.Builder
	normalized.Grow(len(input) + 10)

	shallPrependSpace := !u.vocab.TreatWhitespaceAsSuffix && u.vocab.AddSpacePrefix
	shallAppendSpace := u.vocab.TreatWhitespaceAsSuffix && u.vocab.AddSpacePrefix
	shallMergeSpaces := u.vocab.RemoveExtraWhitespaces

	var isSpacePrepended bool
	var processingNonWs bool

	for len(input) > 0 {
		normRes, consumedInput, err := u.normalizePrefix(input)
		if err != nil {
			return "", err
		}

		for i := 0; i < len(normRes); i++ {
			c := normRes[i]
			if c != ' ' {
				if !processingNonWs {
					processingNonWs = true
					if (shallPrependSpace && !isSpacePrepended) || shallMergeSpaces {
						normalized.WriteString(unigramEscapedSpace)
						isSpacePrepended = true
					}
				}
				normalized.WriteByte(c)
			} else {
				if processingNonWs {
					processingNonWs = false
				}
				if !shallMergeSpaces {
					normalized.WriteString(unigramEscapedSpace)
				}
			}
		}
		input = input[consumedInput:]
	}

	if shallAppendSpace {
		normalized.WriteString(unigramEscapedSpace)
	}

	return normalized.String(), nil
}

func utf8CodeUnitLen(c byte) int {
	return []int{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 3, 4}[c>>4]
}

func (u *Unigram) Encode(s string, addSpecial bool) ([]int32, error) {
	var output []int32

	if addSpecial && u.vocab.AddBOS {
		output = append(output, u.vocab.BOS...)
	}

	normalized, err := u.normalize(s)
	if err != nil {
		return nil, err
	}

	if len(normalized) == 0 {
		return output, nil
	}

	tokenizationResults := make([]*bestTokenization, len(normalized)+1)
	for i := range tokenizationResults {
		tokenizationResults[i] = &bestTokenization{
			tokenId:  u.specialUnkId,
			scoreSum: -math.MaxFloat64,
		}
	}
	tokenizationResults[0].scoreSum = 0

	if err := u.findBestTokenization(normalized, tokenizationResults); err != nil {
		return nil, err
	}

	output = u.backtrackTokenization(normalized, tokenizationResults, output)

	return output, nil
}

func (u *Unigram) findBestTokenization(normalized string, tokenizationResults []*bestTokenization) error {
	for inputOffset := 0; inputOffset < len(normalized); {
		nUtf8CodeUnits := min(utf8CodeUnitLen(normalized[inputOffset]), len(normalized)-inputOffset)
		currentBest := tokenizationResults[inputOffset]

		singleCodepointTokenFound := u.matchTokens(
			normalized,
			inputOffset,
			nUtf8CodeUnits,
			currentBest,
			tokenizationResults,
		)

		if !singleCodepointTokenFound {
			u.handleUnknownToken(inputOffset, nUtf8CodeUnits, currentBest, tokenizationResults)
		}

		inputOffset += nUtf8CodeUnits
	}
	return nil
}

func (u *Unigram) matchTokens(
	normalized string,
	inputOffset int,
	nUtf8CodeUnits int,
	currentBest *bestTokenization,
	tokenizationResults []*bestTokenization,
) bool {
	node := u.tokenMatcher.Traverse(normalized[inputOffset])
	singleCodepointTokenFound := false

	for prefixOffset := inputOffset + 1; prefixOffset <= len(normalized) && node != nil; prefixOffset++ {
		if node.hasValue {
			tokenId := node.value

			if prefixOffset-inputOffset == nUtf8CodeUnits {
				singleCodepointTokenFound = true
			}

			challengerScore := currentBest.scoreSum
			if u.vocab.Types[tokenId] != TOKEN_TYPE_USER_DEFINED {
				challengerScore += float64(u.vocab.Scores[tokenId])
			}

			currentChamp := tokenizationResults[prefixOffset]
			if challengerScore > currentChamp.scoreSum {
				currentChamp.tokenId = tokenId
				currentChamp.inputOffset = inputOffset
				currentChamp.scoreSum = challengerScore
			}
		}

		if prefixOffset >= len(normalized) {
			break
		}
		node = node.Traverse(normalized[prefixOffset])
	}

	return singleCodepointTokenFound
}

func (u *Unigram) handleUnknownToken(
	inputOffset int,
	nUtf8CodeUnits int,
	currentBest *bestTokenization,
	tokenizationResults []*bestTokenization,
) {
	challengerScore := currentBest.scoreSum + float64(u.unknownTokenScore)
	prefixOffset := inputOffset + nUtf8CodeUnits
	currentChamp := tokenizationResults[prefixOffset]

	if challengerScore > currentChamp.scoreSum {
		currentChamp.scoreSum = challengerScore
		currentChamp.inputOffset = inputOffset
		currentChamp.tokenId = u.specialUnkId
	}
}

func (u *Unigram) backtrackTokenization(
	normalized string,
	tokenizationResults []*bestTokenization,
	output []int32,
) []int32 {
	isPrevUnknown := false

	for tokenization := tokenizationResults[len(normalized)]; ; tokenization = tokenizationResults[tokenization.inputOffset] {
		isUnknown := tokenization.tokenId == u.specialUnkId
		if !(isUnknown && isPrevUnknown) {
			output = append(output, tokenization.tokenId)
		}

		if tokenization.inputOffset == 0 {
			break
		}
		isPrevUnknown = isUnknown
	}

	slices.Reverse(output)
	return output
}

func (u *Unigram) Decode(ids []int32) (string, error) {
	var sb strings.Builder
	sb.Grow(len(ids) * 4)

	for i, id := range ids {
		if id < 0 || int(id) >= len(u.vocab.Values) {
			return "", fmt.Errorf("invalid token id: %d", id)
		}

		piece := u.vocab.Values[id]

		if i > 0 && u.needsSeparator(piece) {
			sb.WriteByte(' ')
		}

		sb.WriteString(wordPieceReplacer.Replace(strings.TrimPrefix(piece, ggmlPrefix)))
	}

	return sb.String(), nil
}

func (u *Unigram) needsSeparator(piece string) bool {
	return strings.HasPrefix(piece, ggmlPrefix) ||
		(strings.HasPrefix(piece, "[") && strings.HasSuffix(piece, "]"))
}

func (u *Unigram) Is(id int32, special Special) bool {
	return u.vocab.Is(id, special)
}

func (u *Unigram) Vocabulary() *Vocabulary {
	return u.vocab
}
