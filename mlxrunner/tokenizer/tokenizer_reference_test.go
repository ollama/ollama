package tokenizer_test

import (
	"bytes"
	"encoding/json"
	"os"
	"os/exec"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"github.com/ollama/ollama/mlxrunner/tokenizer"
)

type tokenizerReferenceCase struct {
	name  string
	input string
	want  []int32
}

// Default runs use installed MLX models and skip models whose tokenizer data is missing.
// FETCH_TOKENIZERS=1 downloads missing tokenizer blobs, without model weights.
// VERIFY_TOKENIZERS=1 also fetches missing data and selects Hugging Face tokenizers
// in python3 instead of Go, checking the same inputs and expectations.
func TestTokenizerReference(t *testing.T) {
	verify := os.Getenv("VERIFY_TOKENIZERS") != ""
	if verify {
		if output, err := exec.CommandContext(t.Context(), "python3", "-c", "import tokenizers").CombinedOutput(); err != nil {
			t.Fatalf("VERIFY_TOKENIZERS requires Hugging Face tokenizers in python3: %v\n%s", err, output)
		}
	}
	cases := []struct {
		name, input string
		want        map[string][]int32
	}{
		{name: "empty", input: "", want: map[string][]int32{
			"qwen":     {},
			"nemotron": {},
			"nimble":   {},
			"north":    {},
			"laguna":   {},
			"gemma":    {},
			"muse":     {},
		}},
		{name: "word", input: "hello", want: map[string][]int32{
			"qwen":     {14556},
			"nemotron": {29706},
			"nimble":   {14556},
			"north":    {28373},
			"laguna":   {11339},
			"gemma":    {23391},
			"muse":     {25681},
		}},
		{name: "leading space", input: " hello", want: map[string][]int32{
			"qwen":     {23066},
			"nemotron": {52528},
			"nimble":   {23066},
			"north":    {39821},
			"laguna":   {21433},
			"gemma":    {29104},
			"muse":     {47244},
		}},
		{name: "repeated spaces", input: "a  b", want: map[string][]int32{
			"qwen":     {64, 220, 292},
			"nemotron": {1097, 1032, 1289},
			"nimble":   {64, 220, 292},
			"north":    {69, 225, 290},
			"laguna":   {134, 290, 360},
			"gemma":    {236746, 138, 236763},
			"muse":     {77, 220, 297},
		}},
		{name: "space before number", input: "  1", want: map[string][]int32{
			"qwen":     {220, 220, 16},
			"nemotron": {1032, 1032, 1049},
			"nimble":   {220, 220, 16},
			"north":    {261, 21},
			"laguna":   {290, 290, 86},
			"gemma":    {138, 236770},
			"muse":     {220, 220, 29},
		}},
		{name: "spaces before punctuation", input: "   !", want: map[string][]int32{
			"qwen":     {256, 729},
			"nemotron": {1256, 2662},
			"nimble":   {256, 729},
			"north":    {261, 1350},
			"laguna":   {328, 1319},
			"gemma":    {139, 236888},
			"muse":     {256, 1239},
		}},
		{name: "spaces before combining mark", input: "  \u0301", want: map[string][]int32{
			"qwen":     {220, 220, 52033},
			"nemotron": {1032, 1032, 1204, 1129},
			"nimble":   {220, 220, 52033},
			"north":    {225, 225, 34145},
			"laguna":   {290, 290, 41879},
			"gemma":    {138, 238288},
			"muse":     {220, 184604},
		}},
		{name: "tab before number", input: " \t1", want: map[string][]int32{
			"qwen":     {220, 197, 16},
			"nemotron": {1032, 1009, 1049},
			"nimble":   {220, 197, 16},
			"north":    {10680, 21},
			"laguna":   {290, 267, 86},
			"gemma":    {236743, 255968, 236770},
			"muse":     {220, 197, 29},
		}},
		{name: "whitespace only", input: " \t\n", want: map[string][]int32{
			"qwen":     {16957},
			"nemotron": {29356, 1010},
			"nimble":   {16957},
			"north":    {97799},
			"laguna":   {14875, 268},
			"gemma":    {236743, 255968, 107},
			"muse":     {124731},
		}},
		{name: "trailing spaces", input: "end   ", want: map[string][]int32{
			"qwen":     {400, 262},
			"nemotron": {1474, 1293},
			"nimble":   {400, 262},
			"north":    {432, 298},
			"laguna":   {506, 341},
			"gemma":    {643, 139},
			"muse":     {431, 277},
		}},
		{name: "double newline", input: "a\n\nb", want: map[string][]int32{
			"qwen":     {64, 271, 65},
			"nemotron": {1097, 1267, 1098},
			"nimble":   {64, 271, 65},
			"north":    {69, 300, 70},
			"laguna":   {134, 350, 135},
			"gemma":    {236746, 108, 236763},
			"muse":     {77, 368, 78},
		}},
		{name: "trailing double newline", input: "a\n\n", want: map[string][]int32{
			"qwen":     {64, 271},
			"nemotron": {1097, 1267},
			"nimble":   {64, 271},
			"north":    {69, 300},
			"laguna":   {134, 350},
			"gemma":    {236746, 108},
			"muse":     {77, 368},
		}},
		{name: "newline before number", input: "a\n\n1", want: map[string][]int32{
			"qwen":     {64, 271, 16},
			"nemotron": {1097, 1267, 1049},
			"nimble":   {64, 271, 16},
			"north":    {69, 300, 21},
			"laguna":   {134, 350, 86},
			"gemma":    {236746, 108, 236770},
			"muse":     {77, 368, 29},
		}},
		{name: "crlf", input: "a\r\n\r\nb", want: map[string][]int32{
			"qwen":     {64, 845, 65},
			"nemotron": {1097, 1013, 1010, 1013, 1010, 1098},
			"nimble":   {64, 845, 65},
			"north":    {69, 5026, 70},
			"laguna":   {134, 1336, 135},
			"gemma":    {236746, 251, 107, 251, 107, 236763},
			"muse":     {77, 57966, 78},
		}},
		{name: "punctuation newline", input: "!\n\nb", want: map[string][]int32{
			"qwen":     {0, 271, 65},
			"nemotron": {5338, 1098},
			"nimble":   {0, 271, 65},
			"north":    {3127, 70},
			"laguna":   {70, 350, 135},
			"gemma":    {236888, 108, 236763},
			"muse":     {28655, 78},
		}},
		{name: "spaces then vertical tab", input: "  \u000b", want: map[string][]int32{
			"qwen":     {256, 199},
			"nemotron": {1256, 1011},
			"nimble":   {256, 199},
			"north":    {261, 204},
			"laguna":   {328, 269},
			"gemma":    {138, 249},
			"muse":     {256, 199},
		}},
		{name: "spaces vertical tab word", input: "  \u000bb", want: map[string][]int32{
			"qwen":     {256, 199, 65},
			"nemotron": {1256, 1011, 1098},
			"nimble":   {256, 199, 65},
			"north":    {261, 204, 70},
			"laguna":   {328, 269, 135},
			"gemma":    {138, 249, 236763},
			"muse":     {256, 199, 78},
		}},
		{name: "form feed", input: "\f\f1", want: map[string][]int32{
			"qwen":     {200, 200, 16},
			"nemotron": {1012, 1012, 1049},
			"nimble":   {200, 200, 16},
			"north":    {205, 205, 21},
			"laguna":   {270, 270, 86},
			"gemma":    {250, 250, 236770},
			"muse":     {200, 200, 29},
		}},
		{name: "nonbreaking space", input: "a\u00a0\u00a0b", want: map[string][]int32{
			"qwen":     {64, 3966, 3966, 65},
			"nemotron": {1097, 3997, 3997, 1098},
			"nimble":   {64, 3966, 3966, 65},
			"north":    {69, 3637, 3637, 70},
			"laguna":   {134, 757, 52591},
			"gemma":    {236746, 432, 398, 432, 398, 236763},
			"muse":     {77, 394, 41585},
		}},
		{name: "em space", input: "a\u2003\u2003b", want: map[string][]int32{
			"qwen":     {64, 373, 225, 373, 225, 65},
			"nemotron": {1097, 1287, 1131, 1287, 1131, 1098},
			"nimble":   {64, 373, 225, 373, 225, 65},
			"north":    {69, 29787, 29787, 70},
			"laguna":   {134, 52788, 52788, 135},
			"gemma":    {236746, 464, 366, 369, 464, 366, 369, 236763},
			"muse":     {77, 22393, 22393, 78},
		}},
		{name: "unicode line separator", input: "a\u2028\u2028b", want: map[string][]int32{
			"qwen":     {64, 373, 101, 373, 101, 65},
			"nemotron": {1097, 1287, 1168, 1287, 1168, 1098},
			"nimble":   {64, 373, 101, 373, 101, 65},
			"north":    {69, 488, 106, 488, 106, 70},
			"laguna":   {134, 34036, 34036, 135},
			"gemma":    {236746, 464, 366, 406, 464, 366, 406, 236763},
			"muse":     {77, 178448, 178448, 78},
		}},
		{name: "unicode space newline", input: "\u3000\u3000\nword", want: map[string][]int32{
			"qwen":     {41993, 198, 1119},
			"nemotron": {114840, 114840, 1010, 3494},
			"nimble":   {41993, 198, 1119},
			"north":    {65827, 203, 1936},
			"laguna":   {34318, 268, 1318},
			"gemma":    {465, 366, 366, 465, 366, 366, 107, 3017},
			"muse":     {15884, 198, 1769},
		}},
		{name: "contractions", input: "we'll I'M they're", want: map[string][]int32{
			"qwen":     {868, 3172, 353, 26708, 781, 2224},
			"nemotron": {1808, 7534, 1362, 1039, 1077, 2127, 6185},
			"nimble":   {868, 3172, 353, 26708, 781, 2224},
			"north":    {217525, 212963, 7402},
			"laguna":   {2299, 3091, 397, 46809, 948, 2264},
			"gemma":    {977, 236789, 859, 564, 236789, 236792, 901, 236789, 500},
			"muse":     {900, 8961, 5384, 57, 28661},
		}},
		{name: "digit grouping", input: "1234567", want: map[string][]int32{
			"qwen":     {16, 17, 18, 19, 20, 21, 22},
			"nemotron": {1049, 1050, 1051, 1052, 1053, 1054, 1055},
			"nimble":   {16, 17, 18, 19, 20, 21, 22},
			"north":    {21, 13877, 22180},
			"laguna":   {86, 87, 88, 89, 90, 91, 92},
			"gemma":    {236770, 236778, 236800, 236812, 236810, 236825, 236832},
			"muse":     {7235, 19596, 35},
		}},
		{name: "code", input: "if x >  1:\n    return x\n", want: map[string][]int32{
			"qwen":     {331, 830, 835, 220, 220, 16, 25, 198, 262, 460, 830, 198},
			"nemotron": {1391, 2460, 3006, 1032, 1032, 1049, 1877, 1293, 1850, 2460, 1010},
			"nimble":   {331, 830, 835, 220, 220, 16, 25, 198, 262, 460, 830, 198},
			"north":    {359, 1150, 1518, 261, 21, 638, 298, 758, 1150, 203},
			"laguna":   {406, 854, 981, 290, 290, 86, 95, 268, 341, 658, 854, 268},
			"gemma":    {584, 1123, 1890, 138, 236770, 236787, 107, 140, 2060, 1123, 107},
			"muse":     {364, 831, 1299, 220, 220, 29, 600, 277, 677, 831, 198},
		}},
		{name: "json", input: "{\n  \"count\":  12\n}\n", want: map[string][]int32{
			"qwen":     {90, 198, 220, 328, 1767, 763, 220, 220, 16, 17, 198, 92, 198},
			"nemotron": {2030, 1032, 1429, 12296, 2811, 1032, 1032, 1049, 1050, 1010, 2002},
			"nimble":   {90, 198, 220, 328, 1767, 763, 220, 220, 16, 17, 198, 92, 198},
			"north":    {955, 225, 373, 8227, 1023, 261, 740, 203, 732},
			"laguna":   {160, 268, 290, 444, 3177, 1034, 290, 290, 86, 87, 268, 162, 268},
			"gemma":    {236782, 107, 138, 236775, 2861, 1083, 138, 236770, 236778, 107, 236783, 107},
			"muse":     {785, 220, 392, 7263, 1217, 220, 220, 738, 198, 714},
		}},
		{name: "unicode NFC", input: "cafe\u0301 \u00e9 \u4e2d \U0001f30d", want: map[string][]int32{
			"qwen":     {895, 56868, 3825, 220, 95789, 10838, 234, 235},
			"nemotron": {3173, 3070, 1204, 1129, 1782, 30549, 119685, 1140, 1141},
			"nimble":   {895, 56868, 3825, 220, 95789, 10838, 234, 235},
			"north":    {148414, 34145, 1897, 61196, 96019, 240},
			"laguna":   {88394, 41879, 10757, 32294, 88219, 305},
			"gemma":    {101727, 238288, 1559, 17346, 236743, 244906},
			"muse":     {179383, 7908, 1119, 14351, 85275, 235},
		}},
		{name: "decomposed acute", input: "e\u0301", want: map[string][]int32{
			"qwen":     {933},
			"nemotron": {1101, 1204, 1129},
			"nimble":   {933},
			"north":    {73, 34145},
			"laguna":   {138, 41879},
			"gemma":    {236744, 238288},
			"muse":     {81, 7908},
		}},
		{name: "multiple combining marks", input: "a\u0301\u0308", want: map[string][]int32{
			"qwen":     {1886, 136, 230},
			"nemotron": {1097, 1204, 1129, 129636},
			"nimble":   {1886, 136, 230},
			"north":    {69, 34145, 117207},
			"laguna":   {134, 41879, 206, 300},
			"gemma":    {236746, 238288, 241151},
			"muse":     {77, 7908, 50508},
		}},
		{name: "combining mark decomposition", input: "\u0344", want: map[string][]int32{
			"qwen":     {136, 230, 52033},
			"nemotron": {1205, 1132},
			"nimble":   {136, 230, 52033},
			"north":    {142, 231},
			"laguna":   {207, 296},
			"gemma":    {246735},
			"muse":     {148, 226},
		}},
		{name: "angstrom sign normalization", input: "\u212b", want: map[string][]int32{
			"qwen":     {169111},
			"nemotron": {29246, 1171},
			"nimble":   {169111},
			"north":    {16284, 109},
			"laguna":   {15553, 174},
			"gemma":    {464, 370, 409},
			"muse":     {11840, 117},
		}},
		{name: "Hindi combining marks", input: "\u0939\u093f\u0928\u094d\u0926\u0940", want: map[string][]int32{
			"qwen":     {215989, 150127, 177453},
			"nemotron": {3086, 75790, 2063},
			"nimble":   {90703, 181077, 28640, 148651, 42201},
			"north":    {2990, 140177, 1297},
			"laguna":   {50000, 25825, 171, 15992, 169, 28543},
			"gemma":    {16017, 70249},
			"muse":     {2471, 9028, 98999},
		}},
		{name: "Devanagari virama", input: "\u0915\u094d", want: map[string][]int32{
			"qwen":     {149567},
			"nemotron": {2622, 1891},
			"nimble":   {62516, 28640},
			"north":    {2149, 1218},
			"laguna":   {33892, 15306},
			"gemma":    {1445},
			"muse":     {1541, 823},
		}},
		{name: "joiner before combining mark", input: "a\u200d\u0301", want: map[string][]int32{
			"qwen":     {64, 373, 235, 52033},
			"nemotron": {1097, 29568, 1204, 1129},
			"nimble":   {64, 373, 235, 52033},
			"north":    {69, 36649, 34145},
			"laguna":   {134, 48560, 41879},
			"gemma":    {236746, 237243, 238288},
			"muse":     {77, 12817, 7908},
		}},
		{name: "accent before punctuation", input: "a\u0301!", want: map[string][]int32{
			"qwen":     {1886, 0},
			"nemotron": {1097, 1204, 1129, 1033},
			"nimble":   {1886, 0},
			"north":    {69, 34145, 5},
			"laguna":   {134, 41879, 70},
			"gemma":    {236746, 238288, 236888},
			"muse":     {77, 7908, 13},
		}},
		{name: "leading combining mark", input: "\u0301a", want: map[string][]int32{
			"qwen":     {52033, 64},
			"nemotron": {1204, 1129, 1097},
			"nimble":   {52033, 64},
			"north":    {34145, 69},
			"laguna":   {41879, 134},
			"gemma":    {229580},
			"muse":     {7908, 77},
		}},
		{name: "accent on composed letter", input: "\u00e9\u0301", want: map[string][]int32{
			"qwen":     {933, 52033},
			"nemotron": {1337, 1204, 1129},
			"nimble":   {933, 52033},
			"north":    {487, 34145},
			"laguna":   {2097, 41879},
			"gemma":    {236859, 238288},
			"muse":     {397, 7908},
		}},
		{name: "long s", input: "\u017f", want: map[string][]int32{
			"qwen":     {129, 123},
			"nemotron": {1197, 1191},
			"nimble":   {129, 123},
			"north":    {67879},
			"laguna":   {199, 193},
			"gemma":    {238071},
			"muse":     {129813},
		}},
		{name: "dotted capital i", input: "\u0130", want: map[string][]int32{
			"qwen":     {46186},
			"nemotron": {29522},
			"nimble":   {46186},
			"north":    {26402},
			"laguna":   {67570},
			"gemma":    {237848},
			"muse":     {12327},
		}},
		{name: "nonbreaking and zero width space", input: "\u00a0\u200b", want: map[string][]int32{
			"qwen":     {3966, 15231},
			"nemotron": {3997, 26580},
			"nimble":   {3966, 15231},
			"north":    {3637, 12624},
			"laguna":   {757, 22446},
			"gemma":    {432, 398, 237141},
			"muse":     {394, 4093},
		}},
		{name: "file separator control", input: "a\u001cb", want: map[string][]int32{
			"qwen":     {64, 216, 65},
			"nemotron": {1097, 1028, 1098},
			"nimble":   {64, 216, 65},
			"north":    {69, 221, 70},
			"laguna":   {134, 286, 135},
			"gemma":    {236746, 266, 236763},
			"muse":     {77, 216, 78},
		}},
		{name: "unit separator before number", input: "  \u001f1", want: map[string][]int32{
			"qwen":     {220, 220, 219, 16},
			"nemotron": {1032, 1032, 1031, 1049},
			"nimble":   {220, 220, 219, 16},
			"north":    {225, 225, 224, 21},
			"laguna":   {290, 290, 289, 86},
			"gemma":    {138, 253998, 236770},
			"muse":     {220, 220, 219, 29},
		}},
		{name: "embedded byte order mark", input: "a\ufeffb", want: map[string][]int32{
			"qwen":     {64, 3121, 65},
			"nemotron": {1097, 1239, 1187, 1191, 1098},
			"nimble":   {64, 3121, 65},
			"north":    {69, 158731, 70},
			"laguna":   {134, 9851, 135},
			"gemma":    {236746, 237922, 236763},
			"muse":     {77, 38420, 78},
		}},
		{name: "mixed case identifiers", input: "parseHTTPResponse XMLHttpRequest", want: map[string][]int32{
			"qwen":     {6199, 8957, 2497, 44310},
			"nemotron": {9245, 30499, 6566, 126379, 4967},
			"nimble":   {6199, 8957, 2497, 44310},
			"north":    {7386, 151305, 130902, 3095},
			"laguna":   {7335, 8689, 3165, 46797},
			"gemma":    {7240, 22005, 6126, 99210},
			"muse":     {5329, 19117, 3907, 115915, 2843},
		}},
		{name: "comment after call", input: "call();\n// comment\nnext", want: map[string][]int32{
			"qwen":     {6450, 2061, 198, 320, 3847, 198, 3480},
			"nemotron": {19881, 35674, 7739, 1010, 6651},
			"nimble":   {6450, 2061, 198, 320, 3847, 198, 3480},
			"north":    {10465, 40241, 4889, 203, 8831},
			"laguna":   {7361, 772, 268, 464, 4758, 268, 4954},
			"gemma":    {6639, 1086, 107, 715, 5739, 107, 4874},
			"muse":     {10551, 26440, 7146, 198, 4011},
		}},
		{name: "non-special added reasoning tokens", input: "<think>e\u0301</think>é", want: map[string][]int32{
			"qwen":     {248068, 933, 248069, 933},
			"nemotron": {12, 1101, 1204, 1129, 13, 1337},
			"nimble":   {248068, 933, 248069, 933},
			"north":    {36231, 748, 143978, 34145, 654, 37182, 34, 487},
			"laguna":   {18, 138, 41879, 19, 2097},
			"gemma":    {236820, 36345, 236813, 236744, 238288, 954, 36345, 236813, 236859},
			"muse":     {26911, 1007, 42, 81, 7908, 695, 66873, 42, 397},
		}},
		{name: "unassigned Unicode byte fallback", input: "\U0010ffff\u0378", want: map[string][]int32{
			"qwen":     {176, 237, 123, 123, 137, 116},
			"nemotron": {1244, 1143, 1191, 1191, 1205, 1184},
			"nimble":   {176, 237, 123, 123, 137, 116},
			"north":    {181, 242, 128, 128, 142, 121},
			"laguna":   {246, 307, 193, 193, 207, 186},
			"gemma":    {482, 381, 429, 429, 443, 422},
			"muse":     {187, 237, 136, 136, 148, 129},
		}},
	}
	models := []struct {
		family, model string
		cases         []tokenizerReferenceCase
	}{
		{family: "qwen", model: "qwen3.5:0.8b-mxfp8", cases: []tokenizerReferenceCase{
			{"special token boundary", "a  <|endoftext|>b", []int32{64, 256, 248044, 65}},
			{"repeated special token", "<|endoftext|>x<|endoftext|>", []int32{248044, 87, 248044}},
		}},
		{family: "qwen", model: "qwen3.8:27b-nvfp4", cases: []tokenizerReferenceCase{
			{"special token boundary", "a  <|endoftext|>b", []int32{64, 256, 248044, 65}},
			{"repeated special token", "<|endoftext|>x<|endoftext|>", []int32{248044, 87, 248044}},
		}},
		{family: "nemotron", model: "nemotron-3.5-lightning:30b-a3b-nvfp4", cases: []tokenizerReferenceCase{
			{"special token boundary", "a  <unk>b", []int32{1097, 1256, 0, 1098}},
			{"repeated special token", "<unk>x<unk>", []int32{0, 1120, 0}},
		}},
		{family: "nimble", model: "nimble:9b-nvfp4", cases: []tokenizerReferenceCase{
			{"special token boundary", "a  <|endoftext|>b", []int32{64, 256, 248044, 65}},
			{"repeated special token", "<|endoftext|>x<|endoftext|>", []int32{248044, 87, 248044}},
		}},
		{family: "north", model: "north-mini-code-1.0:mlx-nvfp4", cases: []tokenizerReferenceCase{
			{"special token boundary", "a  <PAD>b", []int32{69, 261, 0, 70}},
			{"repeated special token", "<PAD>x<PAD>", []int32{0, 92, 0}},
		}},
		{family: "laguna", model: "laguna-xs.2:nvfp4", cases: []tokenizerReferenceCase{
			{"special token boundary", "a  \u3008|UNK|\u3009b", []int32{134, 328, 0, 135}},
			{"repeated special token", "\u3008|UNK|\u3009x\u3008|UNK|\u3009", []int32{0, 157, 0}},
		}},
		{family: "gemma", model: "gemma4:e2b-nvfp4", cases: []tokenizerReferenceCase{
			{"special token boundary", "a  <pad>b", []int32{236746, 138, 0, 236763}},
			{"repeated special token", "<pad>x<pad>", []int32{0, 236781, 0}},
		}},
		{family: "muse", model: "muse-glimmer:30b-nvfp4", cases: []tokenizerReferenceCase{
			{"special token boundary", "a  <|begin_of_text|>b", []int32{77, 256, 200000, 78}},
			{"repeated special token", "<|begin_of_text|>x<|begin_of_text|>", []int32{200000, 100, 200000}},
		}},
	}
	for _, tt := range models {
		t.Run(tt.model, func(t *testing.T) {
			data := loadTokenizerReference(t, tt.model)
			var modelCases []tokenizerReferenceCase
			for _, tc := range cases {
				modelCases = append(modelCases, tokenizerReferenceCase{tc.name, tc.input, tc.want[tt.family]})
			}
			modelCases = append(modelCases, tt.cases...)
			var tok *tokenizer.Tokenizer
			var reference [][]int32
			if verify {
				reference = referenceTokenIDs(t, data, modelCases)
			} else {
				var err error
				tok, err = tokenizer.LoadFromBytes(data)
				if err != nil {
					t.Fatal(err)
				}
			}
			for i, tc := range modelCases {
				t.Run(tc.name, func(t *testing.T) {
					var got []int32
					if verify {
						got = reference[i]
					} else {
						got = tok.Encode(tc.input, false)
					}
					if tc.want == nil || !slices.Equal(got, tc.want) {
						t.Errorf("token IDs for %q:\n got: %v\nwant: %v", tc.input, got, tc.want)
						if verify {
							t.Logf("PYTHON OUTPUT (copy-paste as want):\n%#v", got)
						}
					}
				})
			}
		})
	}
}

func referenceTokenIDs(t *testing.T, data []byte, cases []tokenizerReferenceCase) [][]int32 {
	t.Helper()
	path := filepath.Join(t.TempDir(), "tokenizer.json")
	if err := os.WriteFile(path, data, 0o600); err != nil {
		t.Fatal(err)
	}
	inputs := make([]string, len(cases))
	for i, tc := range cases {
		inputs[i] = tc.input
	}
	payload, err := json.Marshal(inputs)
	if err != nil {
		t.Fatal(err)
	}
	// Match Encode(text, false): no padding, truncation, or automatic special tokens.
	const script = `
import json, sys, tokenizers
tokenizer = tokenizers.Tokenizer.from_file(sys.argv[1])
tokenizer.no_padding()
tokenizer.no_truncation()
inputs = json.load(sys.stdin)
print(json.dumps({"version": tokenizers.__version__, "ids": [
    tokenizer.encode(text, add_special_tokens=False).ids for text in inputs
]}))
`
	cmd := exec.CommandContext(t.Context(), "python3", "-c", script, path)
	cmd.Stdin = bytes.NewReader(payload)
	var stderr strings.Builder
	cmd.Stderr = &stderr
	output, err := cmd.Output()
	if err != nil {
		t.Fatalf("Python tokenizer reference failed: %v\n%s", err, stderr.String())
	}
	var result struct {
		Version string    `json:"version"`
		IDs     [][]int32 `json:"ids"`
	}
	if err := json.Unmarshal(output, &result); err != nil {
		t.Fatalf("decode Python reference: %v", err)
	}
	if len(result.IDs) != len(cases) {
		t.Fatalf("Python returned %d results, want %d", len(result.IDs), len(cases))
	}
	t.Logf("Python reference: tokenizers %s", result.Version)
	return result.IDs
}
