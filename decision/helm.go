package decision

import (
	"bytes"
	"encoding/json"
	"fmt"
	"strconv"
	"strings"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/internal/orderedmap"
	"github.com/ollama/ollama/llm"
	"github.com/ollama/ollama/model/renderers"
)

// Saina Helm scores one question per prompt from the output-layer logits of
// its answer codes. Its prompt is the one it was trained on: a fixed
// instruction, then the context, question and coded options as JSON, following
// saina.loader.encode_head_prompt (Apache-2.0). The instruction is part of the
// user message, so the model needs no system prompt.
const helmInstruction = "Answer the multiple-choice question using the supplied context. " +
	"Choose exactly one of the provided options. Reply with only its code, " +
	"with no leading whitespace, explanation, or punctuation.\n"

// helmCodes label the options in order: A–Z, then the two-letter codes that
// are single tokens in Helm's tokenizer. The head rows are stored under these
// tokens, so the list is part of the model's contract.
var helmCodes = strings.Fields(`
A B C D E F G H I J K L M N O P Q R S T U V W X Y Z
AA AB AC AD AE AF AG AH AI AJ AK AL AM AN AO AP AQ AR AS AT AU AV AW AX AY AZ
BA BB BC BD BE BF BG BH BI BJ BK BL BM BN BO BP BR BS BT BU BV BW BX BY
CA CB CC CD CE CF CG CH CI CK CL CM CN CO CP CR CS CT CU CV CW CX CY
DA DB DC DD DE DF DG DH DI DJ DK DL DM DN DO DP DR DS DT DU DV DW DX DY
EA EB EC ED EE EF EG EH EI EK EL EM EN EO EP EQ ER ES ET EU EV EW EX EZ
FA FB FC FD FE FF FG FH FI FK FL FM FN FO FP FR FS FT FU FW FX FY
GA GB GC GD GE GF GG GH GI GL GM GN GO GP GR GS GT GU GV GW GX GY
HA HB HC HD HE HF HG HH HI HK HL HM HN HO HP HQ HR HS HT HU HV HW HX HY HZ
IA IB IC ID IE IF IG IH II IJ IK IL IM IN IO IP IQ IR IS IT IU IV IW IX IZ
JA JB JC JD JE JI JJ JK JM JO JP JR JS JT JU JV
KA KB KC KD KE KF KG KH KI KK KL KM KN KO KP KR KS KT KU KV KW KY
LA LB LC LD LE LF LG LI LK LL LM LN LO LP LR LS LT LU LV LY
MA MB MC MD ME MF MG MH MI MJ MK ML MM MN MO MP MQ MR MS MT MU MV MW MX MY MZ
NA NB NC ND NE NF NG NH NI NJ NK NL NM NN NO NP NQ NR NS NT NU NV NW NX NY NZ
OA OB OC OD OE OF OG OH OI OK OL OM ON OO OP OR OS OT OU OV OW OX
PA PB PC PD PE PF PG PH PI PJ PK PL PM PN PO PP PR PS PT PU PV PW PX PY
QA QB QC QE QG QH QL QM QN QP QQ QR QS QT QU
RA RB RC RD RE RF RG RH RI RJ RK RL RM RN RO RP RR RS RT RU RV RW RX RY
SA SB SC SD SE SF SG SH SI SJ SK SL SM SN SO SP SQ SR SS ST SU SV SW SX SY SZ
TA TB TC TD TE TF TG TH TI TK TL TM TN TO TP TR TS TT TU TV TW TX TY TZ
UA UB UC UD UE UF UG UH UI UK UL UM UN UP UR US UT UU UV UX UY UZ
VA VB VC VD VE VF VG VI VK VL VM VN VO VP VR VS VT VV
WA WB WC WD WE WF WG WH WI WK WL WM WN WO WP WR WS WT WW WX
XA XB XC XD XE XF XH XI XL XM XP XR XS XT XX XY
YA YC YE YG YL YM YN YO YP YS YT YW YY YZ
ZA ZE ZF ZH ZI ZN ZO ZR ZW ZX ZY ZZ`)

// maxHelmChoices matches the option limit of Helm's own API.
const maxHelmChoices = 255

func encodeHelm(req Request, c *Compiled) error {
	state, err := helmContent(req.State)
	if err != nil || strings.TrimSpace(state) == "" {
		return fmt.Errorf("state must be a nonempty string, object, or array")
	}
	for name, q := range req.Questions.All() {
		f, texts, err := helmField(name, q)
		if err != nil {
			return fmt.Errorf("question %q: %w", name, err)
		}
		type option struct {
			Code string `json:"code"`
			Text string `json:"text"`
		}
		options := make([]option, len(texts))
		for i, text := range texts {
			options[i] = option{f.Choices[i].Code, text}
		}
		prompt, err := promptJSON(struct {
			Context  string   `json:"context"`
			Question string   `json:"question"`
			Options  []option `json:"options"`
		}{state, f.Description, options})
		if err != nil {
			return err
		}
		c.messages = append(c.messages, []api.Message{{Role: "user", Content: helmInstruction + prompt}})
		row := llm.ScoreRow{Question: &llm.ScoreQuestion{Type: f.typ, Instructions: f.Description, Options: texts}}
		for _, choice := range f.Choices {
			row.Candidates = append(row.Candidates, choice.Code)
		}
		c.Request.Rows = append(c.Request.Rows, row)
		c.fields = append(c.fields, f)
	}
	return nil
}

// helmField compiles a question into coded choices and the option texts Helm
// scores: "Yes: …"/"No: …" for noul, "key" or "key: description" for choice,
// and "index: description" for score.
func helmField(name string, q Question) (compiledField, []string, error) {
	f := compiledField{Field: Field{Name: name}, typ: q.Type}
	if strings.TrimSpace(name) == "" {
		return f, nil, fmt.Errorf("field name must not be empty")
	}
	var err error
	f.Description, err = helmContent(q.Instructions)
	if err != nil || strings.TrimSpace(f.Description) == "" {
		return f, nil, fmt.Errorf("instructions must be a nonempty string, object, or array")
	}
	var texts []string
	add := func(value any, description, text string) {
		f.Choices = append(f.Choices, Choice{Code: helmCodes[len(f.Choices)], Value: value, Description: description})
		texts = append(texts, text)
	}
	limit := maxHelmChoices
	switch q.Type {
	case "noul":
		yes, no := "The answer is yes", "The answer is no"
		if raw := bytes.TrimSpace(q.Criteria); len(raw) > 0 && string(raw) != "null" {
			var criteria map[string]json.RawMessage
			if err := json.Unmarshal(raw, &criteria); err != nil {
				return f, nil, fmt.Errorf("noul criteria must be an object of true/false descriptions")
			}
			for key, raw := range criteria {
				description, ok, err := helmValue(raw)
				if err != nil || !ok || (key != "true" && key != "false") {
					return f, nil, fmt.Errorf("noul criteria must contain only true/false descriptions")
				}
				if key == "true" {
					yes = description
				} else {
					no = description
				}
			}
		}
		// Helm reads yes first; Answer finds the true choice by value.
		add(true, yes, "Yes: "+yes)
		add(false, no, "No: "+no)
	case "choice":
		criteria := orderedmap.New[string, json.RawMessage]()
		if err := json.Unmarshal(q.Criteria, criteria); err != nil {
			return f, nil, fmt.Errorf("choice criteria must map option keys to descriptions or null")
		}
		for key, raw := range criteria.All() {
			if strings.TrimSpace(key) == "" {
				return f, nil, fmt.Errorf("choice keys must not be empty")
			}
			description, ok, err := helmValue(raw)
			if err != nil {
				return f, nil, fmt.Errorf("choice descriptions must be strings, objects, arrays, or null")
			}
			if !ok {
				add(key, key, key)
			} else {
				add(key, description, key+": "+description)
			}
		}
	case "score":
		limit = 10
		var criteria []*string
		if err := json.Unmarshal(q.Criteria, &criteria); err != nil {
			return f, nil, fmt.Errorf("score criteria must be an array of descriptions")
		}
		for i, description := range criteria {
			if description == nil {
				return f, nil, fmt.Errorf("score descriptions must be strings")
			}
			add(strconv.Itoa(i), *description, strconv.Itoa(i)+": "+*description)
		}
	default:
		return f, nil, fmt.Errorf("type must be choice, noul, or score")
	}
	if len(f.Choices) < 2 || len(f.Choices) > limit {
		return f, nil, fmt.Errorf("criteria must contain 2–%d candidates", limit)
	}
	return f, texts, nil
}

// helmContent renders request content the way Helm's server does: strings as
// they are, objects and arrays as json.dumps(ensure_ascii=False).
func helmContent(raw json.RawMessage) (string, error) {
	text, ok, err := helmValue(raw)
	if err == nil && !ok {
		return "", fmt.Errorf("must be a string, object, or array")
	}
	return text, err
}

// helmValue renders an optional description; ok is false for null or absent.
func helmValue(raw json.RawMessage) (string, bool, error) {
	raw = bytes.TrimSpace(raw)
	if len(raw) == 0 || string(raw) == "null" {
		return "", false, nil
	}
	text, err := content(raw)
	if err != nil {
		return "", false, err
	}
	if raw[0] != '"' {
		text = string(renderers.AddJSONSpaces([]byte(text)))
	}
	return text, true, nil
}
