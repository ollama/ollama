package model

import (
	"encoding/json"
	"testing"
)

func TestThinkingDescriptor(t *testing.T) {
	for _, tt := range []struct {
		name, json string
		valid      bool
	}{
		{"toggle", `{"values":[false,true],"default":true}`, true},
		{"mixed", `{"values":[false,"medium","xhigh"],"default":"medium"}`, true},
		{"nonthinking", `{"values":[false],"default":false}`, true},
		{"unknown", `null`, false},
		{"empty", `{"values":[],"default":false}`, false},
		{"integer", `{"values":[75],"default":75}`, false},
		{"object", `{"values":[{}],"default":false}`, false},
		{"missing default", `{"values":[false,true]}`, false},
		{"unlisted default", `{"values":["high"],"default":"low"}`, false},
		{"duplicate", `{"values":[false,false],"default":false}`, false},
		{"empty name", `{"values":[""],"default":""}`, false},
	} {
		t.Run(tt.name, func(t *testing.T) {
			var thinking *Thinking
			if err := json.Unmarshal([]byte(tt.json), &thinking); err != nil {
				t.Fatal(err)
			}
			if got := thinking.Valid(); got != tt.valid {
				t.Fatalf("Valid() = %v, want %v", got, tt.valid)
			}
			if tt.valid {
				encoded, err := json.Marshal(*thinking)
				if err != nil {
					t.Fatal(err)
				}
				if string(encoded) != tt.json {
					t.Fatalf("JSON = %s, want %s", encoded, tt.json)
				}
				clone := thinking.Clone()
				clone.Values[0] = "changed"
				if thinking.Values[0] == "changed" {
					t.Fatal("clone shares values")
				}
			}
		})
	}
}

func TestThinkValueJSON(t *testing.T) {
	for _, tt := range []struct {
		name  string
		value *ThinkValue
		want  string
	}{
		{"false", &ThinkValue{Value: false}, `false`},
		{"true", &ThinkValue{Value: true}, `true`},
		{"level", &ThinkValue{Value: "xhigh"}, `"xhigh"`},
		{"unset", &ThinkValue{}, `null`},
		{"nil pointer", (*ThinkValue)(nil), `null`},
	} {
		t.Run(tt.name, func(t *testing.T) {
			got, err := json.Marshal(tt.value)
			if err != nil {
				t.Fatal(err)
			}
			if string(got) != tt.want {
				t.Fatalf("JSON = %s, want %s", got, tt.want)
			}
		})
	}
}

func TestThinkValueAccessors(t *testing.T) {
	for _, tt := range []struct {
		name                        string
		value                       *ThinkValue
		valid, isBool, isString, on bool
		level                       string
	}{
		{"nil", nil, true, false, false, false, ""},
		{"unset", &ThinkValue{}, true, false, false, false, ""},
		{"off", &ThinkValue{Value: false}, true, true, false, false, ""},
		{"on", &ThinkValue{Value: true}, true, true, false, true, "medium"},
		{"named", &ThinkValue{Value: "xhigh"}, true, false, true, true, "xhigh"},
		{"empty name", &ThinkValue{Value: ""}, true, false, true, true, ""},
		{"integer", &ThinkValue{Value: 75}, false, false, false, false, ""},
	} {
		t.Run(tt.name, func(t *testing.T) {
			if tt.value.IsValid() != tt.valid || tt.value.IsBool() != tt.isBool || tt.value.IsString() != tt.isString || tt.value.Bool() != tt.on || tt.value.String() != tt.level {
				t.Fatalf("unexpected accessors for %#v", tt.value)
			}
		})
	}
}

func TestThinkingSupports(t *testing.T) {
	thinking := &Thinking{Values: []any{false, true, "medium", "xhigh"}, Default: true}
	for _, tt := range []struct {
		name  string
		value any
		want  bool
	}{
		{"off", false, true},
		{"on", true, true},
		{"named medium", "medium", true},
		{"named xhigh", "xhigh", true},
		{"unknown", "future", false},
		{"string is not boolean", "false", false},
		{"case is exact", "XHIGH", false},
		{"whitespace is exact", " medium ", false},
		{"nil", nil, false},
		{"number", 75, false},
		{"array", []any{false}, false},
		{"object", map[string]any{"value": false}, false},
	} {
		t.Run(tt.name, func(t *testing.T) {
			if got := thinking.Supports(tt.value); got != tt.want {
				t.Fatalf("Supports(%#v) = %v, want %v", tt.value, got, tt.want)
			}
		})
	}
	var unknown *Thinking
	if unknown.Supports(false) {
		t.Fatal("unknown metadata must not advertise controls")
	}
}
