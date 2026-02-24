package convert

import "testing"

func TestSanitizeNonFiniteJSON(t *testing.T) {
	tests := []struct {
		name string
		in   string
		want string
	}{
		{
			name: "infinity token",
			in:   `{"a":[0,Infinity,1]}`,
			want: `{"a":[0,0,1]}`,
		},
		{
			name: "negative infinity token",
			in:   `{"a":-Infinity}`,
			want: `{"a":0}`,
		},
		{
			name: "nan token",
			in:   `{"a":NaN}`,
			want: `{"a":0}`,
		},
		{
			name: "tokens inside strings untouched",
			in:   `{"a":"Infinity -Infinity NaN","b":Infinity}`,
			want: `{"a":"Infinity -Infinity NaN","b":0}`,
		},
		{
			name: "identifier-like token untouched",
			in:   `{"a":InfinityValue}`,
			want: `{"a":InfinityValue}`,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := string(sanitizeNonFiniteJSON([]byte(tt.in)))
			if got != tt.want {
				t.Fatalf("sanitizeNonFiniteJSON() = %q, want %q", got, tt.want)
			}
		})
	}
}
