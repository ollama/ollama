package api

import "testing"

func TestValidatePhoneNumber(t *testing.T) {
	tests := []struct {
		name  string
		phone string
		want  bool
	}{
		// Valid international numbers (E.164 format)
		{"Germany mobile", "+49171234567", true},
		{"Germany with spaces", "+49 171 234567", true},
		{"Germany with hyphens", "+49-171-234567", true},
		{"US number", "+14155551234", true},
		{"US formatted", "+1 (415) 555-1234", true},
		{"Israel mobile", "+972501234567", true},
		{"UK mobile", "+447700123456", true},
		{"Spain mobile", "+34612345678", true},
		{"France mobile", "+33612345678", true},
		{"China mobile", "+8613912345678", true},
		{"Short international", "+12345", true},
		{"Maximum length", "+123456789012345", true},
		
		// Invalid numbers
		{"No plus sign", "14155551234", false},
		{"Starts with zero", "+01234567890", false},
		{"Too long", "+1234567890123456", false},
		{"Empty string", "", false},
		{"Only plus", "+", false},
		{"Letters", "+1415ABC1234", false},
		{"Special chars", "+1-415-555-1234!", false},
		{"Spaces only", "   ", false},
	}
	
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := ValidatePhoneNumber(tt.phone)
			if got != tt.want {
				t.Errorf("ValidatePhoneNumber(%q) = %v, want %v", tt.phone, got, tt.want)
			}
		})
	}
}
