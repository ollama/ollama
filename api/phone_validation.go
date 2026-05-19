package api

import (
	"regexp"
	"strings"
)

// E.164 international phone number format:
// + followed by country code (1-3 digits) and subscriber number (up to 14 digits total)
// This regex validates the structure while being permissive enough for international formats
var phoneRegex = regexp.MustCompile(`^\+[1-9]\d{1,14}$`)

// ValidatePhoneNumber validates a phone number in E.164 international format.
// It accepts numbers like +49171234567 (Germany), +14155551234 (US), +972501234567 (Israel), etc.
// Returns true if the phone number is valid, false otherwise.
func ValidatePhoneNumber(phone string) bool {
	// Remove common formatting characters that users might include
	cleaned := strings.ReplaceAll(phone, " ", "")
	cleaned = strings.ReplaceAll(cleaned, "-", "")
	cleaned = strings.ReplaceAll(cleaned, "(", "")
	cleaned = strings.ReplaceAll(cleaned, ")", "")
	cleaned = strings.ReplaceAll(cleaned, ".", "")
	
	return phoneRegex.MatchString(cleaned)
}
