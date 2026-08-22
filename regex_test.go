package main

import (
	"fmt"
	"regexp"
)

func main() {
	input := `/Users/ollama/Library/Mobile\ Documents/com\~apple\~CloudDocs/screenshots/CleanShot\ 2025-04-17\ at\ 21.26.40@2x.png`
	regexPattern := `(?:[a-zA-Z]:)?(?:\./|/|\\)[\S\\ ]+?\.(?i:jpg|jpeg|png|webp|wav)\b`
	re := regexp.MustCompile(regexPattern)
	
	fmt.Printf("Matches: %q\n", re.FindAllString(input, -1))
}
