//go:build mlx

package grammar

// JSONGrammarEBNF is the EBNF grammar for JSON (character-level).
// Based on https://www.json.org/json-en.html
//
// This grammar operates at the character level. The engine validates
// tokens by matching them as sequences of these character-level terminals.
const JSONGrammarEBNF = `
json = value .

value = object | array | string | number | "true" | "false" | "null" .

object = "{" ws "}" | "{" members "}" .
members = member { "," member } .
member = ws string ws ":" element .

array = "[" ws "]" | "[" elements "]" .
elements = element { "," element } .
element = ws value ws .

string = "\"" { character } "\"" .
character = unescaped | escaped .
unescaped = " " | "!" | "#" … "[" | "]" … "~" .
escaped = "\\" ( "\"" | "\\" | "/" | "b" | "f" | "n" | "r" | "t" | unicode ) .
unicode = "u" hex hex hex hex .
hex = "0" … "9" | "A" … "F" | "a" … "f" .

number = [ "-" ] integer [ fraction ] [ exponent ] .
integer = "0" | onenine { digit } .
fraction = "." digit { digit } .
exponent = ( "e" | "E" ) [ "+" | "-" ] digit { digit } .
digit = "0" … "9" .
onenine = "1" … "9" .

ws = { " " | "\t" | "\n" | "\r" } .
`

// JSONObjectGrammarEBNF is like JSONGrammarEBNF but only allows objects at the top level.
const JSONObjectGrammarEBNF = `
json = object .

value = object | array | string | number | "true" | "false" | "null" .

object = "{" ws "}" | "{" members "}" .
members = member { "," member } .
member = ws string ws ":" element .

array = "[" ws "]" | "[" elements "]" .
elements = element { "," element } .
element = ws value ws .

string = "\"" { character } "\"" .
character = unescaped | escaped .
unescaped = " " | "!" | "#" … "[" | "]" … "~" .
escaped = "\\" ( "\"" | "\\" | "/" | "b" | "f" | "n" | "r" | "t" | unicode ) .
unicode = "u" hex hex hex hex .
hex = "0" … "9" | "A" … "F" | "a" … "f" .

number = [ "-" ] integer [ fraction ] [ exponent ] .
integer = "0" | onenine { digit } .
fraction = "." digit { digit } .
exponent = ( "e" | "E" ) [ "+" | "-" ] digit { digit } .
digit = "0" … "9" .
onenine = "1" … "9" .

ws = { " " | "\t" | "\n" | "\r" } .
`
