#pragma once

#include <cstdint>
#include <string>
#include <vector>

#define CODEPOINT_TYPE_UNIDENTIFIED 0
#define CODEPOINT_TYPE_DIGIT        1
#define CODEPOINT_TYPE_LETTER       2
#define CODEPOINT_TYPE_WHITESPACE   3
#define CODEPOINT_TYPE_ACCENT_MARK  4
#define CODEPOINT_TYPE_PUNCTUATION  5
#define CODEPOINT_TYPE_SYMBOL       6
#define CODEPOINT_TYPE_CONTROL      7

std::string unicode_cpt_to_utf8(uint32_t cp);
std::vector<uint32_t> unicode_cpts_from_utf8(const std::string & utf8);

std::vector<uint32_t> unicode_cpts_normalize_nfd(const std::vector<uint32_t> & cpts);

int unicode_cpt_type(uint32_t cp);
int unicode_cpt_type(const std::string & utf8);

std::string unicode_byte_to_utf8(uint8_t byte);
uint8_t unicode_utf8_to_byte(const std::string & utf8);

// simple tolower that only implements one-to-one mapping, not one-to-many
char32_t unicode_tolower(char32_t cp);
