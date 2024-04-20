#pragma once

#include <cstdint>
#include <map>
#include <utility>
#include <vector>

extern const std::vector<std::pair<uint32_t, uint32_t>> unicode_ranges_digit;
extern const std::vector<std::pair<uint32_t, uint32_t>> unicode_ranges_letter;
extern const std::vector<std::pair<uint32_t, uint32_t>> unicode_ranges_whitespace;
extern const std::vector<std::pair<uint32_t, uint32_t>> unicode_ranges_accent_mark;
extern const std::vector<std::pair<uint32_t, uint32_t>> unicode_ranges_punctuation;
extern const std::vector<std::pair<uint32_t, uint32_t>> unicode_ranges_symbol;
extern const std::vector<std::pair<uint32_t, uint32_t>> unicode_ranges_control;
extern const std::multimap<uint32_t, uint32_t> unicode_map_nfd;
extern const std::map<char32_t, char32_t> unicode_map_lowercase;
