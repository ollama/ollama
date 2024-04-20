#include "unicode.h"
#include "unicode-data.h"

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <map>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

static std::string unicode_cpts_to_utf8(const std::vector<uint32_t> & cps) {
    std::string result;
    for (size_t i = 0; i < cps.size(); ++i) {
        result.append(unicode_cpt_to_utf8(cps[i]));
    }
    return result;
}

static uint32_t unicode_cpt_from_utf8(const std::string & utf8, size_t & offset) {
    assert(offset < utf8.size());
    if (!(utf8[offset + 0] & 0x80)) {
        auto result = utf8[offset + 0];
        offset += 1;
        return result;
    }
    if (!(utf8[offset + 0] & 0x40)) {
        throw std::invalid_argument("invalid character");
    }
    if (!(utf8[offset + 0] & 0x20)) {
        if (offset + 1 >= utf8.size() || ! ((utf8[offset + 1] & 0xc0) == 0x80)) {
            throw std::invalid_argument("invalid character");
        }
        auto result = ((utf8[offset + 0] & 0x1f) << 6) | (utf8[offset + 1] & 0x3f);
        offset += 2;
        return result;
    }
    if (!(utf8[offset + 0] & 0x10)) {
        if (offset + 2 >= utf8.size() || ! ((utf8[offset + 1] & 0xc0) == 0x80) || ! ((utf8[offset + 2] & 0xc0) == 0x80)) {
            throw std::invalid_argument("invalid character");
        }
        auto result = ((utf8[offset + 0] & 0x0f) << 12) | ((utf8[offset + 1] & 0x3f) << 6) | (utf8[offset + 2] & 0x3f);
        offset += 3;
        return result;
    }
    if (!(utf8[offset + 0] & 0x08)) {
        if (offset + 3 >= utf8.size() || ! ((utf8[offset + 1] & 0xc0) == 0x80) || ! ((utf8[offset + 2] & 0xc0) == 0x80) || !((utf8[offset + 3] & 0xc0) == 0x80)) {
            throw std::invalid_argument("invalid character");
        }
        auto result = ((utf8[offset + 0] & 0x07) << 18) | ((utf8[offset + 1] & 0x3f) << 12) | ((utf8[offset + 2] & 0x3f) << 6) | (utf8[offset + 3] & 0x3f);
        offset += 4;
        return result;
    }
    throw std::invalid_argument("invalid string");
}

static std::vector<uint16_t> unicode_cpt_to_utf16(uint32_t cp) {
    std::vector<uint16_t> result;
    if (/* 0x0000 <= cp && */ cp <= 0xffff) {
        result.emplace_back(cp);
    }
    else if (0x10000 <= cp && cp <= 0x10ffff) {
        result.emplace_back(0xd800 | ((cp - 0x10000) >> 10));
        result.emplace_back(0xdc00 | ((cp - 0x10000) & 0x03ff));
    }
    else {
        throw std::invalid_argument("invalid cpt");
    }
    return result;
}

//static std::vector<uint16_t> unicode_cpts_to_utf16(const std::vector<uint32_t> & cps) {
//    std::vector<uint16_t> result;
//    for (size_t i = 0; i < cps.size(); ++i) {
//        auto temp = unicode_cpt_to_utf16(cps[i]);
//        result.insert(result.end(), temp.begin(), temp.end());
//    }
//    return result;
//}

static uint32_t cpt_from_utf16(const std::vector<uint16_t> & utf16, size_t & offset) {
    assert(offset < utf16.size());
    if (((utf16[0] >> 10) << 10) != 0xd800) {
        auto result = utf16[offset + 0];
        offset += 1;
        return result;
    }

    if (offset + 1 >= utf16.size() || !((utf16[1] & 0xdc00) == 0xdc00)) {
        throw std::invalid_argument("invalid character");
    }

    auto result = 0x10000 + (((utf16[0] & 0x03ff) << 10) | (utf16[1] & 0x03ff));
    offset += 2;
    return result;
}

//static std::vector<uint32_t> unicode_cpts_from_utf16(const std::vector<uint16_t> & utf16) {
//    std::vector<uint32_t> result;
//    size_t offset = 0;
//    while (offset < utf16.size()) {
//        result.push_back(cpt_from_utf16(utf16, offset));
//    }
//    return result;
//}

static std::unordered_map<uint32_t, int> unicode_cpt_type_map() {
    std::unordered_map<uint32_t, int> cpt_types;
    for (auto p : unicode_ranges_digit) {
        for (auto i = p.first; i <= p.second; ++ i) {
            cpt_types[i] = CODEPOINT_TYPE_DIGIT;
        }
    }
    for (auto p : unicode_ranges_letter) {
        for (auto i = p.first; i <= p.second; ++ i) {
            cpt_types[i] = CODEPOINT_TYPE_LETTER;
        }
    }
    for (auto p : unicode_ranges_whitespace) {
        for (auto i = p.first; i <= p.second; ++ i) {
            cpt_types[i] = CODEPOINT_TYPE_WHITESPACE;
        }
    }
    for (auto p : unicode_ranges_accent_mark) {
        for (auto i = p.first; i <= p.second; ++ i) {
            cpt_types[i] = CODEPOINT_TYPE_ACCENT_MARK;
        }
    }
    for (auto p : unicode_ranges_punctuation) {
        for (auto i = p.first; i <= p.second; ++ i) {
            cpt_types[i] = CODEPOINT_TYPE_PUNCTUATION;
        }
    }
    for  (auto p : unicode_ranges_symbol) {
        for (auto i = p.first; i <= p.second; ++i) {
            cpt_types[i] = CODEPOINT_TYPE_SYMBOL;
        }
    }
    for (auto p : unicode_ranges_control) {
        for (auto i = p.first; i <= p.second; ++ i) {
            cpt_types[i] = CODEPOINT_TYPE_CONTROL;
        }
    }
    return cpt_types;
}

static std::unordered_map<uint8_t, std::string> unicode_byte_to_utf8_map() {
    std::unordered_map<uint8_t, std::string> map;
    for (int ch = u'!'; ch <= u'~'; ++ch) {
        assert(0 <= ch && ch < 256);
        map[ch] = unicode_cpt_to_utf8(ch);
    }
    for (int ch = u'¡'; ch <= u'¬'; ++ch) {
        assert(0 <= ch && ch < 256);
        map[ch] = unicode_cpt_to_utf8(ch);
    }
    for (int ch = u'®'; ch <= u'ÿ'; ++ch) {
        assert(0 <= ch && ch < 256);
        map[ch] = unicode_cpt_to_utf8(ch);
    }
    auto n = 0;
    for (int ch = 0; ch < 256; ++ch) {
        if (map.find(ch) == map.end()) {
            map[ch] = unicode_cpt_to_utf8(256 + n);
            ++n;
        }
    }
    return map;
}

static std::unordered_map<std::string, uint8_t> unicode_utf8_to_byte_map() {
    std::unordered_map<std::string, uint8_t> map;
    for (int ch = u'!'; ch <= u'~'; ++ch) {
        assert(0 <= ch && ch < 256);
        map[unicode_cpt_to_utf8(ch)] = ch;
    }
    for (int ch = u'¡'; ch <= u'¬'; ++ch) {
        assert(0 <= ch && ch < 256);
        map[unicode_cpt_to_utf8(ch)] = ch;
    }
    for (int ch = u'®'; ch <= u'ÿ'; ++ch) {
        assert(0 <= ch && ch < 256);
        map[unicode_cpt_to_utf8(ch)] = ch;
    }
    auto n = 0;
    for (int ch = 0; ch < 256; ++ch) {
        if (map.find(unicode_cpt_to_utf8(ch)) == map.end()) {
            map[unicode_cpt_to_utf8(256 + n)] = ch;
            ++n;
        }
    }
    return map;
}

//
// interface
//

std::string unicode_cpt_to_utf8(uint32_t cp) {
    std::string result;
    if (/* 0x00 <= cp && */ cp <= 0x7f) {
        result.push_back(cp);
    }
    else if (0x80 <= cp && cp <= 0x7ff) {
        result.push_back(0xc0 | ((cp >> 6) & 0x1f));
        result.push_back(0x80 | (cp & 0x3f));
    }
    else if (0x800 <= cp && cp <= 0xffff) {
        result.push_back(0xe0 | ((cp >> 12) & 0x0f));
        result.push_back(0x80 | ((cp >> 6) & 0x3f));
        result.push_back(0x80 | (cp & 0x3f));
    }
    else if (0x10000 <= cp && cp <= 0x10ffff) {
        result.push_back(0xf0 | ((cp >> 18) & 0x07));
        result.push_back(0x80 | ((cp >> 12) & 0x3f));
        result.push_back(0x80 | ((cp >> 6) & 0x3f));
        result.push_back(0x80 | (cp & 0x3f));
    }
    else {
        throw std::invalid_argument("invalid codepoint");
    }
    return result;
}

std::vector<uint32_t> unicode_cpts_normalize_nfd(const std::vector<uint32_t> & cpts) {
    std::vector<uint32_t> result;
    result.reserve(cpts.size());
    for (size_t i = 0; i < cpts.size(); ++i) {
        auto it = unicode_map_nfd.find(cpts[i]);
        if (it == unicode_map_nfd.end()) {
            result.push_back(cpts[i]);
        } else {
            result.push_back(it->second);
        }
    }
    return result;
}

std::vector<uint32_t> unicode_cpts_from_utf8(const std::string & utf8) {
    std::vector<uint32_t> result;
    size_t offset = 0;
    while (offset < utf8.size()) {
        result.push_back(unicode_cpt_from_utf8(utf8, offset));
    }
    return result;
}

int unicode_cpt_type(uint32_t cp) {
    static std::unordered_map<uint32_t, int> cpt_types = unicode_cpt_type_map();
    const auto it = cpt_types.find(cp);
    return it == cpt_types.end() ? CODEPOINT_TYPE_UNIDENTIFIED : it->second;
}

int unicode_cpt_type(const std::string & utf8) {
    if (utf8.length() == 0) {
        return CODEPOINT_TYPE_UNIDENTIFIED;
    }
    size_t offset = 0;
    return unicode_cpt_type(unicode_cpt_from_utf8(utf8, offset));
}

std::string unicode_byte_to_utf8(uint8_t byte) {
    static std::unordered_map<uint8_t, std::string> map = unicode_byte_to_utf8_map();
    return map.at(byte);
}

uint8_t unicode_utf8_to_byte(const std::string & utf8) {
    static std::unordered_map<std::string, uint8_t> map = unicode_utf8_to_byte_map();
    return map.at(utf8);
}

char32_t unicode_tolower(char32_t cp) {
    auto it = unicode_map_lowercase.find(cp);
    return it == unicode_map_lowercase.end() ? cp : it->second;
}
