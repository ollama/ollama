#pragma once

#include <cstdint>
#include <string>
#include <vector>

// TODO: reimplement this structure in endian-independent way
struct unicode_cpt_flags {
    enum {
        UNDEFINED       = 0x0001,
        NUMBER          = 0x0002,  // regex: \p{N}
        LETTER          = 0x0004,  // regex: \p{L}
        SEPARATOR       = 0x0008,  // regex: \p{Z}
        ACCENT_MARK     = 0x0010,  // regex: \p{M}
        PUNCTUATION     = 0x0020,  // regex: \p{P}
        SYMBOL          = 0x0040,  // regex: \p{S}
        CONTROL         = 0x0080,  // regex: \p{C}
        MASK_CATEGORIES = 0x00FF,
        WHITESPACE      = 0x0100,
        LOWERCASE       = 0x0200,
        UPPERCASE       = 0x0400,
        NFD             = 0x0800,
    };

    // codepoint type
    uint16_t is_undefined   : 1;
    uint16_t is_number      : 1;  // regex: \p{N}
    uint16_t is_letter      : 1;  // regex: \p{L}
    uint16_t is_separator   : 1;  // regex: \p{Z}
    uint16_t is_accent_mark : 1;  // regex: \p{M}
    uint16_t is_punctuation : 1;  // regex: \p{P}
    uint16_t is_symbol      : 1;  // regex: \p{S}
    uint16_t is_control     : 1;  // regex: \p{C}
    // helper flags
    uint16_t is_whitespace  : 1;  // regex: \s
    uint16_t is_lowercase   : 1;
    uint16_t is_uppercase   : 1;
    uint16_t is_nfd         : 1;

    // decode from uint16
    inline unicode_cpt_flags(const uint16_t flags = 0) {
#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
        *reinterpret_cast<uint16_t*>(this) = flags;
#elif __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
        is_undefined   = (flags & UNDEFINED)   ? 1 : 0;
        is_number      = (flags & NUMBER)      ? 1 : 0;
        is_letter      = (flags & LETTER)      ? 1 : 0;
        is_separator   = (flags & SEPARATOR)   ? 1 : 0;
        is_accent_mark = (flags & ACCENT_MARK) ? 1 : 0;
        is_punctuation = (flags & PUNCTUATION) ? 1 : 0;
        is_symbol      = (flags & SYMBOL)      ? 1 : 0;
        is_control     = (flags & CONTROL)     ? 1 : 0;
        is_whitespace  = (flags & WHITESPACE)  ? 1 : 0;
        is_lowercase   = (flags & LOWERCASE)   ? 1 : 0;
        is_uppercase   = (flags & UPPERCASE)   ? 1 : 0;
        is_nfd         = (flags & NFD)         ? 1 : 0;
#else
#error Unexpected or undefined __BYTE_ORDER__
#endif
    }

    inline uint16_t as_uint() const {
#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
        return *reinterpret_cast<const uint16_t*>(this);
#elif __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
        uint16_t result =
              is_undefined   * UNDEFINED
            + is_number      * NUMBER
            + is_letter      * LETTER
            + is_separator   * SEPARATOR
            + is_accent_mark * ACCENT_MARK
            + is_punctuation * PUNCTUATION
            + is_symbol      * SYMBOL
            + is_control     * CONTROL
            + is_whitespace  * WHITESPACE
            + is_lowercase   * LOWERCASE
            + is_uppercase   * UPPERCASE
            + is_nfd         * NFD
            ;

        return result;
#else
#error Unexpected or undefined __BYTE_ORDER__
#endif
    }

    inline uint16_t category_flag() const {
        return this->as_uint() & MASK_CATEGORIES;
    }
};

size_t unicode_len_utf8(char src);

std::string unicode_cpt_to_utf8  (uint32_t cpt);
uint32_t    unicode_cpt_from_utf8(const std::string & utf8, size_t & offset);

std::vector<uint32_t> unicode_cpts_from_utf8(const std::string & utf8);

std::vector<uint32_t> unicode_cpts_normalize_nfd(const std::vector<uint32_t> & cpts);

unicode_cpt_flags unicode_cpt_flags_from_cpt (uint32_t cpt);
unicode_cpt_flags unicode_cpt_flags_from_utf8(const std::string & utf8);

std::string unicode_byte_to_utf8(uint8_t byte);
uint8_t     unicode_utf8_to_byte(const std::string & utf8);

uint32_t unicode_tolower(uint32_t cpt);

bool unicode_cpt_is_han(uint32_t cpt);

std::vector<std::string> unicode_regex_split(const std::string & text, const std::vector<std::string> & regex_exprs);
