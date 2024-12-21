/*
This is free and unencumbered software released into the public domain.

Anyone is free to copy, modify, publish, use, compile, sell, or
distribute this software, either in source code form or as a compiled
binary, for any purpose, commercial or non-commercial, and by any
means.

In jurisdictions that recognize copyright laws, the author or authors
of this software dedicate any and all copyright interest in the
software to the public domain. We make this dedication for the benefit
of the public at large and to the detriment of our heirs and
successors. We intend this dedication to be an overt act of
relinquishment in perpetuity of all present and future rights to this
software under copyright law.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE AUTHORS BE LIABLE FOR ANY CLAIM, DAMAGES OR
OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
OTHER DEALINGS IN THE SOFTWARE.

For more information, please refer to <http://unlicense.org>
*/

#ifndef PUBLIC_DOMAIN_BASE64_HPP_
#define PUBLIC_DOMAIN_BASE64_HPP_

#include <cstdint>
#include <iterator>
#include <stdexcept>
#include <string>

class base64_error : public std::runtime_error
{
public:
    using std::runtime_error::runtime_error;
};

class base64
{
public:
    enum class alphabet
    {
        /** the alphabet is detected automatically */
        auto_,
        /** the standard base64 alphabet is used */
        standard,
        /** like `standard` except that the characters `+` and `/` are replaced by `-` and `_` respectively*/
        url_filename_safe
    };

    enum class decoding_behavior
    {
        /** if the input is not padded, the remaining bits are ignored */
        moderate,
        /** if a padding character is encounter decoding is finished */
        loose
    };

    /**
     Encodes all the elements from `in_begin` to `in_end` to `out`.

     @warning The source and destination cannot overlap. The destination must be able to hold at least
     `required_encode_size(std::distance(in_begin, in_end))`, otherwise the behavior depends on the output iterator.

     @tparam Input_iterator the source; the returned elements are cast to `std::uint8_t` and should not be greater than
     8 bits
     @tparam Output_iterator the destination; the elements written to it are from the type `char`
     @param in_begin the beginning of the source
     @param in_end the ending of the source
     @param out the destination iterator
     @param alphabet which alphabet should be used
     @returns the iterator to the next element past the last element copied
     @throws see `Input_iterator` and `Output_iterator`
    */
    template<typename Input_iterator, typename Output_iterator>
    static Output_iterator encode(Input_iterator in_begin, Input_iterator in_end, Output_iterator out,
                                  alphabet alphabet = alphabet::standard)
    {
        constexpr auto pad = '=';
        const char* alpha  = alphabet == alphabet::url_filename_safe
                                ? "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_"
                                : "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

        while (in_begin != in_end) {
            std::uint8_t i0 = 0, i1 = 0, i2 = 0;

            // first character
            i0 = static_cast<std::uint8_t>(*in_begin);
            ++in_begin;

            *out = alpha[i0 >> 2 & 0x3f];
            ++out;

            // part of first character and second
            if (in_begin != in_end) {
                i1 = static_cast<std::uint8_t>(*in_begin);
                ++in_begin;

                *out = alpha[((i0 & 0x3) << 4) | (i1 >> 4 & 0x0f)];
                ++out;
            } else {
                *out = alpha[(i0 & 0x3) << 4];
                ++out;

                // last padding
                *out = pad;
                ++out;

                // last padding
                *out = pad;
                ++out;

                break;
            }

            // part of second character and third
            if (in_begin != in_end) {
                i2 = static_cast<std::uint8_t>(*in_begin);
                ++in_begin;

                *out = alpha[((i1 & 0xf) << 2) | (i2 >> 6 & 0x03)];
                ++out;
            } else {
                *out = alpha[(i1 & 0xf) << 2];
                ++out;

                // last padding
                *out = pad;
                ++out;

                break;
            }

            // rest of third
            *out = alpha[i2 & 0x3f];
            ++out;
        }

        return out;
    }
    /**
     Encodes a string.

     @param str the string that should be encoded
     @param alphabet which alphabet should be used
     @returns the encoded base64 string
     @throws see base64::encode()
    */
    static std::string encode(const std::string& str, alphabet alphabet = alphabet::standard)
    {
        std::string result;

        result.reserve(required_encode_size(str.length()) + 1);

        encode(str.begin(), str.end(), std::back_inserter(result), alphabet);

        return result;
    }
    /**
     Encodes a char array.

     @param buffer the char array
     @param size the size of the array
     @param alphabet which alphabet should be used
     @returns the encoded string
    */
    static std::string encode(const char* buffer, std::size_t size, alphabet alphabet = alphabet::standard)
    {
        std::string result;

        result.reserve(required_encode_size(size) + 1);

        encode(buffer, buffer + size, std::back_inserter(result), alphabet);

        return result;
    }
    /**
     Decodes all the elements from `in_begin` to `in_end` to `out`. `in_begin` may point to the same location as `out`,
     in other words: inplace decoding is possible.

     @warning The destination must be able to hold at least `required_decode_size(std::distance(in_begin, in_end))`,
     otherwise the behavior depends on the output iterator.

     @tparam Input_iterator the source; the returned elements are cast to `char`
     @tparam Output_iterator the destination; the elements written to it are from the type `std::uint8_t`
     @param in_begin the beginning of the source
     @param in_end the ending of the source
     @param out the destination iterator
     @param alphabet which alphabet should be used
     @param behavior the behavior when an error was detected
     @returns the iterator to the next element past the last element copied
     @throws base64_error depending on the set behavior
     @throws see `Input_iterator` and `Output_iterator`
    */
    template<typename Input_iterator, typename Output_iterator>
    static Output_iterator decode(Input_iterator in_begin, Input_iterator in_end, Output_iterator out,
                                  alphabet alphabet          = alphabet::auto_,
                                  decoding_behavior behavior = decoding_behavior::moderate)
    {
        //constexpr auto pad = '=';
        std::uint8_t last  = 0;
        auto bits          = 0;

        while (in_begin != in_end) {
            auto c = *in_begin;
            ++in_begin;

            if (c == '=') {
                break;
            }

            auto part = _base64_value(alphabet, c);

            // enough bits for one byte
            if (bits + 6 >= 8) {
                *out = (last << (8 - bits)) | (part >> (bits - 2));
                ++out;

                bits -= 2;
            } else {
                bits += 6;
            }

            last = part;
        }

        // check padding
        if (behavior != decoding_behavior::loose) {
            while (in_begin != in_end) {
                auto c = *in_begin;
                ++in_begin;

                if (c != '=') {
                    throw base64_error("invalid base64 character.");
                }
            }
        }

        return out;
    }
    /**
     Decodes a string.

     @param str the base64 encoded string
     @param alphabet which alphabet should be used
     @param behavior the behavior when an error was detected
     @returns the decoded string
     @throws see base64::decode()
    */
    static std::string decode(const std::string& str, alphabet alphabet = alphabet::auto_,
                              decoding_behavior behavior = decoding_behavior::moderate)
    {
        std::string result;

        result.reserve(max_decode_size(str.length()));

        decode(str.begin(), str.end(), std::back_inserter(result), alphabet, behavior);

        return result;
    }
    /**
     Decodes a string.

     @param buffer the base64 encoded buffer
     @param size the size of the buffer
     @param alphabet which alphabet should be used
     @param behavior the behavior when an error was detected
     @returns the decoded string
     @throws see base64::decode()
    */
    static std::string decode(const char* buffer, std::size_t size, alphabet alphabet = alphabet::auto_,
                              decoding_behavior behavior = decoding_behavior::moderate)
    {
        std::string result;

        result.reserve(max_decode_size(size));

        decode(buffer, buffer + size, std::back_inserter(result), alphabet, behavior);

        return result;
    }
    /**
     Decodes a string inplace.

     @param[in,out] str the base64 encoded string
     @param alphabet which alphabet should be used
     @param behavior the behavior when an error was detected
     @throws base64::decode_inplace()
    */
    static void decode_inplace(std::string& str, alphabet alphabet = alphabet::auto_,
                               decoding_behavior behavior = decoding_behavior::moderate)
    {
        str.resize(decode(str.begin(), str.end(), str.begin(), alphabet, behavior) - str.begin());
    }
    /**
     Decodes a char array inplace.

     @param[in,out] str the string array
     @param size the length of the array
     @param alphabet which alphabet should be used
     @param behavior the behavior when an error was detected
     @returns the pointer to the next element past the last element decoded
     @throws base64::decode_inplace()
    */
    static char* decode_inplace(char* str, std::size_t size, alphabet alphabet = alphabet::auto_,
                                decoding_behavior behavior = decoding_behavior::moderate)
    {
        return decode(str, str + size, str, alphabet, behavior);
    }
    /**
     Returns the required decoding size for a given size. The value is calculated with the following formula:

     $$
     \lceil \frac{size}{4} \rceil \cdot 3
     $$

     @param size the size of the encoded input
     @returns the size of the resulting decoded buffer; this the absolute maximum
    */
    static std::size_t max_decode_size(std::size_t size) noexcept
    {
        return (size / 4 + (size % 4 ? 1 : 0)) * 3;
    }
    /**
     Returns the required encoding size for a given size. The value is calculated with the following formula:

     $$
     \lceil \frac{size}{3} \rceil \cdot 4
     $$

     @param size the size of the decoded input
     @returns the size of the resulting encoded buffer
    */
    static std::size_t required_encode_size(std::size_t size) noexcept
    {
        return (size / 3 + (size % 3 ? 1 : 0)) * 4;
    }

private:
    static std::uint8_t _base64_value(alphabet& alphabet, char c)
    {
        if (c >= 'A' && c <= 'Z') {
            return c - 'A';
        } else if (c >= 'a' && c <= 'z') {
            return c - 'a' + 26;
        } else if (c >= '0' && c <= '9') {
            return c - '0' + 52;
        }

        // comes down to alphabet
        if (alphabet == alphabet::standard) {
            if (c == '+') {
                return 62;
            } else if (c == '/') {
                return 63;
            }
        } else if (alphabet == alphabet::url_filename_safe) {
            if (c == '-') {
                return 62;
            } else if (c == '_') {
                return 63;
            }
        } // auto detect
        else {
            if (c == '+') {
                alphabet = alphabet::standard;

                return 62;
            } else if (c == '/') {
                alphabet = alphabet::standard;

                return 63;
            } else if (c == '-') {
                alphabet = alphabet::url_filename_safe;

                return 62;
            } else if (c == '_') {
                alphabet = alphabet::url_filename_safe;

                return 63;
            }
        }

        throw base64_error("invalid base64 character.");
    }
};

#endif // !PUBLIC_DOMAIN_BASE64_HPP_
