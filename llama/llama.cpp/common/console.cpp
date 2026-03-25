#include "console.h"
#include "log.h"
#include <vector>
#include <iostream>
#include <cassert>
#include <cstddef>
#include <cctype>
#include <cwctype>
#include <cstdint>
#include <condition_variable>
#include <mutex>
#include <thread>
#include <stdarg.h>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <fcntl.h>
#include <io.h>
#ifndef ENABLE_VIRTUAL_TERMINAL_PROCESSING
#define ENABLE_VIRTUAL_TERMINAL_PROCESSING 0x0004
#endif
#else
#include <climits>
#include <sys/ioctl.h>
#include <unistd.h>
#include <wchar.h>
#include <stdio.h>
#include <stdlib.h>
#include <signal.h>
#include <termios.h>
#endif

#define ANSI_COLOR_RED     "\x1b[31m"
#define ANSI_COLOR_GREEN   "\x1b[32m"
#define ANSI_COLOR_YELLOW  "\x1b[33m"
#define ANSI_COLOR_BLUE    "\x1b[34m"
#define ANSI_COLOR_MAGENTA "\x1b[35m"
#define ANSI_COLOR_CYAN    "\x1b[36m"
#define ANSI_COLOR_GRAY    "\x1b[90m"
#define ANSI_COLOR_RESET   "\x1b[0m"
#define ANSI_BOLD          "\x1b[1m"

namespace console {

#if defined (_WIN32)
    namespace {
        // Use private-use unicode values to represent special keys that are not reported
        // as characters (e.g. arrows on Windows). These values should never clash with
        // real input and let the rest of the code handle navigation uniformly.
        static constexpr char32_t KEY_ARROW_LEFT       = 0xE000;
        static constexpr char32_t KEY_ARROW_RIGHT      = 0xE001;
        static constexpr char32_t KEY_ARROW_UP         = 0xE002;
        static constexpr char32_t KEY_ARROW_DOWN       = 0xE003;
        static constexpr char32_t KEY_HOME             = 0xE004;
        static constexpr char32_t KEY_END              = 0xE005;
        static constexpr char32_t KEY_CTRL_ARROW_LEFT  = 0xE006;
        static constexpr char32_t KEY_CTRL_ARROW_RIGHT = 0xE007;
        static constexpr char32_t KEY_DELETE           = 0xE008;
    }

    //
    // Console state
    //
#endif

    static bool         advanced_display = false;
    static bool         simple_io        = true;
    static display_type current_display  = DISPLAY_TYPE_RESET;

    static FILE*        out              = stdout;

#if defined (_WIN32)
    static void*        hConsole;
#else
    static FILE*        tty              = nullptr;
    static termios      initial_state;
#endif

    //
    // Init and cleanup
    //

    void init(bool use_simple_io, bool use_advanced_display) {
        advanced_display = use_advanced_display;
        simple_io = use_simple_io;
#if defined(_WIN32)
        // Windows-specific console initialization
        DWORD dwMode = 0;
        hConsole = GetStdHandle(STD_OUTPUT_HANDLE);
        if (hConsole == INVALID_HANDLE_VALUE || !GetConsoleMode(hConsole, &dwMode)) {
            hConsole = GetStdHandle(STD_ERROR_HANDLE);
            if (hConsole != INVALID_HANDLE_VALUE && (!GetConsoleMode(hConsole, &dwMode))) {
                hConsole = nullptr;
                simple_io = true;
            }
        }
        if (hConsole) {
            // Check conditions combined to reduce nesting
            if (advanced_display && !(dwMode & ENABLE_VIRTUAL_TERMINAL_PROCESSING) &&
                !SetConsoleMode(hConsole, dwMode | ENABLE_VIRTUAL_TERMINAL_PROCESSING)) {
                advanced_display = false;
            }
            // Set console output codepage to UTF8
            SetConsoleOutputCP(CP_UTF8);
        }
        HANDLE hConIn = GetStdHandle(STD_INPUT_HANDLE);
        if (hConIn != INVALID_HANDLE_VALUE && GetConsoleMode(hConIn, &dwMode)) {
            // Set console input codepage to UTF16
            _setmode(_fileno(stdin), _O_WTEXT);

            // Set ICANON (ENABLE_LINE_INPUT) and ECHO (ENABLE_ECHO_INPUT)
            if (simple_io) {
                dwMode |= ENABLE_LINE_INPUT | ENABLE_ECHO_INPUT;
            } else {
                dwMode &= ~(ENABLE_LINE_INPUT | ENABLE_ECHO_INPUT);
            }
            if (!SetConsoleMode(hConIn, dwMode)) {
                simple_io = true;
            }
        }
        if (simple_io) {
            _setmode(_fileno(stdin), _O_U8TEXT);
        }
#else
        // POSIX-specific console initialization
        if (!simple_io) {
            struct termios new_termios;
            tcgetattr(STDIN_FILENO, &initial_state);
            new_termios = initial_state;
            new_termios.c_lflag &= ~(ICANON | ECHO);
            new_termios.c_cc[VMIN] = 1;
            new_termios.c_cc[VTIME] = 0;
            tcsetattr(STDIN_FILENO, TCSANOW, &new_termios);

            tty = fopen("/dev/tty", "w+");
            if (tty != nullptr) {
                out = tty;
            }
        }

        setlocale(LC_ALL, "");
#endif
    }

    void cleanup() {
        // Reset console display
        set_display(DISPLAY_TYPE_RESET);

#if !defined(_WIN32)
        // Restore settings on POSIX systems
        if (!simple_io) {
            if (tty != nullptr) {
                out = stdout;
                fclose(tty);
                tty = nullptr;
            }
            tcsetattr(STDIN_FILENO, TCSANOW, &initial_state);
        }
#endif
    }

    //
    // Display and IO
    //

    // Keep track of current display and only emit ANSI code if it changes
    void set_display(display_type display) {
        if (advanced_display && current_display != display) {
            common_log_flush(common_log_main());
            switch(display) {
                case DISPLAY_TYPE_RESET:
                    fprintf(out, ANSI_COLOR_RESET);
                    break;
                case DISPLAY_TYPE_INFO:
                    fprintf(out, ANSI_COLOR_MAGENTA);
                    break;
                case DISPLAY_TYPE_PROMPT:
                    fprintf(out, ANSI_COLOR_YELLOW);
                    break;
                case DISPLAY_TYPE_REASONING:
                    fprintf(out, ANSI_COLOR_GRAY);
                    break;
                case DISPLAY_TYPE_USER_INPUT:
                    fprintf(out, ANSI_BOLD ANSI_COLOR_GREEN);
                    break;
                case DISPLAY_TYPE_ERROR:
                    fprintf(out, ANSI_BOLD ANSI_COLOR_RED);
            }
            current_display = display;
            fflush(out);
        }
    }

    static char32_t getchar32() {
#if defined(_WIN32)
        HANDLE hConsole = GetStdHandle(STD_INPUT_HANDLE);
        wchar_t high_surrogate = 0;

        while (true) {
            INPUT_RECORD record;
            DWORD count;
            if (!ReadConsoleInputW(hConsole, &record, 1, &count) || count == 0) {
                return WEOF;
            }

            if (record.EventType == KEY_EVENT && record.Event.KeyEvent.bKeyDown) {
                wchar_t wc = record.Event.KeyEvent.uChar.UnicodeChar;
                if (wc == 0) {
                    const DWORD ctrl_mask = LEFT_CTRL_PRESSED | RIGHT_CTRL_PRESSED;
                    const bool ctrl_pressed = (record.Event.KeyEvent.dwControlKeyState & ctrl_mask) != 0;
                    switch (record.Event.KeyEvent.wVirtualKeyCode) {
                        case VK_LEFT:   return ctrl_pressed ? KEY_CTRL_ARROW_LEFT  : KEY_ARROW_LEFT;
                        case VK_RIGHT:  return ctrl_pressed ? KEY_CTRL_ARROW_RIGHT : KEY_ARROW_RIGHT;
                        case VK_UP:     return KEY_ARROW_UP;
                        case VK_DOWN:   return KEY_ARROW_DOWN;
                        case VK_HOME:   return KEY_HOME;
                        case VK_END:    return KEY_END;
                        case VK_DELETE: return KEY_DELETE;
                        default:        continue;
                    }
                }

                if ((wc >= 0xD800) && (wc <= 0xDBFF)) { // Check if wc is a high surrogate
                    high_surrogate = wc;
                    continue;
                }
                if ((wc >= 0xDC00) && (wc <= 0xDFFF)) { // Check if wc is a low surrogate
                    if (high_surrogate != 0) { // Check if we have a high surrogate
                        return ((high_surrogate - 0xD800) << 10) + (wc - 0xDC00) + 0x10000;
                    }
                }

                high_surrogate = 0; // Reset the high surrogate
                return static_cast<char32_t>(wc);
            }
        }
#else
        wchar_t wc = getwchar();
        if (static_cast<wint_t>(wc) == WEOF) {
            return WEOF;
        }

#if WCHAR_MAX == 0xFFFF
        if ((wc >= 0xD800) && (wc <= 0xDBFF)) { // Check if wc is a high surrogate
            wchar_t low_surrogate = getwchar();
            if ((low_surrogate >= 0xDC00) && (low_surrogate <= 0xDFFF)) { // Check if the next wchar is a low surrogate
                return (static_cast<char32_t>(wc & 0x03FF) << 10) + (low_surrogate & 0x03FF) + 0x10000;
            }
        }
        if ((wc >= 0xD800) && (wc <= 0xDFFF)) { // Invalid surrogate pair
            return 0xFFFD; // Return the replacement character U+FFFD
        }
#endif

        return static_cast<char32_t>(wc);
#endif
    }

    static void pop_cursor() {
#if defined(_WIN32)
        if (hConsole != NULL) {
            CONSOLE_SCREEN_BUFFER_INFO bufferInfo;
            GetConsoleScreenBufferInfo(hConsole, &bufferInfo);

            COORD newCursorPosition = bufferInfo.dwCursorPosition;
            if (newCursorPosition.X == 0) {
                newCursorPosition.X = bufferInfo.dwSize.X - 1;
                newCursorPosition.Y -= 1;
            } else {
                newCursorPosition.X -= 1;
            }

            SetConsoleCursorPosition(hConsole, newCursorPosition);
            return;
        }
#endif
        putc('\b', out);
    }

    static int estimateWidth(char32_t codepoint) {
#if defined(_WIN32)
        (void)codepoint;
        return 1;
#else
        return wcwidth(codepoint);
#endif
    }

    static int put_codepoint(const char* utf8_codepoint, size_t length, int expectedWidth) {
#if defined(_WIN32)
        CONSOLE_SCREEN_BUFFER_INFO bufferInfo;
        if (!GetConsoleScreenBufferInfo(hConsole, &bufferInfo)) {
            // go with the default
            return expectedWidth;
        }
        COORD initialPosition = bufferInfo.dwCursorPosition;
        DWORD nNumberOfChars = length;
        WriteConsole(hConsole, utf8_codepoint, nNumberOfChars, &nNumberOfChars, NULL);

        CONSOLE_SCREEN_BUFFER_INFO newBufferInfo;
        GetConsoleScreenBufferInfo(hConsole, &newBufferInfo);

        // Figure out our real position if we're in the last column
        if (utf8_codepoint[0] != 0x09 && initialPosition.X == newBufferInfo.dwSize.X - 1) {
            DWORD nNumberOfChars;
            WriteConsole(hConsole, &" \b", 2, &nNumberOfChars, NULL);
            GetConsoleScreenBufferInfo(hConsole, &newBufferInfo);
        }

        int width = newBufferInfo.dwCursorPosition.X - initialPosition.X;
        if (width < 0) {
            width += newBufferInfo.dwSize.X;
        }
        return width;
#else
        // We can trust expectedWidth if we've got one
        if (expectedWidth >= 0 || tty == nullptr) {
            fwrite(utf8_codepoint, length, 1, out);
            return expectedWidth;
        }

        fputs("\033[6n", tty); // Query cursor position
        int x1;
        int y1;
        int x2;
        int y2;
        int results = 0;
        results = fscanf(tty, "\033[%d;%dR", &y1, &x1);

        fwrite(utf8_codepoint, length, 1, tty);

        fputs("\033[6n", tty); // Query cursor position
        results += fscanf(tty, "\033[%d;%dR", &y2, &x2);

        if (results != 4) {
            return expectedWidth;
        }

        int width = x2 - x1;
        if (width < 0) {
            // Calculate the width considering text wrapping
            struct winsize w;
            ioctl(STDOUT_FILENO, TIOCGWINSZ, &w);
            width += w.ws_col;
        }
        return width;
#endif
    }

    static void replace_last(char ch) {
#if defined(_WIN32)
        pop_cursor();
        put_codepoint(&ch, 1, 1);
#else
        fprintf(out, "\b%c", ch);
#endif
    }

    static char32_t decode_utf8(const std::string & input, size_t pos, size_t & advance) {
        unsigned char c = static_cast<unsigned char>(input[pos]);
        if ((c & 0x80u) == 0u) {
            advance = 1;
            return c;
        }
        if ((c & 0xE0u) == 0xC0u && pos + 1 < input.size()) {
            unsigned char c1 = static_cast<unsigned char>(input[pos + 1]);
            if ((c1 & 0xC0u) != 0x80u) {
                advance = 1;
                return 0xFFFD;
            }
            advance = 2;
            return ((c & 0x1Fu) << 6) | (static_cast<unsigned char>(input[pos + 1]) & 0x3Fu);
        }
        if ((c & 0xF0u) == 0xE0u && pos + 2 < input.size()) {
            unsigned char c1 = static_cast<unsigned char>(input[pos + 1]);
            unsigned char c2 = static_cast<unsigned char>(input[pos + 2]);
            if ((c1 & 0xC0u) != 0x80u || (c2 & 0xC0u) != 0x80u) {
                advance = 1;
                return 0xFFFD;
            }
            advance = 3;
            return ((c & 0x0Fu) << 12) |
                   ((static_cast<unsigned char>(input[pos + 1]) & 0x3Fu) << 6) |
                   (static_cast<unsigned char>(input[pos + 2]) & 0x3Fu);
        }
        if ((c & 0xF8u) == 0xF0u && pos + 3 < input.size()) {
            unsigned char c1 = static_cast<unsigned char>(input[pos + 1]);
            unsigned char c2 = static_cast<unsigned char>(input[pos + 2]);
            unsigned char c3 = static_cast<unsigned char>(input[pos + 3]);
            if ((c1 & 0xC0u) != 0x80u || (c2 & 0xC0u) != 0x80u || (c3 & 0xC0u) != 0x80u) {
                advance = 1;
                return 0xFFFD;
            }
            advance = 4;
            return ((c & 0x07u) << 18) |
                   ((static_cast<unsigned char>(input[pos + 1]) & 0x3Fu) << 12) |
                   ((static_cast<unsigned char>(input[pos + 2]) & 0x3Fu) << 6) |
                   (static_cast<unsigned char>(input[pos + 3]) & 0x3Fu);
        }

        advance = 1;
        return 0xFFFD; // replacement character for invalid input
    }

    static void append_utf8(char32_t ch, std::string & out) {
        if (ch <= 0x7F) {
            out.push_back(static_cast<unsigned char>(ch));
        } else if (ch <= 0x7FF) {
            out.push_back(static_cast<unsigned char>(0xC0 | ((ch >> 6) & 0x1F)));
            out.push_back(static_cast<unsigned char>(0x80 | (ch & 0x3F)));
        } else if (ch <= 0xFFFF) {
            out.push_back(static_cast<unsigned char>(0xE0 | ((ch >> 12) & 0x0F)));
            out.push_back(static_cast<unsigned char>(0x80 | ((ch >> 6) & 0x3F)));
            out.push_back(static_cast<unsigned char>(0x80 | (ch & 0x3F)));
        } else if (ch <= 0x10FFFF) {
            out.push_back(static_cast<unsigned char>(0xF0 | ((ch >> 18) & 0x07)));
            out.push_back(static_cast<unsigned char>(0x80 | ((ch >> 12) & 0x3F)));
            out.push_back(static_cast<unsigned char>(0x80 | ((ch >> 6) & 0x3F)));
            out.push_back(static_cast<unsigned char>(0x80 | (ch & 0x3F)));
        } else {
            // Invalid Unicode code point
        }
    }

    // Helper function to remove the last UTF-8 character from a string
    static size_t prev_utf8_char_pos(const std::string & line, size_t pos) {
        if (pos == 0) return 0;
        pos--;
        while (pos > 0 && (line[pos] & 0xC0) == 0x80) {
            pos--;
        }
        return pos;
    }

    static size_t next_utf8_char_pos(const std::string & line, size_t pos) {
        if (pos >= line.length()) return line.length();
        pos++;
        while (pos < line.length() && (line[pos] & 0xC0) == 0x80) {
            pos++;
        }
        return pos;
    }

    static void move_cursor(int delta);
    static void move_word_left(size_t & char_pos, size_t & byte_pos, const std::vector<int> & widths, const std::string & line);
    static void move_word_right(size_t & char_pos, size_t & byte_pos, const std::vector<int> & widths, const std::string & line);
    static void move_to_line_start(size_t & char_pos, size_t & byte_pos, const std::vector<int> & widths);
    static void move_to_line_end(size_t & char_pos, size_t & byte_pos, const std::vector<int> & widths, const std::string & line);

    static void delete_at_cursor(std::string & line, std::vector<int> & widths, size_t & char_pos, size_t & byte_pos) {
        if (char_pos >= widths.size()) {
            return;
        }

        size_t next_pos = next_utf8_char_pos(line, byte_pos);
        int w = widths[char_pos];
        size_t char_len = next_pos - byte_pos;

        line.erase(byte_pos, char_len);
        widths.erase(widths.begin() + char_pos);

        size_t p = byte_pos;
        int tail_width = 0;
        for (size_t i = char_pos; i < widths.size(); ++i) {
            size_t following = next_utf8_char_pos(line, p);
            put_codepoint(line.c_str() + p, following - p, widths[i]);
            tail_width += widths[i];
            p = following;
        }

        for (int i = 0; i < w; ++i) {
            fputc(' ', out);
        }

        move_cursor(-(tail_width + w));
    }

    static void clear_current_line(const std::vector<int> & widths) {
        int total_width = 0;
        for (int w : widths) {
            total_width += (w > 0 ? w : 1);
        }

        if (total_width > 0) {
            std::string spaces(total_width, ' ');
            fwrite(spaces.c_str(), 1, total_width, out);
            move_cursor(-total_width);
        }
    }

    static void set_line_contents(std::string new_line, std::string & line, std::vector<int> & widths, size_t & char_pos,
                                  size_t & byte_pos) {
        move_to_line_start(char_pos, byte_pos, widths);
        clear_current_line(widths);

        line = std::move(new_line);
        widths.clear();
        byte_pos = 0;
        char_pos = 0;

        size_t idx = 0;
        while (idx < line.size()) {
            size_t advance = 0;
            char32_t cp = decode_utf8(line, idx, advance);
            int expected_width = estimateWidth(cp);
            int real_width = put_codepoint(line.c_str() + idx, advance, expected_width);
            if (real_width < 0) real_width = 0;
            widths.push_back(real_width);
            idx += advance;
            ++char_pos;
            byte_pos = idx;
        }
    }

    static void move_to_line_start(size_t & char_pos, size_t & byte_pos, const std::vector<int> & widths) {
        int back_width = 0;
        for (size_t i = 0; i < char_pos; ++i) {
            back_width += widths[i];
        }
        move_cursor(-back_width);
        char_pos = 0;
        byte_pos = 0;
    }

    static void move_to_line_end(size_t & char_pos, size_t & byte_pos, const std::vector<int> & widths, const std::string & line) {
        int forward_width = 0;
        for (size_t i = char_pos; i < widths.size(); ++i) {
            forward_width += widths[i];
        }
        move_cursor(forward_width);
        char_pos = widths.size();
        byte_pos = line.length();
    }

    static bool has_ctrl_modifier(const std::string & params) {
        size_t start = 0;
        while (start < params.size()) {
            size_t end = params.find(';', start);
            size_t len = (end == std::string::npos) ? params.size() - start : end - start;
            if (len > 0) {
                int value = 0;
                for (size_t i = 0; i < len; ++i) {
                    char ch = params[start + i];
                    if (!std::isdigit(static_cast<unsigned char>(ch))) {
                        value = -1;
                        break;
                    }
                    value = value * 10 + (ch - '0');
                }
                if (value == 5) {
                    return true;
                }
            }

            if (end == std::string::npos) {
                break;
            }
            start = end + 1;
        }
        return false;
    }

    static bool is_space_codepoint(char32_t cp) {
        return std::iswspace(static_cast<wint_t>(cp)) != 0;
    }

    static void move_word_left(size_t & char_pos, size_t & byte_pos, const std::vector<int> & widths, const std::string & line) {
        if (char_pos == 0) {
            return;
        }

        size_t new_char_pos = char_pos;
        size_t new_byte_pos = byte_pos;
        int move_width = 0;

        while (new_char_pos > 0) {
            size_t prev_byte = prev_utf8_char_pos(line, new_byte_pos);
            size_t advance = 0;
            char32_t cp = decode_utf8(line, prev_byte, advance);
            if (!is_space_codepoint(cp)) {
                break;
            }
            move_width += widths[new_char_pos - 1];
            new_char_pos--;
            new_byte_pos = prev_byte;
        }

        while (new_char_pos > 0) {
            size_t prev_byte = prev_utf8_char_pos(line, new_byte_pos);
            size_t advance = 0;
            char32_t cp = decode_utf8(line, prev_byte, advance);
            if (is_space_codepoint(cp)) {
                break;
            }
            move_width += widths[new_char_pos - 1];
            new_char_pos--;
            new_byte_pos = prev_byte;
        }

        move_cursor(-move_width);
        char_pos = new_char_pos;
        byte_pos = new_byte_pos;
    }

    static void move_word_right(size_t & char_pos, size_t & byte_pos, const std::vector<int> & widths, const std::string & line) {
        if (char_pos >= widths.size()) {
            return;
        }

        size_t new_char_pos = char_pos;
        size_t new_byte_pos = byte_pos;
        int move_width = 0;

        while (new_char_pos < widths.size()) {
            size_t advance = 0;
            char32_t cp = decode_utf8(line, new_byte_pos, advance);
            if (!is_space_codepoint(cp)) {
                break;
            }
            move_width += widths[new_char_pos];
            new_char_pos++;
            new_byte_pos += advance;
        }

        while (new_char_pos < widths.size()) {
            size_t advance = 0;
            char32_t cp = decode_utf8(line, new_byte_pos, advance);
            if (is_space_codepoint(cp)) {
                break;
            }
            move_width += widths[new_char_pos];
            new_char_pos++;
            new_byte_pos += advance;
        }

        while (new_char_pos < widths.size()) {
            size_t advance = 0;
            char32_t cp = decode_utf8(line, new_byte_pos, advance);
            if (!is_space_codepoint(cp)) {
                break;
            }
            move_width += widths[new_char_pos];
            new_char_pos++;
            new_byte_pos += advance;
        }

        move_cursor(move_width);
        char_pos = new_char_pos;
        byte_pos = new_byte_pos;
    }

    static void move_cursor(int delta) {
        if (delta == 0) return;
#if defined(_WIN32)
        if (hConsole != NULL) {
            CONSOLE_SCREEN_BUFFER_INFO bufferInfo;
            GetConsoleScreenBufferInfo(hConsole, &bufferInfo);
            COORD newCursorPosition = bufferInfo.dwCursorPosition;
            int width = bufferInfo.dwSize.X;
            int newX = newCursorPosition.X + delta;
            int newY = newCursorPosition.Y;

            while (newX >= width) {
                newX -= width;
                newY++;
            }
            while (newX < 0) {
                newX += width;
                newY--;
            }

            newCursorPosition.X = newX;
            newCursorPosition.Y = newY;
            SetConsoleCursorPosition(hConsole, newCursorPosition);
        }
#else
        if (delta < 0) {
            for (int i = 0; i < -delta; i++) fprintf(out, "\b");
        } else {
            for (int i = 0; i < delta; i++) fprintf(out, "\033[C");
        }
#endif
    }

    struct history_t {
        std::vector<std::string> entries;
        size_t viewing_idx = SIZE_MAX;
        std::string backup_line; // current line before viewing history
        void add(const std::string & line) {
            if (line.empty()) {
                return;
            }
            // avoid duplicates with the last entry
            if (entries.empty() || entries.back() != line) {
                entries.push_back(line);
            }
            // also clear viewing state
            end_viewing();
        }
        bool prev(std::string & cur_line) {
            if (entries.empty()) {
                return false;
            }
            if (viewing_idx == SIZE_MAX) {
                return false;
            }
            if (viewing_idx > 0) {
                viewing_idx--;
            }
            cur_line = entries[viewing_idx];
            return true;
        }
        bool next(std::string & cur_line) {
            if (entries.empty() || viewing_idx == SIZE_MAX) {
                return false;
            }
            viewing_idx++;
            if (viewing_idx >= entries.size()) {
                cur_line = backup_line;
                end_viewing();
            } else {
                cur_line = entries[viewing_idx];
            }
            return true;
        }
        void begin_viewing(const std::string & line) {
            backup_line = line;
            viewing_idx = entries.size();
        }
        void end_viewing() {
            viewing_idx = SIZE_MAX;
            backup_line.clear();
        }
        bool is_viewing() const {
            return viewing_idx != SIZE_MAX;
        }
    } history;

    static bool readline_advanced(std::string & line, bool multiline_input) {
        if (out != stdout) {
            fflush(stdout);
        }

        line.clear();
        std::vector<int> widths;
        bool is_special_char = false;
        bool end_of_stream = false;

        size_t byte_pos = 0; // current byte index
        size_t char_pos = 0; // current character index (one char can be multiple bytes)

        char32_t input_char;
        while (true) {
            assert(char_pos <= byte_pos);
            assert(char_pos <= widths.size());
            auto history_prev = [&]() {
                if (!history.is_viewing()) {
                    history.begin_viewing(line);
                }
                std::string new_line;
                if (!history.prev(new_line)) {
                    return;
                }
                set_line_contents(new_line, line, widths, char_pos, byte_pos);
            };
            auto history_next = [&]() {
                if (history.is_viewing()) {
                    std::string new_line;
                    if (!history.next(new_line)) {
                        return;
                    }
                    set_line_contents(new_line, line, widths, char_pos, byte_pos);
                }
            };

            fflush(out); // Ensure all output is displayed before waiting for input
            input_char = getchar32();

            if (input_char == '\r' || input_char == '\n') {
                break;
            }

            if (input_char == (char32_t) WEOF || input_char == 0x04 /* Ctrl+D */) {
                end_of_stream = true;
                break;
            }

            if (is_special_char) {
                replace_last(line.back());
                is_special_char = false;
            }

            if (input_char == '\033') { // Escape sequence
                char32_t code = getchar32();
                if (code == '[') {
                    std::string params;
                    while (true) {
                        code = getchar32();
                        if ((code >= 'A' && code <= 'Z') || (code >= 'a' && code <= 'z') || code == '~' || code == (char32_t) WEOF) {
                            break;
                        }
                        params.push_back(static_cast<char>(code));
                    }

                    const bool ctrl_modifier = has_ctrl_modifier(params);

                    if (code == 'D') { // left
                        if (ctrl_modifier) {
                            move_word_left(char_pos, byte_pos, widths, line);
                        } else if (char_pos > 0) {
                            int w = widths[char_pos - 1];
                            move_cursor(-w);
                            char_pos--;
                            byte_pos = prev_utf8_char_pos(line, byte_pos);
                        }
                    } else if (code == 'C') { // right
                        if (ctrl_modifier) {
                            move_word_right(char_pos, byte_pos, widths, line);
                        } else if (char_pos < widths.size()) {
                            int w = widths[char_pos];
                            move_cursor(w);
                            char_pos++;
                            byte_pos = next_utf8_char_pos(line, byte_pos);
                        }
                    } else if (code == 'H') { // home
                        move_to_line_start(char_pos, byte_pos, widths);
                    } else if (code == 'F') { // end
                        move_to_line_end(char_pos, byte_pos, widths, line);
                    } else if (code == 'A' || code == 'B') {
                        // up/down
                        if (code == 'A') {
                            history_prev();
                            is_special_char = false;
                        } else if (code == 'B') {
                            history_next();
                            is_special_char = false;
                        }
                    } else if ((code == '~' || (code >= 'A' && code <= 'Z') || (code >= 'a' && code <= 'z')) && !params.empty()) {
                        std::string digits;
                        for (char ch : params) {
                            if (ch == ';') {
                                break;
                            }
                            if (std::isdigit(static_cast<unsigned char>(ch))) {
                                digits.push_back(ch);
                            }
                        }

                        if (code == '~') {
                            if (digits == "1" || digits == "7") { // home
                                move_to_line_start(char_pos, byte_pos, widths);
                            } else if (digits == "4" || digits == "8") { // end
                                move_to_line_end(char_pos, byte_pos, widths, line);
                            } else if (digits == "3") { // delete
                                delete_at_cursor(line, widths, char_pos, byte_pos);
                            }
                        }
                    }
                } else if (code == 0x1B) {
                    // Discard the rest of the escape sequence
                    while ((code = getchar32()) != (char32_t) WEOF) {
                        if ((code >= 'A' && code <= 'Z') || (code >= 'a' && code <= 'z') || code == '~') {
                            break;
                        }
                    }
                }
#if defined(_WIN32)
            } else if (input_char == KEY_ARROW_LEFT) {
                if (char_pos > 0) {
                    int w = widths[char_pos - 1];
                    move_cursor(-w);
                    char_pos--;
                    byte_pos = prev_utf8_char_pos(line, byte_pos);
                }
            } else if (input_char == KEY_ARROW_RIGHT) {
                if (char_pos < widths.size()) {
                    int w = widths[char_pos];
                    move_cursor(w);
                    char_pos++;
                    byte_pos = next_utf8_char_pos(line, byte_pos);
                }
            } else if (input_char == KEY_CTRL_ARROW_LEFT) {
                move_word_left(char_pos, byte_pos, widths, line);
            } else if (input_char == KEY_CTRL_ARROW_RIGHT) {
                move_word_right(char_pos, byte_pos, widths, line);
            } else if (input_char == KEY_HOME) {
                move_to_line_start(char_pos, byte_pos, widths);
            } else if (input_char == KEY_END) {
                move_to_line_end(char_pos, byte_pos, widths, line);
            } else if (input_char == KEY_DELETE) {
                delete_at_cursor(line, widths, char_pos, byte_pos);
            } else if (input_char == KEY_ARROW_UP || input_char == KEY_ARROW_DOWN) {
                if (input_char == KEY_ARROW_UP) {
                    history_prev();
                    is_special_char = false;
                } else if (input_char == KEY_ARROW_DOWN) {
                    history_next();
                    is_special_char = false;
                }
#endif
            } else if (input_char == 0x08 || input_char == 0x7F) { // Backspace
                if (char_pos > 0) {
                    int w = widths[char_pos - 1];
                    move_cursor(-w);
                    char_pos--;
                    size_t prev_pos = prev_utf8_char_pos(line, byte_pos);
                    size_t char_len = byte_pos - prev_pos;
                    byte_pos = prev_pos;

                    // remove the character
                    line.erase(byte_pos, char_len);
                    widths.erase(widths.begin() + char_pos);

                    // redraw tail
                    size_t p = byte_pos;
                    int tail_width = 0;
                    for (size_t i = char_pos; i < widths.size(); ++i) {
                        size_t next_p = next_utf8_char_pos(line, p);
                        put_codepoint(line.c_str() + p, next_p - p, widths[i]);
                        tail_width += widths[i];
                        p = next_p;
                    }

                    // clear display
                    for (int i = 0; i < w; ++i) {
                        fputc(' ', out);
                    }
                    move_cursor(-(tail_width + w));
                }
            } else {
                // insert character
                std::string new_char_str;
                append_utf8(input_char, new_char_str);
                int w = estimateWidth(input_char);

                if (char_pos == widths.size()) {
                    // insert at the end
                    line += new_char_str;
                    int real_w = put_codepoint(new_char_str.c_str(), new_char_str.length(), w);
                    if (real_w < 0) real_w = 0;
                    widths.push_back(real_w);
                    byte_pos += new_char_str.length();
                    char_pos++;
                } else {
                    // insert in middle
                    line.insert(byte_pos, new_char_str);

                    int real_w = put_codepoint(new_char_str.c_str(), new_char_str.length(), w);
                    if (real_w < 0) real_w = 0;

                    widths.insert(widths.begin() + char_pos, real_w);

                    // print the tail
                    size_t p = byte_pos + new_char_str.length();
                    int tail_width = 0;
                    for (size_t i = char_pos + 1; i < widths.size(); ++i) {
                        size_t next_p = next_utf8_char_pos(line, p);
                        put_codepoint(line.c_str() + p, next_p - p, widths[i]);
                        tail_width += widths[i];
                        p = next_p;
                    }

                    move_cursor(-tail_width);

                    byte_pos += new_char_str.length();
                    char_pos++;
                }
            }

            if (!line.empty() && (line.back() == '\\' || line.back() == '/')) {
                replace_last(line.back());
                is_special_char = true;
            }
        }

        bool has_more = multiline_input;
        if (is_special_char) {
            replace_last(' ');
            pop_cursor();

            char last = line.back();
            line.pop_back();
            if (last == '\\') {
                line += '\n';
                fputc('\n', out);
                has_more = !has_more;
            } else {
                // llama will just eat the single space, it won't act as a space
                if (line.length() == 1 && line.back() == ' ') {
                    line.clear();
                    pop_cursor();
                }
                has_more = false;
            }
        } else {
            if (end_of_stream) {
                has_more = false;
            } else {
                line += '\n';
                fputc('\n', out);
            }
        }

        if (!end_of_stream && !line.empty()) {
            // remove the trailing newline for history storage
            if (!line.empty() && line.back() == '\n') {
                line.pop_back();
            }
            // TODO: maybe support multiline history entries?
            history.add(line);
        }

        fflush(out);
        return has_more;
    }

    static bool readline_simple(std::string & line, bool multiline_input) {
#if defined(_WIN32)
        std::wstring wline;
        if (!std::getline(std::wcin, wline)) {
            // Input stream is bad or EOF received
            line.clear();
            GenerateConsoleCtrlEvent(CTRL_C_EVENT, 0);
            return false;
        }

        int size_needed = WideCharToMultiByte(CP_UTF8, 0, &wline[0], (int)wline.size(), NULL, 0, NULL, NULL);
        line.resize(size_needed);
        WideCharToMultiByte(CP_UTF8, 0, &wline[0], (int)wline.size(), &line[0], size_needed, NULL, NULL);
#else
        if (!std::getline(std::cin, line)) {
            // Input stream is bad or EOF received
            line.clear();
            return false;
        }
#endif
        if (!line.empty()) {
            char last = line.back();
            if (last == '/') { // Always return control on '/' symbol
                line.pop_back();
                return false;
            }
            if (last == '\\') { // '\\' changes the default action
                line.pop_back();
                multiline_input = !multiline_input;
            }
        }
        line += '\n';

        // By default, continue input if multiline_input is set
        return multiline_input;
    }

    bool readline(std::string & line, bool multiline_input) {
        if (simple_io) {
            return readline_simple(line, multiline_input);
        }
        return readline_advanced(line, multiline_input);
    }

    namespace spinner {
        static const char LOADING_CHARS[] = {'|', '/', '-', '\\'};
        static std::condition_variable cv_stop;
        static std::thread th;
        static size_t frame = 0; // only modified by one thread
        static bool running = false;
        static std::mutex mtx;
        static auto wait_time = std::chrono::milliseconds(100);
        static void draw_next_frame() {
            // don't need lock because only one thread modifies running
            frame = (frame + 1) % sizeof(LOADING_CHARS);
            replace_last(LOADING_CHARS[frame]);
            fflush(out);
        }
        void start() {
            std::unique_lock<std::mutex> lock(mtx);
            if (simple_io || running) {
                return;
            }
            common_log_flush(common_log_main());
            fprintf(out, "%c", LOADING_CHARS[0]);
            fflush(out);
            frame = 1;
            running = true;
            th = std::thread([]() {
                std::unique_lock<std::mutex> lock(mtx);
                while (true) {
                    if (cv_stop.wait_for(lock, wait_time, []{ return !running; })) {
                        break;
                    }
                    draw_next_frame();
                }
            });
        }
        void stop() {
            {
                std::unique_lock<std::mutex> lock(mtx);
                if (simple_io || !running) {
                    return;
                }
                running = false;
                cv_stop.notify_all();
            }
            if (th.joinable()) {
                th.join();
            }
            replace_last(' ');
            pop_cursor();
            fflush(out);
        }
    }

    void log(const char * fmt, ...) {
        va_list args;
        va_start(args, fmt);
        vfprintf(out, fmt, args);
        va_end(args);
    }

    void error(const char * fmt, ...) {
        va_list args;
        va_start(args, fmt);
        display_type cur = current_display;
        set_display(DISPLAY_TYPE_ERROR);
        vfprintf(out, fmt, args);
        set_display(cur); // restore previous color
        va_end(args);
    }

    void flush() {
        fflush(out);
    }
}
