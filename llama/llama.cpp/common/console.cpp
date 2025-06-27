#include "console.h"
#include <vector>
#include <iostream>

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
#define ANSI_COLOR_RESET   "\x1b[0m"
#define ANSI_BOLD          "\x1b[1m"

namespace console {

    //
    // Console state
    //

    static bool      advanced_display = false;
    static bool      simple_io        = true;
    static display_t current_display  = reset;

    static FILE*     out              = stdout;

#if defined (_WIN32)
    static void*     hConsole;
#else
    static FILE*     tty              = nullptr;
    static termios   initial_state;
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
        set_display(reset);

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
    void set_display(display_t display) {
        if (advanced_display && current_display != display) {
            fflush(stdout);
            switch(display) {
                case reset:
                    fprintf(out, ANSI_COLOR_RESET);
                    break;
                case prompt:
                    fprintf(out, ANSI_COLOR_YELLOW);
                    break;
                case user_input:
                    fprintf(out, ANSI_BOLD ANSI_COLOR_GREEN);
                    break;
                case error:
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
                    continue;
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
    static void pop_back_utf8_char(std::string & line) {
        if (line.empty()) {
            return;
        }

        size_t pos = line.length() - 1;

        // Find the start of the last UTF-8 character (checking up to 4 bytes back)
        for (size_t i = 0; i < 3 && pos > 0; ++i, --pos) {
            if ((line[pos] & 0xC0) != 0x80) {
                break; // Found the start of the character
            }
        }
        line.erase(pos);
    }

    static bool readline_advanced(std::string & line, bool multiline_input) {
        if (out != stdout) {
            fflush(stdout);
        }

        line.clear();
        std::vector<int> widths;
        bool is_special_char = false;
        bool end_of_stream = false;

        char32_t input_char;
        while (true) {
            fflush(out); // Ensure all output is displayed before waiting for input
            input_char = getchar32();

            if (input_char == '\r' || input_char == '\n') {
                break;
            }

            if (input_char == (char32_t) WEOF || input_char == 0x04 /* Ctrl+D*/) {
                end_of_stream = true;
                break;
            }

            if (is_special_char) {
                set_display(user_input);
                replace_last(line.back());
                is_special_char = false;
            }

            if (input_char == '\033') { // Escape sequence
                char32_t code = getchar32();
                if (code == '[' || code == 0x1B) {
                    // Discard the rest of the escape sequence
                    while ((code = getchar32()) != (char32_t) WEOF) {
                        if ((code >= 'A' && code <= 'Z') || (code >= 'a' && code <= 'z') || code == '~') {
                            break;
                        }
                    }
                }
            } else if (input_char == 0x08 || input_char == 0x7F) { // Backspace
                if (!widths.empty()) {
                    int count;
                    do {
                        count = widths.back();
                        widths.pop_back();
                        // Move cursor back, print space, and move cursor back again
                        for (int i = 0; i < count; i++) {
                            replace_last(' ');
                            pop_cursor();
                        }
                        pop_back_utf8_char(line);
                    } while (count == 0 && !widths.empty());
                }
            } else {
                int offset = line.length();
                append_utf8(input_char, line);
                int width = put_codepoint(line.c_str() + offset, line.length() - offset, estimateWidth(input_char));
                if (width < 0) {
                    width = 0;
                }
                widths.push_back(width);
            }

            if (!line.empty() && (line.back() == '\\' || line.back() == '/')) {
                set_display(prompt);
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
        set_display(user_input);

        if (simple_io) {
            return readline_simple(line, multiline_input);
        }
        return readline_advanced(line, multiline_input);
    }

}
