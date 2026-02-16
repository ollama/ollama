/*
 * MIT License
 *
 * Copyright (c) 2017 Serge Zaitsev
 * Copyright (c) 2022 Steffen André Langnes
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

/// @file webview.h

#ifndef WEBVIEW_H
#define WEBVIEW_H

/**
 * Used to specify function linkage such as extern, inline, etc.
 *
 * When @c WEBVIEW_API is not already defined, the defaults are as follows:
 *
 * - @c inline when compiling C++ code.
 * - @c extern when compiling C code.
 *
 * The following macros can be used to automatically set an appropriate
 * value for @c WEBVIEW_API:
 *
 * - Define @c WEBVIEW_BUILD_SHARED when building a shared library.
 * - Define @c WEBVIEW_SHARED when using a shared library.
 * - Define @c WEBVIEW_STATIC when building or using a static library.
 */
#ifndef WEBVIEW_API
#if defined(WEBVIEW_SHARED) || defined(WEBVIEW_BUILD_SHARED)
#if defined(_WIN32) || defined(__CYGWIN__)
#if defined(WEBVIEW_BUILD_SHARED)
#define WEBVIEW_API __declspec(dllexport)
#else
#define WEBVIEW_API __declspec(dllimport)
#endif
#else
#define WEBVIEW_API __attribute__((visibility("default")))
#endif
#elif !defined(WEBVIEW_STATIC) && defined(__cplusplus)
#define WEBVIEW_API inline
#else
#define WEBVIEW_API extern
#endif
#endif

/// @name Version
/// @{

#ifndef WEBVIEW_VERSION_MAJOR
/// The current library major version.
#define WEBVIEW_VERSION_MAJOR 0
#endif

#ifndef WEBVIEW_VERSION_MINOR
/// The current library minor version.
#define WEBVIEW_VERSION_MINOR 11
#endif

#ifndef WEBVIEW_VERSION_PATCH
/// The current library patch version.
#define WEBVIEW_VERSION_PATCH 0
#endif

#ifndef WEBVIEW_VERSION_PRE_RELEASE
/// SemVer 2.0.0 pre-release labels prefixed with "-".
#define WEBVIEW_VERSION_PRE_RELEASE ""
#endif

#ifndef WEBVIEW_VERSION_BUILD_METADATA
/// SemVer 2.0.0 build metadata prefixed with "+".
#define WEBVIEW_VERSION_BUILD_METADATA ""
#endif

/// @}

/// @name Used internally
/// @{

/// Utility macro for stringifying a macro argument.
#define WEBVIEW_STRINGIFY(x) #x

/// Utility macro for stringifying the result of a macro argument expansion.
#define WEBVIEW_EXPAND_AND_STRINGIFY(x) WEBVIEW_STRINGIFY(x)

/// @}

/// @name Version
/// @{

/// SemVer 2.0.0 version number in MAJOR.MINOR.PATCH format.
#define WEBVIEW_VERSION_NUMBER                                                 \
  WEBVIEW_EXPAND_AND_STRINGIFY(WEBVIEW_VERSION_MAJOR)                          \
  "." WEBVIEW_EXPAND_AND_STRINGIFY(                                            \
      WEBVIEW_VERSION_MINOR) "." WEBVIEW_EXPAND_AND_STRINGIFY(WEBVIEW_VERSION_PATCH)

/// @}

/// Holds the elements of a MAJOR.MINOR.PATCH version number.
typedef struct {
  /// Major version.
  unsigned int major;
  /// Minor version.
  unsigned int minor;
  /// Patch version.
  unsigned int patch;
} webview_version_t;

/// Holds the library's version information.
typedef struct {
  /// The elements of the version number.
  webview_version_t version;
  /// SemVer 2.0.0 version number in MAJOR.MINOR.PATCH format.
  char version_number[32];
  /// SemVer 2.0.0 pre-release labels prefixed with "-" if specified, otherwise
  /// an empty string.
  char pre_release[48];
  /// SemVer 2.0.0 build metadata prefixed with "+", otherwise an empty string.
  char build_metadata[48];
} webview_version_info_t;

/// Pointer to a webview instance.
typedef void *webview_t;

/// Native handle kind. The actual type depends on the backend.
typedef enum {
  /// Top-level window. @c GtkWindow pointer (GTK), @c NSWindow pointer (Cocoa)
  /// or @c HWND (Win32).
  WEBVIEW_NATIVE_HANDLE_KIND_UI_WINDOW,
  /// Browser widget. @c GtkWidget pointer (GTK), @c NSView pointer (Cocoa) or
  /// @c HWND (Win32).
  WEBVIEW_NATIVE_HANDLE_KIND_UI_WIDGET,
  /// Browser controller. @c WebKitWebView pointer (WebKitGTK), @c WKWebView
  /// pointer (Cocoa/WebKit) or @c ICoreWebView2Controller pointer
  /// (Win32/WebView2).
  WEBVIEW_NATIVE_HANDLE_KIND_BROWSER_CONTROLLER
} webview_native_handle_kind_t;

/// Window size hints
typedef enum {
  /// Width and height are default size.
  WEBVIEW_HINT_NONE,
  /// Width and height are minimum bounds.
  WEBVIEW_HINT_MIN,
  /// Width and height are maximum bounds.
  WEBVIEW_HINT_MAX,
  /// Window size can not be changed by a user.
  WEBVIEW_HINT_FIXED
} webview_hint_t;

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Creates a new webview instance.
 *
 * @param debug Enable developer tools if supported by the backend.
 * @param window Optional native window handle, i.e. @c GtkWindow pointer
 *        @c NSWindow pointer (Cocoa) or @c HWND (Win32). If non-null,
 *        the webview widget is embedded into the given window, and the
 *        caller is expected to assume responsibility for the window as
 *        well as application lifecycle. If the window handle is null,
 *        a new window is created and both the window and application
 *        lifecycle are managed by the webview instance.
 * @remark Win32: The function also accepts a pointer to @c HWND (Win32) in the
 *         window parameter for backward compatibility.
 * @remark Win32/WebView2: @c CoInitializeEx should be called with
 *         @c COINIT_APARTMENTTHREADED before attempting to call this function
 *         with an existing window. Omitting this step may cause WebView2
 *         initialization to fail.
 * @return @c NULL on failure. Creation can fail for various reasons such
 *         as when required runtime dependencies are missing or when window
 *         creation fails.
 */
WEBVIEW_API webview_t webview_create(int debug, void *window);

/**
 * Destroys a webview instance and closes the native window.
 *
 * @param w The webview instance.
 */
WEBVIEW_API void webview_destroy(webview_t w);

/**
 * Runs the main loop until it's terminated.
 *
 * @param w The webview instance.
 */
WEBVIEW_API void webview_run(webview_t w);

/**
 * Stops the main loop. It is safe to call this function from another other
 * background thread.
 *
 * @param w The webview instance.
 */
WEBVIEW_API void webview_terminate(webview_t w);

/**
 * Schedules a function to be invoked on the thread with the run/event loop.
 * Use this function e.g. to interact with the library or native handles.
 *
 * @param w The webview instance.
 * @param fn The function to be invoked.
 * @param arg An optional argument passed along to the callback function.
 */
WEBVIEW_API void
webview_dispatch(webview_t w, void (*fn)(webview_t w, void *arg), void *arg);

/**
 * Returns the native handle of the window associated with the webview instance.
 * The handle can be a @c GtkWindow pointer (GTK), @c NSWindow pointer (Cocoa)
 * or @c HWND (Win32).
 *
 * @param w The webview instance.
 * @return The handle of the native window.
 */
WEBVIEW_API void *webview_get_window(webview_t w);

/**
 * Get a native handle of choice.
 *
 * @param w The webview instance.
 * @param kind The kind of handle to retrieve.
 * @return The native handle or @c NULL.
 * @since 0.11
 */
WEBVIEW_API void *webview_get_native_handle(webview_t w,
                                            webview_native_handle_kind_t kind);

/**
 * Updates the title of the native window.
 *
 * @param w The webview instance.
 * @param title The new title.
 */
WEBVIEW_API void webview_set_title(webview_t w, const char *title);

/**
 * Updates the size of the native window.
 *
 * @param w The webview instance.
 * @param width New width.
 * @param height New height.
 * @param hints Size hints.
 */
WEBVIEW_API void webview_set_size(webview_t w, int width, int height,
                                  webview_hint_t hints);

/**
 * Navigates webview to the given URL. URL may be a properly encoded data URI.
 *
 * Example:
 * @code{.c}
 * webview_navigate(w, "https://github.com/webview/webview");
 * webview_navigate(w, "data:text/html,%3Ch1%3EHello%3C%2Fh1%3E");
 * webview_navigate(w, "data:text/html;base64,PGgxPkhlbGxvPC9oMT4=");
 * @endcode
 *
 * @param w The webview instance.
 * @param url URL.
 */
WEBVIEW_API void webview_navigate(webview_t w, const char *url);

/**
 * Load HTML content into the webview.
 *
 * Example:
 * @code{.c}
 * webview_set_html(w, "<h1>Hello</h1>");
 * @endcode
 *
 * @param w The webview instance.
 * @param html HTML content.
 */
WEBVIEW_API void webview_set_html(webview_t w, const char *html);

/**
 * Injects JavaScript code to be executed immediately upon loading a page.
 * The code will be executed before @c window.onload.
 *
 * @param w The webview instance.
 * @param js JS content.
 */
WEBVIEW_API void webview_init(webview_t w, const char *js);

/**
 * Evaluates arbitrary JavaScript code.
 *
 * Use bindings if you need to communicate the result of the evaluation.
 *
 * @param w The webview instance.
 * @param js JS content.
 */
WEBVIEW_API void webview_eval(webview_t w, const char *js);

/**
 * Binds a function pointer to a new global JavaScript function.
 *
 * Internally, JS glue code is injected to create the JS function by the
 * given name. The callback function is passed a sequential request
 * identifier, a request string and a user-provided argument. The request
 * string is a JSON array of the arguments passed to the JS function.
 *
 * @param w The webview instance.
 * @param name Name of the JS function.
 * @param fn Callback function.
 * @param arg User argument.
 */
WEBVIEW_API void webview_bind(webview_t w, const char *name,
                              void (*fn)(const char *seq, const char *req,
                                         void *arg),
                              void *arg);

/**
 * Removes a binding created with webview_bind().
 *
 * @param w The webview instance.
 * @param name Name of the binding.
 */
WEBVIEW_API void webview_unbind(webview_t w, const char *name);

/**
 * Responds to a binding call from the JS side.
 *
 * @param w The webview instance.
 * @param seq The sequence number of the binding call. Pass along the value
 *            received in the binding handler (see webview_bind()).
 * @param status A status of zero tells the JS side that the binding call was
 *               succesful; any other value indicates an error.
 * @param result The result of the binding call to be returned to the JS side.
 *               This must either be a valid JSON value or an empty string for
 *               the primitive JS value @c undefined.
 */
WEBVIEW_API void webview_return(webview_t w, const char *seq, int status,
                                const char *result);

/**
 * Get the library's version information.
 *
 * @since 0.10
 */
WEBVIEW_API const webview_version_info_t *webview_version(void);

/**
 * Set the zoom level of the webview.
 *
 * @param w The webview instance.
 * @param level The zoom level. 1.0 is normal size, >1.0 zooms in, <1.0 zooms out.
 */
WEBVIEW_API void webview_set_zoom(webview_t w, double level);

/**
 * Get the current zoom level of the webview.
 *
 * @param w The webview instance.
 * @return The current zoom level.
 */
WEBVIEW_API double webview_get_zoom(webview_t w);

// TODO (jmorganca): these forward declarations should be
// in a header file but due to linking issues they live here for now
typedef struct
{
    char *label;
    int enabled;
    int separator;
} menuItem;
int menu_get_item_count();
void *menu_get_items();
void menu_handle_selection(char *item);

#ifdef __cplusplus
}

#ifndef WEBVIEW_HEADER

#if !defined(WEBVIEW_GTK) && !defined(WEBVIEW_COCOA) && !defined(WEBVIEW_EDGE)
#if defined(__APPLE__)
#define WEBVIEW_COCOA
#elif defined(__unix__)
#define WEBVIEW_GTK
#elif defined(_WIN32)
#define WEBVIEW_EDGE
#else
#error "please, specify webview backend"
#endif
#endif

#ifndef WEBVIEW_DEPRECATED
#if __cplusplus >= 201402L
#define WEBVIEW_DEPRECATED(reason) [[deprecated(reason)]]
#elif defined(_MSC_VER)
#define WEBVIEW_DEPRECATED(reason) __declspec(deprecated(reason))
#else
#define WEBVIEW_DEPRECATED(reason) __attribute__((deprecated(reason)))
#endif
#endif

#ifndef WEBVIEW_DEPRECATED_PRIVATE
#define WEBVIEW_DEPRECATED_PRIVATE                                             \
  WEBVIEW_DEPRECATED("Private API should not be used")
#endif

#include <algorithm>
#include <array>
#include <atomic>
#include <cassert>
#include <cstdint>
#include <functional>
#include <future>
#include <map>
#include <string>
#include <utility>
#include <vector>
#include <cstring>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace webview {

using dispatch_fn_t = std::function<void()>;

namespace detail {

// The library's version information.
constexpr const webview_version_info_t library_version_info{
    {WEBVIEW_VERSION_MAJOR, WEBVIEW_VERSION_MINOR, WEBVIEW_VERSION_PATCH},
    WEBVIEW_VERSION_NUMBER,
    WEBVIEW_VERSION_PRE_RELEASE,
    WEBVIEW_VERSION_BUILD_METADATA};

#if defined(_WIN32)
// Converts a narrow (UTF-8-encoded) string into a wide (UTF-16-encoded) string.
inline std::wstring widen_string(const std::string &input) {
  if (input.empty()) {
    return std::wstring();
  }
  UINT cp = CP_UTF8;
  DWORD flags = MB_ERR_INVALID_CHARS;
  auto input_c = input.c_str();
  auto input_length = static_cast<int>(input.size());
  auto required_length =
      MultiByteToWideChar(cp, flags, input_c, input_length, nullptr, 0);
  if (required_length > 0) {
    std::wstring output(static_cast<std::size_t>(required_length), L'\0');
    if (MultiByteToWideChar(cp, flags, input_c, input_length, &output[0],
                            required_length) > 0) {
      return output;
    }
  }
  // Failed to convert string from UTF-8 to UTF-16
  return std::wstring();
}

// Converts a wide (UTF-16-encoded) string into a narrow (UTF-8-encoded) string.
inline std::string narrow_string(const std::wstring &input) {
  struct wc_flags {
    enum TYPE : unsigned int {
      // WC_ERR_INVALID_CHARS
      err_invalid_chars = 0x00000080U
    };
  };
  if (input.empty()) {
    return std::string();
  }
  UINT cp = CP_UTF8;
  DWORD flags = wc_flags::err_invalid_chars;
  auto input_c = input.c_str();
  auto input_length = static_cast<int>(input.size());
  auto required_length = WideCharToMultiByte(cp, flags, input_c, input_length,
                                             nullptr, 0, nullptr, nullptr);
  if (required_length > 0) {
    std::string output(static_cast<std::size_t>(required_length), '\0');
    if (WideCharToMultiByte(cp, flags, input_c, input_length, &output[0],
                            required_length, nullptr, nullptr) > 0) {
      return output;
    }
  }
  // Failed to convert string from UTF-16 to UTF-8
  return std::string();
}
#endif

inline int json_parse_c(const char *s, size_t sz, const char *key, size_t keysz,
                        const char **value, size_t *valuesz) {
  enum {
    JSON_STATE_VALUE,
    JSON_STATE_LITERAL,
    JSON_STATE_STRING,
    JSON_STATE_ESCAPE,
    JSON_STATE_UTF8
  } state = JSON_STATE_VALUE;
  const char *k = nullptr;
  int index = 1;
  int depth = 0;
  int utf8_bytes = 0;

  *value = nullptr;
  *valuesz = 0;

  if (key == nullptr) {
    index = static_cast<decltype(index)>(keysz);
    if (index < 0) {
      return -1;
    }
    keysz = 0;
  }

  for (; sz > 0; s++, sz--) {
    enum {
      JSON_ACTION_NONE,
      JSON_ACTION_START,
      JSON_ACTION_END,
      JSON_ACTION_START_STRUCT,
      JSON_ACTION_END_STRUCT
    } action = JSON_ACTION_NONE;
    auto c = static_cast<unsigned char>(*s);
    switch (state) {
    case JSON_STATE_VALUE:
      if (c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == ',' ||
          c == ':') {
        continue;
      } else if (c == '"') {
        action = JSON_ACTION_START;
        state = JSON_STATE_STRING;
      } else if (c == '{' || c == '[') {
        action = JSON_ACTION_START_STRUCT;
      } else if (c == '}' || c == ']') {
        action = JSON_ACTION_END_STRUCT;
      } else if (c == 't' || c == 'f' || c == 'n' || c == '-' ||
                 (c >= '0' && c <= '9')) {
        action = JSON_ACTION_START;
        state = JSON_STATE_LITERAL;
      } else {
        return -1;
      }
      break;
    case JSON_STATE_LITERAL:
      if (c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == ',' ||
          c == ']' || c == '}' || c == ':') {
        state = JSON_STATE_VALUE;
        s--;
        sz++;
        action = JSON_ACTION_END;
      } else if (c < 32 || c > 126) {
        return -1;
      } // fallthrough
    case JSON_STATE_STRING:
      if (c < 32 || (c > 126 && c < 192)) {
        return -1;
      } else if (c == '"') {
        action = JSON_ACTION_END;
        state = JSON_STATE_VALUE;
      } else if (c == '\\') {
        state = JSON_STATE_ESCAPE;
      } else if (c >= 192 && c < 224) {
        utf8_bytes = 1;
        state = JSON_STATE_UTF8;
      } else if (c >= 224 && c < 240) {
        utf8_bytes = 2;
        state = JSON_STATE_UTF8;
      } else if (c >= 240 && c < 247) {
        utf8_bytes = 3;
        state = JSON_STATE_UTF8;
      } else if (c >= 128 && c < 192) {
        return -1;
      }
      break;
    case JSON_STATE_ESCAPE:
      if (c == '"' || c == '\\' || c == '/' || c == 'b' || c == 'f' ||
          c == 'n' || c == 'r' || c == 't' || c == 'u') {
        state = JSON_STATE_STRING;
      } else {
        return -1;
      }
      break;
    case JSON_STATE_UTF8:
      if (c < 128 || c > 191) {
        return -1;
      }
      utf8_bytes--;
      if (utf8_bytes == 0) {
        state = JSON_STATE_STRING;
      }
      break;
    default:
      return -1;
    }

    if (action == JSON_ACTION_END_STRUCT) {
      depth--;
    }

    if (depth == 1) {
      if (action == JSON_ACTION_START || action == JSON_ACTION_START_STRUCT) {
        if (index == 0) {
          *value = s;
        } else if (keysz > 0 && index == 1) {
          k = s;
        } else {
          index--;
        }
      } else if (action == JSON_ACTION_END ||
                 action == JSON_ACTION_END_STRUCT) {
        if (*value != nullptr && index == 0) {
          *valuesz = (size_t)(s + 1 - *value);
          return 0;
        } else if (keysz > 0 && k != nullptr) {
          if (keysz == (size_t)(s - k - 1) && memcmp(key, k + 1, keysz) == 0) {
            index = 0;
          } else {
            index = 2;
          }
          k = nullptr;
        }
      }
    }

    if (action == JSON_ACTION_START_STRUCT) {
      depth++;
    }
  }
  return -1;
}

constexpr bool is_json_special_char(char c) {
  return c == '"' || c == '\\' || c == '\b' || c == '\f' || c == '\n' ||
         c == '\r' || c == '\t';
}

constexpr bool is_ascii_control_char(char c) { return c >= 0 && c <= 0x1f; }

inline std::string json_escape(const std::string &s, bool add_quotes = true) {
  // Calculate the size of the resulting string.
  // Add space for the double quotes.
  size_t required_length = add_quotes ? 2 : 0;
  for (auto c : s) {
    if (is_json_special_char(c)) {
      // '\' and a single following character
      required_length += 2;
      continue;
    }
    if (is_ascii_control_char(c)) {
      // '\', 'u', 4 digits
      required_length += 6;
      continue;
    }
    ++required_length;
  }
  // Allocate memory for resulting string only once.
  std::string result;
  result.reserve(required_length);
  if (add_quotes) {
    result += '"';
  }
  // Copy string while escaping characters.
  for (auto c : s) {
    if (is_json_special_char(c)) {
      static constexpr char special_escape_table[256] =
          "\0\0\0\0\0\0\0\0btn\0fr\0\0"
          "\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"
          "\0\0\"\0\0\0\0\0\0\0\0\0\0\0\0\0"
          "\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"
          "\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0"
          "\0\0\0\0\0\0\0\0\0\0\0\0\\";
      result += '\\';
      // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-constant-array-index)
      result += special_escape_table[static_cast<unsigned char>(c)];
      continue;
    }
    if (is_ascii_control_char(c)) {
      // Escape as \u00xx
      static constexpr char hex_alphabet[]{"0123456789abcdef"};
      auto uc = static_cast<unsigned char>(c);
      auto h = (uc >> 4) & 0x0f;
      auto l = uc & 0x0f;
      result += "\\u00";
      // NOLINTBEGIN(cppcoreguidelines-pro-bounds-constant-array-index)
      result += hex_alphabet[h];
      result += hex_alphabet[l];
      // NOLINTEND(cppcoreguidelines-pro-bounds-constant-array-index)
      continue;
    }
    result += c;
  }
  if (add_quotes) {
    result += '"';
  }
  // Should have calculated the exact amount of memory needed
  assert(required_length == result.size());
  return result;
}

inline int json_unescape(const char *s, size_t n, char *out) {
  int r = 0;
  if (*s++ != '"') {
    return -1;
  }
  while (n > 2) {
    char c = *s;
    if (c == '\\') {
      s++;
      n--;
      switch (*s) {
      case 'b':
        c = '\b';
        break;
      case 'f':
        c = '\f';
        break;
      case 'n':
        c = '\n';
        break;
      case 'r':
        c = '\r';
        break;
      case 't':
        c = '\t';
        break;
      case '\\':
        c = '\\';
        break;
      case '/':
        c = '/';
        break;
      case '\"':
        c = '\"';
        break;
      default: // TODO: support unicode decoding
        return -1;
      }
    }
    if (out != nullptr) {
      *out++ = c;
    }
    s++;
    n--;
    r++;
  }
  if (*s != '"') {
    return -1;
  }
  if (out != nullptr) {
    *out = '\0';
  }
  return r;
}

inline std::string json_parse(const std::string &s, const std::string &key,
                              const int index) {
  const char *value;
  size_t value_sz;
  if (key.empty()) {
    json_parse_c(s.c_str(), s.length(), nullptr, index, &value, &value_sz);
  } else {
    json_parse_c(s.c_str(), s.length(), key.c_str(), key.length(), &value,
                 &value_sz);
  }
  if (value != nullptr) {
    if (value[0] != '"') {
      return {value, value_sz};
    }
    int n = json_unescape(value, value_sz, nullptr);
    if (n > 0) {
      char *decoded = new char[n + 1];
      json_unescape(value, value_sz, decoded);
      std::string result(decoded, n);
      delete[] decoded;
      return result;
    }
  }
  return "";
}

// Holds a symbol name and associated type for code clarity.
template <typename T> class library_symbol {
public:
  using type = T;

  constexpr explicit library_symbol(const char *name) : m_name(name) {}
  constexpr const char *get_name() const { return m_name; }

private:
  const char *m_name;
};

// Loads a native shared library and allows one to get addresses for those
// symbols.
class native_library {
public:
  native_library() = default;

  explicit native_library(const std::string &name)
      : m_handle{load_library(name)} {}

#ifdef _WIN32
  explicit native_library(const std::wstring &name)
      : m_handle{load_library(name)} {}
#endif

  ~native_library() {
    if (m_handle) {
#ifdef _WIN32
      FreeLibrary(m_handle);
#else
      dlclose(m_handle);
#endif
      m_handle = nullptr;
    }
  }

  native_library(const native_library &other) = delete;
  native_library &operator=(const native_library &other) = delete;
  native_library(native_library &&other) noexcept { *this = std::move(other); }

  native_library &operator=(native_library &&other) noexcept {
    if (this == &other) {
      return *this;
    }
    m_handle = other.m_handle;
    other.m_handle = nullptr;
    return *this;
  }

  // Returns true if the library is currently loaded; otherwise false.
  operator bool() const { return is_loaded(); }

  // Get the address for the specified symbol or nullptr if not found.
  template <typename Symbol>
  typename Symbol::type get(const Symbol &symbol) const {
    if (is_loaded()) {
      // NOLINTBEGIN(cppcoreguidelines-pro-type-reinterpret-cast)
#ifdef _WIN32
#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wcast-function-type"
#endif
      return reinterpret_cast<typename Symbol::type>(
          GetProcAddress(m_handle, symbol.get_name()));
#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif
#else
      return reinterpret_cast<typename Symbol::type>(
          dlsym(m_handle, symbol.get_name()));
#endif
      // NOLINTEND(cppcoreguidelines-pro-type-reinterpret-cast)
    }
    return nullptr;
  }

  // Returns true if the library is currently loaded; otherwise false.
  bool is_loaded() const { return !!m_handle; }

  void detach() { m_handle = nullptr; }

  // Returns true if the library by the given name is currently loaded; otherwise false.
  static inline bool is_loaded(const std::string &name) {
#ifdef _WIN32
    auto handle = GetModuleHandleW(widen_string(name).c_str());
#else
    auto handle = dlopen(name.c_str(), RTLD_NOW | RTLD_NOLOAD);
    if (handle) {
      dlclose(handle);
    }
#endif
    return !!handle;
  }

private:
#ifdef _WIN32
  using mod_handle_t = HMODULE;
#else
  using mod_handle_t = void *;
#endif

  static inline mod_handle_t load_library(const std::string &name) {
#ifdef _WIN32
    return load_library(widen_string(name));
#else
    return dlopen(name.c_str(), RTLD_NOW);
#endif
  }

#ifdef _WIN32
  static inline mod_handle_t load_library(const std::wstring &name) {
    return LoadLibraryW(name.c_str());
  }
#endif

  mod_handle_t m_handle{};
};

class engine_base {
public:
  virtual ~engine_base() = default;

  void navigate(const std::string &url) {
    if (url.empty()) {
      navigate_impl("about:blank");
      return;
    }
    navigate_impl(url);
  }

  using binding_t = std::function<void(std::string, std::string, void *)>;
  class binding_ctx_t {
  public:
    binding_ctx_t(binding_t callback, void *arg)
        : callback(callback), arg(arg) {}
    // This function is called upon execution of the bound JS function
    binding_t callback;
    // This user-supplied argument is passed to the callback
    void *arg;
  };

  using sync_binding_t = std::function<std::string(std::string)>;

  // Synchronous bind
  void bind(const std::string &name, sync_binding_t fn) {
    auto wrapper = [this, fn](const std::string &seq, const std::string &req,
                              void * /*arg*/) { resolve(seq, 0, fn(req)); };
    bind(name, wrapper, nullptr);
  }

  // Asynchronous bind
  void bind(const std::string &name, binding_t fn, void *arg) {
    // NOLINTNEXTLINE(readability-container-contains): contains() requires C++20
    if (bindings.count(name) > 0) {
      return;
    }
    bindings.emplace(name, binding_ctx_t(fn, arg));
    auto js = "(function() { var name = '" + name + "';" + R""(
      var RPC = window._rpc = (window._rpc || {nextSeq: 1});
      window[name] = function() {
        var seq = RPC.nextSeq++;
        var promise = new Promise(function(resolve, reject) {
          RPC[seq] = {
            resolve: resolve,
            reject: reject,
          };
        });
        window.external.invoke(JSON.stringify({
          id: seq,
          method: name,
          params: Array.prototype.slice.call(arguments),
        }));
        return promise;
      }
    })())"";
    init(js);
    eval(js);
  }

  void unbind(const std::string &name) {
    auto found = bindings.find(name);
    if (found != bindings.end()) {
      auto js = "delete window['" + name + "'];";
      init(js);
      eval(js);
      bindings.erase(found);
    }
  }

  void resolve(const std::string &seq, int status, const std::string &result) {
    // NOLINTNEXTLINE(modernize-avoid-bind): Lambda with move requires C++14
    dispatch(std::bind(
        [seq, status, this](std::string escaped_result) {
          std::string js;
          js += "(function(){var seq = \"";
          js += seq;
          js += "\";\n";
          js += "var status = ";
          js += std::to_string(status);
          js += ";\n";
          js += "var result = ";
          js += escaped_result;
          js += ";\
var promise = window._rpc[seq];\
delete window._rpc[seq];\
if (result !== undefined) {\
  try {\
    result = JSON.parse(result);\
  } catch {\
    promise.reject(new Error(\"Failed to parse binding result as JSON\"));\
    return;\
  }\
}\
if (status === 0) {\
  promise.resolve(result);\
} else {\
  promise.reject(result);\
}\
})()";
          eval(js);
        },
        result.empty() ? "undefined" : json_escape(result)));
  }

  void *window() { return window_impl(); }
  void *widget() { return widget_impl(); }
  void *browser_controller() { return browser_controller_impl(); };
  void run() { run_impl(); }
  void terminate() { terminate_impl(); }
  void dispatch(std::function<void()> f) { dispatch_impl(f); }
  void set_title(const std::string &title) { set_title_impl(title); }

  void set_size(int width, int height, webview_hint_t hints) {
    set_size_impl(width, height, hints);
  }

  void set_html(const std::string &html) { set_html_impl(html); }
  void init(const std::string &js) { init_impl(js); }
  void eval(const std::string &js) { eval_impl(js); }
  
  void set_zoom(double level) { 
    const double minZoom = 0.9;  // 90%
    const double maxZoom = 1.2;  // 120%
    level = std::max(minZoom, std::min(maxZoom, level));
    set_zoom_impl(level); 
  }
  double get_zoom() { return get_zoom_impl(); }

protected:
  virtual void navigate_impl(const std::string &url) = 0;
  virtual void *window_impl() = 0;
  virtual void *widget_impl() = 0;
  virtual void *browser_controller_impl() = 0;
  virtual void run_impl() = 0;
  virtual void terminate_impl() = 0;
  virtual void dispatch_impl(std::function<void()> f) = 0;
  virtual void set_title_impl(const std::string &title) = 0;
  virtual void set_size_impl(int width, int height, webview_hint_t hints) = 0;
  virtual void set_html_impl(const std::string &html) = 0;
  virtual void init_impl(const std::string &js) = 0;
  virtual void eval_impl(const std::string &js) = 0;
  virtual void set_zoom_impl(double level) = 0;
  virtual double get_zoom_impl() = 0;

  virtual void on_message(const std::string &msg) {
    auto seq = json_parse(msg, "id", 0);
    auto name = json_parse(msg, "method", 0);
    auto args = json_parse(msg, "params", 0);
    auto found = bindings.find(name);
    if (found == bindings.end()) {
      return;
    }
    const auto &context = found->second;
    context.callback(seq, args, context.arg);
  }

  virtual void on_window_created() { inc_window_count(); }

  virtual void on_window_destroyed(bool skip_termination = false) {
    if (dec_window_count() <= 0) {
      if (!skip_termination) {
        terminate();
      }
    }
  }

private:
  static std::atomic_uint &window_ref_count() {
    static std::atomic_uint ref_count{0};
    return ref_count;
  }

  static unsigned int inc_window_count() { return ++window_ref_count(); }

  static unsigned int dec_window_count() {
    auto &count = window_ref_count();
    if (count > 0) {
      return --count;
    }
    return 0;
  }

  std::map<std::string, binding_ctx_t> bindings;
};

} // namespace detail

WEBVIEW_DEPRECATED_PRIVATE
inline int json_parse_c(const char *s, size_t sz, const char *key, size_t keysz,
                        const char **value, size_t *valuesz) {
  return detail::json_parse_c(s, sz, key, keysz, value, valuesz);
}

WEBVIEW_DEPRECATED_PRIVATE
inline std::string json_escape(const std::string &s) {
  return detail::json_escape(s);
}

WEBVIEW_DEPRECATED_PRIVATE
inline int json_unescape(const char *s, size_t n, char *out) {
  return detail::json_unescape(s, n, out);
}

WEBVIEW_DEPRECATED_PRIVATE
inline std::string json_parse(const std::string &s, const std::string &key,
                              const int index) {
  return detail::json_parse(s, key, index);
}

} // namespace webview

#if defined(WEBVIEW_GTK)
//
// ====================================================================
//
// This implementation uses webkit2gtk backend. It requires gtk+3.0 and
// webkit2gtk-4.0 libraries. Proper compiler flags can be retrieved via:
//
//   pkg-config --cflags --libs gtk+-3.0 webkit2gtk-4.0
//
// ====================================================================
//
#include <cstdlib>

#include <JavaScriptCore/JavaScript.h>
#include <gtk/gtk.h>
#include <webkit2/webkit2.h>

#ifdef GDK_WINDOWING_X11
#include <gdk/gdkx.h>
#endif

#include <fcntl.h>
#include <sys/stat.h>

namespace webview {
namespace detail {

// Namespace containing workaround for WebKit 2.42 when using NVIDIA GPU
// driver.
// See WebKit bug: https://bugs.webkit.org/show_bug.cgi?id=261874
// Please remove all of the code in this namespace when it's no longer needed.
namespace webkit_dmabuf {

// Get environment variable. Not thread-safe.
static inline std::string get_env(const std::string &name) {
  auto *value = std::getenv(name.c_str());
  if (value) {
    return {value};
  }
  return {};
}

// Set environment variable. Not thread-safe.
static inline void set_env(const std::string &name, const std::string &value) {
  ::setenv(name.c_str(), value.c_str(), 1);
}

// Checks whether the NVIDIA GPU driver is used based on whether the kernel
// module is loaded.
static inline bool is_using_nvidia_driver() {
  struct ::stat buffer {};
  if (::stat("/sys/module/nvidia", &buffer) != 0) {
    return false;
  }
  return S_ISDIR(buffer.st_mode);
}

// Checks whether the windowing system is Wayland.
static inline bool is_wayland_display() {
  if (!get_env("WAYLAND_DISPLAY").empty()) {
    return true;
  }
  if (get_env("XDG_SESSION_TYPE") == "wayland") {
    return true;
  }
  if (get_env("DESKTOP_SESSION").find("wayland") != std::string::npos) {
    return true;
  }
  return false;
}

// Checks whether the GDK X11 backend is used.
// See: https://docs.gtk.org/gdk3/class.DisplayManager.html
static inline bool is_gdk_x11_backend() {
#ifdef GDK_WINDOWING_X11
  auto *manager = gdk_display_manager_get();
  auto *display = gdk_display_manager_get_default_display(manager);
  return GDK_IS_X11_DISPLAY(display); // NOLINT(misc-const-correctness)
#else
  return false;
#endif
}

// Checks whether WebKit is affected by bug when using DMA-BUF renderer.
// Returns true if all of the following conditions are met:
//  - WebKit version is >= 2.42 (please narrow this down when there's a fix).
//  - Environment variables are empty or not set:
//    - WEBKIT_DISABLE_DMABUF_RENDERER
//  - Windowing system is not Wayland.
//  - GDK backend is X11.
//  - NVIDIA GPU driver is used.
static inline bool is_webkit_dmabuf_bugged() {
  auto wk_major = webkit_get_major_version();
  auto wk_minor = webkit_get_minor_version();
  // TODO: Narrow down affected WebKit version when there's a fixed version
  auto is_affected_wk_version = wk_major == 2 && wk_minor >= 42;
  if (!is_affected_wk_version) {
    return false;
  }
  if (!get_env("WEBKIT_DISABLE_DMABUF_RENDERER").empty()) {
    return false;
  }
  if (is_wayland_display()) {
    return false;
  }
  if (!is_gdk_x11_backend()) {
    return false;
  }
  if (!is_using_nvidia_driver()) {
    return false;
  }
  return true;
}

// Applies workaround for WebKit DMA-BUF bug if needed.
// See WebKit bug: https://bugs.webkit.org/show_bug.cgi?id=261874
static inline void apply_webkit_dmabuf_workaround() {
  if (!is_webkit_dmabuf_bugged()) {
    return;
  }
  set_env("WEBKIT_DISABLE_DMABUF_RENDERER", "1");
}
} // namespace webkit_dmabuf

namespace webkit_symbols {
using webkit_web_view_evaluate_javascript_t =
    void (*)(WebKitWebView *, const char *, gssize, const char *, const char *,
             GCancellable *, GAsyncReadyCallback, gpointer);

using webkit_web_view_run_javascript_t = void (*)(WebKitWebView *,
                                                  const gchar *, GCancellable *,
                                                  GAsyncReadyCallback,
                                                  gpointer);

constexpr auto webkit_web_view_evaluate_javascript =
    library_symbol<webkit_web_view_evaluate_javascript_t>(
        "webkit_web_view_evaluate_javascript");
constexpr auto webkit_web_view_run_javascript =
    library_symbol<webkit_web_view_run_javascript_t>(
        "webkit_web_view_run_javascript");
} // namespace webkit_symbols

class gtk_webkit_engine : public engine_base {
public:
  gtk_webkit_engine(bool debug, void *window)
      : m_owns_window{!window}, m_window(static_cast<GtkWidget *>(window)) {
    if (m_owns_window) {
      if (gtk_init_check(nullptr, nullptr) == FALSE) {
        return;
      }
      m_window = gtk_window_new(GTK_WINDOW_TOPLEVEL);
      on_window_created();
      g_signal_connect(G_OBJECT(m_window), "destroy",
                       G_CALLBACK(+[](GtkWidget *, gpointer arg) {
                         auto *w = static_cast<gtk_webkit_engine *>(arg);
                         // Widget destroyed along with window.
                         w->m_webview = nullptr;
                         w->m_window = nullptr;
                         w->on_window_destroyed();
                       }),
                       this);
    }
    webkit_dmabuf::apply_webkit_dmabuf_workaround();
    // Initialize webview widget
    m_webview = webkit_web_view_new();
    WebKitUserContentManager *manager =
        webkit_web_view_get_user_content_manager(WEBKIT_WEB_VIEW(m_webview));
    g_signal_connect(manager, "script-message-received::external",
                     G_CALLBACK(+[](WebKitUserContentManager *,
                                    WebKitJavascriptResult *r, gpointer arg) {
                       auto *w = static_cast<gtk_webkit_engine *>(arg);
                       char *s = get_string_from_js_result(r);
                       w->on_message(s);
                       g_free(s);
                     }),
                     this);
    webkit_user_content_manager_register_script_message_handler(manager,
                                                                "external");
    init("window.external={invoke:function(s){window.webkit.messageHandlers."
         "external.postMessage(s);}}");

    gtk_container_add(GTK_CONTAINER(m_window), GTK_WIDGET(m_webview));
    gtk_widget_show(GTK_WIDGET(m_webview));

    WebKitSettings *settings =
        webkit_web_view_get_settings(WEBKIT_WEB_VIEW(m_webview));
    webkit_settings_set_javascript_can_access_clipboard(settings, true);
    if (debug) {
      webkit_settings_set_enable_write_console_messages_to_stdout(settings,
                                                                  true);
      webkit_settings_set_enable_developer_extras(settings, true);
    }

    if (m_owns_window) {
      gtk_widget_grab_focus(GTK_WIDGET(m_webview));
      gtk_widget_show_all(m_window);
    }
  }

  gtk_webkit_engine(const gtk_webkit_engine &) = delete;
  gtk_webkit_engine &operator=(const gtk_webkit_engine &) = delete;
  gtk_webkit_engine(gtk_webkit_engine &&) = delete;
  gtk_webkit_engine &operator=(gtk_webkit_engine &&) = delete;

  virtual ~gtk_webkit_engine() {
    if (m_webview) {
      gtk_widget_destroy(GTK_WIDGET(m_webview));
      m_webview = nullptr;
    }
    if (m_window) {
      if (m_owns_window) {
        // Disconnect handlers to avoid callbacks invoked during destruction.
        g_signal_handlers_disconnect_by_data(GTK_WINDOW(m_window), this);
        gtk_window_close(GTK_WINDOW(m_window));
        on_window_destroyed(true);
      }
      m_window = nullptr;
    }
    if (m_owns_window) {
      // Needed for the window to close immediately.
      deplete_run_loop_event_queue();
    }
  }

  void *window_impl() override { return (void *)m_window; }
  void *widget_impl() override { return (void *)m_webview; }
  void *browser_controller_impl() override { return (void *)m_webview; };
  void run_impl() override { gtk_main(); }
  void terminate_impl() override {
    dispatch_impl([] { gtk_main_quit(); });
  }
  void dispatch_impl(std::function<void()> f) override {
    g_idle_add_full(G_PRIORITY_HIGH_IDLE, (GSourceFunc)([](void *f) -> int {
                      (*static_cast<dispatch_fn_t *>(f))();
                      return G_SOURCE_REMOVE;
                    }),
                    new std::function<void()>(f),
                    [](void *f) { delete static_cast<dispatch_fn_t *>(f); });
  }

  void set_title_impl(const std::string &title) override {
    gtk_window_set_title(GTK_WINDOW(m_window), title.c_str());
  }

  void set_size_impl(int width, int height, webview_hint_t hints) override {
    gtk_window_set_resizable(GTK_WINDOW(m_window), hints != WEBVIEW_HINT_FIXED);
    if (hints == WEBVIEW_HINT_NONE) {
      gtk_window_resize(GTK_WINDOW(m_window), width, height);
    } else if (hints == WEBVIEW_HINT_FIXED) {
      gtk_widget_set_size_request(m_window, width, height);
    } else {
      GdkGeometry g;
      g.min_width = g.max_width = width;
      g.min_height = g.max_height = height;
      GdkWindowHints h =
          (hints == WEBVIEW_HINT_MIN ? GDK_HINT_MIN_SIZE : GDK_HINT_MAX_SIZE);
      // This defines either MIN_SIZE, or MAX_SIZE, but not both:
      gtk_window_set_geometry_hints(GTK_WINDOW(m_window), nullptr, &g, h);
    }
  }

  void navigate_impl(const std::string &url) override {
    webkit_web_view_load_uri(WEBKIT_WEB_VIEW(m_webview), url.c_str());
  }

  void set_html_impl(const std::string &html) override {
    webkit_web_view_load_html(WEBKIT_WEB_VIEW(m_webview), html.c_str(),
                              nullptr);
  }

  void init_impl(const std::string &js) override {
    WebKitUserContentManager *manager =
        webkit_web_view_get_user_content_manager(WEBKIT_WEB_VIEW(m_webview));
    webkit_user_content_manager_add_script(
        manager,
        webkit_user_script_new(js.c_str(), WEBKIT_USER_CONTENT_INJECT_TOP_FRAME,
                               WEBKIT_USER_SCRIPT_INJECT_AT_DOCUMENT_START,
                               nullptr, nullptr));
  }

  void eval_impl(const std::string &js) override {
    auto &lib = get_webkit_library();
    auto wkmajor = webkit_get_major_version();
    auto wkminor = webkit_get_minor_version();
    if ((wkmajor == 2 && wkminor >= 40) || wkmajor > 2) {
      if (auto fn =
              lib.get(webkit_symbols::webkit_web_view_evaluate_javascript)) {
        fn(WEBKIT_WEB_VIEW(m_webview), js.c_str(),
           static_cast<gssize>(js.size()), nullptr, nullptr, nullptr, nullptr,
           nullptr);
      }
    } else if (auto fn =
                   lib.get(webkit_symbols::webkit_web_view_run_javascript)) {
      fn(WEBKIT_WEB_VIEW(m_webview), js.c_str(), nullptr, nullptr, nullptr);
    }
  }

  void set_zoom_impl(double level) override {
    webkit_web_view_set_zoom_level(WEBKIT_WEB_VIEW(m_webview), level);
  }

  double get_zoom_impl() override {
    return webkit_web_view_get_zoom_level(WEBKIT_WEB_VIEW(m_webview));
  }

private:
  static char *get_string_from_js_result(WebKitJavascriptResult *r) {
    char *s;
#if (WEBKIT_MAJOR_VERSION == 2 && WEBKIT_MINOR_VERSION >= 22) ||               \
    WEBKIT_MAJOR_VERSION > 2
    JSCValue *value = webkit_javascript_result_get_js_value(r);
    s = jsc_value_to_string(value);
#else
    JSGlobalContextRef ctx = webkit_javascript_result_get_global_context(r);
    JSValueRef value = webkit_javascript_result_get_value(r);
    JSStringRef js = JSValueToStringCopy(ctx, value, nullptr);
    size_t n = JSStringGetMaximumUTF8CStringSize(js);
    s = g_new(char, n);
    JSStringGetUTF8CString(js, s, n);
    JSStringRelease(js);
#endif
    return s;
  }

  static const native_library &get_webkit_library() {
    static const native_library non_loaded_lib;
    static native_library loaded_lib;

    if (loaded_lib.is_loaded()) {
      return loaded_lib;
    }

    constexpr std::array<const char *, 2> lib_names{"libwebkit2gtk-4.1.so",
                                                    "libwebkit2gtk-4.0.so"};
    auto found =
        std::find_if(lib_names.begin(), lib_names.end(), [](const char *name) {
          return native_library::is_loaded(name);
        });

    if (found == lib_names.end()) {
      return non_loaded_lib;
    }

    loaded_lib = native_library(*found);

    auto loaded = loaded_lib.is_loaded();
    if (!loaded) {
      return non_loaded_lib;
    }

    return loaded_lib;
  }

  // Blocks while depleting the run loop of events.
  void deplete_run_loop_event_queue() {
    bool done{};
    dispatch([&] { done = true; });
    while (!done) {
      gtk_main_iteration();
    }
  }

  bool m_owns_window{};
  GtkWidget *m_window{};
  GtkWidget *m_webview{};
};

} // namespace detail

using browser_engine = detail::gtk_webkit_engine;

} // namespace webview

#elif defined(WEBVIEW_COCOA)

//
// ====================================================================
//
// This implementation uses Cocoa WKWebView backend on macOS. It is
// written using ObjC runtime and uses WKWebView class as a browser runtime.
// You should pass "-framework Webkit" flag to the compiler.
//
// ====================================================================
//

#include <CoreGraphics/CoreGraphics.h>
#include <objc/NSObjCRuntime.h>
#include <objc/objc-runtime.h>

namespace webview {
namespace detail {
namespace objc {

// A convenient template function for unconditionally casting the specified
// C-like function into a function that can be called with the given return
// type and arguments. Caller takes full responsibility for ensuring that
// the function call is valid. It is assumed that the function will not
// throw exceptions.
template <typename Result, typename Callable, typename... Args>
Result invoke(Callable callable, Args... args) noexcept {
  return reinterpret_cast<Result (*)(Args...)>(callable)(args...);
}

// Calls objc_msgSend.
template <typename Result, typename... Args>
Result msg_send(Args... args) noexcept {
  return invoke<Result>(objc_msgSend, args...);
}

// Wrapper around NSAutoreleasePool that drains the pool on destruction.
class autoreleasepool {
public:
  autoreleasepool()
      : m_pool(msg_send<id>(objc_getClass("NSAutoreleasePool"),
                            sel_registerName("new"))) {}

  ~autoreleasepool() {
    if (m_pool) {
      msg_send<void>(m_pool, sel_registerName("drain"));
    }
  }

  autoreleasepool(const autoreleasepool &) = delete;
  autoreleasepool &operator=(const autoreleasepool &) = delete;
  autoreleasepool(autoreleasepool &&) = delete;
  autoreleasepool &operator=(autoreleasepool &&) = delete;

private:
  id m_pool{};
};

inline id autoreleased(id object) {
  msg_send<void>(object, sel_registerName("autorelease"));
  return object;
}

} // namespace objc

enum NSBackingStoreType : NSUInteger { NSBackingStoreBuffered = 2 };

enum NSWindowStyleMask : NSUInteger {
  NSWindowStyleMaskTitled = 1,
  NSWindowStyleMaskClosable = 2,
  NSWindowStyleMaskMiniaturizable = 4,
  NSWindowStyleMaskResizable = 8
};

enum NSApplicationActivationPolicy : NSInteger {
  NSApplicationActivationPolicyRegular = 0
};

enum WKUserScriptInjectionTime : NSInteger {
  WKUserScriptInjectionTimeAtDocumentStart = 0
};

enum NSModalResponse : NSInteger { NSModalResponseOK = 1 };

// Convenient conversion of string literals.
inline id operator"" _cls(const char *s, std::size_t) {
  return (id)objc_getClass(s);
}
inline SEL operator"" _sel(const char *s, std::size_t) {
  return sel_registerName(s);
}
inline id operator"" _str(const char *s, std::size_t) {
  return objc::msg_send<id>("NSString"_cls, "stringWithUTF8String:"_sel, s);
}

class cocoa_wkwebview_engine : public engine_base {
public:
  cocoa_wkwebview_engine(bool debug, void *window)
      : m_debug{debug}, m_window{static_cast<id>(window)}, m_owns_window{
                                                               !window} {
    auto app = get_shared_application();
    // See comments related to application lifecycle in create_app_delegate().
    if (!m_owns_window) {
      set_up_window();
    } else {
      // Only set the app delegate if it hasn't already been set.
      auto delegate = objc::msg_send<id>(app, "delegate"_sel);
      if (delegate) {
        set_up_window();
      } else {
        m_app_delegate = create_app_delegate();
        objc_setAssociatedObject(m_app_delegate, "webview", (id)this,
                                 OBJC_ASSOCIATION_ASSIGN);
        objc::msg_send<void>(app, "setDelegate:"_sel, m_app_delegate);

        // Start the main run loop so that the app delegate gets the
        // NSApplicationDidFinishLaunchingNotification notification after the run
        // loop has started in order to perform further initialization.
        // We need to return from this constructor so this run loop is only
        // temporary.
        // Skip the main loop if this isn't the first instance of this class
        // because the launch event is only sent once. Instead, proceed to
        // create a window.
        if (get_and_set_is_first_instance()) {
          objc::msg_send<void>(app, "run"_sel);
        } else {
          set_up_window();
        }
      }
    }
  }

  cocoa_wkwebview_engine(const cocoa_wkwebview_engine &) = delete;
  cocoa_wkwebview_engine &operator=(const cocoa_wkwebview_engine &) = delete;
  cocoa_wkwebview_engine(cocoa_wkwebview_engine &&) = delete;
  cocoa_wkwebview_engine &operator=(cocoa_wkwebview_engine &&) = delete;

  virtual ~cocoa_wkwebview_engine() {
    objc::autoreleasepool arp;
    if (m_window) {
      if (m_webview) {
        if (m_webview == objc::msg_send<id>(m_window, "contentView"_sel)) {
          objc::msg_send<void>(m_window, "setContentView:"_sel, nullptr);
        }
        objc::msg_send<void>(m_webview, "release"_sel);
        m_webview = nullptr;
      }
      if (m_owns_window) {
        // Replace delegate to avoid callbacks and other bad things during
        // destruction.
        objc::msg_send<void>(m_window, "setDelegate:"_sel, nullptr);
        objc::msg_send<void>(m_window, "close"_sel);
        on_window_destroyed(true);
      }
      m_window = nullptr;
    }
    if (m_window_delegate) {
      objc::msg_send<void>(m_window_delegate, "release"_sel);
      m_window_delegate = nullptr;
    }
    if (m_app_delegate) {
      auto app = get_shared_application();
      objc::msg_send<void>(app, "setDelegate:"_sel, nullptr);
      // Make sure to release the delegate we created.
      objc::msg_send<void>(m_app_delegate, "release"_sel);
      m_app_delegate = nullptr;
    }
    if (m_owns_window) {
      // Needed for the window to close immediately.
      deplete_run_loop_event_queue();
    }
    // TODO: Figure out why m_manager is still alive after the autoreleasepool
    // has been drained.
  }

  void *window_impl() override { return (void *)m_window; }
  void *widget_impl() override { return (void *)m_webview; }
  void *browser_controller_impl() override { return (void *)m_webview; };
  void terminate_impl() override { stop_run_loop(); }
  void run_impl() override {
    auto app = get_shared_application();
    objc::msg_send<void>(app, "run"_sel);
  }
  void dispatch_impl(std::function<void()> f) override {
    dispatch_async_f(dispatch_get_main_queue(), new dispatch_fn_t(f),
                     (dispatch_function_t)([](void *arg) {
                       auto f = static_cast<dispatch_fn_t *>(arg);
                       (*f)();
                       delete f;
                     }));
  }
  void set_title_impl(const std::string &title) override {
    objc::autoreleasepool arp;

    objc::msg_send<void>(m_window, "setTitle:"_sel,
                         objc::msg_send<id>("NSString"_cls,
                                            "stringWithUTF8String:"_sel,
                                            title.c_str()));
  }
  void set_size_impl(int width, int height, webview_hint_t hints) override {
    objc::autoreleasepool arp;

    auto style = static_cast<NSWindowStyleMask>(
        NSWindowStyleMaskTitled | NSWindowStyleMaskClosable |
        NSWindowStyleMaskMiniaturizable);
    if (hints != WEBVIEW_HINT_FIXED) {
      style =
          static_cast<NSWindowStyleMask>(style | NSWindowStyleMaskResizable);
    }
    objc::msg_send<void>(m_window, "setStyleMask:"_sel, style);

    if (hints == WEBVIEW_HINT_MIN) {
      objc::msg_send<void>(m_window, "setContentMinSize:"_sel,
                           CGSizeMake(width, height));
    } else if (hints == WEBVIEW_HINT_MAX) {
      objc::msg_send<void>(m_window, "setContentMaxSize:"_sel,
                           CGSizeMake(width, height));
    } else {
      objc::msg_send<void>(m_window, "setFrame:display:animate:"_sel,
                           CGRectMake(0, 0, width, height), YES, NO);
    }
    objc::msg_send<void>(m_window, "center"_sel);
  }
  void navigate_impl(const std::string &url) override {
    objc::autoreleasepool arp;

    auto nsurl = objc::msg_send<id>(
        "NSURL"_cls, "URLWithString:"_sel,
        objc::msg_send<id>("NSString"_cls, "stringWithUTF8String:"_sel,
                           url.c_str()));

    objc::msg_send<void>(
        m_webview, "loadRequest:"_sel,
        objc::msg_send<id>("NSURLRequest"_cls, "requestWithURL:"_sel, nsurl));
  }
  void set_html_impl(const std::string &html) override {
    objc::autoreleasepool arp;
    objc::msg_send<void>(m_webview, "loadHTMLString:baseURL:"_sel,
                         objc::msg_send<id>("NSString"_cls,
                                            "stringWithUTF8String:"_sel,
                                            html.c_str()),
                         nullptr);
  }
  void init_impl(const std::string &js) override {
    objc::autoreleasepool arp;
    auto script = objc::autoreleased(objc::msg_send<id>(
        objc::msg_send<id>("WKUserScript"_cls, "alloc"_sel),
        "initWithSource:injectionTime:forMainFrameOnly:"_sel,
        objc::msg_send<id>("NSString"_cls, "stringWithUTF8String:"_sel,
                           js.c_str()),
        WKUserScriptInjectionTimeAtDocumentStart, YES));
    objc::msg_send<void>(m_manager, "addUserScript:"_sel, script);
  }
  void eval_impl(const std::string &js) override {
    objc::autoreleasepool arp;
    objc::msg_send<void>(m_webview, "evaluateJavaScript:completionHandler:"_sel,
                         objc::msg_send<id>("NSString"_cls,
                                            "stringWithUTF8String:"_sel,
                                            js.c_str()),
                         nullptr);
  }

  void set_zoom_impl(double level) override {
    objc::autoreleasepool arp;
    objc::msg_send<void>(m_webview, "setPageZoom:"_sel, level);
  }

  double get_zoom_impl() override {
    objc::autoreleasepool arp;
    return objc::msg_send<double>(m_webview, "pageZoom"_sel);
  }

private:
  id create_app_delegate() {
    objc::autoreleasepool arp;
    constexpr auto class_name = "WebviewAppDelegate";
    // Avoid crash due to registering same class twice
    auto cls = objc_lookUpClass(class_name);
    if (!cls) {
      // Note: Avoid registering the class name "AppDelegate" as it is the
      // default name in projects created with Xcode, and using the same name
      // causes objc_registerClassPair to crash.
      cls = objc_allocateClassPair((Class) "NSResponder"_cls, class_name, 0);
      class_addProtocol(cls, objc_getProtocol("NSTouchBarProvider"));
      class_addMethod(cls,
                      "applicationShouldTerminateAfterLastWindowClosed:"_sel,
                      (IMP)(+[](id, SEL, id) -> BOOL { return NO; }), "c@:@");
      class_addMethod(cls, "applicationDidFinishLaunching:"_sel,
                      (IMP)(+[](id self, SEL, id notification) {
                        auto app =
                            objc::msg_send<id>(notification, "object"_sel);
                        auto w = get_associated_webview(self);
                        w->on_application_did_finish_launching(self, app);
                      }),
                      "v@:@");
      objc_registerClassPair(cls);
    }
    return objc::msg_send<id>((id)cls, "new"_sel);
  }
  id create_script_message_handler() {
    objc::autoreleasepool arp;
    constexpr auto class_name = "WebviewWKScriptMessageHandler";
    // Avoid crash due to registering same class twice
    auto cls = objc_lookUpClass(class_name);
    if (!cls) {
      cls = objc_allocateClassPair((Class) "NSResponder"_cls, class_name, 0);
      class_addProtocol(cls, objc_getProtocol("WKScriptMessageHandler"));
      class_addMethod(
          cls, "userContentController:didReceiveScriptMessage:"_sel,
          (IMP)(+[](id self, SEL, id, id msg) {
            auto w = get_associated_webview(self);
            w->on_message(objc::msg_send<const char *>(
                objc::msg_send<id>(msg, "body"_sel), "UTF8String"_sel));
          }),
          "v@:@@");
      objc_registerClassPair(cls);
    }
    auto instance = objc::msg_send<id>((id)cls, "new"_sel);
    objc_setAssociatedObject(instance, "webview", (id)this,
                             OBJC_ASSOCIATION_ASSIGN);
    return instance;
  }
  static id create_webkit_ui_delegate() {
    objc::autoreleasepool arp;
    constexpr auto class_name = "WebviewWKUIDelegate";
    // Avoid crash due to registering same class twice
    auto cls = objc_lookUpClass(class_name);
    if (!cls) {
      cls = objc_allocateClassPair((Class) "NSObject"_cls, class_name, 0);
      class_addProtocol(cls, objc_getProtocol("WKUIDelegate"));
      class_addMethod(
          cls,
          "webView:runOpenPanelWithParameters:initiatedByFrame:completionHandler:"_sel,
          (IMP)(+[](id, SEL, id, id parameters, id, id completion_handler) {
            auto allows_multiple_selection =
                objc::msg_send<BOOL>(parameters, "allowsMultipleSelection"_sel);
            auto allows_directories =
                objc::msg_send<BOOL>(parameters, "allowsDirectories"_sel);

            // Show a panel for selecting files.
            auto panel = objc::msg_send<id>("NSOpenPanel"_cls, "openPanel"_sel);
            objc::msg_send<void>(panel, "setCanChooseFiles:"_sel, YES);
            objc::msg_send<void>(panel, "setCanChooseDirectories:"_sel,
                                 allows_directories);
            objc::msg_send<void>(panel, "setAllowsMultipleSelection:"_sel,
                                 allows_multiple_selection);
            auto modal_response =
                objc::msg_send<NSModalResponse>(panel, "runModal"_sel);

            // Get the URLs for the selected files. If the modal was canceled
            // then we pass null to the completion handler to signify
            // cancellation.
            id urls = modal_response == NSModalResponseOK
                          ? objc::msg_send<id>(panel, "URLs"_sel)
                          : nullptr;

            // Invoke the completion handler block.
            auto sig = objc::msg_send<id>(
                "NSMethodSignature"_cls, "signatureWithObjCTypes:"_sel, "v@?@");
            auto invocation = objc::msg_send<id>(
                "NSInvocation"_cls, "invocationWithMethodSignature:"_sel, sig);
            objc::msg_send<void>(invocation, "setTarget:"_sel,
                                 completion_handler);
            objc::msg_send<void>(invocation, "setArgument:atIndex:"_sel, &urls,
                                 1);
            objc::msg_send<void>(invocation, "invoke"_sel);
          }),
          "v@:@@@@");
      objc_registerClassPair(cls);
    }
    return objc::msg_send<id>((id)cls, "new"_sel);
  }
  static id create_window_delegate() {
    objc::autoreleasepool arp;
    constexpr auto class_name = "WebviewNSWindowDelegate";
    // Avoid crash due to registering same class twice
    auto cls = objc_lookUpClass(class_name);
    if (!cls) {
      cls = objc_allocateClassPair((Class) "NSObject"_cls, class_name, 0);
      class_addProtocol(cls, objc_getProtocol("NSWindowDelegate"));
      class_addMethod(cls, "windowWillClose:"_sel,
                      (IMP)(+[](id self, SEL, id notification) {
                        auto window =
                            objc::msg_send<id>(notification, "object"_sel);
                        auto w = get_associated_webview(self);
                        w->on_window_will_close(self, window);
                      }),
                      "v@:@");
      objc_registerClassPair(cls);
    }
    return objc::msg_send<id>((id)cls, "new"_sel);
  }
  static id get_shared_application() {
    return objc::msg_send<id>("NSApplication"_cls, "sharedApplication"_sel);
  }
  static cocoa_wkwebview_engine *get_associated_webview(id object) {
    auto w =
        (cocoa_wkwebview_engine *)objc_getAssociatedObject(object, "webview");
    assert(w);
    return w;
  }
  static id get_main_bundle() noexcept {
    return objc::msg_send<id>("NSBundle"_cls, "mainBundle"_sel);
  }
  static bool is_app_bundled() noexcept {
    auto bundle = get_main_bundle();
    if (!bundle) {
      return false;
    }
    auto bundle_path = objc::msg_send<id>(bundle, "bundlePath"_sel);
    auto bundled =
        objc::msg_send<BOOL>(bundle_path, "hasSuffix:"_sel, ".app"_str);
    return !!bundled;
  }
  void on_application_did_finish_launching(id /*delegate*/, id app) {
    // See comments related to application lifecycle in create_app_delegate().
    if (m_owns_window) {
      // Stop the main run loop so that we can return
      // from the constructor.
      stop_run_loop();
    }

    // Activate the app if it is not bundled.
    // Bundled apps launched from Finder are activated automatically but
    // otherwise not. Activating the app even when it has been launched from
    // Finder does not seem to be harmful but calling this function is rarely
    // needed as proper activation is normally taken care of for us.
    // Bundled apps have a default activation policy of
    // NSApplicationActivationPolicyRegular while non-bundled apps have a
    // default activation policy of NSApplicationActivationPolicyProhibited.
    if (!is_app_bundled()) {
      // "setActivationPolicy:" must be invoked before
      // "activateIgnoringOtherApps:" for activation to work.
      objc::msg_send<void>(app, "setActivationPolicy:"_sel,
                           NSApplicationActivationPolicyRegular);
      // Activate the app regardless of other active apps.
      // This can be obtrusive so we only do it when necessary.
      objc::msg_send<void>(app, "activateIgnoringOtherApps:"_sel, YES);
    }

    set_up_window();
  }
  void on_window_will_close(id /*delegate*/, id /*window*/) {
    // Widget destroyed along with window.
    m_webview = nullptr;
    m_window = nullptr;
    dispatch([this] { on_window_destroyed(); });
  }
  void set_up_window() {
    objc::autoreleasepool arp;

    // Main window
    if (m_owns_window) {
      m_window = objc::msg_send<id>("NSWindow"_cls, "alloc"_sel);
      auto style = NSWindowStyleMaskTitled;
      m_window = objc::msg_send<id>(
          m_window, "initWithContentRect:styleMask:backing:defer:"_sel,
          CGRectMake(0, 0, 0, 0), style, NSBackingStoreBuffered, NO);

      m_window_delegate = create_window_delegate();
      objc_setAssociatedObject(m_window_delegate, "webview", (id)this,
                               OBJC_ASSOCIATION_ASSIGN);
      objc::msg_send<void>(m_window, "setDelegate:"_sel, m_window_delegate);

      on_window_created();
    }

    set_up_web_view();

    objc::msg_send<void>(m_window, "setContentView:"_sel, m_webview);

    if (m_owns_window) {
      // objc::msg_send<void>(m_window, "makeKeyAndOrderFront:"_sel, nullptr);
    }
  }
  void set_up_web_view() {
    objc::autoreleasepool arp;

    auto config = objc::autoreleased(
        objc::msg_send<id>("WKWebViewConfiguration"_cls, "new"_sel));

    m_manager = objc::msg_send<id>(config, "userContentController"_sel);
    m_webview = objc::msg_send<id>("WKWebView"_cls, "alloc"_sel);

    auto preferences = objc::msg_send<id>(config, "preferences"_sel);
    auto yes_value =
        objc::msg_send<id>("NSNumber"_cls, "numberWithBool:"_sel, YES);

    if (m_debug) {
      // Equivalent Obj-C:
      // [[config preferences] setValue:@YES forKey:@"developerExtrasEnabled"];
      objc::msg_send<id>(preferences, "setValue:forKey:"_sel, yes_value,
                         "developerExtrasEnabled"_str);
    }

    // Equivalent Obj-C:
    // [[config preferences] setValue:@YES forKey:@"fullScreenEnabled"];
    objc::msg_send<id>(preferences, "setValue:forKey:"_sel, yes_value,
                       "fullScreenEnabled"_str);

    // Equivalent Obj-C:
    // [[config preferences] setValue:@YES forKey:@"javaScriptCanAccessClipboard"];
    objc::msg_send<id>(preferences, "setValue:forKey:"_sel, yes_value,
                       "javaScriptCanAccessClipboard"_str);

    // Equivalent Obj-C:
    // [[config preferences] setValue:@YES forKey:@"DOMPasteAllowed"];
    objc::msg_send<id>(preferences, "setValue:forKey:"_sel, yes_value,
                       "DOMPasteAllowed"_str);

    auto ui_delegate = objc::autoreleased(create_webkit_ui_delegate());
    objc::msg_send<void>(m_webview, "initWithFrame:configuration:"_sel,
                         CGRectMake(0, 0, 0, 0), config);
    objc::msg_send<void>(m_webview, "setUIDelegate:"_sel, ui_delegate);

    if (m_debug) {
      // Explicitly make WKWebView inspectable via Safari on OS versions that
      // disable the feature by default (macOS 13.3 and later) and support
      // enabling it. According to Apple, the behavior on older OS versions is
      // for content to always be inspectable in "debug builds".
      // Testing shows that this is true for macOS 12.6 but somehow not 10.15.
      // https://webkit.org/blog/13936/enabling-the-inspection-of-web-content-in-apps/
#if defined(__has_builtin)
#if __has_builtin(__builtin_available)
      if (__builtin_available(macOS 13.3, iOS 16.4, tvOS 16.4, *)) {
        objc::msg_send<void>(
            m_webview, "setInspectable:"_sel,
            objc::msg_send<id>("NSNumber"_cls, "numberWithBool:"_sel, YES));
      }
#else
#error __builtin_available not supported by compiler
#endif
#else
#error __has_builtin not supported by compiler
#endif
    }

    auto script_message_handler =
        objc::autoreleased(create_script_message_handler());
    objc::msg_send<void>(m_manager, "addScriptMessageHandler:name:"_sel,
                         script_message_handler, "external"_str);

    init(R""(
      window.external = {
        invoke: function(s) {
          window.webkit.messageHandlers.external.postMessage(s);
        },
      };
      )"");
  }
  void stop_run_loop() {
    objc::autoreleasepool arp;
    auto app = get_shared_application();
    // Request the run loop to stop. This doesn't immediately stop the loop.
    objc::msg_send<void>(app, "stop:"_sel, nullptr);
    // The run loop will stop after processing an NSEvent.
    // Event type: NSEventTypeApplicationDefined (macOS 10.12+),
    //             NSApplicationDefined (macOS 10.0–10.12)
    int type = 15;
    auto event = objc::msg_send<id>(
        "NSEvent"_cls,
        "otherEventWithType:location:modifierFlags:timestamp:windowNumber:context:subtype:data1:data2:"_sel,
        type, CGPointMake(0, 0), 0, 0, 0, nullptr, 0, 0, 0);
    objc::msg_send<void>(app, "postEvent:atStart:"_sel, event, YES);
  }
  static bool get_and_set_is_first_instance() noexcept {
    static std::atomic_bool first{true};
    bool temp = first;
    if (temp) {
      first = false;
    }
    return temp;
  }

  // Blocks while depleting the run loop of events.
  void deplete_run_loop_event_queue() {
    objc::autoreleasepool arp;
    auto app = get_shared_application();
    bool done{};
    dispatch([&] { done = true; });
    auto mask = NSUIntegerMax; // NSEventMaskAny
    // NSDefaultRunLoopMode
    auto mode = objc::msg_send<id>("NSString"_cls, "stringWithUTF8String:"_sel,
                                   "kCFRunLoopDefaultMode");
    while (!done) {
      objc::autoreleasepool arp;
      auto event = objc::msg_send<id>(
          app, "nextEventMatchingMask:untilDate:inMode:dequeue:"_sel, mask,
          nullptr, mode, YES);
      if (event) {
        objc::msg_send<void>(app, "sendEvent:"_sel, event);
      }
    }
  }

  bool m_debug{};
  id m_app_delegate{};
  id m_window_delegate{};
  id m_window{};
  id m_webview{};
  id m_manager{};
  bool m_owns_window{};
};

} // namespace detail

using browser_engine = detail::cocoa_wkwebview_engine;

} // namespace webview

#elif defined(WEBVIEW_EDGE)

//
// ====================================================================
//
// This implementation uses Win32 API to create a native window. It
// uses Edge/Chromium webview2 backend as a browser engine.
//
// ====================================================================
//

#define WIN32_LEAN_AND_MEAN
#include <shlobj.h>
#include <shlwapi.h>
#include <stdlib.h>
#include <windows.h>
#include <shellapi.h>
#include <wrl.h>

#include "WebView2.h"

#ifdef _MSC_VER
#pragma comment(lib, "advapi32.lib")
#pragma comment(lib, "ole32.lib")
#pragma comment(lib, "shell32.lib")
#pragma comment(lib, "shlwapi.lib")
#pragma comment(lib, "user32.lib")
#pragma comment(lib, "version.lib")
#endif

namespace webview {
namespace detail {

using msg_cb_t = std::function<void(const std::string)>;

// Parses a version string with 1-4 integral components, e.g. "1.2.3.4".
// Missing or invalid components default to 0, and excess components are ignored.
template <typename T>
std::array<unsigned int, 4>
parse_version(const std::basic_string<T> &version) noexcept {
  auto parse_component = [](auto sb, auto se) -> unsigned int {
    try {
      auto n = std::stol(std::basic_string<T>(sb, se));
      return n < 0 ? 0 : n;
    } catch (std::exception &) {
      return 0;
    }
  };
  auto end = version.end();
  auto sb = version.begin(); // subrange begin
  auto se = sb;              // subrange end
  unsigned int ci = 0;       // component index
  std::array<unsigned int, 4> components{};
  while (sb != end && se != end && ci < components.size()) {
    if (*se == static_cast<T>('.')) {
      components[ci++] = parse_component(sb, se);
      sb = ++se;
      continue;
    }
    ++se;
  }
  if (sb < se && ci < components.size()) {
    components[ci] = parse_component(sb, se);
  }
  return components;
}

template <typename T, std::size_t Length>
auto parse_version(const T (&version)[Length]) noexcept {
  return parse_version(std::basic_string<T>(version, Length));
}

std::wstring get_file_version_string(const std::wstring &file_path) noexcept {
  DWORD dummy_handle; // Unused
  DWORD info_buffer_length =
      GetFileVersionInfoSizeW(file_path.c_str(), &dummy_handle);
  if (info_buffer_length == 0) {
    return std::wstring();
  }
  std::vector<char> info_buffer;
  info_buffer.reserve(info_buffer_length);
  if (!GetFileVersionInfoW(file_path.c_str(), 0, info_buffer_length,
                           info_buffer.data())) {
    return std::wstring();
  }
  auto sub_block = L"\\StringFileInfo\\040904B0\\ProductVersion";
  LPWSTR version = nullptr;
  unsigned int version_length = 0;
  if (!VerQueryValueW(info_buffer.data(), sub_block,
                      reinterpret_cast<LPVOID *>(&version), &version_length)) {
    return std::wstring();
  }
  if (!version || version_length == 0) {
    return std::wstring();
  }
  return std::wstring(version, version_length);
}

// A wrapper around COM library initialization. Calls CoInitializeEx in the
// constructor and CoUninitialize in the destructor.
class com_init_wrapper {
public:
  com_init_wrapper() = default;

  com_init_wrapper(DWORD dwCoInit) {
    // We can safely continue as long as COM was either successfully
    // initialized or already initialized.
    // RPC_E_CHANGED_MODE means that CoInitializeEx was already called with
    // a different concurrency model.
    switch (CoInitializeEx(nullptr, dwCoInit)) {
    case S_OK:
    case S_FALSE:
      m_initialized = true;
      break;
    }
  }

  ~com_init_wrapper() {
    if (m_initialized) {
      CoUninitialize();
      m_initialized = false;
    }
  }

  com_init_wrapper(const com_init_wrapper &other) = delete;
  com_init_wrapper &operator=(const com_init_wrapper &other) = delete;
  com_init_wrapper(com_init_wrapper &&other) { *this = std::move(other); }

  com_init_wrapper &operator=(com_init_wrapper &&other) {
    if (this == &other) {
      return *this;
    }
    m_initialized = std::exchange(other.m_initialized, false);
    return *this;
  }

  bool is_initialized() const { return m_initialized; }

private:
  bool m_initialized = false;
};

namespace ntdll_symbols {
using RtlGetVersion_t =
    unsigned int /*NTSTATUS*/ (WINAPI *)(RTL_OSVERSIONINFOW *);

constexpr auto RtlGetVersion = library_symbol<RtlGetVersion_t>("RtlGetVersion");
} // namespace ntdll_symbols

namespace user32_symbols {
using DPI_AWARENESS_CONTEXT = HANDLE;
using SetProcessDpiAwarenessContext_t = BOOL(WINAPI *)(DPI_AWARENESS_CONTEXT);
using SetProcessDPIAware_t = BOOL(WINAPI *)();
using GetDpiForWindow_t = UINT(WINAPI *)(HWND);
using EnableNonClientDpiScaling_t = BOOL(WINAPI *)(HWND);
using AdjustWindowRectExForDpi_t = BOOL(WINAPI *)(LPRECT, DWORD, BOOL, DWORD,
                                                  UINT);
using GetWindowDpiAwarenessContext_t = DPI_AWARENESS_CONTEXT(WINAPI *)(HWND);
using AreDpiAwarenessContextsEqual_t = BOOL(WINAPI *)(DPI_AWARENESS_CONTEXT,
                                                      DPI_AWARENESS_CONTEXT);

// Use intptr_t as the underlying type because we need to
// reinterpret_cast<DPI_AWARENESS_CONTEXT> which is a pointer.
// Available since Windows 10, version 1607
enum class dpi_awareness : intptr_t {
  per_monitor_v2_aware = -4, // Available since Windows 10, version 1703
  per_monitor_aware = -3
};

constexpr auto SetProcessDpiAwarenessContext =
    library_symbol<SetProcessDpiAwarenessContext_t>(
        "SetProcessDpiAwarenessContext");
constexpr auto SetProcessDPIAware =
    library_symbol<SetProcessDPIAware_t>("SetProcessDPIAware");
constexpr auto GetDpiForWindow =
    library_symbol<GetDpiForWindow_t>("GetDpiForWindow");
constexpr auto EnableNonClientDpiScaling =
    library_symbol<EnableNonClientDpiScaling_t>("EnableNonClientDpiScaling");
constexpr auto AdjustWindowRectExForDpi =
    library_symbol<AdjustWindowRectExForDpi_t>("AdjustWindowRectExForDpi");
constexpr auto GetWindowDpiAwarenessContext =
    library_symbol<GetWindowDpiAwarenessContext_t>(
        "GetWindowDpiAwarenessContext");
constexpr auto AreDpiAwarenessContextsEqual =
    library_symbol<AreDpiAwarenessContextsEqual_t>(
        "AreDpiAwarenessContextsEqual");
} // namespace user32_symbols

// MARGINS structure for DwmExtendFrameIntoClientArea
typedef struct _MARGINS {
  int cxLeftWidth;
  int cxRightWidth;
  int cyTopHeight;
  int cyBottomHeight;
} MARGINS;

namespace dwmapi_symbols {
typedef enum {
  // This undocumented value is used instead of DWMWA_USE_IMMERSIVE_DARK_MODE
  // on Windows 10 older than build 19041 (2004/20H1).
  DWMWA_USE_IMMERSIVE_DARK_MODE_BEFORE_V10_0_19041 = 19,
  // Documented as being supported since Windows 11 build 22000 (21H2) but it
  // works since Windows 10 build 19041 (2004/20H1).
  DWMWA_USE_IMMERSIVE_DARK_MODE = 20
} DWMWINDOWATTRIBUTE;
using DwmSetWindowAttribute_t = HRESULT(WINAPI *)(HWND, DWORD, LPCVOID, DWORD);
using DwmExtendFrameIntoClientArea_t = HRESULT(WINAPI *)(HWND, const MARGINS *);

constexpr auto DwmSetWindowAttribute =
    library_symbol<DwmSetWindowAttribute_t>("DwmSetWindowAttribute");
constexpr auto DwmExtendFrameIntoClientArea =
    library_symbol<DwmExtendFrameIntoClientArea_t>("DwmExtendFrameIntoClientArea");
} // namespace dwmapi_symbols

namespace shcore_symbols {
typedef enum { PROCESS_PER_MONITOR_DPI_AWARE = 2 } PROCESS_DPI_AWARENESS;
using SetProcessDpiAwareness_t = HRESULT(WINAPI *)(PROCESS_DPI_AWARENESS);

constexpr auto SetProcessDpiAwareness =
    library_symbol<SetProcessDpiAwareness_t>("SetProcessDpiAwareness");
} // namespace shcore_symbols

class reg_key {
public:
  explicit reg_key(HKEY root_key, const wchar_t *sub_key, DWORD options,
                   REGSAM sam_desired) {
    HKEY handle;
    auto status =
        RegOpenKeyExW(root_key, sub_key, options, sam_desired, &handle);
    if (status == ERROR_SUCCESS) {
      m_handle = handle;
    }
  }

  explicit reg_key(HKEY root_key, const std::wstring &sub_key, DWORD options,
                   REGSAM sam_desired)
      : reg_key(root_key, sub_key.c_str(), options, sam_desired) {}

  virtual ~reg_key() {
    if (m_handle) {
      RegCloseKey(m_handle);
      m_handle = nullptr;
    }
  }

  reg_key(const reg_key &other) = delete;
  reg_key &operator=(const reg_key &other) = delete;
  reg_key(reg_key &&other) = delete;
  reg_key &operator=(reg_key &&other) = delete;

  bool is_open() const { return !!m_handle; }
  bool get_handle() const { return m_handle; }

  template <typename Container>
  void query_bytes(const wchar_t *name, Container &result) const {
    DWORD buf_length = 0;
    // Get the size of the data in bytes.
    auto status = RegQueryValueExW(m_handle, name, nullptr, nullptr, nullptr,
                                   &buf_length);
    if (status != ERROR_SUCCESS && status != ERROR_MORE_DATA) {
      result.resize(0);
      return;
    }
    // Read the data.
    result.resize(buf_length / sizeof(typename Container::value_type));
    auto *buf = reinterpret_cast<LPBYTE>(&result[0]);
    status =
        RegQueryValueExW(m_handle, name, nullptr, nullptr, buf, &buf_length);
    if (status != ERROR_SUCCESS) {
      result.resize(0);
      return;
    }
  }

  std::wstring query_string(const wchar_t *name) const {
    std::wstring result;
    query_bytes(name, result);
    // Remove trailing null-characters.
    for (std::size_t length = result.size(); length > 0; --length) {
      if (result[length - 1] != 0) {
        result.resize(length);
        break;
      }
    }
    return result;
  }

  unsigned int query_uint(const wchar_t *name,
                          unsigned int default_value) const {
    std::vector<char> data;
    query_bytes(name, data);
    if (data.size() < sizeof(DWORD)) {
      return default_value;
    }
    return static_cast<unsigned int>(*reinterpret_cast<DWORD *>(data.data()));
  }

private:
  HKEY m_handle = nullptr;
};

// Compare the specified version against the OS version.
// Returns less than 0 if the OS version is less.
// Returns 0 if the versions are equal.
// Returns greater than 0 if the specified version is greater.
inline int compare_os_version(unsigned int major, unsigned int minor,
                              unsigned int build) {
  // Use RtlGetVersion both to bypass potential issues related to
  // VerifyVersionInfo and manifests, and because both GetVersion and
  // GetVersionEx are deprecated.
  auto ntdll = native_library(L"ntdll.dll");
  if (auto fn = ntdll.get(ntdll_symbols::RtlGetVersion)) {
    RTL_OSVERSIONINFOW vi{};
    vi.dwOSVersionInfoSize = sizeof(vi);
    if (fn(&vi) != 0) {
      return false;
    }
    if (vi.dwMajorVersion == major) {
      if (vi.dwMinorVersion == minor) {
        return static_cast<int>(vi.dwBuildNumber) - static_cast<int>(build);
      }
      return static_cast<int>(vi.dwMinorVersion) - static_cast<int>(minor);
    }
    return static_cast<int>(vi.dwMajorVersion) - static_cast<int>(major);
  }
  return false;
}

inline bool is_per_monitor_v2_awareness_available() {
  // Windows 10, version 1703
  return compare_os_version(10, 0, 15063) >= 0;
}

inline bool enable_dpi_awareness() {
  auto user32 = native_library(L"user32.dll");
  if (auto fn = user32.get(user32_symbols::SetProcessDpiAwarenessContext)) {
    auto dpi_awareness =
        reinterpret_cast<user32_symbols::DPI_AWARENESS_CONTEXT>(
            is_per_monitor_v2_awareness_available()
                ? user32_symbols::dpi_awareness::per_monitor_v2_aware
                : user32_symbols::dpi_awareness::per_monitor_aware);
    if (fn(dpi_awareness)) {
      return true;
    }
    return GetLastError() == ERROR_ACCESS_DENIED;
  }
  if (auto shcore = native_library(L"shcore.dll")) {
    if (auto fn = shcore.get(shcore_symbols::SetProcessDpiAwareness)) {
      auto result = fn(shcore_symbols::PROCESS_PER_MONITOR_DPI_AWARE);
      return result == S_OK || result == E_ACCESSDENIED;
    }
  }
  if (auto fn = user32.get(user32_symbols::SetProcessDPIAware)) {
    return !!fn();
  }
  return true;
}

inline bool enable_non_client_dpi_scaling_if_needed(HWND window) {
  auto user32 = native_library(L"user32.dll");
  auto get_ctx_fn = user32.get(user32_symbols::GetWindowDpiAwarenessContext);
  if (!get_ctx_fn) {
    return true;
  }
  auto awareness = get_ctx_fn(window);
  if (!awareness) {
    return false;
  }
  auto ctx_equal_fn = user32.get(user32_symbols::AreDpiAwarenessContextsEqual);
  if (!ctx_equal_fn) {
    return true;
  }
  // EnableNonClientDpiScaling is only needed with per monitor v1 awareness.
  auto per_monitor = reinterpret_cast<user32_symbols::DPI_AWARENESS_CONTEXT>(
      user32_symbols::dpi_awareness::per_monitor_aware);
  if (!ctx_equal_fn(awareness, per_monitor)) {
    return true;
  }
  auto enable_fn = user32.get(user32_symbols::EnableNonClientDpiScaling);
  if (!enable_fn) {
    return true;
  }
  return !!enable_fn(window);
}

constexpr int get_default_window_dpi() {
  constexpr const int default_dpi = 96; // USER_DEFAULT_SCREEN_DPI
  return default_dpi;
}

inline int get_window_dpi(HWND window) {
  auto user32 = native_library(L"user32.dll");
  if (auto fn = user32.get(user32_symbols::GetDpiForWindow)) {
    auto dpi = static_cast<int>(fn(window));
    return dpi;
  }
  return get_default_window_dpi();
}

constexpr int scale_value_for_dpi(int value, int from_dpi, int to_dpi) {
  return (value * to_dpi) / from_dpi;
}

constexpr SIZE scale_size(int width, int height, int from_dpi, int to_dpi) {
  auto scaled_width = scale_value_for_dpi(width, from_dpi, to_dpi);
  auto scaled_height = scale_value_for_dpi(height, from_dpi, to_dpi);
  return {scaled_width, scaled_height};
}

inline SIZE make_window_frame_size(HWND window, int width, int height,
                                   int dpi) {
  auto style = GetWindowLong(window, GWL_STYLE);
  RECT r{0, 0, width, height};
  auto user32 = native_library(L"user32.dll");
  if (auto fn = user32.get(user32_symbols::AdjustWindowRectExForDpi)) {
    fn(&r, style, FALSE, 0, static_cast<UINT>(dpi));
  } else {
    AdjustWindowRect(&r, style, 0);
  }
  auto frame_width = r.right - r.left;
  auto frame_height = r.bottom - r.top;
  return {frame_width, frame_height};
}

inline bool is_dark_theme_enabled() {
  constexpr auto *sub_key =
      L"SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\Themes\\Personalize";
  reg_key key(HKEY_CURRENT_USER, sub_key, 0, KEY_READ);
  if (!key.is_open()) {
    // Default is light theme
    return false;
  }
  return key.query_uint(L"AppsUseLightTheme", 1) == 0;
}

inline void apply_window_theme(HWND window) {
  auto dark_theme_enabled = is_dark_theme_enabled();

  // Use "immersive dark mode" on systems that support it.
  // Changes the color of the window's title bar (light or dark).
  BOOL use_dark_mode{dark_theme_enabled ? TRUE : FALSE};
  static native_library dwmapi{L"dwmapi.dll"};
  if (auto fn = dwmapi.get(dwmapi_symbols::DwmSetWindowAttribute)) {
    // Try the modern, documented attribute before the older, undocumented one.
    if (fn(window, dwmapi_symbols::DWMWA_USE_IMMERSIVE_DARK_MODE,
           &use_dark_mode, sizeof(use_dark_mode)) != S_OK) {
      fn(window,
         dwmapi_symbols::DWMWA_USE_IMMERSIVE_DARK_MODE_BEFORE_V10_0_19041,
         &use_dark_mode, sizeof(use_dark_mode));
    }
  }

  // adds dark/light colors to the titlebar instead of transparent
  if (auto fn = dwmapi.get(dwmapi_symbols::DwmExtendFrameIntoClientArea)) {
    MARGINS margins = { 0, 0, 0, 32 }; // No extension to preserve button alignment
    fn(window, &margins);
  }
}

// Enable built-in WebView2Loader implementation by default.
#ifndef WEBVIEW_MSWEBVIEW2_BUILTIN_IMPL
#define WEBVIEW_MSWEBVIEW2_BUILTIN_IMPL 1
#endif

// Link WebView2Loader.dll explicitly by default only if the built-in
// implementation is enabled.
#ifndef WEBVIEW_MSWEBVIEW2_EXPLICIT_LINK
#define WEBVIEW_MSWEBVIEW2_EXPLICIT_LINK WEBVIEW_MSWEBVIEW2_BUILTIN_IMPL
#endif

// Explicit linking of WebView2Loader.dll should be used along with
// the built-in implementation.
#if WEBVIEW_MSWEBVIEW2_BUILTIN_IMPL == 1 &&                                    \
    WEBVIEW_MSWEBVIEW2_EXPLICIT_LINK != 1
#undef WEBVIEW_MSWEBVIEW2_EXPLICIT_LINK
#error Please set WEBVIEW_MSWEBVIEW2_EXPLICIT_LINK=1.
#endif

#if WEBVIEW_MSWEBVIEW2_BUILTIN_IMPL == 1
// Gets the last component of a Windows native file path.
// For example, if the path is "C:\a\b" then the result is "b".
template <typename T>
std::basic_string<T>
get_last_native_path_component(const std::basic_string<T> &path) {
  auto pos = path.find_last_of(static_cast<T>('\\'));
  if (pos != std::basic_string<T>::npos) {
    return path.substr(pos + 1);
  }
  return std::basic_string<T>();
}
#endif /* WEBVIEW_MSWEBVIEW2_BUILTIN_IMPL */

template <typename T> struct cast_info_t {
  using type = T;
  IID iid;
};

namespace mswebview2 {
static constexpr IID IID_ICoreWebView2ScriptDialogOpeningEventHandler{
       0xef381bf9,
       0xafa8,
       0x4e37,
       {0x91, 0xc4, 0x8a, 0xc4, 0x85, 0x24, 0xbd, 0xfb}};
static constexpr IID
    IID_ICoreWebView2CreateCoreWebView2ControllerCompletedHandler{
        0x6C4819F3,
        0xC9B7,
        0x4260,
        {0x81, 0x27, 0xC9, 0xF5, 0xBD, 0xE7, 0xF6, 0x8C}};
static constexpr IID
    IID_ICoreWebView2CreateCoreWebView2EnvironmentCompletedHandler{
        0x4E8A3389,
        0xC9D8,
        0x4BD2,
        {0xB6, 0xB5, 0x12, 0x4F, 0xEE, 0x6C, 0xC1, 0x4D}};
static constexpr IID IID_ICoreWebView2PermissionRequestedEventHandler{
    0x15E1C6A3,
    0xC72A,
    0x4DF3,
    {0x91, 0xD7, 0xD0, 0x97, 0xFB, 0xEC, 0x6B, 0xFD}};
static constexpr IID IID_ICoreWebView2WebMessageReceivedEventHandler{
    0x57213F19,
    0x00E6,
    0x49FA,
    {0x8E, 0x07, 0x89, 0x8E, 0xA0, 0x1E, 0xCB, 0xD2}};
static constexpr IID IID_ICoreWebView2NewWindowRequestedEventHandler{
    0xD4C185FE,
    0xC81C,
    0x481B,
    {0xB9, 0x4A, 0x11, 0xD1, 0xF9, 0x64, 0xEA, 0x5C}};
static constexpr IID IID_ICoreWebView2NavigationStartingEventHandler{
    0x9ADBE429,
    0xF36D,
    0x432B,
    {0x9D, 0xDC, 0xCE, 0xC5, 0x21, 0xB0, 0x03, 0x49}};
static constexpr IID IID_ICoreWebView2ContextMenuRequestedEventHandler{
    0x04d3fe1d,
    0xab87,
    0x42fb,
    {0xa8,0x98,0xda,0x24,0x1d,0x35,0xb6,0x3c}};

#if WEBVIEW_MSWEBVIEW2_BUILTIN_IMPL == 1
enum class webview2_runtime_type { installed = 0, embedded = 1 };

namespace webview2_symbols {
using CreateWebViewEnvironmentWithOptionsInternal_t =
    HRESULT(STDMETHODCALLTYPE *)(
        bool, webview2_runtime_type, PCWSTR, IUnknown *,
        ICoreWebView2CreateCoreWebView2EnvironmentCompletedHandler *);
using DllCanUnloadNow_t = HRESULT(STDMETHODCALLTYPE *)();

static constexpr auto CreateWebViewEnvironmentWithOptionsInternal =
    library_symbol<CreateWebViewEnvironmentWithOptionsInternal_t>(
        "CreateWebViewEnvironmentWithOptionsInternal");
static constexpr auto DllCanUnloadNow =
    library_symbol<DllCanUnloadNow_t>("DllCanUnloadNow");
} // namespace webview2_symbols
#endif /* WEBVIEW_MSWEBVIEW2_BUILTIN_IMPL */

#if WEBVIEW_MSWEBVIEW2_EXPLICIT_LINK == 1
namespace webview2_symbols {
using CreateCoreWebView2EnvironmentWithOptions_t = HRESULT(STDMETHODCALLTYPE *)(
    PCWSTR, PCWSTR, ICoreWebView2EnvironmentOptions *,
    ICoreWebView2CreateCoreWebView2EnvironmentCompletedHandler *);
using GetAvailableCoreWebView2BrowserVersionString_t =
    HRESULT(STDMETHODCALLTYPE *)(PCWSTR, LPWSTR *);

static constexpr auto CreateCoreWebView2EnvironmentWithOptions =
    library_symbol<CreateCoreWebView2EnvironmentWithOptions_t>(
        "CreateCoreWebView2EnvironmentWithOptions");
static constexpr auto GetAvailableCoreWebView2BrowserVersionString =
    library_symbol<GetAvailableCoreWebView2BrowserVersionString_t>(
        "GetAvailableCoreWebView2BrowserVersionString");
} // namespace webview2_symbols
#endif /* WEBVIEW_MSWEBVIEW2_EXPLICIT_LINK */

class loader {
public:
  HRESULT create_environment_with_options(
      PCWSTR browser_dir, PCWSTR user_data_dir,
      ICoreWebView2EnvironmentOptions *env_options,
      ICoreWebView2CreateCoreWebView2EnvironmentCompletedHandler
          *created_handler) const {
#if WEBVIEW_MSWEBVIEW2_EXPLICIT_LINK == 1
    if (m_lib.is_loaded()) {
      if (auto fn = m_lib.get(
              webview2_symbols::CreateCoreWebView2EnvironmentWithOptions)) {
        return fn(browser_dir, user_data_dir, env_options, created_handler);
      }
    }
#if WEBVIEW_MSWEBVIEW2_BUILTIN_IMPL == 1
    return create_environment_with_options_impl(browser_dir, user_data_dir,
                                                env_options, created_handler);
#else
    return S_FALSE;
#endif
#else
    return ::CreateCoreWebView2EnvironmentWithOptions(
        browser_dir, user_data_dir, env_options, created_handler);
#endif /* WEBVIEW_MSWEBVIEW2_EXPLICIT_LINK */
  }

  HRESULT
  get_available_browser_version_string(PCWSTR browser_dir,
                                       LPWSTR *version) const {
#if WEBVIEW_MSWEBVIEW2_EXPLICIT_LINK == 1
    if (m_lib.is_loaded()) {
      if (auto fn = m_lib.get(
              webview2_symbols::GetAvailableCoreWebView2BrowserVersionString)) {
        return fn(browser_dir, version);
      }
    }
#if WEBVIEW_MSWEBVIEW2_BUILTIN_IMPL == 1
    return get_available_browser_version_string_impl(browser_dir, version);
#else
    return S_FALSE;
#endif
#else
    return ::GetAvailableCoreWebView2BrowserVersionString(browser_dir, version);
#endif /* WEBVIEW_MSWEBVIEW2_EXPLICIT_LINK */
  }

private:
#if WEBVIEW_MSWEBVIEW2_BUILTIN_IMPL == 1
  struct client_info_t {
    bool found = false;
    std::wstring dll_path;
    std::wstring version;
    webview2_runtime_type runtime_type;
  };

  HRESULT create_environment_with_options_impl(
      PCWSTR browser_dir, PCWSTR user_data_dir,
      ICoreWebView2EnvironmentOptions *env_options,
      ICoreWebView2CreateCoreWebView2EnvironmentCompletedHandler
          *created_handler) const {
    auto found_client = find_available_client(browser_dir);
    if (!found_client.found) {
      return -1;
    }
    auto client_dll = native_library(found_client.dll_path.c_str());
    if (auto fn = client_dll.get(
            webview2_symbols::CreateWebViewEnvironmentWithOptionsInternal)) {
      return fn(true, found_client.runtime_type, user_data_dir, env_options,
                created_handler);
    }
    if (auto fn = client_dll.get(webview2_symbols::DllCanUnloadNow)) {
      if (!fn()) {
        client_dll.detach();
      }
    }
    return ERROR_SUCCESS;
  }

  HRESULT
  get_available_browser_version_string_impl(PCWSTR browser_dir,
                                            LPWSTR *version) const {
    if (!version) {
      return -1;
    }
    auto found_client = find_available_client(browser_dir);
    if (!found_client.found) {
      return -1;
    }
    auto info_length_bytes =
        found_client.version.size() * sizeof(found_client.version[0]);
    auto info = static_cast<LPWSTR>(CoTaskMemAlloc(info_length_bytes));
    if (!info) {
      return -1;
    }
    CopyMemory(info, found_client.version.c_str(), info_length_bytes);
    *version = info;
    return 0;
  }

  client_info_t find_available_client(PCWSTR browser_dir) const {
    if (browser_dir) {
      return find_embedded_client(api_version, browser_dir);
    }
    auto found_client =
        find_installed_client(api_version, true, default_release_channel_guid);
    if (!found_client.found) {
      found_client = find_installed_client(api_version, false,
                                           default_release_channel_guid);
    }
    return found_client;
  }

  std::wstring make_client_dll_path(const std::wstring &dir) const {
    auto dll_path = dir;
    if (!dll_path.empty()) {
      auto last_char = dir[dir.size() - 1];
      if (last_char != L'\\' && last_char != L'/') {
        dll_path += L'\\';
      }
    }
    dll_path += L"EBWebView\\";
#if defined(_M_X64) || defined(__x86_64__)
    dll_path += L"x64";
#elif defined(_M_IX86) || defined(__i386__)
    dll_path += L"x86";
#elif defined(_M_ARM64) || defined(__aarch64__)
    dll_path += L"arm64";
#else
#error WebView2 integration for this platform is not yet supported.
#endif
    dll_path += L"\\EmbeddedBrowserWebView.dll";
    return dll_path;
  }

  client_info_t
  find_installed_client(unsigned int min_api_version, bool system,
                        const std::wstring &release_channel) const {
    std::wstring sub_key = client_state_reg_sub_key;
    sub_key += release_channel;
    auto root_key = system ? HKEY_LOCAL_MACHINE : HKEY_CURRENT_USER;
    reg_key key(root_key, sub_key, 0, KEY_READ | KEY_WOW64_32KEY);
    if (!key.is_open()) {
      return {};
    }
    auto ebwebview_value = key.query_string(L"EBWebView");

    auto client_version_string =
        get_last_native_path_component(ebwebview_value);
    auto client_version = parse_version(client_version_string);
    if (client_version[2] < min_api_version) {
      // Our API version is greater than the runtime API version.
      return {};
    }

    auto client_dll_path = make_client_dll_path(ebwebview_value);
    return {true, client_dll_path, client_version_string,
            webview2_runtime_type::installed};
  }

  client_info_t find_embedded_client(unsigned int min_api_version,
                                     const std::wstring &dir) const {
    auto client_dll_path = make_client_dll_path(dir);

    auto client_version_string = get_file_version_string(client_dll_path);
    auto client_version = parse_version(client_version_string);
    if (client_version[2] < min_api_version) {
      // Our API version is greater than the runtime API version.
      return {};
    }

    return {true, client_dll_path, client_version_string,
            webview2_runtime_type::embedded};
  }

  // The minimum WebView2 API version we need regardless of the SDK release
  // actually used. The number comes from the SDK release version,
  // e.g. 1.0.1150.38. To be safe the SDK should have a number that is greater
  // than or equal to this number. The Edge browser webview client must
  // have a number greater than or equal to this number.
  static constexpr unsigned int api_version = 1150;

  static constexpr auto client_state_reg_sub_key =
      L"SOFTWARE\\Microsoft\\EdgeUpdate\\ClientState\\";

  // GUID for the stable release channel.
  static constexpr auto stable_release_guid =
      L"{F3017226-FE2A-4295-8BDF-00C3A9A7E4C5}";

  static constexpr auto default_release_channel_guid = stable_release_guid;
#endif /* WEBVIEW_MSWEBVIEW2_BUILTIN_IMPL */

#if WEBVIEW_MSWEBVIEW2_EXPLICIT_LINK == 1
  native_library m_lib{L"WebView2Loader.dll"};
#endif
};

namespace cast_info {
static constexpr auto controller_completed =
    cast_info_t<ICoreWebView2CreateCoreWebView2ControllerCompletedHandler>{
        IID_ICoreWebView2CreateCoreWebView2ControllerCompletedHandler};

static constexpr auto environment_completed =
    cast_info_t<ICoreWebView2CreateCoreWebView2EnvironmentCompletedHandler>{
        IID_ICoreWebView2CreateCoreWebView2EnvironmentCompletedHandler};

static constexpr auto message_received =
    cast_info_t<ICoreWebView2WebMessageReceivedEventHandler>{
        IID_ICoreWebView2WebMessageReceivedEventHandler};

static constexpr auto permission_requested =
    cast_info_t<ICoreWebView2PermissionRequestedEventHandler>{
        IID_ICoreWebView2PermissionRequestedEventHandler};

static constexpr auto new_window_requested =
    cast_info_t<ICoreWebView2NewWindowRequestedEventHandler>{
        IID_ICoreWebView2NewWindowRequestedEventHandler};

static constexpr auto navigation_starting =
    cast_info_t<ICoreWebView2NavigationStartingEventHandler>{
        IID_ICoreWebView2NavigationStartingEventHandler};
static constexpr auto script_dialog_opening =
    cast_info_t<ICoreWebView2ScriptDialogOpeningEventHandler>{
        IID_ICoreWebView2ScriptDialogOpeningEventHandler};

static constexpr auto context_menu_requested =
    cast_info_t<ICoreWebView2ContextMenuRequestedEventHandler>{
        IID_ICoreWebView2ContextMenuRequestedEventHandler};
} // namespace cast_info
} // namespace mswebview2

class webview2_com_handler
    : public ICoreWebView2CreateCoreWebView2EnvironmentCompletedHandler,
      public ICoreWebView2CreateCoreWebView2ControllerCompletedHandler,
      public ICoreWebView2WebMessageReceivedEventHandler,
      public ICoreWebView2PermissionRequestedEventHandler,
      public ICoreWebView2NewWindowRequestedEventHandler,
      public ICoreWebView2NavigationStartingEventHandler,
      public ICoreWebView2ScriptDialogOpeningEventHandler,
      public ICoreWebView2ContextMenuRequestedEventHandler,
      public ICoreWebView2CustomItemSelectedEventHandler {
  using webview2_com_handler_cb_t =
      std::function<void(ICoreWebView2Controller *, ICoreWebView2_11 *webview)>;

public:
  webview2_com_handler(HWND hwnd, msg_cb_t msgCb, webview2_com_handler_cb_t cb)
      : m_window(hwnd), m_msgCb(msgCb), m_cb(cb) {}

  virtual ~webview2_com_handler() = default;
  webview2_com_handler(const webview2_com_handler &other) = delete;
  webview2_com_handler &operator=(const webview2_com_handler &other) = delete;
  webview2_com_handler(webview2_com_handler &&other) = delete;
  webview2_com_handler &operator=(webview2_com_handler &&other) = delete;

  ULONG STDMETHODCALLTYPE AddRef() { return ++m_ref_count; }
  ULONG STDMETHODCALLTYPE Release() {
    if (m_ref_count > 1) {
      return --m_ref_count;
    }
    delete this;
    return 0;
  }

  HRESULT STDMETHODCALLTYPE Invoke(
    ICoreWebView2 *sender,
    ICoreWebView2NewWindowRequestedEventArgs *args) {
    
    LPWSTR uri;
    args->get_Uri(&uri);
    
    if (uri) {
        std::wstring wuri(uri);
        std::string uri_str = narrow_string(wuri);
        std::string uri_lower = uri_str;
        std::transform(uri_lower.begin(), uri_lower.end(), uri_lower.begin(), ::tolower);
        bool is_localhost = 
            uri_lower.find("localhost") != std::string::npos ||
            uri_lower.find("127.0.0.1") != std::string::npos ||
            uri_lower.find("::1") != std::string::npos;
        
        args->put_Handled(TRUE);
        
        if (is_localhost) {
            std::string path = "/";
            size_t protocol_end = uri_str.find("://");
            if (protocol_end != std::string::npos) {
                size_t path_start = uri_str.find("/", protocol_end + 3);
                if (path_start != std::string::npos) {
                    path = uri_str.substr(path_start);
                }
            }
            
            std::string js = "history.pushState({}, '', '" + path + "'); window.dispatchEvent(new PopStateEvent('popstate'));";
            std::wstring wjs = widen_string(js);
            sender->ExecuteScript(wjs.c_str(), nullptr);
        } else {
            ShellExecuteW(nullptr, L"open", uri, nullptr, nullptr, SW_SHOWNORMAL);
        }

        CoTaskMemFree(uri);
    }

    return S_OK;
  }

  HRESULT STDMETHODCALLTYPE Invoke(
    ICoreWebView2 *sender,
    ICoreWebView2NavigationStartingEventArgs *args) {

    LPWSTR uri;
    args->get_Uri(&uri);

    if (uri) {
        std::wstring wuri(uri);
        std::string uri_str = narrow_string(wuri);

        std::string uri_lower = uri_str;
        std::transform(uri_lower.begin(), uri_lower.end(), uri_lower.begin(), ::tolower);

        bool is_external =
            (uri_lower.find("http://") == 0 || uri_lower.find("https://") == 0) &&
            uri_lower.find("localhost") == std::string::npos &&
            uri_lower.find("127.0.0.1") == std::string::npos &&
            uri_lower.find("::1") == std::string::npos;

        if (is_external) {
            args->put_Cancel(TRUE);
            ShellExecuteW(nullptr, L"open", uri, nullptr, nullptr, SW_SHOWNORMAL);
        }

        CoTaskMemFree(uri);
    }

    return S_OK;
  }

  HRESULT STDMETHODCALLTYPE Invoke(
    ICoreWebView2 *sender, 
    ICoreWebView2ScriptDialogOpeningEventArgs *args) {

    LPWSTR message;
    COREWEBVIEW2_SCRIPT_DIALOG_KIND kind;
    
    args->get_Message(&message);
    args->get_Kind(&kind);
    
    if (kind == COREWEBVIEW2_SCRIPT_DIALOG_KIND_CONFIRM) {
      int result = MessageBoxW(m_window, message, L"Confirm", MB_OKCANCEL | MB_ICONQUESTION);
      
      if (result == IDOK) {
        args->Accept();
      }
      
      CoTaskMemFree(message);

      SetFocus(m_window);

      return S_OK;
    }

    CoTaskMemFree(message);
    return S_OK;
  }

  HRESULT STDMETHODCALLTYPE Invoke(
    ICoreWebView2 *sender,
    ICoreWebView2ContextMenuRequestedEventArgs *args) {

    ICoreWebView2ContextMenuItemCollection* items = nullptr;
    if (FAILED(args->get_MenuItems(&items)) || !items) {
      return S_OK;
    }

    UINT count = 0;
    items->get_Count(&count);

    // remove unwanted default items
    for (int idx = static_cast<int>(count) - 1; idx >= 0; --idx) {
        ICoreWebView2ContextMenuItem* item = nullptr;
        items->GetValueAtIndex(idx, &item);
        if (!item) continue;

        LPWSTR name = nullptr;
        item->get_Name(&name);

        if (name != nullptr) {
            std::wstring itemName(name);
            std::transform(itemName.begin(), itemName.end(), itemName.begin(), ::towlower);
            if (itemName.find(L"saveas") != std::wstring::npos ||
                itemName.find(L"print") != std::wstring::npos ||
                itemName.find(L"copylinktohighlight") != std::wstring::npos ||
                itemName.find(L"back") != std::wstring::npos ||
                itemName.find(L"forward") != std::wstring::npos ||
                itemName.find(L"reload") != std::wstring::npos ||
                itemName.find(L"share") != std::wstring::npos ||
                itemName.find(L"screenshot") != std::wstring::npos ||
                itemName.find(L"webcapture") != std::wstring::npos ||
                itemName.find(L"openlinkinnewwindow") != std::wstring::npos ||
                itemName.find(L"savelinkas") != std::wstring::npos ||
                itemName.find(L"copylinklocation") != std::wstring::npos) {
                items->RemoveValueAtIndex(idx);
            }
        }

        if (name) {
            CoTaskMemFree(name);
        }

        if (item) {
            item->Release();
        }
    }

    items->get_Count(&count);

    // Add custom menu items at the top
    int item_count = menu_get_item_count();
    if (item_count > 0) {
        // std::wstring message = L"Found " + std::to_wstring(customItemCount) + L" custom menu items";
        // MessageBoxW(m_window, message.c_str(), L"Custom Menu Items", MB_OK | MB_ICONINFORMATION);

        menuItem* customItems = static_cast<menuItem*>(menu_get_items());
        if (customItems) {
            ICoreWebView2_11 *webview = nullptr;
            if (sender) {
                sender->QueryInterface(IID_ICoreWebView2_11,
                                            reinterpret_cast<void **>(&webview));
            }

            ICoreWebView2Environment* env_base = nullptr;
            ICoreWebView2Environment9* env = nullptr;
            if (SUCCEEDED(webview->get_Environment(&env_base)) && env_base) {
                env_base->QueryInterface(IID_ICoreWebView2Environment9, (void**)&env);
                env_base->Release();
                
                int insertIndex = 0;
                for (int i = 0; i < item_count; i++) {
                    if (customItems[i].separator) {
                        ICoreWebView2ContextMenuItem* separator = nullptr;
                        env->CreateContextMenuItem(L"", nullptr, COREWEBVIEW2_CONTEXT_MENU_ITEM_KIND_SEPARATOR, &separator);
                        if (separator) {
                            items->InsertValueAtIndex(insertIndex++, separator);
                            separator->Release();
                        }
                    } else if (customItems[i].label) {
                        std::string label(customItems[i].label);
                        std::wstring wlabel = widen_string(label);
                        
                        ICoreWebView2ContextMenuItem* menuItem = nullptr;
                        env->CreateContextMenuItem(wlabel.c_str(), nullptr, COREWEBVIEW2_CONTEXT_MENU_ITEM_KIND_COMMAND, &menuItem);
                        
                        if (menuItem) {
                            menuItem->put_IsEnabled(customItems[i].enabled);
                            
                            // Add click handler
                            EventRegistrationToken token;
                            menuItem->add_CustomItemSelected(this, &token);
                            
                            // Insert at the top (current insertIndex)
                            items->InsertValueAtIndex(insertIndex++, menuItem);
                            menuItem->Release();
                        }
                    }
                }

                items->get_Count(&count);
                if (count > 0) {
                    ICoreWebView2ContextMenuItem* separator = nullptr;
                    env->CreateContextMenuItem(L"", nullptr, COREWEBVIEW2_CONTEXT_MENU_ITEM_KIND_SEPARATOR, &separator);
                    if (separator) {
                        items->InsertValueAtIndex(insertIndex, separator);
                        separator->Release();
                    }
                }
                
                env->Release();
            }
        }
    }

    return S_OK;
  }

  HRESULT STDMETHODCALLTYPE Invoke(
    ICoreWebView2ContextMenuItem* sender,
    IUnknown* /*args*/) {
    LPWSTR label = nullptr;
    if (sender && SUCCEEDED(sender->get_Label(&label)) && label) {
      std::wstring wlabel(label);
      std::string itemLabel = narrow_string(wlabel);
      menu_handle_selection((char*)itemLabel.c_str());
      CoTaskMemFree(label);
    }

    return S_OK;
  }

  HRESULT STDMETHODCALLTYPE QueryInterface(REFIID riid, LPVOID *ppv) {
    using namespace mswebview2::cast_info;

    if (!ppv) {
      return E_POINTER;
    }

    // All of the COM interfaces we implement should be added here regardless
    // of whether they are required.
    // This is just to be on the safe side in case the WebView2 Runtime ever
    // requests a pointer to an interface we implement.
    // The WebView2 Runtime must at the very least be able to get a pointer to
    // ICoreWebView2CreateCoreWebView2EnvironmentCompletedHandler when we use
    // our custom WebView2 loader implementation, and observations have shown
    // that it is the only interface requested in this case. None have been
    // observed to be requested when using the official WebView2 loader.

    if (cast_if_equal_iid(riid, controller_completed, ppv) ||
        cast_if_equal_iid(riid, environment_completed, ppv) ||
        cast_if_equal_iid(riid, message_received, ppv) ||
        cast_if_equal_iid(riid, permission_requested, ppv) ||
        cast_if_equal_iid(riid, new_window_requested, ppv) ||
        cast_if_equal_iid(riid, navigation_starting, ppv) ||
        cast_if_equal_iid(riid, script_dialog_opening, ppv) ||
        cast_if_equal_iid(riid, context_menu_requested, ppv)) {
      return S_OK;
    }

    return E_NOINTERFACE;
  }

  HRESULT STDMETHODCALLTYPE Invoke(HRESULT res, ICoreWebView2Environment *env) {
    if (SUCCEEDED(res)) {
      res = env->CreateCoreWebView2Controller(m_window, this);
      if (SUCCEEDED(res)) {
        return S_OK;
      }
    }
    try_create_environment();
    return S_OK;
  }
  HRESULT STDMETHODCALLTYPE Invoke(HRESULT res,
                                   ICoreWebView2Controller *controller) {
    if (FAILED(res)) {
      // See try_create_environment() regarding
      // HRESULT_FROM_WIN32(ERROR_INVALID_STATE).
      // The result is E_ABORT if the parent window has been destroyed already.
      switch (res) {
      case HRESULT_FROM_WIN32(ERROR_INVALID_STATE):
      case E_ABORT:
        return S_OK;
      }
      try_create_environment();
      return S_OK;
    }

    ICoreWebView2 *webview_base = nullptr;
    controller->get_CoreWebView2(&webview_base);

      ICoreWebView2_11 *webview = nullptr;
      if (webview_base) {
        webview_base->QueryInterface(IID_ICoreWebView2_11,
                                    reinterpret_cast<void **>(&webview));
        webview_base->Release();
      }
    if (!webview) {
      return E_FAIL;
    }

    ::EventRegistrationToken token;
    webview->add_WebMessageReceived(this, &token);
    webview->add_PermissionRequested(this, &token);
    webview->add_NewWindowRequested(this, &token);
    webview->add_NavigationStarting(this, &token);
    webview->add_ScriptDialogOpening(this, &token);
    webview->add_ContextMenuRequested(this, &token);

    m_cb(controller, webview);
    return S_OK;
  }
  HRESULT STDMETHODCALLTYPE Invoke(
      ICoreWebView2 *sender, ICoreWebView2WebMessageReceivedEventArgs *args) {
    LPWSTR message;
    args->TryGetWebMessageAsString(&message);
    m_msgCb(narrow_string(message));
    sender->PostWebMessageAsString(message);

    CoTaskMemFree(message);
    return S_OK;
  }
  HRESULT STDMETHODCALLTYPE
  Invoke(ICoreWebView2 * /*sender*/,
         ICoreWebView2PermissionRequestedEventArgs *args) {
    COREWEBVIEW2_PERMISSION_KIND kind;
    args->get_PermissionKind(&kind);
    if (kind == COREWEBVIEW2_PERMISSION_KIND_CLIPBOARD_READ) {
      args->put_State(COREWEBVIEW2_PERMISSION_STATE_ALLOW);
    }
    return S_OK;
  }

  // Checks whether the specified IID equals the IID of the specified type and
  // if so casts the "this" pointer to T and returns it. Returns nullptr on
  // mismatching IIDs.
  // If ppv is specified then the pointer will also be assigned to *ppv.
  template <typename T>
  T *cast_if_equal_iid(REFIID riid, const cast_info_t<T> &info,
                       LPVOID *ppv = nullptr) noexcept {
    T *ptr = nullptr;
    if (IsEqualIID(riid, info.iid)) {
      ptr = static_cast<T *>(this);
      ptr->AddRef();
    }
    if (ppv) {
      *ppv = ptr;
    }
    return ptr;
  }

  // Set the function that will perform the initiating logic for creating
  // the WebView2 environment.
  void set_attempt_handler(std::function<HRESULT()> attempt_handler) noexcept {
    m_attempt_handler = attempt_handler;
  }

  // Retry creating a WebView2 environment.
  // The initiating logic for creating the environment is defined by the
  // caller of set_attempt_handler().
  void try_create_environment() noexcept {
    // WebView creation fails with HRESULT_FROM_WIN32(ERROR_INVALID_STATE) if
    // a running instance using the same user data folder exists, and the
    // Environment objects have different EnvironmentOptions.
    // Source: https://docs.microsoft.com/en-us/microsoft-edge/webview2/reference/win32/icorewebview2environment?view=webview2-1.0.1150.38
    if (m_attempts < m_max_attempts) {
      ++m_attempts;
      auto res = m_attempt_handler();
      if (SUCCEEDED(res)) {
        return;
      }
      // Not entirely sure if this error code only applies to
      // CreateCoreWebView2Controller so we check here as well.
      if (res == HRESULT_FROM_WIN32(ERROR_INVALID_STATE)) {
        return;
      }
      try_create_environment();
      return;
    }
    // Give up.
    m_cb(nullptr, nullptr);
  }

private:
  HWND m_window;
  msg_cb_t m_msgCb;
  webview2_com_handler_cb_t m_cb;
  std::atomic<ULONG> m_ref_count{1};
  std::function<HRESULT()> m_attempt_handler;
  unsigned int m_max_attempts = 5;
  unsigned int m_attempts = 0;
};

class win32_edge_engine : public engine_base {
public:
  win32_edge_engine(bool debug, void *window) : m_owns_window{!window} {
    if (!is_webview2_available()) {
      return;
    }

    HINSTANCE hInstance = GetModuleHandle(nullptr);

    if (m_owns_window) {
      m_com_init = {COINIT_APARTMENTTHREADED};
      if (!m_com_init.is_initialized()) {
        return;
      }
      enable_dpi_awareness();

      HICON icon = (HICON)LoadImage(
          hInstance, IDI_APPLICATION, IMAGE_ICON, GetSystemMetrics(SM_CXICON),
          GetSystemMetrics(SM_CYICON), LR_DEFAULTCOLOR);

      // Create a top-level window.
      WNDCLASSEXW wc;
      ZeroMemory(&wc, sizeof(WNDCLASSEX));
      wc.cbSize = sizeof(WNDCLASSEX);
      wc.hInstance = hInstance;
      wc.lpszClassName = L"webview";
      wc.hIcon = icon;
      wc.lpfnWndProc = (WNDPROC)(+[](HWND hwnd, UINT msg, WPARAM wp,
                                     LPARAM lp) -> LRESULT {
        win32_edge_engine *w{};

        if (msg == WM_NCCREATE) {
          auto *lpcs{reinterpret_cast<LPCREATESTRUCT>(lp)};
          w = static_cast<win32_edge_engine *>(lpcs->lpCreateParams);
          w->m_window = hwnd;
          SetWindowLongPtrW(hwnd, GWLP_USERDATA, reinterpret_cast<LONG_PTR>(w));
          enable_non_client_dpi_scaling_if_needed(hwnd);
          apply_window_theme(hwnd);
        } else {
          w = reinterpret_cast<win32_edge_engine *>(
              GetWindowLongPtrW(hwnd, GWLP_USERDATA));
        }

        if (!w) {
          return DefWindowProcW(hwnd, msg, wp, lp);
        }

        switch (msg) {
        case WM_SIZE:
          w->resize_widget();
          break;
        case WM_CLOSE:
          DestroyWindow(hwnd);
          break;
        case WM_DESTROY:
          w->m_window = nullptr;
          SetWindowLongPtrW(hwnd, GWLP_USERDATA, 0);
          w->on_window_destroyed();
          break;
        case WM_GETMINMAXINFO: {
          auto lpmmi = (LPMINMAXINFO)lp;
          if (w->m_maxsz.x > 0 && w->m_maxsz.y > 0) {
            lpmmi->ptMaxSize = w->m_maxsz;
            lpmmi->ptMaxTrackSize = w->m_maxsz;
          }
          if (w->m_minsz.x > 0 && w->m_minsz.y > 0) {
            lpmmi->ptMinTrackSize = w->m_minsz;
          }
        } break;
        case 0x02E4 /*WM_GETDPISCALEDSIZE*/: {
          auto dpi = static_cast<int>(wp);
          auto *size{reinterpret_cast<SIZE *>(lp)};
          *size = w->get_scaled_size(w->m_dpi, dpi);
          return TRUE;
        }
        case 0x02E0 /*WM_DPICHANGED*/: {
          // Windows 10: The size we get here is exactly what we supplied to WM_GETDPISCALEDSIZE.
          // Windows 11: The size we get here is NOT what we supplied to WM_GETDPISCALEDSIZE.
          // Due to this difference, don't use the suggested bounds.
          auto dpi = static_cast<int>(HIWORD(wp));
          w->on_dpi_changed(dpi);
          break;
        }
        case WM_SETTINGCHANGE: {
          auto *area = reinterpret_cast<const wchar_t *>(lp);
          if (area) {
            w->on_system_setting_change(area);
          }
          break;
        }
        case WM_ACTIVATE:
          if (LOWORD(wp) != WA_INACTIVE) {
            w->focus_webview();
          }
          break;
        default:
          return DefWindowProcW(hwnd, msg, wp, lp);
        }
        return 0;
      });
      RegisterClassExW(&wc);

      CreateWindowW(L"webview", L"", WS_OVERLAPPEDWINDOW, CW_USEDEFAULT,
                    CW_USEDEFAULT, 0, 0, nullptr, nullptr, hInstance, this);
      if (m_window == nullptr) {
        return;
      }
      on_window_created();

      m_dpi = get_window_dpi(m_window);
      constexpr const int initial_width = 640;
      constexpr const int initial_height = 480;
      set_size(initial_width, initial_height, WEBVIEW_HINT_NONE);
    } else {
      m_window = IsWindow(static_cast<HWND>(window))
                     ? static_cast<HWND>(window)
                     : *(static_cast<HWND *>(window));
      m_dpi = get_window_dpi(m_window);
    }

    // Create a window that WebView2 will be embedded into.
    WNDCLASSEXW widget_wc{};
    widget_wc.cbSize = sizeof(WNDCLASSEX);
    widget_wc.hInstance = hInstance;
    widget_wc.lpszClassName = L"webview_widget";
    widget_wc.lpfnWndProc = (WNDPROC)(+[](HWND hwnd, UINT msg, WPARAM wp,
                                          LPARAM lp) -> LRESULT {
      win32_edge_engine *w{};

      if (msg == WM_NCCREATE) {
        auto *lpcs{reinterpret_cast<LPCREATESTRUCT>(lp)};
        w = static_cast<win32_edge_engine *>(lpcs->lpCreateParams);
        w->m_widget = hwnd;
        SetWindowLongPtrW(hwnd, GWLP_USERDATA, reinterpret_cast<LONG_PTR>(w));
      } else {
        w = reinterpret_cast<win32_edge_engine *>(
            GetWindowLongPtrW(hwnd, GWLP_USERDATA));
      }

      if (!w) {
        return DefWindowProcW(hwnd, msg, wp, lp);
      }

      switch (msg) {
      case WM_SIZE:
        w->resize_webview();
        break;
      case WM_DESTROY:
        w->m_widget = nullptr;
        SetWindowLongPtrW(hwnd, GWLP_USERDATA, 0);
        break;
      default:
        return DefWindowProcW(hwnd, msg, wp, lp);
      }
      return 0;
    });
    RegisterClassExW(&widget_wc);
    CreateWindowExW(WS_EX_CONTROLPARENT, L"webview_widget", nullptr, WS_CHILD,
                    0, 0, 0, 0, m_window, nullptr, hInstance, this);

    // Create a message-only window for internal messaging.
    WNDCLASSEXW message_wc{};
    message_wc.cbSize = sizeof(WNDCLASSEX);
    message_wc.hInstance = hInstance;
    message_wc.lpszClassName = L"webview_message";
    message_wc.lpfnWndProc = (WNDPROC)(+[](HWND hwnd, UINT msg, WPARAM wp,
                                           LPARAM lp) -> LRESULT {
      win32_edge_engine *w{};

      if (msg == WM_NCCREATE) {
        auto *lpcs{reinterpret_cast<LPCREATESTRUCT>(lp)};
        w = static_cast<win32_edge_engine *>(lpcs->lpCreateParams);
        w->m_message_window = hwnd;
        SetWindowLongPtrW(hwnd, GWLP_USERDATA, reinterpret_cast<LONG_PTR>(w));
      } else {
        w = reinterpret_cast<win32_edge_engine *>(
            GetWindowLongPtrW(hwnd, GWLP_USERDATA));
      }

      if (!w) {
        return DefWindowProcW(hwnd, msg, wp, lp);
      }

      switch (msg) {
      case WM_APP:
        if (auto f = (dispatch_fn_t *)(lp)) {
          (*f)();
          delete f;
        }
        break;
      case WM_DESTROY:
        w->m_message_window = nullptr;
        SetWindowLongPtrW(hwnd, GWLP_USERDATA, 0);
        break;
      default:
        return DefWindowProcW(hwnd, msg, wp, lp);
      }
      return 0;
    });
    RegisterClassExW(&message_wc);
    CreateWindowExW(0, L"webview_message", nullptr, 0, 0, 0, 0, 0, HWND_MESSAGE,
                    nullptr, hInstance, this);

    if (m_owns_window) {
      // ShowWindow(m_window, SW_SHOW);
      UpdateWindow(m_window);
      SetFocus(m_window);
    }

    auto cb =
        std::bind(&win32_edge_engine::on_message, this, std::placeholders::_1);

    embed(m_widget, debug, cb);
  }

  virtual ~win32_edge_engine() {
    if (m_com_handler) {
      m_com_handler->Release();
      m_com_handler = nullptr;
    }
    if (m_webview) {
      m_webview->Release();
      m_webview = nullptr;
    }
    if (m_controller) {
      m_controller->Release();
      m_controller = nullptr;
    }
    // Replace wndproc to avoid callbacks and other bad things during
    // destruction.
    auto wndproc = reinterpret_cast<LONG_PTR>(
        +[](HWND hwnd, UINT msg, WPARAM wp, LPARAM lp) -> LRESULT {
          return DefWindowProcW(hwnd, msg, wp, lp);
        });
    if (m_widget) {
      SetWindowLongPtrW(m_widget, GWLP_WNDPROC, wndproc);
    }
    if (m_window && m_owns_window) {
      SetWindowLongPtrW(m_window, GWLP_WNDPROC, wndproc);
    }
    if (m_widget) {
      DestroyWindow(m_widget);
      m_widget = nullptr;
    }
    if (m_window) {
      if (m_owns_window) {
        DestroyWindow(m_window);
        on_window_destroyed(true);
      }
      m_window = nullptr;
    }
    if (m_owns_window) {
      // Not strictly needed for windows to close immediately but aligns
      // behavior across backends.
      deplete_run_loop_event_queue();
    }
    // We need the message window in order to deplete the event queue.
    if (m_message_window) {
      SetWindowLongPtrW(m_message_window, GWLP_WNDPROC, wndproc);
      DestroyWindow(m_message_window);
      m_message_window = nullptr;
    }
  }

  win32_edge_engine(const win32_edge_engine &other) = delete;
  win32_edge_engine &operator=(const win32_edge_engine &other) = delete;
  win32_edge_engine(win32_edge_engine &&other) = delete;
  win32_edge_engine &operator=(win32_edge_engine &&other) = delete;

  void run_impl() override {
    MSG msg;
    while (GetMessageW(&msg, nullptr, 0, 0) > 0) {
      TranslateMessage(&msg);
      DispatchMessageW(&msg);
    }
  }
  void *window_impl() override { return (void *)m_window; }
  void *widget_impl() override { return (void *)m_widget; }
  void *browser_controller_impl() override { return (void *)m_controller; }
  void terminate_impl() override { PostQuitMessage(0); }
  void dispatch_impl(dispatch_fn_t f) override {
    PostMessageW(m_message_window, WM_APP, 0, (LPARAM) new dispatch_fn_t(f));
  }

  void set_title_impl(const std::string &title) override {
    SetWindowTextW(m_window, widen_string(title).c_str());
  }

  void set_size_impl(int width, int height, webview_hint_t hints) override {
    auto style = GetWindowLong(m_window, GWL_STYLE);
    if (hints == WEBVIEW_HINT_FIXED) {
      style &= ~(WS_THICKFRAME | WS_MAXIMIZEBOX);
    } else {
      style |= (WS_THICKFRAME | WS_MAXIMIZEBOX);
    }
    SetWindowLong(m_window, GWL_STYLE, style);

    if (hints == WEBVIEW_HINT_MAX) {
      m_maxsz.x = width;
      m_maxsz.y = height;
    } else if (hints == WEBVIEW_HINT_MIN) {
      m_minsz.x = width;
      m_minsz.y = height;
    } else {
      auto dpi = get_window_dpi(m_window);
      m_dpi = dpi;
      auto scaled_size =
          scale_size(width, height, get_default_window_dpi(), dpi);
      auto frame_size =
          make_window_frame_size(m_window, scaled_size.cx, scaled_size.cy, dpi);
      SetWindowPos(m_window, nullptr, 0, 0, frame_size.cx, frame_size.cy,
                   SWP_NOZORDER | SWP_NOACTIVATE | SWP_NOMOVE |
                       SWP_FRAMECHANGED);
    }
  }

  void navigate_impl(const std::string &url) override {
    auto wurl = widen_string(url);
    m_webview->Navigate(wurl.c_str());
  }

  void init_impl(const std::string &js) override {
    auto wjs = widen_string(js);
    m_webview->AddScriptToExecuteOnDocumentCreated(wjs.c_str(), nullptr);
  }

  void eval_impl(const std::string &js) override {
    auto wjs = widen_string(js);
    m_webview->ExecuteScript(wjs.c_str(), nullptr);
  }

  void set_html_impl(const std::string &html) override {
    m_webview->NavigateToString(widen_string(html).c_str());
  }

  void set_zoom_impl(double level) override {
    if (m_controller) {
      m_controller->put_ZoomFactor(level);
    }
  }

  double get_zoom_impl() override {
    double zoom = 1.0;
    if (m_controller) {
      m_controller->get_ZoomFactor(&zoom);
    }
    return zoom;
  }

private:
  bool embed(HWND wnd, bool debug, msg_cb_t cb) {
    std::atomic_flag flag = ATOMIC_FLAG_INIT;
    flag.test_and_set();

    wchar_t currentExePath[MAX_PATH];
    GetModuleFileNameW(nullptr, currentExePath, MAX_PATH);
    wchar_t *currentExeName = PathFindFileNameW(currentExePath);

    wchar_t dataPath[MAX_PATH];
    if (!SUCCEEDED(
            SHGetFolderPathW(nullptr, CSIDL_APPDATA, nullptr, 0, dataPath))) {
      return false;
    }
    wchar_t userDataFolder[MAX_PATH];
    PathCombineW(userDataFolder, dataPath, currentExeName);

    m_com_handler = new webview2_com_handler(
        wnd, cb,
        [&](ICoreWebView2Controller *controller, ICoreWebView2_11 *webview) {
          if (!controller || !webview) {
            flag.clear();
            return;
          }
          controller->AddRef();
          webview->AddRef();
          m_controller = controller;
          m_webview = webview;
          flag.clear();
        });

    m_com_handler->set_attempt_handler([&] {
      return m_webview2_loader.create_environment_with_options(
          nullptr, userDataFolder, nullptr, m_com_handler);
    });
    m_com_handler->try_create_environment();

    // Pump the message loop until WebView2 has finished initialization.
    bool got_quit_msg = false;
    MSG msg;
    while (flag.test_and_set() && GetMessageW(&msg, nullptr, 0, 0) >= 0) {
      if (msg.message == WM_QUIT) {
        got_quit_msg = true;
        break;
      }
      TranslateMessage(&msg);
      DispatchMessageW(&msg);
    }
    if (got_quit_msg) {
      return false;
    }
    if (!m_controller || !m_webview) {
      return false;
    }
    ICoreWebView2Settings *settings = nullptr;
    auto res = m_webview->get_Settings(&settings);
    if (res != S_OK) {
      return false;
    }
    res = settings->put_AreDevToolsEnabled(debug ? TRUE : FALSE);
    if (res != S_OK) {
      return false;
    }
    res = settings->put_IsStatusBarEnabled(FALSE);
    if (res != S_OK) {
      return false;
    }
    res = settings->put_AreDefaultScriptDialogsEnabled(FALSE);
    if (res != S_OK) {
      return false;
    }
    res = settings->put_IsZoomControlEnabled(FALSE);
    if (res != S_OK) {
      return false;
    }
    init("window.external={invoke:s=>window.chrome.webview.postMessage(s)}");
    resize_webview();
    m_controller->put_IsVisible(TRUE);
    ShowWindow(m_widget, SW_SHOW);
    UpdateWindow(m_widget);
    if (m_owns_window) {
      focus_webview();
    }
    return true;
  }

  void resize_widget() {
    if (m_widget) {
      RECT r{};
      if (GetClientRect(GetParent(m_widget), &r)) {
        MoveWindow(m_widget, r.left, r.top, r.right - r.left, r.bottom - r.top,
                   TRUE);
      }
    }
  }

  void resize_webview() {
    if (m_widget && m_controller) {
      RECT bounds{};
      if (GetClientRect(m_widget, &bounds)) {
        m_controller->put_Bounds(bounds);
      }
    }
  }

  void focus_webview() {
    if (m_controller) {
      m_controller->MoveFocus(COREWEBVIEW2_MOVE_FOCUS_REASON_PROGRAMMATIC);
    }
  }

  bool is_webview2_available() const noexcept {
    LPWSTR version_info = nullptr;
    auto res = m_webview2_loader.get_available_browser_version_string(
        nullptr, &version_info);
    // The result will be equal to HRESULT_FROM_WIN32(ERROR_FILE_NOT_FOUND)
    // if the WebView2 runtime is not installed.
    auto ok = SUCCEEDED(res) && version_info;
    if (version_info) {
      CoTaskMemFree(version_info);
    }
    return ok;
  }

  void on_dpi_changed(int dpi) {
    auto scaled_size = get_scaled_size(m_dpi, dpi);
    auto frame_size =
        make_window_frame_size(m_window, scaled_size.cx, scaled_size.cy, dpi);
    SetWindowPos(m_window, nullptr, 0, 0, frame_size.cx, frame_size.cy,
                 SWP_NOZORDER | SWP_NOACTIVATE | SWP_NOMOVE | SWP_FRAMECHANGED);
    m_dpi = dpi;
  }

  SIZE get_size() const {
    RECT bounds;
    GetClientRect(m_window, &bounds);
    auto width = bounds.right - bounds.left;
    auto height = bounds.bottom - bounds.top;
    return {width, height};
  }

  SIZE get_scaled_size(int from_dpi, int to_dpi) const {
    auto size = get_size();
    return scale_size(size.cx, size.cy, from_dpi, to_dpi);
  }

  void on_system_setting_change(const wchar_t *area) {
    // Detect light/dark mode change in system.
    if (lstrcmpW(area, L"ImmersiveColorSet") == 0) {
      apply_window_theme(m_window);
    }
  }

  // Blocks while depleting the run loop of events.
  void deplete_run_loop_event_queue() {
    bool done{};
    dispatch([&] { done = true; });
    while (!done) {
      MSG msg;
      if (GetMessageW(&msg, nullptr, 0, 0) > 0) {
        TranslateMessage(&msg);
        DispatchMessageW(&msg);
      }
    }
  }

  // The app is expected to call CoInitializeEx before
  // CreateCoreWebView2EnvironmentWithOptions.
  // Source: https://docs.microsoft.com/en-us/microsoft-edge/webview2/reference/win32/webview2-idl#createcorewebview2environmentwithoptions
  com_init_wrapper m_com_init;
  HWND m_window = nullptr;
  HWND m_widget = nullptr;
  HWND m_message_window = nullptr;
  POINT m_minsz = POINT{0, 0};
  POINT m_maxsz = POINT{0, 0};
  DWORD m_main_thread = GetCurrentThreadId();
  ICoreWebView2_11 *m_webview = nullptr;
  ICoreWebView2Controller *m_controller = nullptr;
  webview2_com_handler *m_com_handler = nullptr;
  mswebview2::loader m_webview2_loader;
  int m_dpi{};
  bool m_owns_window{};
};

} // namespace detail

using browser_engine = detail::win32_edge_engine;

} // namespace webview

#endif /* WEBVIEW_GTK, WEBVIEW_COCOA, WEBVIEW_EDGE */

namespace webview {
using webview = browser_engine;
} // namespace webview

WEBVIEW_API webview_t webview_create(int debug, void *wnd) {
  auto w = new webview::webview(debug, wnd);
  if (!w->window()) {
    delete w;
    return nullptr;
  }
  return w;
}

WEBVIEW_API void webview_destroy(webview_t w) {
  delete static_cast<webview::webview *>(w);
}

WEBVIEW_API void webview_run(webview_t w) {
  static_cast<webview::webview *>(w)->run();
}

WEBVIEW_API void webview_terminate(webview_t w) {
  static_cast<webview::webview *>(w)->terminate();
}

WEBVIEW_API void webview_dispatch(webview_t w, void (*fn)(webview_t, void *),
                                  void *arg) {
  static_cast<webview::webview *>(w)->dispatch([=]() { fn(w, arg); });
}

WEBVIEW_API void *webview_get_window(webview_t w) {
  return static_cast<webview::webview *>(w)->window();
}

WEBVIEW_API void *webview_get_native_handle(webview_t w,
                                            webview_native_handle_kind_t kind) {
  auto *w_ = static_cast<webview::webview *>(w);
  switch (kind) {
  case WEBVIEW_NATIVE_HANDLE_KIND_UI_WINDOW:
    return w_->window();
  case WEBVIEW_NATIVE_HANDLE_KIND_UI_WIDGET:
    return w_->widget();
  case WEBVIEW_NATIVE_HANDLE_KIND_BROWSER_CONTROLLER:
    return w_->browser_controller();
  default:
    return nullptr;
  }
}

WEBVIEW_API void webview_set_title(webview_t w, const char *title) {
  static_cast<webview::webview *>(w)->set_title(title);
}

WEBVIEW_API void webview_set_size(webview_t w, int width, int height,
                                  webview_hint_t hints) {
  static_cast<webview::webview *>(w)->set_size(width, height, hints);
}

WEBVIEW_API void webview_navigate(webview_t w, const char *url) {
  static_cast<webview::webview *>(w)->navigate(url);
}

WEBVIEW_API void webview_set_html(webview_t w, const char *html) {
  static_cast<webview::webview *>(w)->set_html(html);
}

WEBVIEW_API void webview_init(webview_t w, const char *js) {
  static_cast<webview::webview *>(w)->init(js);
}

WEBVIEW_API void webview_eval(webview_t w, const char *js) {
  static_cast<webview::webview *>(w)->eval(js);
}

WEBVIEW_API void webview_bind(webview_t w, const char *name,
                              void (*fn)(const char *seq, const char *req,
                                         void *arg),
                              void *arg) {
  static_cast<webview::webview *>(w)->bind(
      name,
      [=](const std::string &seq, const std::string &req, void *arg) {
        fn(seq.c_str(), req.c_str(), arg);
      },
      arg);
}

WEBVIEW_API void webview_unbind(webview_t w, const char *name) {
  static_cast<webview::webview *>(w)->unbind(name);
}

WEBVIEW_API void webview_return(webview_t w, const char *seq, int status,
                                const char *result) {
  static_cast<webview::webview *>(w)->resolve(seq, status, result);
}

WEBVIEW_API const webview_version_info_t *webview_version(void) {
  return &webview::detail::library_version_info;
}

WEBVIEW_API void webview_set_zoom(webview_t w, double level) {
  static_cast<webview::webview *>(w)->set_zoom(level);
}

WEBVIEW_API double webview_get_zoom(webview_t w) {
  return static_cast<webview::webview *>(w)->get_zoom();
}

#endif /* WEBVIEW_HEADER */
#endif /* __cplusplus */
#endif /* WEBVIEW_H */
