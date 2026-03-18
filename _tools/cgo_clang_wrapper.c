// cgo-clang wrapper: filters out GCC-specific flags that clang doesn't support
// Build with: clang cgo_clang_wrapper.c -o cgo-clang.exe
#include <windows.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

static int should_skip(const char *arg) {
    if (strcmp(arg, "-mthreads") == 0) return 1;
    if (strcmp(arg, "-lmingwthrd") == 0) return 1;
    if (strcmp(arg, "-lmingw32") == 0) return 1;
    if (strcmp(arg, "-lpthread") == 0) return 1;
    if (strcmp(arg, "-lmsvcrt") == 0) return 1;
    if (strcmp(arg, "-static") == 0) return 1;
    if (strcmp(arg, "-static-libgcc") == 0) return 1;
    if (strcmp(arg, "-static-libstdc++") == 0) return 1;
    if (strncmp(arg, "-fmessage-length", 16) == 0) return 1;
    if (strncmp(arg, "-fdebug-prefix-map", 18) == 0) return 1;
    if (strncmp(arg, "-fno-diagnostics-show-note", 25) == 0) return 1;
    return 0;
}

int main(int argc, char **argv) {
    // Build command line for clang
    char cmdline[32768] = "\"C:\\Program Files\\LLVM\\bin\\clang.exe\" --target=aarch64-pc-windows-msvc";
    size_t pos = strlen(cmdline);

    for (int i = 1; i < argc; i++) {
        if (should_skip(argv[i])) continue;

        // Check if arg needs quoting
        int needs_quote = (strchr(argv[i], ' ') != NULL);
        if (needs_quote) {
            pos += snprintf(cmdline + pos, sizeof(cmdline) - pos, " \"%s\"", argv[i]);
        } else {
            pos += snprintf(cmdline + pos, sizeof(cmdline) - pos, " %s", argv[i]);
        }
    }

    // Execute
    STARTUPINFOA si = { .cb = sizeof(si) };
    PROCESS_INFORMATION pi;

    if (!CreateProcessA(NULL, cmdline, NULL, NULL, TRUE, 0, NULL, NULL, &si, &pi)) {
        fprintf(stderr, "cgo-clang: CreateProcess failed (%lu)\n", GetLastError());
        return 1;
    }

    WaitForSingleObject(pi.hProcess, INFINITE);
    DWORD exit_code;
    GetExitCodeProcess(pi.hProcess, &exit_code);
    CloseHandle(pi.hProcess);
    CloseHandle(pi.hThread);
    return (int)exit_code;
}
