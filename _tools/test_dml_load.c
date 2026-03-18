#include <windows.h>
#include <stdio.h>

typedef void* (*PFN_ggml_backend_init)(void);

int main() {
    printf("Testing ggml-directml.dll load...\n");

    HMODULE mod = LoadLibraryA("C:\\Users\\smithdavi\\AppData\\Local\\Programs\\Ollama\\lib\\ollama\\directml\\ggml-directml.dll");
    if (!mod) {
        printf("ERROR: Failed to load ggml-directml.dll, error=%lu\n", GetLastError());
        return 1;
    }
    printf("OK: ggml-directml.dll loaded\n");

    PFN_ggml_backend_init init_fn = (PFN_ggml_backend_init)GetProcAddress(mod, "ggml_backend_init");
    if (!init_fn) {
        printf("ERROR: ggml_backend_init not found, error=%lu\n", GetLastError());
    } else {
        printf("OK: ggml_backend_init found at %p\n", init_fn);
    }

    void* score_fn = GetProcAddress(mod, "ggml_backend_score");
    if (!score_fn) {
        printf("INFO: ggml_backend_score not found (optional)\n");
    } else {
        printf("OK: ggml_backend_score found at %p\n", score_fn);
    }

    // Also try loading DirectML.dll to see if it can be found
    HMODULE dml = LoadLibraryA("DirectML.dll");
    if (!dml) {
        printf("ERROR: DirectML.dll not loadable, error=%lu\n", GetLastError());
    } else {
        printf("OK: DirectML.dll loaded\n");
        void* create_fn = GetProcAddress(dml, "DMLCreateDevice1");
        if (!create_fn) {
            printf("ERROR: DMLCreateDevice1 not found, error=%lu\n", GetLastError());
            create_fn = GetProcAddress(dml, "DMLCreateDevice");
            if (create_fn) {
                printf("OK: DMLCreateDevice found (older API)\n");
            }
        } else {
            printf("OK: DMLCreateDevice1 found at %p\n", create_fn);
        }
        FreeLibrary(dml);
    }

    FreeLibrary(mod);
    printf("Done.\n");
    return 0;
}
