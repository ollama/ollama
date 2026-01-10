#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <stdexcept>
#include <array>
#include <vector>
#include <map>
#include <thread>
#include <mutex>
#include <future>
#include <queue>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cassert>
#include <algorithm>
#include <sys/stat.h>
#include <sys/types.h>
#include <filesystem>

#ifdef _WIN32
    #define NOMINMAX
    #include <windows.h>
    #include <direct.h> // For _mkdir on Windows
#else
    #include <unistd.h>
    #include <sys/wait.h>
    #include <fcntl.h>
#endif

#define ASYNCIO_CONCURRENCY 64

std::mutex lock;
std::vector<std::pair<std::string, std::string>> shader_fnames;
std::locale c_locale("C");

std::string GLSLC = "glslc";
std::string input_filepath = "";
std::string output_dir = "/tmp";
std::string target_hpp = "";
std::string target_cpp = "";

const std::vector<std::string> type_names = {
    "f32",
    "f16",
    "q4_0",
    "q4_1",
    "q5_0",
    "q5_1",
    "q8_0",
    "q2_k",
    "q3_k",
    "q4_k",
    "q5_k",
    "q6_k",
    "iq1_s",
    "iq1_m",
    "iq2_xxs",
    "iq2_xs",
    "iq2_s",
    "iq3_xxs",
    "iq3_s",
    "iq4_xs",
    "iq4_nl",
    "mxfp4",
    "bf16",
};

enum MatMulIdType {
    NONE,
    DEFAULT,
    SUBGROUP,
};

namespace {

void execute_command(std::vector<std::string>& command, std::string& stdout_str, std::string& stderr_str) {
#ifdef _WIN32
    HANDLE stdout_read, stdout_write;
    HANDLE stderr_read, stderr_write;
    SECURITY_ATTRIBUTES sa = { sizeof(SECURITY_ATTRIBUTES), NULL, TRUE };

    if (!CreatePipe(&stdout_read, &stdout_write, &sa, 0) ||
        !SetHandleInformation(stdout_read, HANDLE_FLAG_INHERIT, 0)) {
        throw std::runtime_error("Failed to create stdout pipe");
    }

    if (!CreatePipe(&stderr_read, &stderr_write, &sa, 0) ||
        !SetHandleInformation(stderr_read, HANDLE_FLAG_INHERIT, 0)) {
        throw std::runtime_error("Failed to create stderr pipe");
    }

    PROCESS_INFORMATION pi;
    STARTUPINFOA si = {};
    si.cb = sizeof(STARTUPINFOA);
    si.dwFlags = STARTF_USESTDHANDLES;
    si.hStdOutput = stdout_write;
    si.hStdError = stderr_write;

    std::string cmd;
    for (const auto& part : command) {
        cmd += part + " ";
    }

    if (!CreateProcessA(NULL, cmd.data(), NULL, NULL, TRUE, 0, NULL, NULL, &si, &pi)) {
        throw std::runtime_error("Failed to create process");
    }

    CloseHandle(stdout_write);
    CloseHandle(stderr_write);

    std::array<char, 128> buffer;
    DWORD bytes_read;

    while (ReadFile(stdout_read, buffer.data(), (DWORD)buffer.size(), &bytes_read, NULL) && bytes_read > 0) {
        stdout_str.append(buffer.data(), bytes_read);
    }

    while (ReadFile(stderr_read, buffer.data(), (DWORD)buffer.size(), &bytes_read, NULL) && bytes_read > 0) {
        stderr_str.append(buffer.data(), bytes_read);
    }

    CloseHandle(stdout_read);
    CloseHandle(stderr_read);
    WaitForSingleObject(pi.hProcess, INFINITE);
    CloseHandle(pi.hProcess);
    CloseHandle(pi.hThread);
#else
    int stdout_pipe[2];
    int stderr_pipe[2];

    if (pipe(stdout_pipe) != 0 || pipe(stderr_pipe) != 0) {
        throw std::runtime_error("Failed to create pipes");
    }

    pid_t pid = fork();
    if (pid < 0) {
        throw std::runtime_error("Failed to fork process");
    }

    std::vector<char*> argv;
    for (std::string& part : command) {
        argv.push_back(part.data());
    }
    argv.push_back(nullptr);

    if (pid == 0) {
        close(stdout_pipe[0]);
        close(stderr_pipe[0]);
        dup2(stdout_pipe[1], STDOUT_FILENO);
        dup2(stderr_pipe[1], STDERR_FILENO);
        close(stdout_pipe[1]);
        close(stderr_pipe[1]);
        execvp(argv[0], argv.data());
        _exit(EXIT_FAILURE);
    } else {
        close(stdout_pipe[1]);
        close(stderr_pipe[1]);

        std::array<char, 128> buffer;
        ssize_t bytes_read;

        while ((bytes_read = read(stdout_pipe[0], buffer.data(), buffer.size())) > 0) {
            stdout_str.append(buffer.data(), bytes_read);
        }

        while ((bytes_read = read(stderr_pipe[0], buffer.data(), buffer.size())) > 0) {
            stderr_str.append(buffer.data(), bytes_read);
        }

        close(stdout_pipe[0]);
        close(stderr_pipe[0]);
        waitpid(pid, nullptr, 0);
    }
#endif
}

bool directory_exists(const std::string& path) {
    struct stat info;
    if (stat(path.c_str(), &info) != 0) {
        return false; // Path doesn't exist or can't be accessed
    }
    return (info.st_mode & S_IFDIR) != 0; // Check if it is a directory
}

bool create_directory(const std::string& path) {
#ifdef _WIN32
    return _mkdir(path.c_str()) == 0 || errno == EEXIST; // EEXIST means the directory already exists
#else
    return mkdir(path.c_str(), 0755) == 0 || errno == EEXIST; // 0755 is the directory permissions
#endif
}

std::string to_uppercase(const std::string& input) {
    std::string result = input;
    for (char& c : result) {
        c = std::toupper(c);
    }
    return result;
}

bool string_starts_with(const std::string& str, const std::string& prefix) {
    if (prefix.size() > str.size()) {
        return false;
    }
    return std::equal(prefix.begin(), prefix.end(), str.begin());
}

bool string_ends_with(const std::string& str, const std::string& suffix) {
    if (suffix.size() > str.size()) {
        return false;
    }
    return std::equal(suffix.rbegin(), suffix.rend(), str.rbegin());
}

bool is_quantized_type(const std::string& type_name) {
    return type_name != "f32" && type_name != "f16" && type_name != "bf16";
}

bool is_legacy_quant(const std::string& type_name) {
    return type_name == "q4_0" || type_name == "q4_1" || type_name == "q5_0" || type_name == "q5_1" || type_name == "q8_0";
}

bool is_k_quant(const std::string& type_name) {
    return string_ends_with(type_name, "_k");
}

bool is_iq_quant(const std::string& type_name) {
    return string_starts_with(type_name, "iq");
}

static const char path_separator = '/';

std::string join_paths(const std::string& path1, const std::string& path2) {
    return path1 + path_separator + path2;
}

std::string basename(const std::string &path) {
    return path.substr(path.find_last_of("/\\") + 1);
}

std::stringstream make_generic_stringstream() {
    std::stringstream ss;
    ss.imbue(c_locale);
    return ss;
}

std::string read_binary_file(const std::string& path, bool may_not_exist = false) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) {
        if (!may_not_exist) {
            std::cerr << "Error opening file: " << path << " (" << strerror(errno) << ")\n";
        }
        return {};
    }

    fseek(f, 0, SEEK_END);
    size_t size = ftell(f);
    fseek(f, 0, SEEK_SET);

    std::string data(size, '\0');
    size_t read_size = fread(data.data(), 1, size, f);
    fclose(f);
    if (read_size != size) {
        std::cerr << "Error reading file: " << path << " (" << strerror(errno) << ")\n";
        return {};
    }

    return data;
}

void write_binary_file(const std::string& path, const std::string& content) {
    FILE* f = fopen(path.c_str(), "wb");
    if (!f) {
        std::cerr << "Error opening file for writing: " << path << " (" << strerror(errno) << ")\n";
        return;
    }

    size_t write_size = fwrite(content.data(), 1, content.size(), f);
    fclose(f);
    if (write_size != content.size()) {
        std::cerr << "Error writing file: " << path << " (" << strerror(errno) << ")\n";
        return;
    }
}

void write_file_if_changed(const std::string& path, const std::string& content) {
    std::string existing = read_binary_file(path, true);
    if (existing != content) {
        write_binary_file(path, content);
    }
}


// variables to track number of compiles in progress
static uint32_t compile_count = 0;
static std::mutex compile_count_mutex;
static std::condition_variable compile_count_cond;
static bool generate_dep_file = true;

void decrement_compile_count(uint32_t * count) {
    if (count) {
        std::lock_guard<std::mutex> guard(compile_count_mutex);
        assert(compile_count > 0);
        compile_count--;
        compile_count_cond.notify_all();
    }
}

using compile_count_guard = std::unique_ptr<uint32_t, decltype(&decrement_compile_count)>;

compile_count_guard acquire_compile_slot() {
    // wait until fewer than N compiles are in progress.
    // 16 is an arbitrary limit, the goal is to avoid "failed to create pipe" errors.
    uint32_t N = std::max(1u, std::min(16u, std::thread::hardware_concurrency()));
    std::unique_lock<std::mutex> guard(compile_count_mutex);
    compile_count_cond.wait(guard, [N] { return compile_count < N; });
    compile_count++;
    return compile_count_guard(&compile_count, &decrement_compile_count);
}

void string_to_spv_func(std::string name, std::string in_path, std::string out_path, std::map<std::string, std::string> defines, bool coopmat, bool dep_file, compile_count_guard slot) {
    std::string target_env = (name.find("_cm2") != std::string::npos) ? "--target-env=vulkan1.3" : "--target-env=vulkan1.2";

    #ifdef _WIN32
        std::vector<std::string> cmd = {GLSLC, "-fshader-stage=compute", target_env, "\"" + in_path + "\"", "-o", "\"" + out_path + "\""};
    #else
        std::vector<std::string> cmd = {GLSLC, "-fshader-stage=compute", target_env, in_path, "-o", out_path};
    #endif

    // disable spirv-opt for coopmat shaders for https://github.com/ggerganov/llama.cpp/issues/10734
    // disable spirv-opt for bf16 shaders for https://github.com/ggml-org/llama.cpp/issues/15344
    // disable spirv-opt for rope shaders for https://github.com/ggml-org/llama.cpp/issues/16860
    if (!coopmat && name.find("bf16") == std::string::npos && name.find("rope") == std::string::npos) {
        cmd.push_back("-O");
    }

    if (dep_file) {
        cmd.push_back("-MD");
        cmd.push_back("-MF");
#ifdef _WIN32
        cmd.push_back("\"" + target_cpp + ".d\"");
#else
        cmd.push_back(target_cpp + ".d");
#endif
    }

    #ifdef GGML_VULKAN_SHADER_DEBUG_INFO
        cmd.push_back("-g");
    #endif

    for (const auto& define : defines) {
        cmd.push_back("-D" + define.first + "=" + define.second);
    }

    std::string command;
    for (const auto& part : cmd) {
        command += part + " ";
    }

    std::string stdout_str, stderr_str;
    try {
        // std::cout << "Executing command: ";
        // for (const auto& part : cmd) {
        //     std::cout << part << " ";
        // }
        // std::cout << std::endl;

        execute_command(cmd, stdout_str, stderr_str);
        if (!stderr_str.empty()) {
            std::cerr << "cannot compile " << name << "\n\n";
            for (const auto& part : cmd) {
                std::cerr << part << " ";
            }
            std::cerr << "\n\n" << stderr_str << std::endl;
            return;
        }

        if (dep_file) {
            // replace .spv output path with the embed .cpp path which is used as output in CMakeLists.txt
            std::string dep = read_binary_file(target_cpp + ".d", true);
            if (!dep.empty()) {
                size_t pos = dep.find(out_path);
                if (pos != std::string::npos) {
                    dep.replace(pos, out_path.length(), target_cpp);
                }
                write_binary_file(target_cpp + ".d", dep);
            }
        }

        std::lock_guard<std::mutex> guard(lock);
        shader_fnames.push_back(std::make_pair(name, out_path));
    } catch (const std::exception& e) {
        std::cerr << "Error executing command for " << name << ": " << e.what() << std::endl;
    }
}

std::map<std::string, std::string> merge_maps(const std::map<std::string, std::string>& a, const std::map<std::string, std::string>& b) {
    std::map<std::string, std::string> result = a;
    result.insert(b.begin(), b.end());
    return result;
}

static std::vector<std::future<void>> compiles;
void string_to_spv(std::string name, const std::string& source, const std::map<std::string, std::string>& defines, bool fp16 = true, bool coopmat = false, bool coopmat2 = false, bool f16acc = false) {
    name = name + (f16acc ? "_f16acc" : "") + (coopmat ? "_cm1" : "") + (coopmat2 ? "_cm2" : (fp16 ? "" : "_fp32"));
    std::string out_path = join_paths(output_dir, name + ".spv");

    if (input_filepath == "") {
        // No input source to compile, only generate header for all shaders
        shader_fnames.push_back(std::pair(name, out_path));
        return;
    } else if (basename(input_filepath) != source) {
        // Only compile shader variants matching the input filename
        return;
    }

    compile_count_guard slot = acquire_compile_slot();
    compiles.push_back(std::async(
        string_to_spv_func, name, input_filepath, out_path, defines, coopmat, generate_dep_file, std::move(slot)));
    // Don't write the same dep file from multiple processes
    generate_dep_file = false;
}

void matmul_shaders(bool fp16, MatMulIdType matmul_id_type, bool coopmat, bool coopmat2, bool f16acc) {
    std::string load_vec = coopmat2 ? "1" : fp16 ? "8" : "4";
    std::string aligned_b_type_f32 = coopmat2 ? "float" : fp16 ? "mat2x4" : "vec4";
    std::string aligned_b_type_f16 = coopmat2 ? "float16_t" : fp16 ? "f16mat2x4" : "f16vec4";

    std::map<std::string, std::string> base_dict;
    std::string shader_name = "matmul";

    if (matmul_id_type == MatMulIdType::DEFAULT) {
        base_dict["MUL_MAT_ID"] = "1";
        shader_name = "matmul_id";
    } else if (matmul_id_type == MatMulIdType::SUBGROUP) {
        base_dict["MUL_MAT_ID"] = "1";
        base_dict["MUL_MAT_ID_USE_SUBGROUPS"] = "1";
        shader_name = "matmul_id_subgroup";
    }

    if (fp16) {
        base_dict["FLOAT16"] = "1";
    }

    base_dict["ACC_TYPE"     ] = f16acc ? "float16_t" : "float";
    base_dict["ACC_TYPE_VEC2"] = f16acc ? "f16vec2"   : "vec2";
    if (f16acc) {
        base_dict["ACC_TYPE_MAX"] = "float16_t(65504.0)";
    }

    if (coopmat) {
        base_dict["COOPMAT"] = "1";
    }

    const std::string source_name = coopmat2 ? "mul_mm_cm2.comp" : "mul_mm.comp";

    auto const &FLOAT_TYPE = [&](int vec, const std::string &t) -> std::string {
        switch (vec) {
        case 1:
            if (t == "bf16") {
                // scalar path promotes to float
                if (!coopmat && !coopmat2) {
                    return "float";
                }
                return "bfloat16_t";
            }
            if (coopmat2 || fp16) {
                return "float16_t";
            }
            return "float";
        case 2:
            if (t == "bf16") {
                // scalar path promotes to float
                if (!coopmat && !coopmat2) {
                    return "vec2";
                }
                return "bf16vec2";
            }
            if (coopmat2 || fp16) {
                return "f16vec2";
            }
            return "vec2";
        case 4:
            if (t == "bf16") {
                // scalar path promotes to float
                if (!coopmat && !coopmat2) {
                    return "vec4";
                }
                return "bf16vec4";
            }
            if (coopmat2 || fp16) {
                return "f16vec4";
            }
            return "vec4";
        case 8:
            if (t == "bf16") {
                // scalar path promotes to float
                if (!coopmat && !coopmat2) {
                    return "mat2x4";
                }
                throw std::runtime_error("bf16 vec8 not supported");
            }
            if (coopmat2 || fp16) {
                return "f16mat2x4";
            }
            return "mat2x4";
        default:
            throw std::runtime_error("invalid vector size");
        }
    };

    const std::map<std::string, std::string> float_type_dict_f16 = {
        {"FLOAT_TYPE",      FLOAT_TYPE(1, "f16")},
        {"FLOAT_TYPE_VEC2", FLOAT_TYPE(2, "f16")},
        {"FLOAT_TYPE_VEC4", FLOAT_TYPE(4, "f16")},
        {"FLOAT_TYPE_VEC8", FLOAT_TYPE(8, "f16")},
    };

    // Shaders with f16 B_TYPE
    string_to_spv(shader_name + "_f32_f16",         source_name, merge_maps(merge_maps(base_dict, float_type_dict_f16), {{"DATA_A_F32", "1"},                                                     {"B_TYPE", "float16_t"},        {"D_TYPE", "float"}, }), fp16, coopmat, coopmat2, f16acc);
    string_to_spv(shader_name + "_f32_f16_aligned", source_name, merge_maps(merge_maps(base_dict, float_type_dict_f16), {{"DATA_A_F32", "1"}, {"LOAD_VEC_A", load_vec}, {"LOAD_VEC_B", load_vec}, {"B_TYPE", aligned_b_type_f16}, {"D_TYPE", "float"}, {"ALIGNED", "1"}}), fp16, coopmat, coopmat2, f16acc);

    string_to_spv(shader_name + "_f16",             source_name, merge_maps(merge_maps(base_dict, float_type_dict_f16), {{"DATA_A_F16", "1"},                                                     {"B_TYPE", "float16_t"},        {"D_TYPE", "float"}}), fp16, coopmat, coopmat2, f16acc);
    string_to_spv(shader_name + "_f16_aligned",     source_name, merge_maps(merge_maps(base_dict, float_type_dict_f16), {{"DATA_A_F16", "1"}, {"LOAD_VEC_A", load_vec}, {"LOAD_VEC_B", load_vec}, {"B_TYPE", aligned_b_type_f16}, {"D_TYPE", "float"}, {"ALIGNED", "1"}}), fp16, coopmat, coopmat2, f16acc);

    // bf16
    {
        // For aligned matmul loads
        std::string load_vec_a = coopmat2 ? "1" : "4";

        // scalar path promotes to float
        std::string to_float_type = (coopmat || coopmat2) ? "uintBitsToBFloat16EXT" : "bf16_to_fp32";

        const std::map<std::string, std::string> float_type_dict_bf16 = {
            {"FLOAT_TYPE",      FLOAT_TYPE(1, "bf16")},
            {"FLOAT_TYPE_VEC2", FLOAT_TYPE(2, "bf16")},
            {"FLOAT_TYPE_VEC4", FLOAT_TYPE(4, "bf16")},
        };

        // If bfloat16 is not supported, then only compile the scalar (promote to fp32) shader
#if !defined(GGML_VULKAN_BFLOAT16_GLSLC_SUPPORT)
        if (!(coopmat || coopmat2))
#endif
        {
            string_to_spv(shader_name + "_bf16",         source_name, merge_maps(merge_maps(base_dict, float_type_dict_bf16), {{"TO_FLOAT_TYPE", to_float_type}, {"DATA_A_BF16", "1"},                             {"B_TYPE", coopmat2 ? "bfloat16_t" : "uint16_t"}, {"D_TYPE", "float"}, {"B_IS_FLOAT", "1"}, {"DATA_B_BF16", "1"}}),                   fp16, coopmat, coopmat2, f16acc);
            string_to_spv(shader_name + "_bf16_aligned", source_name, merge_maps(merge_maps(base_dict, float_type_dict_bf16), {{"TO_FLOAT_TYPE", to_float_type}, {"DATA_A_BF16", "1"}, {"LOAD_VEC_A", load_vec_a}, {"LOAD_VEC_B", "4"}, {"B_TYPE", coopmat2 ? "bfloat16_t" : "u16vec4"},  {"D_TYPE", "float"}, {"B_IS_FLOAT", "1"}, {"DATA_B_BF16", "1"}, {"ALIGNED", "1"}}), fp16, coopmat, coopmat2, f16acc);
        }
    }

    for (const auto& tname : type_names) {
        std::string load_vec_quant = "2";
        if ((tname == "q4_0") || (tname == "q4_1") || (tname == "q5_1") || (tname == "iq1_s") || (tname == "iq1_m") || (tname == "iq2_xxs") || (tname == "iq2_xs") || (tname == "iq2_s"))
            load_vec_quant = "8";
        else if ((tname == "q5_0") || (tname == "q8_0") || (tname == "q2_k") || (tname == "q4_k") || (tname == "q5_k") || (tname == "iq3_xxs") || (tname == "iq3_s") || (tname == "iq4_nl") || (tname == "mxfp4"))
            load_vec_quant = "4";

        if (tname == "bf16") {
            continue;
        }

        std::string data_a_key = "DATA_A_" + to_uppercase(tname);
        // For unaligned, load one at a time for f32/f16, or two at a time for quants
        std::string load_vec_a_unaligned = (coopmat2 || tname == "f32" || tname == "f16" || tname == "bf16") ? "1" : load_vec_quant;
        // For aligned matmul loads
        std::string load_vec_a = (coopmat2 || tname == "f32" || tname == "f16" || tname == "bf16") ? load_vec : load_vec_quant;

        const std::map<std::string, std::string> float_type_dict = {
            {"FLOAT_TYPE",      FLOAT_TYPE(1, tname)},
            {"FLOAT_TYPE_VEC2", FLOAT_TYPE(2, tname)},
            {"FLOAT_TYPE_VEC4", FLOAT_TYPE(4, tname)},
            {"FLOAT_TYPE_VEC8", FLOAT_TYPE(8, tname)},
        };

        // don't generate f32 variants for coopmat2
        if (!coopmat2) {
            string_to_spv(shader_name + "_" + tname + "_f32",         source_name, merge_maps(merge_maps(base_dict, float_type_dict), {{data_a_key, "1"}, {"LOAD_VEC_A", load_vec_a_unaligned},                           {"B_TYPE", "float"},            {"D_TYPE", "float"}}), fp16, coopmat, coopmat2, f16acc);
            string_to_spv(shader_name + "_" + tname + "_f32_aligned", source_name, merge_maps(merge_maps(base_dict, float_type_dict), {{data_a_key, "1"}, {"LOAD_VEC_A", load_vec_a},           {"LOAD_VEC_B", load_vec}, {"B_TYPE", aligned_b_type_f32}, {"D_TYPE", "float"}, {"ALIGNED", "1"}}), fp16, coopmat, coopmat2, f16acc);
        }

        if (tname != "f16" && tname != "f32") {
            string_to_spv(shader_name + "_" + tname + "_f16",         source_name,  merge_maps(merge_maps(base_dict, float_type_dict), {{data_a_key, "1"}, {"LOAD_VEC_A", load_vec_a_unaligned},                           {"B_TYPE", "float16_t"},        {"D_TYPE", "float"}}), fp16, coopmat, coopmat2, f16acc);
            string_to_spv(shader_name + "_" + tname + "_f16_aligned", source_name,  merge_maps(merge_maps(base_dict, float_type_dict), {{data_a_key, "1"}, {"LOAD_VEC_A", load_vec_a},           {"LOAD_VEC_B", load_vec}, {"B_TYPE", aligned_b_type_f16}, {"D_TYPE", "float"}, {"ALIGNED", "1"}}), fp16, coopmat, coopmat2, f16acc);
        }

#if defined(GGML_VULKAN_INTEGER_DOT_GLSLC_SUPPORT)
        // Integer dot mmq performs better with f32 accumulators
        if (!f16acc && !coopmat && !coopmat2 && (is_legacy_quant(tname) || is_k_quant(tname) || tname == "mxfp4")) {
            string_to_spv(shader_name + "_" + tname + "_q8_1", "mul_mmq.comp", merge_maps(merge_maps(base_dict, float_type_dict), {{data_a_key, "1"}, {"D_TYPE", "float"},}), fp16, coopmat, coopmat2, f16acc);
        }
#endif
    }
}

void process_shaders() {
    std::map<std::string, std::string> base_dict = {{"FLOAT_TYPE", "float"}, {"FLOAT_TYPE_VEC2", "vec2"}};

    // matmul
    for (const MatMulIdType& matmul_id_type : {MatMulIdType::NONE, MatMulIdType::DEFAULT, MatMulIdType::SUBGROUP}) {
        // No coopmats
        // fp32
        matmul_shaders(false, matmul_id_type, false, false, false);

        // fp16, fp32acc and fp16acc
        matmul_shaders(true, matmul_id_type, false, false, false);
        matmul_shaders(true, matmul_id_type, false, false, true);

        if (matmul_id_type != MatMulIdType::DEFAULT) {
#if defined(GGML_VULKAN_COOPMAT_GLSLC_SUPPORT)
            // Coopmat, fp32acc and fp16acc
            matmul_shaders(true, matmul_id_type, true, false, false);
            matmul_shaders(true, matmul_id_type, true, false, true);
#endif

#if defined(GGML_VULKAN_COOPMAT2_GLSLC_SUPPORT)
            // Coopmat2, fp32acc and fp16acc
            matmul_shaders(true, matmul_id_type, false, true, false);
            matmul_shaders(true, matmul_id_type, false, true, true);
#endif
        }
    }

    // flash attention
    for (const auto& f16acc : {false, true}) {
        std::map<std::string, std::string> fa_base_dict = base_dict;
        fa_base_dict["ACC_TYPE"] = f16acc ? "float16_t" : "float";
        fa_base_dict["ACC_TYPEV4"] = f16acc ? "f16vec4" : "vec4";
        if (f16acc) {
            fa_base_dict["ACC_TYPE_MAX"] = "float16_t(65504.0)";
        }

        for (const auto& tname : type_names) {
            if (tname == "bf16") continue;

#if defined(GGML_VULKAN_COOPMAT2_GLSLC_SUPPORT)
            if (tname == "f16") {
                string_to_spv("flash_attn_f32_f16_" + tname, "flash_attn_cm2.comp",
                    merge_maps(fa_base_dict, {{"Q_TYPE", "float"}, {"D_TYPE", "float"}}), true, false, true, f16acc);
            } else {
                std::string data_a_key = "DATA_A_" + to_uppercase(tname);
                string_to_spv("flash_attn_f32_f16_" + tname, "flash_attn_cm2.comp",
                    merge_maps(fa_base_dict, {{data_a_key, "1"}, {"Q_TYPE", "float"}, {"D_TYPE", "float"}, {"DEQUANTFUNC", "dequantFunc"+to_uppercase(tname) }, {"BLOCK_SIZE", "QUANT_K_"+to_uppercase(tname) }}), true, false, true, f16acc);
            }
#endif
#if defined(GGML_VULKAN_COOPMAT_GLSLC_SUPPORT)
            if (tname == "f16") {
                string_to_spv("flash_attn_f32_f16_" + tname, "flash_attn_cm1.comp",
                    merge_maps(fa_base_dict, {{"Q_TYPE", "float"}, {"D_TYPE", "float"}, {"COOPMAT", "1"}}), true, true, false, f16acc);
            } else if (tname == "q4_0" || tname == "q8_0" || tname == "f32") {
                std::string data_a_key = "DATA_A_" + to_uppercase(tname);
                string_to_spv("flash_attn_f32_f16_" + tname, "flash_attn_cm1.comp",
                    merge_maps(fa_base_dict, {{data_a_key, "1"}, {"Q_TYPE", "float"}, {"D_TYPE", "float"}, {"BLOCK_SIZE", "QUANT_K_"+to_uppercase(tname)}, {"COOPMAT", "1"}}), true, true, false, f16acc);
            }
#endif
            if (tname == "f16") {
                string_to_spv("flash_attn_f32_f16_" + tname, "flash_attn.comp",
                    merge_maps(fa_base_dict, {{"Q_TYPE", "float"}, {"D_TYPE", "float"}}), true, false, false, f16acc);
            } else if (tname == "q4_0" || tname == "q8_0" || tname == "f32") {
                std::string data_a_key = "DATA_A_" + to_uppercase(tname);
                string_to_spv("flash_attn_f32_f16_" + tname, "flash_attn.comp",
                    merge_maps(fa_base_dict, {{data_a_key, "1"}, {"Q_TYPE", "float"}, {"D_TYPE", "float"}, {"BLOCK_SIZE", "QUANT_K_"+to_uppercase(tname) }}), true, false, false, f16acc);
            }
        }
    }

    for (const auto& tname : type_names) {
        // mul mat vec
        std::string data_a_key = "DATA_A_" + to_uppercase(tname);
        std::string shader = (string_ends_with(tname, "_k") || string_starts_with(tname, "iq1_") || string_starts_with(tname, "iq2_") || string_starts_with(tname, "iq3_")) ? "mul_mat_vec_" + tname + ".comp" : "mul_mat_vec.comp";

        string_to_spv("mul_mat_vec_" + tname + "_f32_f32", shader, merge_maps(base_dict, {{data_a_key, "1"}, {"B_TYPE", "float"}, {"B_TYPE_VEC2", "vec2"}, {"B_TYPE_VEC4", "vec4"}, {"D_TYPE", "float"}}));
        string_to_spv("mul_mat_vec_" + tname + "_f16_f32", shader, merge_maps(base_dict, {{data_a_key, "1"}, {"B_TYPE", "float16_t"}, {"B_TYPE_VEC2", "f16vec2"}, {"B_TYPE_VEC4", "f16vec4"}, {"D_TYPE", "float"}}));

        string_to_spv("mul_mat_vec_" + tname + "_f32_f32_subgroup", shader, merge_maps(base_dict, {{data_a_key, "1"}, {"B_TYPE", "float"}, {"B_TYPE_VEC2", "vec2"}, {"B_TYPE_VEC4", "vec4"}, {"D_TYPE", "float"}, {"USE_SUBGROUP_ADD", "1"}}));
        string_to_spv("mul_mat_vec_" + tname + "_f16_f32_subgroup", shader, merge_maps(base_dict, {{data_a_key, "1"}, {"B_TYPE", "float16_t"}, {"B_TYPE_VEC2", "f16vec2"}, {"B_TYPE_VEC4", "f16vec4"}, {"D_TYPE", "float"}, {"USE_SUBGROUP_ADD", "1"}}));

        string_to_spv("mul_mat_vec_" + tname + "_f32_f32_subgroup_no_shmem", shader, merge_maps(base_dict, {{data_a_key, "1"}, {"B_TYPE", "float"}, {"B_TYPE_VEC2", "vec2"}, {"B_TYPE_VEC4", "vec4"}, {"D_TYPE", "float"}, {"USE_SUBGROUP_ADD_NO_SHMEM", "1"}}));
        string_to_spv("mul_mat_vec_" + tname + "_f16_f32_subgroup_no_shmem", shader, merge_maps(base_dict, {{data_a_key, "1"}, {"B_TYPE", "float16_t"}, {"B_TYPE_VEC2", "f16vec2"}, {"B_TYPE_VEC4", "f16vec4"}, {"D_TYPE", "float"}, {"USE_SUBGROUP_ADD_NO_SHMEM", "1"}}));

        string_to_spv("mul_mat_vec_id_" + tname + "_f32_f32", shader, merge_maps(base_dict, {{"MUL_MAT_ID", "1"}, {data_a_key, "1"}, {"B_TYPE", "float"}, {"B_TYPE_VEC2", "vec2"}, {"B_TYPE_VEC4", "vec4"}, {"D_TYPE", "float"}}));
        string_to_spv("mul_mat_vec_id_" + tname + "_f32_f32_subgroup", shader, merge_maps(base_dict, {{"MUL_MAT_ID", "1"}, {data_a_key, "1"}, {"B_TYPE", "float"}, {"B_TYPE_VEC2", "vec2"}, {"B_TYPE_VEC4", "vec4"}, {"D_TYPE", "float"}, {"USE_SUBGROUP_ADD", "1"}}));
        string_to_spv("mul_mat_vec_id_" + tname + "_f32_f32_subgroup_no_shmem", shader, merge_maps(base_dict, {{"MUL_MAT_ID", "1"}, {data_a_key, "1"}, {"B_TYPE", "float"}, {"B_TYPE_VEC2", "vec2"}, {"B_TYPE_VEC4", "vec4"}, {"D_TYPE", "float"}, {"USE_SUBGROUP_ADD_NO_SHMEM", "1"}}));

        // mul mat vec with integer dot product
#if defined(GGML_VULKAN_INTEGER_DOT_GLSLC_SUPPORT)
        if (is_legacy_quant(tname) || tname == "mxfp4" || is_k_quant(tname) || tname == "iq1_s" || tname == "iq1_m") {
            string_to_spv("mul_mat_vec_" + tname + "_q8_1_f32", "mul_mat_vecq.comp", merge_maps(base_dict, {{data_a_key, "1"}, {"D_TYPE", "float"}, {"FLOAT_TYPE", "float"}, {"FLOAT_TYPE_VEC2", "vec2"}, {"ACC_TYPE", "float"}}));
            string_to_spv("mul_mat_vec_" + tname + "_q8_1_f32_subgroup", "mul_mat_vecq.comp", merge_maps(base_dict, {{data_a_key, "1"}, {"D_TYPE", "float"}, {"FLOAT_TYPE", "float"}, {"FLOAT_TYPE_VEC2", "vec2"}, {"ACC_TYPE", "float"}, {"USE_SUBGROUP_ADD", "1"}}));
            string_to_spv("mul_mat_vec_" + tname + "_q8_1_f32_subgroup_no_shmem", "mul_mat_vecq.comp", merge_maps(base_dict, {{data_a_key, "1"}, {"D_TYPE", "float"}, {"FLOAT_TYPE", "float"}, {"FLOAT_TYPE_VEC2", "vec2"}, {"ACC_TYPE", "float"}, {"USE_SUBGROUP_ADD_NO_SHMEM", "1"}}));

            string_to_spv("mul_mat_vec_id_" + tname + "_q8_1_f32", "mul_mat_vecq.comp", merge_maps(base_dict, {{"MUL_MAT_ID", "1"}, {data_a_key, "1"}, {"D_TYPE", "float"}, {"FLOAT_TYPE", "float"}, {"FLOAT_TYPE_VEC2", "vec2"}, {"ACC_TYPE", "float"}}));
            string_to_spv("mul_mat_vec_id_" + tname + "_q8_1_f32_subgroup", "mul_mat_vecq.comp", merge_maps(base_dict, {{"MUL_MAT_ID", "1"}, {data_a_key, "1"}, {"D_TYPE", "float"}, {"FLOAT_TYPE", "float"}, {"FLOAT_TYPE_VEC2", "vec2"}, {"ACC_TYPE", "float"}, {"USE_SUBGROUP_ADD", "1"}}));
            string_to_spv("mul_mat_vec_id_" + tname + "_q8_1_f32_subgroup_no_shmem", "mul_mat_vecq.comp", merge_maps(base_dict, {{"MUL_MAT_ID", "1"}, {data_a_key, "1"}, {"D_TYPE", "float"}, {"FLOAT_TYPE", "float"}, {"FLOAT_TYPE_VEC2", "vec2"}, {"ACC_TYPE", "float"}, {"USE_SUBGROUP_ADD_NO_SHMEM", "1"}}));
        }
#endif

        // Dequant shaders
        if (tname != "f16" && tname != "bf16") {
            string_to_spv("dequant_" + tname, "dequant_" + tname + ".comp", merge_maps(base_dict, {{data_a_key, "1"}, {"D_TYPE", "float16_t"}}));
        }

        shader = (tname == "f32" || tname == "f16" || tname == "bf16") ? "get_rows.comp" : "get_rows_quant.comp";

        if (tname == "f16") {
            string_to_spv("get_rows_" + tname, shader, merge_maps(base_dict, {{"TEMP_TYPE", "FLOAT_TYPE"}, {data_a_key, "1"}, {"B_TYPE", "int"}, {"D_TYPE", "float16_t"}, {"OPTIMIZATION_ERROR_WORKAROUND", "1"}}));
        } else {
            string_to_spv("get_rows_" + tname, shader, merge_maps(base_dict, {{"TEMP_TYPE", "FLOAT_TYPE"}, {data_a_key, "1"}, {"B_TYPE", "int"}, {"D_TYPE", "float16_t"}}));
        }
        string_to_spv("get_rows_" + tname + "_f32", shader, merge_maps(base_dict, {{"TEMP_TYPE", "FLOAT_TYPE"}, {data_a_key, "1"}, {"B_TYPE", "int"}, {"D_TYPE", "float"}}));
    }

    string_to_spv("get_rows_i32", "get_rows.comp", {{"TEMP_TYPE", "uint"}, {"A_TYPE", "uint"}, {"B_TYPE", "int"}, {"D_TYPE", "uint"}});

    string_to_spv("mul_mat_vec_p021_f16_f32_subgroup_add", "mul_mat_vec_p021.comp", {{"A_TYPE", "float16_t"}, {"A_TYPE_VEC4", "f16vec4"}, {"B_TYPE", "float"}, {"B_TYPE_VEC4", "vec4"}, {"D_TYPE", "float"}, {"USE_SUBGROUP_ADD", "1"}});
    string_to_spv("mul_mat_vec_p021_f16_f32",              "mul_mat_vec_p021.comp", {{"A_TYPE", "float16_t"}, {"A_TYPE_VEC4", "f16vec4"}, {"B_TYPE", "float"}, {"B_TYPE_VEC4", "vec4"}, {"D_TYPE", "float"}});
    string_to_spv("mul_mat_vec_nc_f16_f32", "mul_mat_vec_nc.comp", {{"A_TYPE", "float16_t"}, {"A_TYPE_VEC4", "f16vec4"}, {"B_TYPE", "float"}, {"B_TYPE_VEC4", "vec4"}, {"D_TYPE", "float"}});

    // Norms
    string_to_spv("norm_f32", "norm.comp", merge_maps(base_dict, {{"A_TYPE", "float"}, {"D_TYPE", "float"}}));
    string_to_spv("group_norm_f32", "group_norm.comp", merge_maps(base_dict, {{"A_TYPE", "float"}, {"D_TYPE", "float"}}));
    string_to_spv("rms_norm_f32", "rms_norm.comp", merge_maps(base_dict, {{"A_TYPE", "float"}, {"B_TYPE", "float"}, {"D_TYPE", "float"}}));
    string_to_spv("rms_norm_partials_f32", "rms_norm_partials.comp", merge_maps(base_dict, {{"A_TYPE", "float"}, {"B_TYPE", "float"}, {"D_TYPE", "float"}}));
    string_to_spv("rms_norm_mul_rope_f32_f32", "rms_norm.comp", merge_maps(base_dict, {{"A_TYPE", "float"}, {"B_TYPE", "float"}, {"D_TYPE", "float"}, {"ROPE_D_TYPE", "float"}, {"RMS_NORM_ROPE_FUSION", "1"}}));
    string_to_spv("rms_norm_mul_rope_f32_f16_rte", "rms_norm.comp", merge_maps(base_dict, {{"A_TYPE", "float"}, {"B_TYPE", "float"}, {"D_TYPE", "float"}, {"ROPE_D_TYPE", "float16_t"}, {"RMS_NORM_ROPE_FUSION", "1"}, {"RTE16", "1"}}));
    string_to_spv("rms_norm_back_f32", "rms_norm_back.comp", merge_maps(base_dict, {{"A_TYPE", "float"}, {"B_TYPE", "float"}, {"D_TYPE", "float"}}));
    string_to_spv("l2_norm_f32", "l2_norm.comp", merge_maps(base_dict, {{"A_TYPE", "float"}, {"D_TYPE", "float"}}));

    string_to_spv("cpy_f32_f32", "copy.comp", {{"A_TYPE", "float"}, {"D_TYPE", "float"}});
    string_to_spv("cpy_f32_f16", "copy.comp", {{"A_TYPE", "float"}, {"D_TYPE", "float16_t"}});
    string_to_spv("cpy_f16_f16", "copy.comp", {{"A_TYPE", "float16_t"}, {"D_TYPE", "float16_t"}, {"OPTIMIZATION_ERROR_WORKAROUND", "1"}});
    string_to_spv("cpy_f16_f32", "copy.comp", {{"A_TYPE", "float16_t"}, {"D_TYPE", "float"}, {"OPTIMIZATION_ERROR_WORKAROUND", "1"}});
    string_to_spv("cpy_f32_bf16","copy.comp", {{"A_TYPE", "float"}, {"D_TYPE", "uint16_t"}, {"DATA_D_BF16", "1"}});
    string_to_spv("contig_cpy_f32_f32", "contig_copy.comp", {{"A_TYPE", "float"}, {"D_TYPE", "float"}});
    string_to_spv("contig_cpy_f32_i32", "contig_copy.comp", {{"A_TYPE", "float"}, {"D_TYPE", "int"}});
    string_to_spv("contig_cpy_i32_f32", "contig_copy.comp", {{"A_TYPE", "int"}, {"D_TYPE", "float"}});
    string_to_spv("contig_cpy_f32_f16", "contig_copy.comp", {{"A_TYPE", "float"}, {"D_TYPE", "float16_t"}});
    string_to_spv("contig_cpy_f16_f16", "contig_copy.comp", {{"A_TYPE", "float16_t"}, {"D_TYPE", "float16_t"}, {"OPTIMIZATION_ERROR_WORKAROUND", "1"}});
    string_to_spv("contig_cpy_f16_f32", "contig_copy.comp", {{"A_TYPE", "float16_t"}, {"D_TYPE", "float"}, {"OPTIMIZATION_ERROR_WORKAROUND", "1"}});
    string_to_spv("contig_cpy_f32_bf16","contig_copy.comp",{{"A_TYPE", "float"}, {"D_TYPE", "uint16_t"}, {"DATA_D_BF16", "1"}});
    string_to_spv("cpy_f32_i32", "copy.comp", {{"A_TYPE", "float"}, {"D_TYPE", "int"}});
    string_to_spv("cpy_i32_f32", "copy.comp", {{"A_TYPE", "int"}, {"D_TYPE", "float"}});

    string_to_spv("cpy_transpose_16", "copy_transpose.comp", {{"A_TYPE", "uint16_t"}, {"D_TYPE", "uint16_t"}});
    string_to_spv("cpy_transpose_32", "copy_transpose.comp", {{"A_TYPE", "uint"}, {"D_TYPE", "uint"}});

    for (std::string t : {"q4_0", "q4_1", "q5_0", "q5_1", "q8_0", "iq4_nl"}) {
        string_to_spv("cpy_f32_" + t, "copy_to_quant.comp", {{"DATA_A_" + to_uppercase(t), "1"}, {"D_TYPE", "float"}, {"FLOAT_TYPE", "float"}});
        string_to_spv("cpy_f32_" + t + "_rte", "copy_to_quant.comp", {{"DATA_A_" + to_uppercase(t), "1"}, {"D_TYPE", "float"}, {"FLOAT_TYPE", "float"}, {"RTE16", "1"}});
        string_to_spv("cpy_" + t + "_f32", "copy_from_quant.comp", {{"DATA_A_" + to_uppercase(t), "1"}, {"D_TYPE", "float"}, {"FLOAT_TYPE", "float"}});
    }

    for (std::string t : {"f32", "f16", "bf16", "q4_0", "q4_1", "q5_0", "q5_1", "q8_0", "iq4_nl"}) {
        string_to_spv("set_rows_" + t + "_i32",     "copy_to_quant.comp", {{"SET_ROWS", "1"}, {"DATA_A_" + to_uppercase(t), "1"}, {"B_TYPE", "uint"}, {"B_SIZE", "32"}, {"D_TYPE", "float"}, {"FLOAT_TYPE", "float"}});
        string_to_spv("set_rows_" + t + "_i32_rte", "copy_to_quant.comp", {{"SET_ROWS", "1"}, {"DATA_A_" + to_uppercase(t), "1"}, {"B_TYPE", "uint"}, {"B_SIZE", "32"}, {"D_TYPE", "float"}, {"FLOAT_TYPE", "float"}, {"RTE16", "1"}});
        string_to_spv("set_rows_" + t + "_i64",     "copy_to_quant.comp", {{"SET_ROWS", "1"}, {"DATA_A_" + to_uppercase(t), "1"}, {"B_TYPE", "uvec2"}, {"B_SIZE", "64"}, {"D_TYPE", "float"}, {"FLOAT_TYPE", "float"}});
        string_to_spv("set_rows_" + t + "_i64_rte", "copy_to_quant.comp", {{"SET_ROWS", "1"}, {"DATA_A_" + to_uppercase(t), "1"}, {"B_TYPE", "uvec2"}, {"B_SIZE", "64"}, {"D_TYPE", "float"}, {"FLOAT_TYPE", "float"}, {"RTE16", "1"}});
    }

    auto get_type_str = [](bool f16) {
        return f16 ? "float16_t" : "float";
    };
    auto get_suffix = [](bool src0_f16, bool src1_f16, bool dst_f16) {
        std::string s;
        s += std::string(src0_f16 ? "_f16" : "_f32");
        s += std::string(src1_f16 ? "_f16" : "_f32");
        s += std::string(dst_f16 ? "_f16" : "_f32");
        return s;
    };
    for (std::string op : {"add", "sub", "mul", "div", "add_rms", }) {
    for (auto src0_f16 : {false, true}) {
    for (auto src1_f16 : {false, true}) {
    for (auto dst_f16  : {false, true}) {
    for (auto rte      : {false, true}) {
        auto source = op == "add_rms" ? std::string("add") : op;
        auto name = op + get_suffix(src0_f16, src1_f16, dst_f16) + (rte ? "_rte" : "");
        auto add_rms = op == "add_rms" ? "1" : "0";
        string_to_spv(name.c_str(), source + ".comp", {{"A_TYPE", get_type_str(src0_f16)}, {"B_TYPE", get_type_str(src1_f16)}, {"D_TYPE", get_type_str(dst_f16)}, {"FLOAT_TYPE", "float"}, {"RTE16", rte ? "1" : "0"}, {"ADD_RMS" , add_rms}});
    }
    }
    }
    }
    }

    string_to_spv("sub_f32", "sub.comp", {{"A_TYPE", "float"}, {"B_TYPE", "float"}, {"D_TYPE", "float"}, {"FLOAT_TYPE", "float"}});

    string_to_spv("acc_f32", "acc.comp", {{"A_TYPE", "float"}, {"B_TYPE", "float"}, {"D_TYPE", "float"}, {"FLOAT_TYPE", "float"}});

    string_to_spv("split_k_reduce", "mul_mat_split_k_reduce.comp", {});
    string_to_spv("fa_split_k_reduce", "flash_attn_split_k_reduce.comp", {});

    string_to_spv("quantize_q8_1", "quantize_q8_1.comp", {});
    string_to_spv("quantize_q8_1_subgroup", "quantize_q8_1.comp", {{"USE_SUBGROUPS", "1"}});

    string_to_spv("quantize_q8_1_x4", "quantize_q8_1.comp", {{"QBLOCK_X4", "1"}});
    string_to_spv("quantize_q8_1_x4_subgroup", "quantize_q8_1.comp", {{"QBLOCK_X4", "1"}, {"USE_SUBGROUPS", "1"}});

    string_to_spv("mul_f32", "mul.comp", {{"A_TYPE", "float"}, {"B_TYPE", "float"}, {"D_TYPE", "float"}, {"FLOAT_TYPE", "float"}});

    string_to_spv("div_f32", "div.comp", {{"A_TYPE", "float"}, {"B_TYPE", "float"}, {"D_TYPE", "float"}, {"FLOAT_TYPE", "float"}});

    string_to_spv("repeat_f32", "repeat.comp", {{"A_TYPE", "float"}, {"D_TYPE", "float"}});
    string_to_spv("repeat_back_f32", "repeat_back.comp", {{"A_TYPE", "float"}, {"D_TYPE", "float"}});

    string_to_spv("scale_f32", "scale.comp", {{"A_TYPE", "float"}, {"D_TYPE", "float"}, {"FLOAT_TYPE", "float"}});

    string_to_spv("sqr_f32", "square.comp", {{"A_TYPE", "float"}, {"D_TYPE", "float"}, {"FLOAT_TYPE", "float"}});

    string_to_spv("sqrt_f32", "sqrt.comp", {{"A_TYPE", "float"}, {"D_TYPE", "float"}, {"FLOAT_TYPE", "float"}});

    string_to_spv("sin_f32", "sin.comp", {{"A_TYPE", "float"}, {"D_TYPE", "float"}, {"FLOAT_TYPE", "float"}});

    string_to_spv("cos_f32", "cos.comp", {{"A_TYPE", "float"}, {"D_TYPE", "float"}, {"FLOAT_TYPE", "float"}});

    string_to_spv("clamp_f32", "clamp.comp", {{"A_TYPE", "float"}, {"D_TYPE", "float"}, {"FLOAT_TYPE", "float"}});

    string_to_spv("pad_f32", "pad.comp", {{"A_TYPE", "float"}, {"D_TYPE", "float"}});

    string_to_spv("concat_f32", "concat.comp", {{"A_TYPE", "float"}, {"B_TYPE", "float"}, {"D_TYPE", "float"}});
    string_to_spv("concat_f16", "concat.comp", {{"A_TYPE", "float16_t"}, {"B_TYPE", "float16_t"}, {"D_TYPE", "float16_t"}, {"OPTIMIZATION_ERROR_WORKAROUND", "1"}});
    string_to_spv("concat_i32", "concat.comp", {{"A_TYPE", "int"}, {"B_TYPE", "int"}, {"D_TYPE", "int"}});

    string_to_spv("upscale_f32", "upscale.comp", {{"A_TYPE", "float"}, {"B_TYPE", "float"}, {"D_TYPE", "float"}});

    for (auto rte : {false, true}) {
        std::string suffix = rte ? "_rte" : "";
        string_to_spv("exp_f16" + suffix,        "exp.comp",         {{"A_TYPE", "float16_t"},   {"D_TYPE", "float16_t"},   {"RTE16", rte ? "1" : "0"}});
        string_to_spv("exp_f32" + suffix,        "exp.comp",         {{"A_TYPE", "float"},       {"D_TYPE", "float"}    ,   {"RTE16", rte ? "1" : "0"}});

        string_to_spv("log_f16" + suffix,        "log.comp",         {{"A_TYPE", "float16_t"},   {"D_TYPE", "float16_t"},   {"RTE16", rte ? "1" : "0"}});
        string_to_spv("log_f32" + suffix,        "log.comp",         {{"A_TYPE", "float"},       {"D_TYPE", "float"},       {"RTE16", rte ? "1" : "0"}});
    }
    string_to_spv("gelu_f16",       "gelu.comp",        {{"A_TYPE", "float16_t"},   {"D_TYPE", "float16_t"}});
    string_to_spv("gelu_f32",       "gelu.comp",        {{"A_TYPE", "float"},       {"D_TYPE", "float"}});
    string_to_spv("gelu_erf_f16",   "gelu_erf.comp",    {{"A_TYPE", "float16_t"},   {"D_TYPE", "float16_t"}});
    string_to_spv("gelu_erf_f32",   "gelu_erf.comp",    {{"A_TYPE", "float"},       {"D_TYPE", "float"}});
    string_to_spv("gelu_quick_f16", "gelu_quick.comp",  {{"A_TYPE", "float16_t"},   {"D_TYPE", "float16_t"}});
    string_to_spv("gelu_quick_f32", "gelu_quick.comp",  {{"A_TYPE", "float"},       {"D_TYPE", "float"}});
    string_to_spv("silu_f16",       "silu.comp",        {{"A_TYPE", "float16_t"},   {"D_TYPE", "float16_t"}});
    string_to_spv("silu_f32",       "silu.comp",        {{"A_TYPE", "float"},       {"D_TYPE", "float"}});
    string_to_spv("relu_f16",       "relu.comp",        {{"A_TYPE", "float16_t"},   {"D_TYPE", "float16_t"}});
    string_to_spv("relu_f32",       "relu.comp",        {{"A_TYPE", "float"},       {"D_TYPE", "float"}});
    string_to_spv("neg_f16",        "neg.comp",         {{"A_TYPE", "float16_t"},   {"D_TYPE", "float16_t"}});
    string_to_spv("neg_f32",        "neg.comp",         {{"A_TYPE", "float"},       {"D_TYPE", "float"}});
    string_to_spv("tanh_f16",       "tanh.comp",        {{"A_TYPE", "float16_t"},   {"D_TYPE", "float16_t"}});
    string_to_spv("tanh_f32",       "tanh.comp",        {{"A_TYPE", "float"},       {"D_TYPE", "float"}});
    string_to_spv("sigmoid_f16",    "sigmoid.comp",     {{"A_TYPE", "float16_t"},   {"D_TYPE", "float16_t"}});
    string_to_spv("sigmoid_f32",    "sigmoid.comp",     {{"A_TYPE", "float"},       {"D_TYPE", "float"}});
    string_to_spv("hardsigmoid_f16","hardsigmoid.comp", {{"A_TYPE", "float16_t"},   {"D_TYPE", "float16_t"}});
    string_to_spv("hardsigmoid_f32","hardsigmoid.comp", {{"A_TYPE", "float"},       {"D_TYPE", "float"}});
    string_to_spv("hardswish_f16",  "hardswish.comp",   {{"A_TYPE", "float16_t"},   {"D_TYPE", "float16_t"}});
    string_to_spv("hardswish_f32",  "hardswish.comp",   {{"A_TYPE", "float"},       {"D_TYPE", "float"}});
    string_to_spv("abs_f16",        "abs.comp",         {{"A_TYPE", "float16_t"},   {"D_TYPE", "float16_t"}});
    string_to_spv("abs_f32",        "abs.comp",         {{"A_TYPE", "float"},       {"D_TYPE", "float"}});
    string_to_spv("xielu_f16",      "xielu.comp",       {{"A_TYPE", "float16_t"},   {"D_TYPE", "float16_t"}});
    string_to_spv("xielu_f32",      "xielu.comp",       {{"A_TYPE", "float"},       {"D_TYPE", "float"}});

    string_to_spv("tri_f16",        "tri.comp",         {{"A_TYPE", "float16_t"},   {"D_TYPE", "float16_t"}});
    string_to_spv("tri_f32",        "tri.comp",         {{"A_TYPE", "float"},       {"D_TYPE", "float"}});
    string_to_spv("diag_f16",       "diag.comp",        {{"A_TYPE", "float16_t"},   {"D_TYPE", "float16_t"}});
    string_to_spv("diag_f32",       "diag.comp",        {{"A_TYPE", "float"},       {"D_TYPE", "float"}});

    string_to_spv("softplus_f16",   "softplus.comp",    {{"A_TYPE", "float16_t"},   {"D_TYPE", "float16_t"}});
    string_to_spv("softplus_f32",   "softplus.comp",    {{"A_TYPE", "float"},       {"D_TYPE", "float"}});

    string_to_spv("add1_f16_f16",   "add1.comp",        {{"A_TYPE", "float16_t"},   {"B_TYPE", "float16_t"}, {"D_TYPE", "float16_t"}, {"FLOAT_TYPE", "float"}});
    string_to_spv("add1_f16_f32",   "add1.comp",        {{"A_TYPE", "float16_t"},   {"B_TYPE", "float"}, {"D_TYPE", "float16_t"}, {"FLOAT_TYPE", "float"}});
    string_to_spv("add1_f32_f32",   "add1.comp",        {{"A_TYPE", "float"},       {"B_TYPE", "float"}, {"D_TYPE", "float"}, {"FLOAT_TYPE", "float"}});
    string_to_spv("arange_f32",     "arange.comp",      {{"A_TYPE", "float"},       {"D_TYPE", "float"}, {"FLOAT_TYPE", "float"}});
    string_to_spv("fill_f32",       "fill.comp",        {{"D_TYPE", "float"},       {"FLOAT_TYPE", "float"}});
    string_to_spv("step_f16",       "step.comp",        {{"A_TYPE", "float16_t"},   {"D_TYPE", "float16_t"}});
    string_to_spv("step_f32",       "step.comp",        {{"A_TYPE", "float"},       {"D_TYPE", "float"}});
    string_to_spv("round_f16",      "round.comp",       {{"A_TYPE", "float16_t"},   {"D_TYPE", "float16_t"}});
    string_to_spv("round_f32",      "round.comp",       {{"A_TYPE", "float"},       {"D_TYPE", "float"}});
    string_to_spv("ceil_f16",       "ceil.comp",        {{"A_TYPE", "float16_t"},   {"D_TYPE", "float16_t"}});
    string_to_spv("ceil_f32",       "ceil.comp",        {{"A_TYPE", "float"},       {"D_TYPE", "float"}});
    string_to_spv("floor_f16",      "floor.comp",       {{"A_TYPE", "float16_t"},   {"D_TYPE", "float16_t"}});
    string_to_spv("floor_f32",      "floor.comp",       {{"A_TYPE", "float"},       {"D_TYPE", "float"}});
    string_to_spv("trunc_f16",      "trunc.comp",       {{"A_TYPE", "float16_t"},   {"D_TYPE", "float16_t"}});
    string_to_spv("trunc_f32",      "trunc.comp",       {{"A_TYPE", "float"},       {"D_TYPE", "float"}});

    for (auto rte : {false, true}) {
        std::string suffix = rte ? "_rte" : "";
        string_to_spv("geglu_f16" + suffix,      "geglu.comp",       {{"A_TYPE", "float16_t"},   {"D_TYPE", "float16_t"},   {"RTE16", rte ? "1" : "0"}});
        string_to_spv("geglu_f32" + suffix,      "geglu.comp",       {{"A_TYPE", "float"},       {"D_TYPE", "float"},       {"RTE16", rte ? "1" : "0"}});
        string_to_spv("reglu_f16" + suffix,      "reglu.comp",       {{"A_TYPE", "float16_t"},   {"D_TYPE", "float16_t"},   {"RTE16", rte ? "1" : "0"}});
        string_to_spv("reglu_f32" + suffix,      "reglu.comp",       {{"A_TYPE", "float"},       {"D_TYPE", "float"},       {"RTE16", rte ? "1" : "0"}});
        string_to_spv("swiglu_f16" + suffix,     "swiglu.comp",      {{"A_TYPE", "float16_t"},   {"D_TYPE", "float16_t"},   {"RTE16", rte ? "1" : "0"}});
        string_to_spv("swiglu_f32" + suffix,     "swiglu.comp",      {{"A_TYPE", "float"},       {"D_TYPE", "float"},       {"RTE16", rte ? "1" : "0"}});
        string_to_spv("swiglu_oai_f16" + suffix, "swiglu_oai.comp",  {{"A_TYPE", "float16_t"},   {"D_TYPE", "float16_t"},   {"RTE16", rte ? "1" : "0"}});
        string_to_spv("swiglu_oai_f32" + suffix, "swiglu_oai.comp",  {{"A_TYPE", "float"},       {"D_TYPE", "float"},       {"RTE16", rte ? "1" : "0"}});
        string_to_spv("geglu_erf_f16" + suffix,  "geglu_erf.comp",   {{"A_TYPE", "float16_t"},   {"D_TYPE", "float16_t"},   {"RTE16", rte ? "1" : "0"}});
        string_to_spv("geglu_erf_f32" + suffix,  "geglu_erf.comp",   {{"A_TYPE", "float"},       {"D_TYPE", "float"},       {"RTE16", rte ? "1" : "0"}});
        string_to_spv("geglu_quick_f16" + suffix,"geglu_quick.comp", {{"A_TYPE", "float16_t"},   {"D_TYPE", "float16_t"},   {"RTE16", rte ? "1" : "0"}});
        string_to_spv("geglu_quick_f32" + suffix,"geglu_quick.comp", {{"A_TYPE", "float"},       {"D_TYPE", "float"},       {"RTE16", rte ? "1" : "0"}});
    }

    string_to_spv("leaky_relu_f32", "leaky_relu.comp",  {{"A_TYPE", "float"}, {"D_TYPE", "float"}});
    string_to_spv("silu_back_f32",  "silu_back.comp",   {{"A_TYPE", "float"}, {"B_TYPE", "float"}, {"D_TYPE", "float"}});

    string_to_spv("diag_mask_inf_f32", "diag_mask_inf.comp", {{"A_TYPE", "float"}, {"D_TYPE", "float"}});

    string_to_spv("soft_max_f32", "soft_max.comp", merge_maps(base_dict, {{"A_TYPE", "float"}, {"B_TYPE", "float"}, {"D_TYPE", "float"}}));
    string_to_spv("soft_max_f32_f16", "soft_max.comp", merge_maps(base_dict, {{"A_TYPE", "float"}, {"B_TYPE", "float16_t"}, {"D_TYPE", "float"}}));
    string_to_spv("soft_max_back_f32", "soft_max_back.comp", merge_maps(base_dict, {{"A_TYPE", "float"}, {"B_TYPE", "float"}, {"D_TYPE", "float"}}));

    string_to_spv("soft_max_large1_f32", "soft_max_large1.comp", merge_maps(base_dict, {{"A_TYPE", "float"}, {"B_TYPE", "float"}, {"D_TYPE", "float"}}));
    string_to_spv("soft_max_large2_f32", "soft_max_large2.comp", merge_maps(base_dict, {{"A_TYPE", "float"}, {"B_TYPE", "float"}, {"D_TYPE", "float"}}));
    string_to_spv("soft_max_large3_f32", "soft_max_large3.comp", merge_maps(base_dict, {{"A_TYPE", "float"}, {"B_TYPE", "float"}, {"D_TYPE", "float"}}));
    string_to_spv("soft_max_large1_f32_f16", "soft_max_large1.comp", merge_maps(base_dict, {{"A_TYPE", "float"}, {"B_TYPE", "float16_t"}, {"D_TYPE", "float"}}));
    string_to_spv("soft_max_large2_f32_f16", "soft_max_large2.comp", merge_maps(base_dict, {{"A_TYPE", "float"}, {"B_TYPE", "float16_t"}, {"D_TYPE", "float"}}));
    string_to_spv("soft_max_large3_f32_f16", "soft_max_large3.comp", merge_maps(base_dict, {{"A_TYPE", "float"}, {"B_TYPE", "float16_t"}, {"D_TYPE", "float"}}));

    string_to_spv("rope_norm_f32", "rope_norm.comp", {{"A_TYPE", "float"}, {"ROPE_D_TYPE", "float"}});
    string_to_spv("rope_norm_f16", "rope_norm.comp", {{"A_TYPE", "float16_t"}, {"ROPE_D_TYPE", "float16_t"}});
    string_to_spv("rope_norm_f16_rte", "rope_norm.comp", {{"A_TYPE", "float16_t"}, {"ROPE_D_TYPE", "float16_t"}, {"RTE16", "1"}});
    string_to_spv("rope_norm_f32_f16", "rope_norm.comp", {{"A_TYPE", "float"}, {"ROPE_D_TYPE", "float16_t"}});
    string_to_spv("rope_norm_f32_f16_rte", "rope_norm.comp", {{"A_TYPE", "float"}, {"ROPE_D_TYPE", "float16_t"}, {"RTE16", "1"}});

    string_to_spv("rope_neox_f32", "rope_neox.comp", {{"A_TYPE", "float"}, {"ROPE_D_TYPE", "float"}});
    string_to_spv("rope_neox_f16", "rope_neox.comp", {{"A_TYPE", "float16_t"}, {"ROPE_D_TYPE", "float16_t"}});
    string_to_spv("rope_neox_f16_rte", "rope_neox.comp", {{"A_TYPE", "float16_t"}, {"ROPE_D_TYPE", "float16_t"}, {"RTE16", "1"}});
    string_to_spv("rope_neox_f32_f16", "rope_neox.comp", {{"A_TYPE", "float"}, {"ROPE_D_TYPE", "float16_t"}});
    string_to_spv("rope_neox_f32_f16_rte", "rope_neox.comp", {{"A_TYPE", "float"}, {"ROPE_D_TYPE", "float16_t"}, {"RTE16", "1"}});

    string_to_spv("rope_multi_f32", "rope_multi.comp", {{"A_TYPE", "float"}, {"ROPE_D_TYPE", "float"}});
    string_to_spv("rope_multi_f16", "rope_multi.comp", {{"A_TYPE", "float16_t"}, {"ROPE_D_TYPE", "float16_t"}});
    string_to_spv("rope_multi_f16_rte", "rope_multi.comp", {{"A_TYPE", "float16_t"}, {"ROPE_D_TYPE", "float16_t"}, {"RTE16", "1"}});
    string_to_spv("rope_multi_f32_f16", "rope_multi.comp", {{"A_TYPE", "float"}, {"ROPE_D_TYPE", "float16_t"}});
    string_to_spv("rope_multi_f32_f16_rte", "rope_multi.comp", {{"A_TYPE", "float"}, {"ROPE_D_TYPE", "float16_t"}, {"RTE16", "1"}});

    string_to_spv("rope_vision_f32", "rope_vision.comp", {{"A_TYPE", "float"}, {"ROPE_D_TYPE", "float"}});
    string_to_spv("rope_vision_f16", "rope_vision.comp", {{"A_TYPE", "float16_t"}, {"ROPE_D_TYPE", "float16_t"}});
    string_to_spv("rope_vision_f16_rte", "rope_vision.comp", {{"A_TYPE", "float16_t"}, {"ROPE_D_TYPE", "float16_t"}, {"RTE16", "1"}});

    string_to_spv("argsort_f32", "argsort.comp", {{"A_TYPE", "float"}});
    string_to_spv("argsort_large_f32", "argsort_large.comp", {{"A_TYPE", "float"}});

    string_to_spv("topk_argsort_f32", "topk_argsort.comp", {{"A_TYPE", "float"}});
    string_to_spv("topk_nary_search_f32", "topk_nary_search.comp", {{"A_TYPE", "float"}});

    string_to_spv("argmax_f32", "argmax.comp", merge_maps(base_dict, {{"A_TYPE", "float"}, {"D_TYPE", "int"}}));
    string_to_spv("sum_rows_f32", "sum_rows.comp", merge_maps(base_dict, {{"A_TYPE", "float"}, {"D_TYPE", "float"}}));
    string_to_spv("count_equal_i32", "count_equal.comp", merge_maps(base_dict, {{"A_TYPE", "int"}, {"B_TYPE", "int"}, {"D_TYPE", "int"}}));
    string_to_spv("cumsum_f32", "cumsum.comp", merge_maps(base_dict, {{"A_TYPE", "float"}, {"D_TYPE", "float"}}));
    string_to_spv("cumsum_multipass1_f32", "cumsum_multipass1.comp", merge_maps(base_dict, {{"A_TYPE", "float"}, {"D_TYPE", "float"}}));
    string_to_spv("cumsum_multipass2_f32", "cumsum_multipass2.comp", merge_maps(base_dict, {{"A_TYPE", "float"}, {"D_TYPE", "float"}}));

    string_to_spv("count_experts", "count_experts.comp", merge_maps(base_dict, {{"A_TYPE", "uint"}, {"D_TYPE", "uint"}}));

    for (std::string dim_str : {"", "_3d"}) {
        for (bool bda : {false, true}) {
            std::string bda_str = bda ? "_bda" : "";
            std::string bda_def = bda ? "1" : "0";
            string_to_spv("im2col" + dim_str + "_f32" + bda_str, "im2col" + dim_str + ".comp", merge_maps(base_dict, {{"A_TYPE", "float"}, {"D_TYPE", "float"}, {"D_SIZE", "4"}, {"BDA", bda_def}}));
            string_to_spv("im2col" + dim_str + "_f32_f16" + bda_str, "im2col" + dim_str + ".comp", merge_maps(base_dict, {{"A_TYPE", "float"}, {"D_TYPE", "float16_t"}, {"D_SIZE", "2"}, {"BDA", bda_def}}));
            string_to_spv("im2col" + dim_str + "_f32_f16_rte" + bda_str, "im2col" + dim_str + ".comp", merge_maps(base_dict, {{"A_TYPE", "float"}, {"D_TYPE", "float16_t"}, {"D_SIZE", "2"}, {"RTE16", "1"}, {"BDA", bda_def}}));
        }
    }

    string_to_spv("timestep_embedding_f32", "timestep_embedding.comp", merge_maps(base_dict, {{"A_TYPE", "float"}, {"D_TYPE", "float"}}));

    string_to_spv("conv_transpose_1d_f32", "conv_transpose_1d.comp", {{"A_TYPE", "float"},  {"B_TYPE", "float"}, {"D_TYPE", "float"}});

    string_to_spv("pool2d_f32", "pool2d.comp", merge_maps(base_dict, {{"A_TYPE", "float"}, {"D_TYPE", "float"}}));

    string_to_spv("rwkv_wkv6_f32", "wkv6.comp", merge_maps(base_dict, {{"A_TYPE", "float"}}));

    string_to_spv("rwkv_wkv7_f32", "wkv7.comp", merge_maps(base_dict, {{"A_TYPE", "float"}}));

    string_to_spv("opt_step_adamw_f32", "opt_step_adamw.comp", merge_maps(base_dict, {{"A_TYPE", "float"}}));
    string_to_spv("opt_step_sgd_f32", "opt_step_sgd.comp", merge_maps(base_dict, {{"A_TYPE", "float"}}));

    string_to_spv("solve_tri_f32", "solve_tri.comp", merge_maps(base_dict, {{"A_TYPE", "float"}, {"B_TYPE", "float"}, {"D_TYPE", "float"}}));

    for (auto transpose : {false, true}) {
        for (auto unroll : {false, true}) {
            for (auto a_f16 : {false, true}) {
                std::map<std::string, std::string> defines = {
                    {"A_TYPE", a_f16 ? "float16_t" : "float"}, {"B_TYPE", "float"}, {"D_TYPE", "float"},
                    {"USE_COLLECTIVES", "1"}, {"UNROLL", unroll ? "[[unroll]]" : ""},
                };
                if (transpose) defines["TRANSPOSE"] = "1";
                std::string name = std::string(transpose ? "conv_transpose_2d": "conv2d")
                    + (a_f16 ? "_f16" : "") + "_f32";
                string_to_spv(name + (unroll ? "_unroll" : ""), "conv2d_mm.comp", defines);
#if defined(GGML_VULKAN_COOPMAT2_GLSLC_SUPPORT)
                if (unroll) {
                    defines["COOPMAT2"] = "1";
                    string_to_spv(name, "conv2d_mm.comp", defines, true, false, true);
                }
#endif
            }
        }
    }

    string_to_spv("conv2d_dw_whcn_f32", "conv2d_dw.comp", merge_maps(base_dict, {{"A_TYPE", "float"}, {"B_TYPE", "float"}, {"D_TYPE", "float"}, {"WHCN", "1"}}));
    string_to_spv("conv2d_dw_cwhn_f32", "conv2d_dw.comp", merge_maps(base_dict, {{"A_TYPE", "float"}, {"B_TYPE", "float"}, {"D_TYPE", "float"}, {"CWHN", "1"}}));
    string_to_spv("conv2d_dw_whcn_f16_f32", "conv2d_dw.comp", merge_maps(base_dict, {{"A_TYPE", "float16_t"}, {"B_TYPE", "float"}, {"D_TYPE", "float"}, {"WHCN", "1"}}));
    string_to_spv("conv2d_dw_cwhn_f16_f32", "conv2d_dw.comp", merge_maps(base_dict, {{"A_TYPE", "float16_t"}, {"B_TYPE", "float"}, {"D_TYPE", "float"}, {"CWHN", "1"}}));

    string_to_spv("roll_f32", "roll.comp", merge_maps(base_dict, {{"A_TYPE", "float"}, {"D_TYPE", "float"}}));

    string_to_spv("add_id_f32", "add_id.comp", merge_maps(base_dict, {{"A_TYPE", "float"}, {"B_TYPE", "float"}, {"D_TYPE", "float"}}));

    string_to_spv("multi_add_f32", "multi_add.comp", {{"A_TYPE", "float"}, {"B_TYPE", "float"}, {"D_TYPE", "float"}, {"FLOAT_TYPE", "float"}, {"RTE16", "1"}, {"ADD_RMS" , "0"}});
    string_to_spv("multi_add_rms_f32", "multi_add.comp", {{"A_TYPE", "float"}, {"B_TYPE", "float"}, {"D_TYPE", "float"}, {"FLOAT_TYPE", "float"}, {"RTE16", "1"}, {"ADD_RMS" , "1"}});

    string_to_spv("ssm_scan_f32",          "ssm_scan.comp", {{"A_TYPE", "float"}});
    string_to_spv("ssm_scan_subgroup_f32", "ssm_scan.comp", {{"A_TYPE", "float"}, {"USE_SUBGROUP_ADD", "1"}});

    string_to_spv("ssm_conv_f32", "ssm_conv.comp", {{"A_TYPE", "float"}});

    string_to_spv("topk_moe_f32", "topk_moe.comp", {});

    for (auto &c : compiles) {
        c.wait();
    }
}

void write_output_files() {
    std::stringstream hdr = make_generic_stringstream();
    std::stringstream src = make_generic_stringstream();

    hdr << "#include <cstdint>\n\n";
    src << "#include \"" << basename(target_hpp) << "\"\n\n";

    std::sort(shader_fnames.begin(), shader_fnames.end());
    for (const auto& pair : shader_fnames) {
        const std::string& name = pair.first;
        #ifdef _WIN32
            std::string path = pair.second;
            std::replace(path.begin(), path.end(), '/', '\\' );
        #else
            const std::string& path = pair.second;
        #endif

        hdr << "extern const uint64_t " << name << "_len;\n";
        hdr << "extern const unsigned char " << name << "_data[];\n\n";

        if (input_filepath != "") {
            std::string data = read_binary_file(path);
            if (data.empty()) {
                continue;
            }

            src << "const uint64_t " << name << "_len = " << data.size() << ";\n";
            src << "const unsigned char " << name << "_data[" << data.size() << "] = {\n" << std::hex;
            auto bytes = reinterpret_cast<const uint8_t*>(data.data());
            for (size_t i = 0; i < data.size(); ++i) {
                src << "0x" << static_cast<int>(bytes[i]) << ",";
                if ((i + 1) % 12 == 0) src << "\n";
            }
            src << std::dec << "\n};\n\n";
        }
    }

    std::string suffixes[2] = {"_f32", "_f16"};
    for (std::string op : {"add", "sub", "mul", "div", "add_rms"}) {
        hdr << "extern const void * " << op << "_data[2][2][2][2];\n";
        hdr << "extern const uint64_t " << op << "_len[2][2][2][2];\n";

        std::string op_file = op == "add_rms" ? "add.comp" : std::string(op) + ".comp";
        if (basename(input_filepath) != op_file) {
            continue;
        }
        std::stringstream data = make_generic_stringstream();
        std::stringstream len  = make_generic_stringstream();
        data << "const void * " << op << "_data[2][2][2][2] = ";
        len  << "const uint64_t " << op << "_len[2][2][2][2] = ";
        for (uint32_t t0 = 0; t0 < 2; ++t0) {
            if (t0 == 0) {
                data << "{";
                len  << "{";
            }
            for (uint32_t t1 = 0; t1 < 2; ++t1) {
                if (t1 == 0) {
                    data << "{";
                    len  << "{";
                }
                for (uint32_t t2 = 0; t2 < 2; ++t2) {
                    if (t2 == 0) {
                        data << "{";
                        len  << "{";
                    }
                    for (uint32_t rte = 0; rte < 2; ++rte) {
                        if (rte == 0) {
                            data << "{";
                            len  << "{";
                        }
                        data << op << suffixes[t0] << suffixes[t1] << suffixes[t2] << ((rte != 0) ? "_rte" : "");
                        len  << op << suffixes[t0] << suffixes[t1] << suffixes[t2] << ((rte != 0) ? "_rte" : "");
                        data << "_data,";
                        len  << "_len,";
                        if (rte == 1) {
                            data << "}, ";
                            len  << "}, ";
                        }
                    }
                    if (t2 == 1) {
                        data << "}, ";
                        len  << "}, ";
                    }
                }
                if (t1 == 1) {
                    data << "}, ";
                    len  << "}, ";
                }
            }
            if (t0 == 1) {
                data << "};\n";
                len  << "};\n";
            }
        }
        src << data.str();
        src << len.str();
    }

    std::vector<std::string> btypes = {"f16", "f32"};

#if defined(GGML_VULKAN_INTEGER_DOT_GLSLC_SUPPORT)
    btypes.push_back("q8_1");
#endif

    for (const std::string& btype : btypes) {
    for (const auto& tname : type_names) {
        if (btype == "q8_1" && !is_legacy_quant(tname) && tname != "mxfp4" && !is_k_quant(tname) && tname != "iq1_s" && tname != "iq1_m") {
            continue;
        }
        hdr << "extern const void * arr_dmmv_"   << tname << "_" << btype << "_f32_data[3];\n";
        hdr << "extern const uint64_t arr_dmmv_" << tname << "_" << btype << "_f32_len[3];\n";
        if (basename(input_filepath) == "mul_mat_vec.comp") {
            src << "const void * arr_dmmv_"   << tname << "_" << btype << "_f32_data[3] = {mul_mat_vec_" << tname << "_" << btype << "_f32_data, mul_mat_vec_" << tname << "_" << btype << "_f32_subgroup_data, mul_mat_vec_" << tname << "_" << btype << "_f32_subgroup_no_shmem_data};\n";
            src << "const uint64_t arr_dmmv_" << tname << "_" << btype << "_f32_len[3] =  {mul_mat_vec_" << tname << "_" << btype << "_f32_len,  mul_mat_vec_" << tname << "_" << btype << "_f32_subgroup_len, mul_mat_vec_"  << tname << "_" << btype << "_f32_subgroup_no_shmem_len};\n";
        }

        if (btype == "f16") {
            continue;
        }
        hdr << "extern const void * arr_dmmv_id_"   << tname << "_" << btype << "_f32_data[3];\n";
        hdr << "extern const uint64_t arr_dmmv_id_" << tname << "_" << btype << "_f32_len[3];\n";
        if (basename(input_filepath) == "mul_mat_vec.comp") {
            src << "const void * arr_dmmv_id_"   << tname << "_" << btype << "_f32_data[3] = {mul_mat_vec_id_" << tname << "_" << btype << "_f32_data, mul_mat_vec_id_" << tname << "_" << btype << "_f32_subgroup_data, mul_mat_vec_id_" << tname << "_" << btype << "_f32_subgroup_no_shmem_data};\n";
            src << "const uint64_t arr_dmmv_id_" << tname << "_" << btype << "_f32_len[3] =  {mul_mat_vec_id_" << tname << "_" << btype << "_f32_len,  mul_mat_vec_id_" << tname << "_" << btype << "_f32_subgroup_len, mul_mat_vec_id_"  << tname << "_" << btype << "_f32_subgroup_no_shmem_len};\n";
        }
    }
    }

    if (input_filepath == "") {
        write_file_if_changed(target_hpp, hdr.str());
    }
    if (target_cpp != "") {
        write_binary_file(target_cpp, src.str());
    }
}

} // namespace

int main(int argc, char** argv) {
    std::map<std::string, std::string> args;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg.rfind("--", 0) == 0) {
            if (i + 1 < argc && argv[i + 1][0] != '-') {
                args[arg] = argv[i + 1];
                ++i;
            } else {
                args[arg] = "";
            }
        }
    }

    if (args.find("--glslc") != args.end()) {
        GLSLC = args["--glslc"]; // Path to glslc
    }
    if (args.find("--source") != args.end()) {
        input_filepath = args["--source"]; // The shader source file to compile
    }
    if (args.find("--output-dir") != args.end()) {
        output_dir = args["--output-dir"]; // Directory for containing SPIR-V output
    }
    if (args.find("--target-hpp") != args.end()) {
        target_hpp = args["--target-hpp"]; // Path to generated header file
    }
    if (args.find("--target-cpp") != args.end()) {
        target_cpp = args["--target-cpp"]; // Path to generated cpp file
    }

    if (!directory_exists(output_dir)) {
        if (!create_directory(output_dir)) {
            std::cerr << "Error creating output directory: " << output_dir << "\n";
            return EXIT_FAILURE;
        }
    }

    process_shaders();

    write_output_files();

    return EXIT_SUCCESS;
}
