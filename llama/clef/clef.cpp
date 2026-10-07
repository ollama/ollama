// Clef joint schema head, ported from Cloudflare's Apache-2.0 reference.
// https://huggingface.co/Cloudflare/clef/blob/main/joint_schema_model.py
#include "clef.h"
#include "ggml-backend.h"
#include "ggml.h"
#include "gguf.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <map>
#include <stdexcept>

#if !defined(_WIN32)
#include <sys/types.h>
#endif

namespace {
struct matrix {
    int rows = 0, cols = 0;
    std::vector<float> data;
    matrix() = default;
    matrix(int r, int c) : rows(r), cols(c), data(size_t(r) * c) {}
    float &at(int r, int c) { return data[size_t(r) * cols + c]; }
    float at(int r, int c) const { return data[size_t(r) * cols + c]; }
};
void require(bool condition, const char *message) {
    if (!condition)
        throw std::runtime_error(message);
}
// Use the backend registry so builds with dynamically selected CPU variants
// work without linking to a particular ggml-cpu library.
struct cpu_backend {
    ggml_backend_t handle = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    cpu_backend() {
        require(handle != nullptr, "Clef: CPU backend unavailable");
        auto reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(handle));
        auto set_threads = reinterpret_cast<ggml_backend_set_n_threads_t>(
            ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_n_threads"));
        if (set_threads)
            set_threads(handle, 8);
    }
    ~cpu_backend() { ggml_backend_free(handle); }
};
struct graph {
    ggml_context *ctx;
    explicit graph(size_t bytes) {
        ctx = ggml_init({bytes + (2 << 20), nullptr, true});
        require(ctx != nullptr, "Clef: cannot allocate graph");
    }
    ~graph() { ggml_free(ctx); }
    ggml_tensor *input(const matrix &m) {
        auto t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, m.cols, m.rows);
        t->data = const_cast<float *>(m.data.data());
        return t;
    }
    matrix run(ggml_tensor *output, int rows, int cols) {
        auto gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, output);
        static thread_local cpu_backend cpu;
        require(ggml_backend_graph_compute(cpu.handle, gf) == GGML_STATUS_SUCCESS,
                "Clef: graph evaluation failed");
        matrix result(rows, cols);
        memcpy(result.data.data(), output->data, result.data.size() * sizeof(float));
        return result;
    }
};
matrix linear(const matrix &x, const matrix &w) {
    require(x.cols == w.cols, "Clef: linear shape mismatch");
    graph g(size_t(x.rows) * w.rows * sizeof(float));
    auto tx = g.input(x), tw = g.input(w);
    ggml_set_no_alloc(g.ctx, false);
    return g.run(ggml_mul_mat(g.ctx, tw, tx), x.rows, w.rows);
}
matrix add(matrix a, const matrix &b) {
    require(a.cols == b.cols && (b.rows == 1 || a.rows == b.rows), "Clef: addition shape mismatch");
    for (int r = 0; r < a.rows; ++r)
        for (int c = 0; c < a.cols; ++c)
            a.at(r, c) += b.at(b.rows == 1 ? 0 : r, c);
    return a;
}
matrix slice(const matrix &m, int start, int end) {
    require(start >= 0 && start < end && end <= m.rows, "Clef: invalid span");
    matrix out(end - start, m.cols);
    std::copy(m.data.begin() + size_t(start) * m.cols, m.data.begin() + size_t(end) * m.cols,
              out.data.begin());
    return out;
}
matrix mean(const matrix &m, int start, int end) {
    require(start >= 0 && start < end && end <= m.rows, "Clef: invalid span");
    matrix out(1, m.cols);
    for (int r = start; r < end; ++r)
        for (int c = 0; c < m.cols; ++c)
            out.at(0, c) += m.at(r, c) / (end - start);
    return out;
}
void append(matrix &a, const matrix &b) {
    require(a.rows == 0 || a.cols == b.cols, "Clef: append shape mismatch");
    a.cols = b.cols;
    a.rows += b.rows;
    a.data.insert(a.data.end(), b.data.begin(), b.data.end());
}
float cosine(const matrix &a, int ar, const matrix &b, int br, double eps = 1e-8) {
    require(a.cols == b.cols, "Clef: cosine shape mismatch");
    double dot = 0, aa = 0, bb = 0;
    for (int c = 0; c < a.cols; ++c) {
        dot += double(a.at(ar, c)) * b.at(br, c);
        aa += double(a.at(ar, c)) * a.at(ar, c);
        bb += double(b.at(br, c)) * b.at(br, c);
    }
    return dot / (std::max(std::sqrt(aa), eps) * std::max(std::sqrt(bb), eps));
}
} // namespace

struct clef_head::impl {
    FILE *file = nullptr;
    gguf_context *meta = nullptr;
    ggml_context *tensors = nullptr;
    std::map<std::string, matrix> weights;
    int hidden, width, heads, routing_layers, layers;
    int output_index;
    ggml_tensor *output;

    explicit impl(const std::string &path) : file(ggml_fopen(path.c_str(), "rb")) {
        try {
            meta = gguf_init_from_file(path.c_str(), {true, &tensors});
            require(meta && tensors && file, "Clef: cannot open model");
            const int arch_key = gguf_find_key(meta, "general.architecture");
            require(arch_key >= 0 && gguf_get_kv_type(meta, arch_key) == GGUF_TYPE_STRING,
                    "Clef: missing architecture");
            const std::string prefix = std::string(gguf_get_val_str(meta, arch_key)) + ".decision.";
            auto integer = [&](const std::string &key) {
                int i = gguf_find_key(meta, (prefix + key).c_str());
                require(i >= 0 && gguf_get_kv_type(meta, i) == GGUF_TYPE_UINT32,
                        "Clef: missing head configuration");
                return int(gguf_get_val_u32(meta, i));
            };
            hidden = integer("hidden_size");
            width = integer("width");
            heads = integer("heads");
            routing_layers = integer("routing_layers");
            layers = integer("layers");
            require(hidden > 0 && hidden <= 16384 && width > 0 && width <= 4096 && heads > 0 &&
                        width % heads == 0 && routing_layers >= 0 && routing_layers <= 32 && layers >= 0 &&
                        layers <= 32,
                    "Clef: invalid head configuration");
            for (int i = 0; i < gguf_get_n_tensors(meta); ++i) {
                std::string name = gguf_get_tensor_name(meta, i);
                if (name.rfind("clef.", 0) != 0)
                    continue;
                auto t = ggml_get_tensor(tensors, name.c_str());
                require(t->ne[2] == 1 && t->ne[3] == 1 && t->type == GGML_TYPE_F32,
                        "Clef: head tensors must be float32 matrices");
                matrix m(t->ne[1], t->ne[0]);
                read(i, 0, m.data.data(), m.data.size() * sizeof(float));
                weights.emplace(name.substr(5), std::move(m));
            }
            output_index = gguf_find_tensor(meta, "output.weight");
            require(output_index >= 0, "Clef: output embeddings are missing");
            output = ggml_get_tensor(tensors, "output.weight");
            require(output->ne[0] == hidden, "Clef: output embedding shape mismatch");
        } catch (...) {
            if (tensors)
                ggml_free(tensors);
            if (meta)
                gguf_free(meta);
            if (file)
                std::fclose(file);
            throw;
        }
    }
    ~impl() {
        ggml_free(tensors);
        gguf_free(meta);
        std::fclose(file);
    }
    void read(int index, size_t offset, void *data, size_t bytes) {
        // The head can sit past 2 GiB, where iostream seeks truncate on some
        // Windows C++ runtimes, so seek with an explicit 64-bit offset.
        const size_t position = gguf_get_data_offset(meta) + gguf_get_tensor_offset(meta, index) + offset;
#if defined(_WIN32)
        const int seek_rc = _fseeki64(file, static_cast<__int64>(position), SEEK_SET);
#else
        const int seek_rc = fseeko(file, static_cast<off_t>(position), SEEK_SET);
#endif
        require(seek_rc == 0 && std::fread(data, 1, bytes, file) == bytes, "Clef: truncated tensor data");
    }
    const matrix &w(const std::string &key) { return weights.at(key); }
    matrix norm(const matrix &x, const std::string &key) {
        const auto &weight = w(key + ".weight"), &bias = w(key + ".bias");
        require(weight.data.size() == size_t(x.cols) && bias.data.size() == size_t(x.cols),
                "Clef: layer norm shape mismatch");
        matrix out = x;
        for (int r = 0; r < x.rows; ++r) {
            double m = 0, v = 0;
            for (int c = 0; c < x.cols; ++c)
                m += x.at(r, c) / x.cols;
            for (int c = 0; c < x.cols; ++c)
                v += (x.at(r, c) - m) * (x.at(r, c) - m) / x.cols;
            for (int c = 0; c < x.cols; ++c)
                out.at(r, c) = (x.at(r, c) - m) / std::sqrt(v + 1e-5) * weight.data[c] + bias.data[c];
        }
        return out;
    }
    matrix project(const matrix &x, const std::string &key, bool bias = false) {
        matrix out = linear(x, w(key + ".weight"));
        return bias ? add(std::move(out), w(key + ".bias")) : out;
    }
    matrix feedforward(matrix x, const std::string &first, const std::string &last) {
        x = project(x, first, true);
        for (auto &v : x.data)
            v = 0.5f * v * (1.0f + std::erf(v / std::sqrt(2.0f)));
        return project(x, last, true);
    }
    matrix attention(const matrix &query, const matrix &memory, const std::string &key) {
        const auto &weight = w(key + ".in_proj_weight"), &bias = w(key + ".in_proj_bias");
        require(weight.rows == 3 * width && weight.cols == width && bias.data.size() == size_t(3 * width),
                "Clef: attention shape mismatch");
        auto projection = [&](const matrix &x, int n) {
            auto y = linear(x, slice(weight, n * width, (n + 1) * width));
            for (int r = 0; r < y.rows; ++r)
                for (int c = 0; c < width; ++c)
                    y.at(r, c) += bias.data[n * width + c];
            return y;
        };
        auto q = projection(query, 0), k = projection(memory, 1), v = projection(memory, 2);
        const int d = width / heads, nq = q.rows, nk = k.rows;
        graph g((size_t(nq) * nk * heads * 3 + size_t(nq + nk) * width * 8) * sizeof(float));
        auto tq = g.input(q), tk = g.input(k), tv = g.input(v);
        ggml_set_no_alloc(g.ctx, false);
        auto split = [&](ggml_tensor *t, int n) {
            return ggml_cont(g.ctx, ggml_permute(g.ctx, ggml_reshape_3d(g.ctx, t, d, heads, n), 0, 2, 1, 3));
        };
        tq = split(tq, nq);
        tk = split(tk, nk);
        tv = split(tv, nk);
        auto scores =
            ggml_soft_max(g.ctx, ggml_scale(g.ctx, ggml_mul_mat(g.ctx, tk, tq), 1.0f / std::sqrt(float(d))));
        tv = ggml_cont(g.ctx, ggml_transpose(g.ctx, tv));
        auto out = ggml_mul_mat(g.ctx, tv, scores);
        out = ggml_cont(g.ctx, ggml_permute(g.ctx, out, 0, 2, 1, 3));
        auto result = g.run(ggml_reshape_2d(g.ctx, out, width, nq), nq, width);
        return project(result, key + ".out_proj", true);
    }
    matrix lexical(const std::vector<int32_t> &tokens, int start, int end) {
        require(start >= 0 && start < end && end <= int(tokens.size()), "Clef: invalid lexical span");
        matrix result(1, hidden);
        const auto traits = ggml_get_type_traits(output->type);
        require(traits->to_float != nullptr || output->type == GGML_TYPE_F32,
                "Clef: unsupported output embedding type");
        size_t bytes = ggml_row_size(output->type, hidden);
        std::vector<uint8_t> raw(bytes);
        std::vector<float> row(hidden);
        for (int i = start; i < end; ++i) {
            require(tokens[i] >= 0 && tokens[i] < output->ne[1], "Clef: invalid token");
            read(output_index, size_t(tokens[i]) * bytes, raw.data(), bytes);
            if (output->type == GGML_TYPE_F32)
                memcpy(row.data(), raw.data(), bytes);
            else
                traits->to_float(raw.data(), row.data(), hidden);
            for (int c = 0; c < hidden; ++c)
                result.at(0, c) += row[c] / (end - start);
        }
        return result;
    }
    std::vector<std::vector<float>> score(const std::vector<std::vector<float>> &states,
                                          const std::vector<int32_t> &tokens, const common_json &fields) {
        require(states.size() == tokens.size() && !states.empty(), "Clef: missing backbone hidden states");
        require(fields.is_array() && !fields.empty() && fields.size() <= 64, "Clef: expected 1-64 fields");
        matrix x;
        for (const auto &row : states) {
            require(row.size() == size_t(hidden), "Clef: hidden size mismatch");
            matrix m(1, hidden);
            m.data = row;
            append(x, m);
        }
        x = norm(x, "hidden_norm");
        auto memory = project(x, "memory_projection"), global = slice(x, x.rows - 1, x.rows);
        matrix questions, lexical_options, queries;
        std::vector<int> counts, types;
        auto span = [](const common_json &value) {
            require(value.is_array() && value.size() == 2 && value[0].is_number_integer() &&
                        value[1].is_number_integer(),
                    "Clef: malformed span");
            return std::pair<int, int>{value[0].get<int>(), value[1].get<int>()};
        };
        for (const auto &field : fields) {
            int type = field.at("type").get<int>();
            require(type >= 0 && type < 3, "Clef: invalid question type");
            types.push_back(type);
            auto [qs, qe] = span(field.at("question"));
            auto question = mean(x, qs, qe);
            append(questions, question);
            const auto &options = field.at("options");
            require(options.is_array() && options.size() >= 2 && options.size() <= 26,
                    "Clef: expected 2-26 options");
            counts.push_back(options.size());
            matrix contexts, lex;
            for (const auto &option : options) {
                auto [s, e] = span(option);
                append(contexts, mean(x, s, e));
                append(lex, lexical(tokens, s, e));
            }
            append(lexical_options, lex);
            append(queries, add(add(project(contexts, "option_context_projection"),
                                    project(lex, "option_lexical_projection")),
                                project(question, "option_question_projection")));
        }
        for (int i = 0; i < routing_layers; ++i) {
            auto key = "evidence_layers." + std::to_string(i);
            queries = add(queries, attention(norm(queries, key + ".query_norm"),
                                             norm(memory, key + ".memory_norm"), key + ".attention"));
            queries = add(queries, feedforward(norm(queries, key + ".feedforward_norm"),
                                               key + ".feedforward.0", key + ".feedforward.3"));
        }
        auto base = project(questions, "question_projection");
        matrix summaries;
        int offset = 0;
        for (int f = 0; f < base.rows; ++f) {
            std::vector<float> scores(counts[f]);
            for (int j = 0; j < counts[f]; ++j)
                for (int c = 0; c < width; ++c)
                    scores[j] += queries.at(offset + j, c) * base.at(f, c) / std::sqrt(float(width));
            float peak = *std::max_element(scores.begin(), scores.end()), sum = 0;
            for (auto &v : scores) {
                v = std::exp(v - peak);
                sum += v;
            }
            matrix summary(1, width);
            for (int j = 0; j < counts[f]; ++j)
                for (int c = 0; c < width; ++c)
                    summary.at(0, c) += scores[j] / sum * queries.at(offset + j, c);
            append(summaries, summary);
            offset += counts[f];
        }
        auto fs =
            add(add(base, norm(summaries, "option_summary_norm")), project(global, "global_projection"));
        const auto &te = w("type_embedding.weight");
        require(te.rows == 3 && te.cols == width, "Clef: type embedding shape mismatch");
        for (int f = 0; f < fs.rows; ++f)
            for (int c = 0; c < width; ++c)
                fs.at(f, c) += te.at(types[f], c);
        for (int i = 0; i < layers; ++i) {
            auto key = "layers." + std::to_string(i);
            auto normalized = norm(fs, key + ".norm1");
            fs = add(fs, attention(normalized, normalized, key + ".self_attn"));
            fs = add(fs, attention(norm(fs, key + ".norm2"), memory, key + ".multihead_attn"));
            fs = add(fs, feedforward(norm(fs, key + ".norm3"), key + ".linear1", key + ".linear2"));
        }
        fs = norm(fs, "field_norm");
        auto options = norm(queries, "option_norm");
        float prior_scale = std::exp(std::min(w("prior_logit_scale").data.at(0), std::log(100.0f)));
        float joint_scale = std::exp(std::min(w("joint_logit_scale").data.at(0), std::log(100.0f)));
        float gate = 1.0f / (1.0f + std::exp(-w("residual_gate").data.at(0)));
        std::vector<std::vector<float>> result;
        offset = 0;
        for (int f = 0; f < fs.rows; ++f) {
            matrix features(counts[f], 4 * width);
            for (int j = 0; j < counts[f]; ++j)
                for (int c = 0; c < width; ++c) {
                    float a = fs.at(f, c), b = options.at(offset + j, c);
                    features.at(j, c) = a;
                    features.at(j, width + c) = b;
                    features.at(j, 2 * width + c) = a * b;
                    features.at(j, 3 * width + c) = std::abs(a - b);
                }
            auto residual = feedforward(features, "residual_scorer.0", "residual_scorer.3");
            auto anchor = add(slice(questions, f, f + 1), global);
            std::vector<float> logits;
            for (int j = 0; j < counts[f]; ++j) {
                float v = prior_scale * cosine(lexical_options, offset + j, anchor, 0, 1e-12) +
                          gate * (joint_scale * cosine(fs, f, options, offset + j) + residual.at(j, 0));
                require(std::isfinite(v), "Clef: non-finite logit");
                logits.push_back(v);
            }
            result.push_back(std::move(logits));
            offset += counts[f];
        }
        return result;
    }
};
clef_head::clef_head(const std::string &path) : p(new impl(path)) {}
clef_head::~clef_head() = default;
std::vector<std::vector<float>> clef_head::score(const std::vector<std::vector<float>> &hidden,
                                                 const std::vector<int32_t> &tokens,
                                                 const common_json &fields) {
    return p->score(hidden, tokens, fields);
}
