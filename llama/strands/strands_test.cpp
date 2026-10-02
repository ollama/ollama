// Run: cmake --build build/llama-server-local --target strands-head-test
//      build/llama-server-local/bin/strands-head-test /tmp/strands-test.gguf
#include "strands.h"
#include "ggml.h"
#include "gguf.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <stdexcept>

int main(int argc, char **argv) {
    if (argc != 2) return 2;
    try {
        auto ctx = ggml_init({1 << 20, nullptr, false});
        auto meta = gguf_init_empty();
        gguf_set_val_str(meta, "general.architecture", "qwen35");
        gguf_set_val_u32(meta, "qwen35.decision.hidden_size", 4);
        gguf_set_val_u32(meta, "qwen35.decision.pointer_dim", 2);
        gguf_set_val_f32(meta, "qwen35.decision.temperature.noul", .5f);
        gguf_set_val_f32(meta, "qwen35.decision.temperature.choice", 1.f);
        gguf_set_val_f32(meta, "qwen35.decision.temperature.score", 2.f);
        auto weight = [&](const char *name, int rows, int cols, std::initializer_list<float> data) {
            auto t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, cols, rows);
            ggml_set_name(t, name);
            std::copy(data.begin(), data.end(), static_cast<float *>(t->data));
            gguf_add_tensor(meta, t);
        };
        weight("strands.norm.weight", 1, 4, {1.1f,.9f,1.3f,.8f});
        weight("strands.norm.bias", 1, 4, {.1f,-.2f,.3f,-.4f});
        weight("strands.q.weight", 2, 4, {.1f,.2f,.3f,.4f,-.4f,.2f,.1f,.3f});
        weight("strands.q.bias", 1, 2, {.2f,-.3f});
        weight("strands.k.weight", 2, 4, {.3f,-.1f,.2f,.4f,.2f,.4f,-.3f,.1f});
        weight("strands.k.bias", 1, 2, {-.1f,.3f});
        if (!gguf_write_to_file(meta, argv[1], false)) throw std::runtime_error("could not write fixture");
        gguf_free(meta);
        ggml_free(ctx);

        strands_head head(argv[1]);
        const std::vector<std::vector<float>> hidden{{1,2,4,-1},{3,-2,0,1},{-1,4,2,0}};
        auto fields = common_json::parse(R"([{"type":1,"question":[2,3],"options":[[0,1],[1,2]]}])");
        // PyTorch LayerNorm + Linear + scaled dot product, independently evaluated.
        const float expected[] = {-.11312298476696014f, .061370573937892914f};
        const float temperature[] = {.5f,1.f,2.f};
        for (int type = 0; type < 3; ++type) {
            fields[0]["type"] = type;
            auto logits = head.score(hidden, fields);
            for (int i = 0; i < 2; ++i)
                if (std::abs(logits.at(0).at(i) - expected[i]/temperature[type]) > 1e-6f)
                    throw std::runtime_error("pointer/calibration parity failed");
        }
        for (const auto &invalid : {
                R"([{"type":3,"question":[2,3],"options":[[0,1],[1,2]]}])",
                R"([{"type":1,"question":[0,1],"options":[[0,1],[1,2]]}])",
                R"([{"type":1,"question":[2,3],"options":[[0,1],[2,3]]}])",
                R"([{"type":1,"question":[2,3],"options":[[-1,1],[1,2]]}])"}) {
            bool rejected = false;
            try { head.score(hidden, common_json::parse(invalid)); }
            catch (const std::exception &) { rejected = true; }
            if (!rejected) throw std::runtime_error("invalid span/type accepted");
        }
        std::remove(argv[1]);
        std::cout << "Strands pointer head and calibration tests passed\n";
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        std::remove(argv[1]);
        return 1;
    }
}
