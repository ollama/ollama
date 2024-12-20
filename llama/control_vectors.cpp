#include "control_vectors.h"
#include "common.h"

extern "C" {

int32_t llama_apply_control_vector(
    const struct llama_model * model,
    struct llama_context * lctx,
    char* path,
    float strength
) {
    // TODO support multiple control vectors at once
    struct common_control_vector_load_info info;
    info.strength = strength;
    info.fname = std::string(path);

    std::vector<common_control_vector_load_info> infos;
    infos.push_back(info);

    const auto cvec = common_control_vector_load(infos);
    if (cvec.n_embd == -1) {
        return -1;
    }

    int32_t control_vector_layer_start = 1;
    int32_t control_vector_layer_end = llama_n_layer(model);

    int err = llama_control_vector_apply(
        lctx,
        cvec.data.data(),
        cvec.data.size(),
        cvec.n_embd,
        control_vector_layer_start,
        control_vector_layer_end
    );

    return err;
}

}
