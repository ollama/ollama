#pragma once

#include <string>
#include <vector>
#include <cstdint>

enum llm_chat_template {
    LLM_CHAT_TEMPLATE_CHATML,
    LLM_CHAT_TEMPLATE_LLAMA_2,
    LLM_CHAT_TEMPLATE_LLAMA_2_SYS,
    LLM_CHAT_TEMPLATE_LLAMA_2_SYS_BOS,
    LLM_CHAT_TEMPLATE_LLAMA_2_SYS_STRIP,
    LLM_CHAT_TEMPLATE_MISTRAL_V1,
    LLM_CHAT_TEMPLATE_MISTRAL_V3,
    LLM_CHAT_TEMPLATE_MISTRAL_V3_TEKKEN,
    LLM_CHAT_TEMPLATE_MISTRAL_V7,
    LLM_CHAT_TEMPLATE_PHI_3,
    LLM_CHAT_TEMPLATE_FALCON_3,
    LLM_CHAT_TEMPLATE_ZEPHYR,
    LLM_CHAT_TEMPLATE_MONARCH,
    LLM_CHAT_TEMPLATE_GEMMA,
    LLM_CHAT_TEMPLATE_ORION,
    LLM_CHAT_TEMPLATE_OPENCHAT,
    LLM_CHAT_TEMPLATE_VICUNA,
    LLM_CHAT_TEMPLATE_VICUNA_ORCA,
    LLM_CHAT_TEMPLATE_DEEPSEEK,
    LLM_CHAT_TEMPLATE_DEEPSEEK_2,
    LLM_CHAT_TEMPLATE_DEEPSEEK_3,
    LLM_CHAT_TEMPLATE_COMMAND_R,
    LLM_CHAT_TEMPLATE_LLAMA_3,
    LLM_CHAT_TEMPLATE_CHATGML_3,
    LLM_CHAT_TEMPLATE_CHATGML_4,
    LLM_CHAT_TEMPLATE_MINICPM,
    LLM_CHAT_TEMPLATE_EXAONE_3,
    LLM_CHAT_TEMPLATE_RWKV_WORLD,
    LLM_CHAT_TEMPLATE_GRANITE,
    LLM_CHAT_TEMPLATE_GIGACHAT,
    LLM_CHAT_TEMPLATE_MEGREZ,
    LLM_CHAT_TEMPLATE_UNKNOWN,
};

struct llama_chat_message;

llm_chat_template llm_chat_template_from_str(const std::string & name);

llm_chat_template llm_chat_detect_template(const std::string & tmpl);

int32_t llm_chat_apply_template(
    llm_chat_template tmpl,
    const std::vector<const llama_chat_message *> & chat,
    std::string & dest, bool add_ass);
