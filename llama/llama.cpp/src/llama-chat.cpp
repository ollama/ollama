#include "llama-chat.h"

#include "llama.h"

#include <map>
#include <sstream>
#include <algorithm>

#if __cplusplus >= 202000L
    #define LU8(x) (const char*)(u8##x)
#else
    #define LU8(x) u8##x
#endif

// trim whitespace from the beginning and end of a string
static std::string trim(const std::string & str) {
    size_t start = 0;
    size_t end = str.size();
    while (start < end && isspace(static_cast<unsigned char>(str[start]))) {
        start += 1;
    }
    while (end > start && isspace(static_cast<unsigned char>(str[end - 1]))) {
        end -= 1;
    }
    return str.substr(start, end - start);
}

static const std::map<std::string, llm_chat_template> LLM_CHAT_TEMPLATES = {
    { "chatml",            LLM_CHAT_TEMPLATE_CHATML            },
    { "llama2",            LLM_CHAT_TEMPLATE_LLAMA_2           },
    { "llama2-sys",        LLM_CHAT_TEMPLATE_LLAMA_2_SYS       },
    { "llama2-sys-bos",    LLM_CHAT_TEMPLATE_LLAMA_2_SYS_BOS   },
    { "llama2-sys-strip",  LLM_CHAT_TEMPLATE_LLAMA_2_SYS_STRIP },
    { "mistral-v1",        LLM_CHAT_TEMPLATE_MISTRAL_V1        },
    { "mistral-v3",        LLM_CHAT_TEMPLATE_MISTRAL_V3        },
    { "mistral-v3-tekken", LLM_CHAT_TEMPLATE_MISTRAL_V3_TEKKEN },
    { "mistral-v7",        LLM_CHAT_TEMPLATE_MISTRAL_V7        },
    { "mistral-v7-tekken", LLM_CHAT_TEMPLATE_MISTRAL_V7_TEKKEN },
    { "phi3",              LLM_CHAT_TEMPLATE_PHI_3             },
    { "phi4",              LLM_CHAT_TEMPLATE_PHI_4             },
    { "falcon3",           LLM_CHAT_TEMPLATE_FALCON_3          },
    { "zephyr",            LLM_CHAT_TEMPLATE_ZEPHYR            },
    { "monarch",           LLM_CHAT_TEMPLATE_MONARCH           },
    { "gemma",             LLM_CHAT_TEMPLATE_GEMMA             },
    { "orion",             LLM_CHAT_TEMPLATE_ORION             },
    { "openchat",          LLM_CHAT_TEMPLATE_OPENCHAT          },
    { "vicuna",            LLM_CHAT_TEMPLATE_VICUNA            },
    { "vicuna-orca",       LLM_CHAT_TEMPLATE_VICUNA_ORCA       },
    { "deepseek",          LLM_CHAT_TEMPLATE_DEEPSEEK          },
    { "deepseek2",         LLM_CHAT_TEMPLATE_DEEPSEEK_2        },
    { "deepseek3",         LLM_CHAT_TEMPLATE_DEEPSEEK_3        },
    { "command-r",         LLM_CHAT_TEMPLATE_COMMAND_R         },
    { "llama3",            LLM_CHAT_TEMPLATE_LLAMA_3           },
    { "chatglm3",          LLM_CHAT_TEMPLATE_CHATGLM_3         },
    { "chatglm4",          LLM_CHAT_TEMPLATE_CHATGLM_4         },
    { "glmedge",           LLM_CHAT_TEMPLATE_GLMEDGE           },
    { "minicpm",           LLM_CHAT_TEMPLATE_MINICPM           },
    { "exaone3",           LLM_CHAT_TEMPLATE_EXAONE_3          },
    { "exaone4",           LLM_CHAT_TEMPLATE_EXAONE_4          },
    { "rwkv-world",        LLM_CHAT_TEMPLATE_RWKV_WORLD        },
    { "granite",           LLM_CHAT_TEMPLATE_GRANITE           },
    { "gigachat",          LLM_CHAT_TEMPLATE_GIGACHAT          },
    { "megrez",            LLM_CHAT_TEMPLATE_MEGREZ            },
    { "yandex",            LLM_CHAT_TEMPLATE_YANDEX            },
    { "bailing",           LLM_CHAT_TEMPLATE_BAILING           },
    { "llama4",            LLM_CHAT_TEMPLATE_LLAMA4            },
    { "smolvlm",           LLM_CHAT_TEMPLATE_SMOLVLM           },
    { "hunyuan-moe",       LLM_CHAT_TEMPLATE_HUNYUAN_MOE       },
    { "gpt-oss",           LLM_CHAT_TEMPLATE_OPENAI_MOE        },
    { "hunyuan-dense",     LLM_CHAT_TEMPLATE_HUNYUAN_DENSE     },
    { "kimi-k2",           LLM_CHAT_TEMPLATE_KIMI_K2           },
    { "seed_oss",          LLM_CHAT_TEMPLATE_SEED_OSS          },
};

llm_chat_template llm_chat_template_from_str(const std::string & name) {
    return LLM_CHAT_TEMPLATES.at(name);
}

llm_chat_template llm_chat_detect_template(const std::string & tmpl) {
    try {
        return llm_chat_template_from_str(tmpl);
    } catch (const std::out_of_range &) {
        // ignore
    }

    auto tmpl_contains = [&tmpl](const char * haystack) -> bool {
        return tmpl.find(haystack) != std::string::npos;
    };
    if (tmpl_contains("<|im_start|>")) {
        return tmpl_contains("<|im_sep|>")
            ? LLM_CHAT_TEMPLATE_PHI_4
            : tmpl_contains("<end_of_utterance>")
                ? LLM_CHAT_TEMPLATE_SMOLVLM // SmolVLM uses <|im_start|> as BOS, but it is NOT chatml
                : LLM_CHAT_TEMPLATE_CHATML;
    } else if (tmpl.find("mistral") == 0 || tmpl_contains("[INST]")) {
        if (tmpl_contains("[SYSTEM_PROMPT]")) {
            return LLM_CHAT_TEMPLATE_MISTRAL_V7;
        } else if (
            // catches official 'v1' template
            tmpl_contains("' [INST] ' + system_message")
            // catches official 'v3' and 'v3-tekken' templates
            || tmpl_contains("[AVAILABLE_TOOLS]")
        ) {
            // Official mistral 'v1', 'v3' and 'v3-tekken' templates
            // See: https://github.com/mistralai/cookbook/blob/main/concept-deep-dive/tokenization/chat_templates.md
            // See: https://github.com/mistralai/cookbook/blob/main/concept-deep-dive/tokenization/templates.md
            if (tmpl_contains(" [INST]")) {
                return LLM_CHAT_TEMPLATE_MISTRAL_V1;
            } else if (tmpl_contains("\"[INST]\"")) {
                return LLM_CHAT_TEMPLATE_MISTRAL_V3_TEKKEN;
            }
            return LLM_CHAT_TEMPLATE_MISTRAL_V3;
        } else {
            // llama2 template and its variants
            // [variant] support system message
            // See: https://huggingface.co/blog/llama2#how-to-prompt-llama-2
            bool support_system_message = tmpl_contains("<<SYS>>");
            bool add_bos_inside_history = tmpl_contains("bos_token + '[INST]");
            bool strip_message = tmpl_contains("content.strip()");
            if (strip_message) {
                return LLM_CHAT_TEMPLATE_LLAMA_2_SYS_STRIP;
            } else if (add_bos_inside_history) {
                return LLM_CHAT_TEMPLATE_LLAMA_2_SYS_BOS;
            } else if (support_system_message) {
                return LLM_CHAT_TEMPLATE_LLAMA_2_SYS;
            } else {
                return LLM_CHAT_TEMPLATE_LLAMA_2;
            }
        }
    } else if (tmpl_contains("<|assistant|>") && tmpl_contains("<|end|>")) {
        return LLM_CHAT_TEMPLATE_PHI_3;
    } else if (tmpl_contains("[gMASK]<sop>")) {
        return LLM_CHAT_TEMPLATE_CHATGLM_4;
    } else if (tmpl_contains("<|assistant|>") && tmpl_contains("<|user|>")) {
        return tmpl_contains("</s>") ? LLM_CHAT_TEMPLATE_FALCON_3 : LLM_CHAT_TEMPLATE_GLMEDGE;
    } else if (tmpl_contains("<|{{ item['role'] }}|>") && tmpl_contains("<|begin_of_image|>")) {
        return LLM_CHAT_TEMPLATE_GLMEDGE;
    } else if (tmpl_contains("<|user|>") && tmpl_contains("<|endoftext|>")) {
        return LLM_CHAT_TEMPLATE_ZEPHYR;
    } else if (tmpl_contains("bos_token + message['role']")) {
        return LLM_CHAT_TEMPLATE_MONARCH;
    } else if (tmpl_contains("<start_of_turn>")) {
        return LLM_CHAT_TEMPLATE_GEMMA;
    } else if (tmpl_contains("'\\n\\nAssistant: ' + eos_token")) {
        // OrionStarAI/Orion-14B-Chat
        return LLM_CHAT_TEMPLATE_ORION;
    } else if (tmpl_contains("GPT4 Correct ")) {
        // openchat/openchat-3.5-0106
        return LLM_CHAT_TEMPLATE_OPENCHAT;
    } else if (tmpl_contains("USER: ") && tmpl_contains("ASSISTANT: ")) {
        // eachadea/vicuna-13b-1.1 (and Orca variant)
        if (tmpl_contains("SYSTEM: ")) {
            return LLM_CHAT_TEMPLATE_VICUNA_ORCA;
        }
        return LLM_CHAT_TEMPLATE_VICUNA;
    } else if (tmpl_contains("### Instruction:") && tmpl_contains("<|EOT|>")) {
        // deepseek-ai/deepseek-coder-33b-instruct
        return LLM_CHAT_TEMPLATE_DEEPSEEK;
    } else if (tmpl_contains("<|START_OF_TURN_TOKEN|>") && tmpl_contains("<|USER_TOKEN|>")) {
        // CohereForAI/c4ai-command-r-plus
        return LLM_CHAT_TEMPLATE_COMMAND_R;
    } else if (tmpl_contains("<|start_header_id|>") && tmpl_contains("<|end_header_id|>")) {
        return LLM_CHAT_TEMPLATE_LLAMA_3;
    } else if (tmpl_contains("[gMASK]sop")) {
        // chatglm3-6b
        return LLM_CHAT_TEMPLATE_CHATGLM_3;
    } else if (tmpl_contains(LU8("<用户>"))) {
        // MiniCPM-3B-OpenHermes-2.5-v2-GGUF
        return LLM_CHAT_TEMPLATE_MINICPM;
    } else if (tmpl_contains("'Assistant: ' + message['content'] + eos_token")) {
        return LLM_CHAT_TEMPLATE_DEEPSEEK_2;
    } else if (tmpl_contains(LU8("<｜Assistant｜>")) && tmpl_contains(LU8("<｜User｜>")) && tmpl_contains(LU8("<｜end▁of▁sentence｜>"))) {
        return LLM_CHAT_TEMPLATE_DEEPSEEK_3;
    } else if (tmpl_contains("[|system|]") && tmpl_contains("[|assistant|]") && tmpl_contains("[|endofturn|]")) {
        if (tmpl_contains("[|tool|]")) {
            return LLM_CHAT_TEMPLATE_EXAONE_4;
        }
        // ref: https://huggingface.co/LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct/discussions/8#66bae61b1893d14ee8ed85bb
        // EXAONE-3.0-7.8B-Instruct
        return LLM_CHAT_TEMPLATE_EXAONE_3;
    } else if (tmpl_contains("rwkv-world") || tmpl_contains("{{- 'User: ' + message['content']|trim + '\\n\\n' -}}")) {
        return LLM_CHAT_TEMPLATE_RWKV_WORLD;
    } else if (tmpl_contains("<|start_of_role|>")) {
        return LLM_CHAT_TEMPLATE_GRANITE;
    } else if (tmpl_contains("message['role'] + additional_special_tokens[0] + message['content'] + additional_special_tokens[1]")) {
        return LLM_CHAT_TEMPLATE_GIGACHAT;
    } else if (tmpl_contains("<|role_start|>")) {
        return LLM_CHAT_TEMPLATE_MEGREZ;
    } else if (tmpl_contains(" Ассистент:")) {
        return LLM_CHAT_TEMPLATE_YANDEX;
    } else if (tmpl_contains("<role>ASSISTANT</role>") && tmpl_contains("'HUMAN'")) {
        return LLM_CHAT_TEMPLATE_BAILING;
    } else if (tmpl_contains("<|header_start|>") && tmpl_contains("<|header_end|>")) {
        return LLM_CHAT_TEMPLATE_LLAMA4;
    } else if (tmpl_contains("<|endofuserprompt|>")) {
        return LLM_CHAT_TEMPLATE_DOTS1;
    } else if (tmpl_contains("<|extra_0|>") && tmpl_contains("<|extra_4|>")) {
        return LLM_CHAT_TEMPLATE_HUNYUAN_MOE;
    } else if (tmpl_contains("<|start|>") && tmpl_contains("<|channel|>")) {
        return LLM_CHAT_TEMPLATE_OPENAI_MOE;
    } else if (tmpl_contains("<｜hy_Assistant｜>") && tmpl_contains("<｜hy_place▁holder▁no▁3｜>")) {
        return LLM_CHAT_TEMPLATE_HUNYUAN_DENSE;
    } else if (tmpl_contains("<|im_assistant|>assistant<|im_middle|>")) {
        return LLM_CHAT_TEMPLATE_KIMI_K2;
    } else if (tmpl_contains("<seed:bos>")) {
        return LLM_CHAT_TEMPLATE_SEED_OSS;
    }
    return LLM_CHAT_TEMPLATE_UNKNOWN;
}

// Simple version of "llama_apply_chat_template" that only works with strings
// This function uses heuristic checks to determine commonly used template. It is not a jinja parser.
int32_t llm_chat_apply_template(
    llm_chat_template tmpl,
    const std::vector<const llama_chat_message *> & chat,
    std::string & dest, bool add_ass) {
    // Taken from the research: https://github.com/ggerganov/llama.cpp/issues/5527
    std::stringstream ss;
    if (tmpl == LLM_CHAT_TEMPLATE_CHATML) {
        // chatml template
        for (auto message : chat) {
            ss << "<|im_start|>" << message->role << "\n" << message->content << "<|im_end|>\n";
        }
        if (add_ass) {
            ss << "<|im_start|>assistant\n";
        }
    } else if (tmpl == LLM_CHAT_TEMPLATE_MISTRAL_V7 || tmpl == LLM_CHAT_TEMPLATE_MISTRAL_V7_TEKKEN) {
        // Official mistral 'v7' template
        // See: https://huggingface.co/mistralai/Mistral-Large-Instruct-2411#basic-instruct-template-v7
        //      https://huggingface.co/mistralai/Mistral-Small-3.1-24B-Instruct-2503#basic-instruct-template-v7-tekken
        const char * trailing_space = tmpl == LLM_CHAT_TEMPLATE_MISTRAL_V7 ? " " : "";
        for (auto message : chat) {
            std::string role(message->role);
            std::string content(message->content);
            if (role == "system") {
                ss << "[SYSTEM_PROMPT]" << trailing_space << content << "[/SYSTEM_PROMPT]";
            } else if (role == "user") {
                ss << "[INST]" << trailing_space << content << "[/INST]";
            } else {
                ss << trailing_space << content << "</s>";
            }
        }
    } else if (tmpl == LLM_CHAT_TEMPLATE_MISTRAL_V1
            || tmpl == LLM_CHAT_TEMPLATE_MISTRAL_V3
            || tmpl == LLM_CHAT_TEMPLATE_MISTRAL_V3_TEKKEN) {
        // See: https://github.com/mistralai/cookbook/blob/main/concept-deep-dive/tokenization/chat_templates.md
        // See: https://github.com/mistralai/cookbook/blob/main/concept-deep-dive/tokenization/templates.md
        std::string leading_space = tmpl == LLM_CHAT_TEMPLATE_MISTRAL_V1 ? " " : "";
        std::string trailing_space = tmpl == LLM_CHAT_TEMPLATE_MISTRAL_V3_TEKKEN ? "" : " ";
        bool trim_assistant_message = tmpl == LLM_CHAT_TEMPLATE_MISTRAL_V3;
        bool is_inside_turn = false;
        for (auto message : chat) {
            if (!is_inside_turn) {
                ss << leading_space << "[INST]" << trailing_space;
                is_inside_turn = true;
            }
            std::string role(message->role);
            std::string content(message->content);
            if (role == "system") {
                ss << content << "\n\n";
            } else if (role == "user") {
                ss << content << leading_space << "[/INST]";
            } else {
                ss << trailing_space << (trim_assistant_message ? trim(content) : content) << "</s>";
                is_inside_turn = false;
            }
        }
    } else if (
            tmpl == LLM_CHAT_TEMPLATE_LLAMA_2
            || tmpl == LLM_CHAT_TEMPLATE_LLAMA_2_SYS
            || tmpl == LLM_CHAT_TEMPLATE_LLAMA_2_SYS_BOS
            || tmpl == LLM_CHAT_TEMPLATE_LLAMA_2_SYS_STRIP) {
        // llama2 template and its variants
        // [variant] support system message
        // See: https://huggingface.co/blog/llama2#how-to-prompt-llama-2
        bool support_system_message = tmpl != LLM_CHAT_TEMPLATE_LLAMA_2;
        // [variant] add BOS inside history
        bool add_bos_inside_history = tmpl == LLM_CHAT_TEMPLATE_LLAMA_2_SYS_BOS;
        // [variant] trim spaces from the input message
        bool strip_message = tmpl == LLM_CHAT_TEMPLATE_LLAMA_2_SYS_STRIP;
        // construct the prompt
        bool is_inside_turn = true; // skip BOS at the beginning
        ss << "[INST] ";
        for (auto message : chat) {
            std::string content = strip_message ? trim(message->content) : message->content;
            std::string role(message->role);
            if (!is_inside_turn) {
                is_inside_turn = true;
                ss << (add_bos_inside_history ? "<s>[INST] " : "[INST] ");
            }
            if (role == "system") {
                if (support_system_message) {
                    ss << "<<SYS>>\n" << content << "\n<</SYS>>\n\n";
                } else {
                    // if the model does not support system message, we still include it in the first message, but without <<SYS>>
                    ss << content << "\n";
                }
            } else if (role == "user") {
                ss << content << " [/INST]";
            } else {
                ss << content << "</s>";
                is_inside_turn = false;
            }
        }
    } else if (tmpl == LLM_CHAT_TEMPLATE_PHI_3) {
        // Phi 3
        for (auto message : chat) {
            std::string role(message->role);
            ss << "<|" << role << "|>\n" << message->content << "<|end|>\n";
        }
        if (add_ass) {
            ss << "<|assistant|>\n";
        }
    } else if (tmpl == LLM_CHAT_TEMPLATE_PHI_4) {
        // chatml template
        for (auto message : chat) {
            ss << "<|im_start|>" << message->role << "<|im_sep|>" << message->content << "<|im_end|>";
        }
        if (add_ass) {
            ss << "<|im_start|>assistant<|im_sep|>";
        }
    } else if (tmpl == LLM_CHAT_TEMPLATE_FALCON_3) {
        // Falcon 3
        for (auto message : chat) {
            std::string role(message->role);
            ss << "<|" << role << "|>\n" << message->content << "\n";
        }
        if (add_ass) {
            ss << "<|assistant|>\n";
        }
    } else if (tmpl == LLM_CHAT_TEMPLATE_ZEPHYR) {
        // zephyr template
        for (auto message : chat) {
            ss << "<|" << message->role << "|>" << "\n" << message->content << "<|endoftext|>\n";
        }
        if (add_ass) {
            ss << "<|assistant|>\n";
        }
    } else if (tmpl == LLM_CHAT_TEMPLATE_MONARCH) {
        // mlabonne/AlphaMonarch-7B template (the <s> is included inside history)
        for (auto message : chat) {
            std::string bos = (message == chat.front()) ? "" : "<s>"; // skip BOS for first message
            ss << bos << message->role << "\n" << message->content << "</s>\n";
        }
        if (add_ass) {
            ss << "<s>assistant\n";
        }
    } else if (tmpl == LLM_CHAT_TEMPLATE_GEMMA) {
        // google/gemma-7b-it
        std::string system_prompt = "";
        for (auto message : chat) {
            std::string role(message->role);
            if (role == "system") {
                // there is no system message for gemma, but we will merge it with user prompt, so nothing is broken
                system_prompt += trim(message->content);
                continue;
            }
            // in gemma, "assistant" is "model"
            role = role == "assistant" ? "model" : message->role;
            ss << "<start_of_turn>" << role << "\n";
            if (!system_prompt.empty() && role != "model") {
                ss << system_prompt << "\n\n";
                system_prompt = "";
            }
            ss << trim(message->content) << "<end_of_turn>\n";
        }
        if (add_ass) {
            ss << "<start_of_turn>model\n";
        }
    } else if (tmpl == LLM_CHAT_TEMPLATE_ORION) {
        // OrionStarAI/Orion-14B-Chat
        std::string system_prompt = "";
        for (auto message : chat) {
            std::string role(message->role);
            if (role == "system") {
                // there is no system message support, we will merge it with user prompt
                system_prompt += message->content;
                continue;
            } else if (role == "user") {
                ss << "Human: ";
                if (!system_prompt.empty()) {
                    ss << system_prompt << "\n\n";
                    system_prompt = "";
                }
                ss << message->content << "\n\nAssistant: </s>";
            } else {
                ss << message->content << "</s>";
            }
        }
    } else if (tmpl == LLM_CHAT_TEMPLATE_OPENCHAT) {
        // openchat/openchat-3.5-0106,
        for (auto message : chat) {
            std::string role(message->role);
            if (role == "system") {
                ss << message->content << "<|end_of_turn|>";
            } else {
                role[0] = toupper(role[0]);
                ss << "GPT4 Correct " << role << ": " << message->content << "<|end_of_turn|>";
            }
        }
        if (add_ass) {
            ss << "GPT4 Correct Assistant:";
        }
    } else if (tmpl == LLM_CHAT_TEMPLATE_VICUNA || tmpl == LLM_CHAT_TEMPLATE_VICUNA_ORCA) {
        // eachadea/vicuna-13b-1.1 (and Orca variant)
        for (auto message : chat) {
            std::string role(message->role);
            if (role == "system") {
                // Orca-Vicuna variant uses a system prefix
                if (tmpl == LLM_CHAT_TEMPLATE_VICUNA_ORCA) {
                    ss << "SYSTEM: " << message->content << "\n";
                } else {
                    ss << message->content << "\n\n";
                }
            } else if (role == "user") {
                ss << "USER: " << message->content << "\n";
            } else if (role == "assistant") {
                ss << "ASSISTANT: " << message->content << "</s>\n";
            }
        }
        if (add_ass) {
            ss << "ASSISTANT:";
        }
    } else if (tmpl == LLM_CHAT_TEMPLATE_DEEPSEEK) {
        // deepseek-ai/deepseek-coder-33b-instruct
        for (auto message : chat) {
            std::string role(message->role);
            if (role == "system") {
                ss << message->content;
            } else if (role == "user") {
                ss << "### Instruction:\n" << message->content << "\n";
            } else if (role == "assistant") {
                ss << "### Response:\n" << message->content << "\n<|EOT|>\n";
            }
        }
        if (add_ass) {
            ss << "### Response:\n";
        }
    } else if (tmpl == LLM_CHAT_TEMPLATE_COMMAND_R) {
        // CohereForAI/c4ai-command-r-plus
        for (auto message : chat) {
            std::string role(message->role);
            if (role == "system") {
                ss << "<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>" << trim(message->content) << "<|END_OF_TURN_TOKEN|>";
            } else if (role == "user") {
                ss << "<|START_OF_TURN_TOKEN|><|USER_TOKEN|>" << trim(message->content) << "<|END_OF_TURN_TOKEN|>";
            } else if (role == "assistant") {
                ss << "<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>" << trim(message->content) << "<|END_OF_TURN_TOKEN|>";
            }
        }
        if (add_ass) {
            ss << "<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>";
        }
    } else if (tmpl == LLM_CHAT_TEMPLATE_LLAMA_3) {
        // Llama 3
        for (auto message : chat) {
            std::string role(message->role);
            ss << "<|start_header_id|>" << role << "<|end_header_id|>\n\n" << trim(message->content) << "<|eot_id|>";
        }
        if (add_ass) {
            ss << "<|start_header_id|>assistant<|end_header_id|>\n\n";
        }
    } else if (tmpl == LLM_CHAT_TEMPLATE_CHATGLM_3) {
        // chatglm3-6b
        ss << "[gMASK]" << "sop";
        for (auto message : chat) {
            std::string role(message->role);
            ss << "<|" << role << "|>" << "\n " << message->content;
        }
        if (add_ass) {
            ss << "<|assistant|>";
        }
    } else if (tmpl == LLM_CHAT_TEMPLATE_CHATGLM_4) {
        ss << "[gMASK]" << "<sop>";
        for (auto message : chat) {
            std::string role(message->role);
            ss << "<|" << role << "|>" << "\n" << message->content;
        }
        if (add_ass) {
            ss << "<|assistant|>\n";
        }
    } else if (tmpl == LLM_CHAT_TEMPLATE_GLMEDGE) {
        for (auto message : chat) {
            std::string role(message->role);
            ss << "<|" << role << "|>" << "\n" << message->content;
        }
        if (add_ass) {
            ss << "<|assistant|>";
        }
    } else if (tmpl == LLM_CHAT_TEMPLATE_MINICPM) {
        // MiniCPM-3B-OpenHermes-2.5-v2-GGUF
        for (auto message : chat) {
            std::string role(message->role);
            if (role == "user") {
                ss << LU8("<用户>");
                ss << trim(message->content);
                ss << "<AI>";
            } else {
                ss << trim(message->content);
            }
        }
    } else if (tmpl == LLM_CHAT_TEMPLATE_DEEPSEEK_2) {
        // DeepSeek-V2
        for (auto message : chat) {
            std::string role(message->role);
            if (role == "system") {
                ss << message->content << "\n\n";
            } else if (role == "user") {
                ss << "User: " << message->content << "\n\n";
            } else if (role == "assistant") {
                ss << "Assistant: " << message->content << LU8("<｜end▁of▁sentence｜>");
            }
        }
        if (add_ass) {
            ss << "Assistant:";
        }
    } else if (tmpl == LLM_CHAT_TEMPLATE_DEEPSEEK_3) {
        // DeepSeek-V3
        for (auto message : chat) {
            std::string role(message->role);
            if (role == "system") {
                ss << message->content << "\n\n";
            } else if (role == "user") {
                ss << LU8("<｜User｜>") << message->content;
            } else if (role == "assistant") {
                ss << LU8("<｜Assistant｜>") << message->content << LU8("<｜end▁of▁sentence｜>");
            }
        }
        if (add_ass) {
            ss << LU8("<｜Assistant｜>");
        }
    } else if (tmpl == LLM_CHAT_TEMPLATE_EXAONE_3) {
        // ref: https://huggingface.co/LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct/discussions/8#66bae61b1893d14ee8ed85bb
        // EXAONE-3.0-7.8B-Instruct
        for (auto message : chat) {
            std::string role(message->role);
            if (role == "system") {
                ss << "[|system|]" << trim(message->content) << "[|endofturn|]\n";
            } else if (role == "user") {
                ss << "[|user|]" << trim(message->content) << "\n";
            } else if (role == "assistant") {
                ss << "[|assistant|]" << trim(message->content) << "[|endofturn|]\n";
            }
        }
        if (add_ass) {
            ss << "[|assistant|]";
        }
    } else if (tmpl == LLM_CHAT_TEMPLATE_EXAONE_4) {
        for (auto message : chat) {
            std::string role(message->role);
            if (role == "system") {
                ss << "[|system|]" << trim(message->content) << "[|endofturn|]\n";
            } else if (role == "user") {
                ss << "[|user|]" << trim(message->content) << "\n";
            } else if (role == "assistant") {
                ss << "[|assistant|]" << trim(message->content) << "[|endofturn|]\n";
            } else if (role == "tool") {
                ss << "[|tool|]" << trim(message->content) << "[|endofturn|]\n";
            }
        }
        if (add_ass) {
            ss << "[|assistant|]";
        }
    } else if (tmpl == LLM_CHAT_TEMPLATE_RWKV_WORLD) {
        // this template requires the model to have "\n\n" as EOT token
        for (size_t i = 0; i < chat.size(); i++) {
            std::string role(chat[i]->role);
            if (role == "system") {
                ss << "System: " << trim(chat[i]->content) << "\n\n";
            } else if (role == "user") {
                ss << "User: " << trim(chat[i]->content) << "\n\n";
                if (i == chat.size() - 1) {
                    ss << "Assistant:";
                }
            } else if (role == "assistant") {
                ss << "Assistant: " << trim(chat[i]->content) << "\n\n";
            }
        }
    } else if (tmpl == LLM_CHAT_TEMPLATE_GRANITE) {
        // IBM Granite template
        for (const auto & message : chat) {
            std::string role(message->role);
            ss << "<|start_of_role|>" << role << "<|end_of_role|>";
            if (role == "assistant_tool_call") {
                ss << "<|tool_call|>";
            }
            ss << message->content << "<|end_of_text|>\n";
        }
        if (add_ass) {
            ss << "<|start_of_role|>assistant<|end_of_role|>\n";
        }
    } else if (tmpl == LLM_CHAT_TEMPLATE_GIGACHAT) {
        // GigaChat template
        bool has_system = !chat.empty() && std::string(chat[0]->role) == "system";

        // Handle system message if present
        if (has_system) {
            ss << "<s>" << chat[0]->content << "<|message_sep|>";
        } else {
            ss << "<s>";
        }

        // Process remaining messages
        for (size_t i = has_system ? 1 : 0; i < chat.size(); i++) {
            std::string role(chat[i]->role);
            if (role == "user") {
                ss << "user<|role_sep|>" << chat[i]->content << "<|message_sep|>"
                << "available functions<|role_sep|>[]<|message_sep|>";
            } else if (role == "assistant") {
                ss << "assistant<|role_sep|>" << chat[i]->content << "<|message_sep|>";
            }
        }

        // Add generation prompt if needed
        if (add_ass) {
            ss << "assistant<|role_sep|>";
        }
    }  else if (tmpl == LLM_CHAT_TEMPLATE_MEGREZ) {
        // Megrez template
        for (auto message : chat) {
            std::string role(message->role);
            ss << "<|role_start|>" << role << "<|role_end|>" << message->content << "<|turn_end|>";
        }

        if (add_ass) {
            ss << "<|role_start|>assistant<|role_end|>";
        }
    } else if (tmpl == LLM_CHAT_TEMPLATE_YANDEX) {
        // Yandex template ("\n\n" is defined as EOT token)

        for (size_t i = 0; i < chat.size(); i++) {
            std::string role(chat[i]->role);
            if (role == "user") {
                ss << " Пользователь: " << chat[i]->content << "\n\n";
            } else if (role == "assistant") {
                ss << " Ассистент: " << chat[i]->content << "\n\n";
            }
        }

        // Add generation prompt if needed
        if (add_ass) {
            ss << " Ассистент:[SEP]";
        }
    }  else if (tmpl == LLM_CHAT_TEMPLATE_BAILING) {
        // Bailing (Ling) template
        for (auto message : chat) {
            std::string role(message->role);

            if (role == "user") {
                role = "HUMAN";
            } else {
                std::transform(role.begin(), role.end(), role.begin(), ::toupper);
            }

            ss << "<role>" << role << "</role>" << message->content;
        }

        if (add_ass) {
            ss << "<role>ASSISTANT</role>";
        }
    } else if (tmpl == LLM_CHAT_TEMPLATE_LLAMA4) {
        // Llama 4
        for (auto message : chat) {
            std::string role(message->role);
            ss << "<|header_start|>" << role << "<|header_end|>\n\n" << trim(message->content) << "<|eot|>";
        }
        if (add_ass) {
            ss << "<|header_start|>assistant<|header_end|>\n\n";
        }
    } else if (tmpl == LLM_CHAT_TEMPLATE_SMOLVLM) {
        // SmolVLM
        ss << "<|im_start|>"; // uses <|im_start|> as BOS, but the actual content is NOT chatml
        for (auto message : chat) {
            std::string role(message->role);
            if (role == "system") {
                ss << message->content << "\n\n";
            } else if (role == "user") {
                ss << "User: " << message->content << "<end_of_utterance>\n";
            } else {
                ss << "Assistant: " << message->content << "<end_of_utterance>\n";
            }
        }
        if (add_ass) {
            ss << "Assistant:";
        }
    } else if (tmpl == LLM_CHAT_TEMPLATE_DOTS1) {
        // dots.llm1.inst (DOTS1)
        for (auto message : chat) {
            std::string role(message->role);
            if (role == "system") {
                ss << "<|system|>" << message->content << "<|endofsystem|>";
            } else if (role == "user") {
                ss << "<|userprompt|>" << message->content << "<|endofuserprompt|>";
            } else {
                ss << "<|response|>" << message->content << "<|endofresponse|>";
            }
        }
        if (add_ass) {
            ss << "<|response|>";
        }
    } else if (tmpl == LLM_CHAT_TEMPLATE_HUNYUAN_MOE) {
        // tencent/Hunyuan-A13B-Instruct
        for (auto message : chat) {
            std::string role(message->role);
            if (role == "system") {
                ss << "<|startoftext|>" << message->content << "<|extra_4|>";
            } else if (role == "assistant") {
                ss << message->content << "<|eos|>";
            } else {
                ss << "<|startoftext|>" << message->content << "<|extra_0|>";
            }
        }
    } else if (tmpl == LLM_CHAT_TEMPLATE_OPENAI_MOE) {
        // OpenAI MoE (based on Harmony chat template)
        for (auto message : chat) {
            std::string role(message->role);
            ss << "<|start|>" << role << "<|message|>" << message->content;
            ss << (role == "assistant" ? "<|return|>" : "<|end|>");
        }
        if (add_ass) {
            ss << "<|start|>assistant";
        }
    } else if (tmpl == LLM_CHAT_TEMPLATE_HUNYUAN_DENSE) {
        // tencent/Hunyuan-4B-Instruct
        for (size_t i = 0; i < chat.size(); i++) {
            std::string role(chat[i]->role);
            if (i == 0) {
                if (role == "system") {
                    ss << chat[i]->content << "<｜hy_place▁holder▁no▁3｜>";
                }
            }

            if (role == "assistant") {
                ss << "<｜hy_Assistant｜>" << chat[i]->content << "<｜hy_place▁holder▁no▁2｜>";
            } else if (role == "user") {
                ss << "<｜hy_User｜>" << chat[i]->content << "<｜hy_Assistant｜>";
            }
        }
    } else if (tmpl == LLM_CHAT_TEMPLATE_KIMI_K2) {
        // moonshotai/Kimi-K2-Instruct
        for (auto message : chat) {
            std::string role(message->role);
            if (role == "system") {
                ss << "<|im_system|>system<|im_middle|>";
            } else if (role == "user") {
                ss << "<|im_user|>user<|im_middle|>";
            } else if (role == "assistant") {
                ss << "<|im_assistant|>assistant<|im_middle|>";
            } else if (role == "tool") {
                ss << "<|im_system|>tool<|im_middle|>";
            }

            ss << message->content << "<|im_end|>";
        }
        if (add_ass) {
            ss << "<|im_assistant|>assistant<|im_middle|>";
        }
    } else if (tmpl == LLM_CHAT_TEMPLATE_SEED_OSS) {
        for (auto message: chat) {
            std::string role(message->role);
            ss << "<seed:bos>" << role << "\n" << (role == "assistant" ? trim(message->content) : message->content) << "<seed:eos>";
        }
        if (add_ass) {
            ss << "<seed:bos>assistant\n";
        }
    } else {
        // template not supported
        return -1;
    }
    dest = ss.str();
    return dest.size();
}

// public interface

int32_t llama_chat_builtin_templates(const char ** output, size_t len) {
    auto it = LLM_CHAT_TEMPLATES.begin();
    for (size_t i = 0; i < std::min(len, LLM_CHAT_TEMPLATES.size()); i++) {
        output[i] = it->first.c_str();
        std::advance(it, 1);
    }
    return (int32_t) LLM_CHAT_TEMPLATES.size();
}
