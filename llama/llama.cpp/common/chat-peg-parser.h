#pragma once

#include "chat.h"
#include "peg-parser.h"

class common_chat_peg_builder : public common_peg_parser_builder {
  public:
    static constexpr const char * REASONING_BLOCK = "reasoning-block";
    static constexpr const char * REASONING = "reasoning";
    static constexpr const char * CONTENT = "content";

    common_peg_parser reasoning_block(const common_peg_parser & p) { return tag(REASONING_BLOCK, p); }
    common_peg_parser reasoning(const common_peg_parser & p) { return tag(REASONING, p); }
    common_peg_parser content(const common_peg_parser & p) { return tag(CONTENT, p); }
};

inline common_peg_arena build_chat_peg_parser(const std::function<common_peg_parser(common_chat_peg_builder & builder)> & fn) {
    common_chat_peg_builder builder;
    builder.set_root(fn(builder));
    return builder.build();
}

class common_chat_peg_mapper {
  public:
    common_chat_msg & result;

    common_chat_peg_mapper(common_chat_msg & msg) : result(msg) {}

    virtual void from_ast(const common_peg_ast_arena & arena, const common_peg_parse_result & result);
    virtual void map(const common_peg_ast_node & node);
};

class common_chat_peg_native_builder : public common_chat_peg_builder {
  public:
    static constexpr const char * TOOL = "tool";
    static constexpr const char * TOOL_OPEN = "tool-open";
    static constexpr const char * TOOL_CLOSE = "tool-close";
    static constexpr const char * TOOL_ID = "tool-id";
    static constexpr const char * TOOL_NAME = "tool-name";
    static constexpr const char * TOOL_ARGS = "tool-args";

    common_peg_parser tool(const common_peg_parser & p) { return tag(TOOL, p); }
    common_peg_parser tool_open(const common_peg_parser & p) { return atomic(tag(TOOL_OPEN, p)); }
    common_peg_parser tool_close(const common_peg_parser & p) { return atomic(tag(TOOL_CLOSE, p)); }
    common_peg_parser tool_id(const common_peg_parser & p) { return atomic(tag(TOOL_ID, p)); }
    common_peg_parser tool_name(const common_peg_parser & p) { return atomic(tag(TOOL_NAME, p)); }
    common_peg_parser tool_args(const common_peg_parser & p) { return tag(TOOL_ARGS, p); }
};

class common_chat_peg_native_mapper : public common_chat_peg_mapper {
    common_chat_tool_call * current_tool;

  public:
    common_chat_peg_native_mapper(common_chat_msg & msg) : common_chat_peg_mapper(msg) {}

    void map(const common_peg_ast_node & node) override;
};

inline common_peg_arena build_chat_peg_native_parser(const std::function<common_peg_parser(common_chat_peg_native_builder & builder)> & fn) {
    common_chat_peg_native_builder builder;
    builder.set_root(fn(builder));
    return builder.build();
}

class common_chat_peg_constructed_builder : public common_chat_peg_builder {
  public:
    static constexpr const char * TOOL = "tool";
    static constexpr const char * TOOL_OPEN = "tool-open";
    static constexpr const char * TOOL_CLOSE = "tool-close";
    static constexpr const char * TOOL_NAME = "tool-name";
    static constexpr const char * TOOL_ARG = "tool-arg";
    static constexpr const char * TOOL_ARG_OPEN = "tool-arg-open";
    static constexpr const char * TOOL_ARG_CLOSE = "tool-arg-close";
    static constexpr const char * TOOL_ARG_NAME = "tool-arg-name";
    static constexpr const char * TOOL_ARG_STRING_VALUE = "tool-arg-string-value";
    static constexpr const char * TOOL_ARG_JSON_VALUE = "tool-arg-json-value";

    common_peg_parser tool(const common_peg_parser & p) { return tag(TOOL, p); }
    common_peg_parser tool_open(const common_peg_parser & p) { return atomic(tag(TOOL_OPEN, p)); }
    common_peg_parser tool_close(const common_peg_parser & p) { return atomic(tag(TOOL_CLOSE, p)); }
    common_peg_parser tool_name(const common_peg_parser & p) { return atomic(tag(TOOL_NAME, p)); }
    common_peg_parser tool_arg(const common_peg_parser & p) { return tag(TOOL_ARG, p); }
    common_peg_parser tool_arg_open(const common_peg_parser & p) { return atomic(tag(TOOL_ARG_OPEN, p)); }
    common_peg_parser tool_arg_close(const common_peg_parser & p) { return atomic(tag(TOOL_ARG_CLOSE, p)); }
    common_peg_parser tool_arg_name(const common_peg_parser & p) { return atomic(tag(TOOL_ARG_NAME, p)); }
    common_peg_parser tool_arg_string_value(const common_peg_parser & p) { return tag(TOOL_ARG_STRING_VALUE, p); }
    common_peg_parser tool_arg_json_value(const common_peg_parser & p) { return tag(TOOL_ARG_JSON_VALUE, p); }
};

class common_chat_peg_constructed_mapper : public common_chat_peg_mapper {
    common_chat_tool_call * current_tool;
    int arg_count = 0;
    bool needs_closing_quote = false;

  public:
    common_chat_peg_constructed_mapper(common_chat_msg & msg) : common_chat_peg_mapper(msg) {}

    void map(const common_peg_ast_node & node) override;
};

inline common_peg_arena build_chat_peg_constructed_parser(const std::function<common_peg_parser(common_chat_peg_constructed_builder & builder)> & fn) {
    common_chat_peg_constructed_builder builder;
    builder.set_root(fn(builder));
    return builder.build();
}
