#pragma once

#include "common.h"
#include "arg.h"

#include <string>
#include <vector>
#include <map>

//
// INI preset parser and writer
//

constexpr const char * COMMON_PRESET_DEFAULT_NAME = "default";

struct common_preset {
    std::string name;
    // TODO: support repeated args in the future
    std::map<common_arg, std::string> options;

    // convert preset to CLI argument list
    std::vector<std::string> to_args() const;

    // convert preset to INI format string
    std::string to_ini() const;

    // TODO: maybe implement to_env() if needed
};

// interface for multiple presets in one file
using common_presets = std::map<std::string, common_preset>;
common_presets common_presets_load(const std::string & path, common_params_context & ctx_params);
