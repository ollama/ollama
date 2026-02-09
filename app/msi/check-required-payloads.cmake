if(NOT DEFINED REQUIRED_PAYLOADS)
    message(FATAL_ERROR "REQUIRED_PAYLOADS was not provided")
endif()

string(REPLACE "|" ";" _required_payloads "${REQUIRED_PAYLOADS}")
set(_missing_payloads)

foreach(_payload IN LISTS _required_payloads)
    if(NOT EXISTS "${_payload}")
        list(APPEND _missing_payloads "${_payload}")
    endif()
endforeach()

if(_missing_payloads)
    string(REPLACE ";" "\n  " _missing_payloads_text "${_missing_payloads}")
    message(FATAL_ERROR
        "Missing required MSI payload(s):\n"
        "  ${_missing_payloads_text}\n"
        "Build the runtime payloads before building MSI packages.")
endif()
