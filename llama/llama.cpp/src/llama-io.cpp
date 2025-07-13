#include "llama-io.h"

void llama_io_write_i::write_string(const std::string & str) {
    uint32_t str_size = str.size();

    write(&str_size,  sizeof(str_size));
    write(str.data(), str_size);
}

void llama_io_read_i::read_string(std::string & str) {
    uint32_t str_size;
    read_to(&str_size, sizeof(str_size));

    str.assign((const char *) read(str_size), str_size);
}
