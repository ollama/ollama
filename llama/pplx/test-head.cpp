#include "pplx.h"
#include "json.h"
#include <iostream>

int main(int argc, char ** argv) {
    if (argc != 2) return 2;
    try {
        const pplx_head head(argv[1]);
        std::string line;
        std::getline(std::cin, line);
        const auto input = common_json::parse(line);
        std::cout << common_json(head.score(input.at("hidden").get<std::vector<float>>(), input.at("options").get<int>())).dump();
    } catch (const std::exception & e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
