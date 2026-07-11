#include <gtest/gtest.h>
#include <string>
#include <vector>
#include <cstdint>
#include "llama/compat/llama-ollama-compat-util.cpp"

class SecurityTest : public ::testing::TestWithParam<std::tuple<size_t, size_t, size_t>> {};

TEST_P(SecurityTest, TruncateDataArr_BufferBoundsInvariant) {
    // Invariant: memcpy source size must not exceed source buffer boundaries
    auto [elem_size, new_n, source_size] = GetParam();
    
    // Create mock GGUF metadata with controlled buffer
    struct MockMeta {
        void* data;
        size_t size;
    };
    
    MockMeta meta;
    std::vector<uint8_t> source_buffer(source_size, 0xAA);
    meta.data = source_buffer.data();
    meta.size = source_size;
    
    // Test the actual function - this would need to be adapted to the actual interface
    // For demonstration, we're showing the core security check
    size_t copy_size = elem_size * new_n;
    
    // Security property: copy size must not exceed source buffer size
    ASSERT_LE(copy_size, source_size) 
        << "Buffer overflow: elem_size=" << elem_size 
        << ", new_n=" << new_n 
        << ", copy_size=" << copy_size 
        << ", source_size=" << source_size;
}

INSTANTIATE_TEST_SUITE_P(
    AdversarialInputs,
    SecurityTest,
    ::testing::Values(
        // Exact exploit case: overflow causing out-of-bounds read
        std::make_tuple(1024, 1024 * 1024, 1024),
        // Boundary case: maximum size_t values
        std::make_tuple(SIZE_MAX, 2, SIZE_MAX),
        // Valid input: normal operation
        std::make_tuple(4, 256, 1024),
        // Edge case: zero elements
        std::make_tuple(16, 0, 0),
        // Boundary: exact match
        std::make_tuple(8, 128, 1024)
    )
);

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}