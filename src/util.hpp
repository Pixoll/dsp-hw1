#pragma once

#include <random>

#define MIN(a, b) ((a) < (b) ? a : b)
#define MAX(a, b) ((a) > (b) ? a : b)

template<typename IntType>
class int_generator {
    std::mt19937_64 m_rng{std::random_device{}()};
    std::uniform_int_distribution<IntType> m_uniform_distribution;

public:
    int_generator(int64_t min, int64_t max) : m_uniform_distribution(min, max) {
    }

    IntType operator()() {
        return m_uniform_distribution(m_rng);
    }
};
