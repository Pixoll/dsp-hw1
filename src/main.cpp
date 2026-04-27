#include <algorithm>
#include <array>
#include <functional>
#include <iostream>
#include <iterator>
#include <omp.h>
#include <random>
#include <vector>

#include "util.hpp"
#include "parallel/k_way_mergesort.hpp"
#include "parallel/mergesort.hpp"
#include "parallel/ranks_k_way_mergesort.hpp"
#include "parallel/ranks_mergesort.hpp"
#include "sequential/k_way_mergesort.hpp"
#include "sequential/mergesort.hpp"

struct Result {
    const double time;
    const bool correct;
};

template<typename T>
std::ostream &operator<<(std::ostream &out, const std::vector<T> &v) {
    if (v.empty()) {
        return out;
    }

    out << '[';
    for (int i = 0; i < v.size(); ++i) {
        out << v[i] << ", ";
    }
    return out << "\b\b]";
}

std::ostream &operator<<(std::ostream &out, const Result &r) {
    return out << r.correct << " (" << r.time << " s)";
}

Result benchmark(
    const std::vector<int> &array,
    const std::vector<int> &sorted,
    const std::function<void(std::vector<int> &)> &fn
) {
    std::vector copy(array);

    const double start_time = omp_get_wtime();
    fn(copy);
    const double end_time = omp_get_wtime();

    return {
        .time = end_time - start_time,
        .correct = copy == sorted,
    };
}

static constexpr int MIN_THREADS = 2;
static constexpr int K = 8;

int main() {
    constexpr std::array sizes{20, 22, 24, 26};
    const int max_threads = omp_get_max_threads();

    if (max_threads < MIN_THREADS) {
        std::cerr << "You need at least 2 threads for parallelism" << std::endl;
        return 1;
    }

    int threads = MIN_THREADS;
    std::cout << std::boolalpha;

    while (true) {
        omp_set_num_threads(threads);

        for (const int &size: sizes) {
            std::cout <<
                "==================================================\n"
                "\n"
                "threads    " << threads << "\n"
                "size       2^" << size << "\n"
                << std::endl;

            std::vector<int> array(1 << size);
            int_generator<int> generate_int(std::numeric_limits<int>::min(), std::numeric_limits<int>::max());
            for (int &n: array) {
                n = generate_int();
            }

            std::vector sorted(array);
            std::ranges::sort(sorted);

            std::cout << "sorting control" << std::endl;
            const Result &control_result = benchmark(
                array,
                sorted,
                [](std::vector<int> &a) {
                    std::ranges::sort(a);
                }
            );

            std::cout << "sorting regular sequential" << std::endl;
            const Result &regular_sequential_result = benchmark(
                array,
                sorted,
                [](std::vector<int> &a) {
                    sequential_mergesort(a);
                }
            );

            std::cout << "sorting regular parallel" << std::endl;
            const Result &regular_parallel_result = benchmark(
                array,
                sorted,
                [](std::vector<int> &a) {
                    parallel_mergesort(a);
                }
            );

            std::cout << "sorting k-way sequential" << std::endl;
            const Result &k_way_sequential_result = benchmark(
                array,
                sorted,
                [](std::vector<int> &a) {
                    sequential_k_way_mergesort(a, K);
                }
            );

            std::cout << "sorting k-way parallel" << std::endl;
            const Result &k_way_parallel_result = benchmark(
                array,
                sorted,
                [](std::vector<int> &a) {
                    parallel_k_way_mergesort(a, K);
                }
                );

            std::cout << "sorting ranks parallel" << std::endl;
            const Result &ranks_parallel_result = benchmark(
                array,
                sorted,
                [](std::vector<int> &a) {
                    parallel_ranks_mergesort(a);
                }
                );

            std::cout << "sorting ranks k-way parallel" << std::endl;
            const Result &ranks_k_way_parallel_result = benchmark(
                array,
                sorted,
                [](std::vector<int> &a) {
                    parallel_ranks_k_way_mergesort(a, K);
                }
            );

            std::cout << "\n"
                "control               " << control_result << "\n"
                "regular sequential    " << regular_sequential_result << "\n"
                "regular parallel      " << regular_parallel_result << "\n"
                "k-way sequential      " << k_way_sequential_result << "\n"
                "k-way parallel        " << k_way_parallel_result << "\n"
                "ranks parallel        " << ranks_parallel_result << "\n"
                "ranks k-way parallel  " << ranks_k_way_parallel_result << "\n"
                << std::endl;
        }

        if (threads >= max_threads) {
            break;
        }

        threads = std::min(threads * 2, max_threads);
    }

    return 0;
}
