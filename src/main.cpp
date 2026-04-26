#include <algorithm>
#include <functional>
#include <iostream>
#include <iterator>
#include <omp.h>
#include <random>
#include <vector>

#include "util.hpp"
#include "parallel_merge_sort/k_way_mergesort.hpp"
#include "parallel_merge_sort/mergesort.hpp"
#include "parallel_merge_sort/parallel_mergesort_ranks.hpp"
#include "sequential_merge_sort/k_way_mergesort.hpp"
#include "sequential_merge_sort/mergesort.hpp"

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

int main() {
    std::vector<int> array(1 << 25);
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

    std::cout << "sorting sequential_mergesort" << std::endl;
    const Result &sequential_sort_result = benchmark(
        array,
        sorted,
        [](std::vector<int> &a) {
            sequential_mergesort(a);
        }
    );

    std::cout << "sorting parallel_mergesort" << std::endl;
    const Result &parallel_sort_result = benchmark(
        array,
        sorted,
        [](std::vector<int> &a) {
            parallel_mergesort(a);
        }
    );

    std::cout << "sorting sequential_k_way_mergesort" << std::endl;
    const Result &sequential_k_way_sort_result = benchmark(
        array,
        sorted,
        [](std::vector<int> &a) {
            sequential_k_way_mergesort(a, 4);
        }
    );

    std::cout << "sorting parallel_k_way_mergesort" << std::endl;
    const Result &parallel_k_way_sort_result = benchmark(
        array,
        sorted,
        [](std::vector<int> &a) {
            parallel_k_way_mergesort(a, 4);
        }
    );

    std::cout << "sorting parallel_mergesort_ranks" << std::endl;
    const Result &parallel_ranks_sort_result = benchmark(
        array,
        sorted,
        [](std::vector<int> &a) {
            parallel_mergesort_ranks(a);
        }
    );

    std::cout << std::boolalpha
        << "control sort:          " << control_result << '\n'
        << "sequential sort:       " << sequential_sort_result << '\n'
        << "parallel sort:         " << parallel_sort_result << '\n'
        << "sequential k-way sort: " << sequential_k_way_sort_result << '\n'
        << "parallel k-way sort:   " << parallel_k_way_sort_result << '\n'
        << "parallel ranks sort:   " << parallel_ranks_sort_result << '\n'
        << std::flush;

    return 0;
}
