#include <algorithm>
#include <array>
#include <functional>
#include <iomanip>
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
    double time = -1;
    double speedup = -1;
    double efficiency = -1;
    int p = -1;
    int k = -1;
    int g_threshold = -1;
    bool correct = false;

    void calculate_measurements(const Result &ref) {
        if (p > 1 && timed() && ref.p == 1 && ref.timed()) {
            speedup = ref.time / time;
            efficiency = speedup / p;
        }
    }

    [[nodiscard]] bool timed() const {
        return time != -1;
    }

    [[nodiscard]] bool measured() const {
        return speedup != -1 && efficiency != -1;
    }
};

template<typename T>
std::ostream &operator<<(std::ostream &out, const std::vector<T> &v);

std::ostream &operator<<(std::ostream &out, const Result &r);

Result benchmark(
    const std::function<void(std::vector<int> &)> &fn,
    const std::vector<int> &array,
    const std::vector<int> &sorted
);

Result benchmark(
    const std::function<void(std::vector<int> &, int)> &fn,
    const std::vector<int> &array,
    int k,
    const std::vector<int> &sorted
);

Result benchmark(
    const std::function<void(std::vector<int> &, int)> &fn,
    const std::vector<int> &array,
    int g_threshold,
    const std::vector<int> &sorted,
    int p,
    const Result &ref
);

Result benchmark(
    const std::function<void(std::vector<int> &, int, int)> &fn,
    const std::vector<int> &array,
    int k,
    int g_threshold,
    const std::vector<int> &sorted,
    int p,
    const Result &ref
);

constexpr int PRECISION = 9;

int main() {
    const int max_threads = omp_get_max_threads();
    std::vector<int> threads;

    for (int t = 1; t <= max_threads; t *= 2) {
        threads.emplace_back(t);
    }
    if (threads.back() != max_threads) {
        threads.emplace_back(max_threads);
    }

    std::cout << std::boolalpha << std::fixed << std::setprecision(PRECISION);

    for (int size = 10; size <= 26; size += 2) {
        std::cout <<
            "==================================================\n"
            "\n"
            "size       2^" << size << "\n"
            << std::endl;

        constexpr std::array ks{4, 8, 16, 32};
        std::vector<int> array(1 << size);
        int_generator<int> generate_int(std::numeric_limits<int>::min(), std::numeric_limits<int>::max());
        for (int &n: array) {
            n = generate_int();
        }

        std::vector sorted(array);
        std::ranges::sort(sorted);

        std::cout << "sorting control" << std::endl;
        const Result &control_result = benchmark(std::ranges::sort, array, sorted);

        std::cout << "sorting regular sequential" << std::endl;
        const Result &regular_sequential_result = benchmark(sequential_mergesort, array, sorted);

        std::array<Result, ks.size()> k_way_sequential_results;
        for (int i = 0; i < ks.size(); i++) {
            const int k = ks[i];
            std::cout << "sorting k-way (" << k << ") sequential" << std::endl;
            k_way_sequential_results[i] = benchmark(sequential_k_way_mergesort, array, k, sorted);
        }

        std::cout << "\n"
            "sequential results:\n"
            "control       " << control_result << "\n"
            "regular       " << regular_sequential_result << "\n";

        for (const auto &result: k_way_sequential_results) {
            std::cout << "k-way (" << std::setw(2) << result.k << ")    " << result << "\n";
        }

        std::cout << std::endl;

        Result p1_regular_parallel_result;
        std::array<Result, ks.size()> p1_k_way_parallel_results;
        Result p1_ranks_parallel_result;
        std::array<Result, ks.size()> p1_ranks_k_way_parallel_results;

        for (const int &t: threads) {
            std::cout <<
                "--+----+----+----+----+----+----+----+----+----+--\n"
                "\n"
                "threads    " << t << "\n"
                << std::endl;

            omp_set_num_threads(t);

            constexpr std::array g_thresholds{1 << 10, 1 << 12, 1 << 14};

            for (const int &g_threshold: g_thresholds) {
                std::cout <<
                    "--^----v----^----^----v----^----^----v----^----^--\n"
                    "\n"
                    "granularity threshold    " << g_threshold << "\n"
                    << std::endl;

                std::cout << "sorting regular parallel" << std::endl;
                const Result &regular_parallel_result = benchmark(
                    parallel_mergesort,
                    array,
                    g_threshold,
                    sorted,
                    t,
                    p1_regular_parallel_result
                );
                if (!p1_regular_parallel_result.timed()) {
                    p1_regular_parallel_result = regular_parallel_result;
                }

                std::array<Result, ks.size()> k_way_parallel_results;
                for (int i = 0; i < ks.size(); i++) {
                    const int k = ks[i];
                    std::cout << "sorting k-way (" << k << ") parallel" << std::endl;
                    k_way_parallel_results[i] = benchmark(
                        parallel_k_way_mergesort,
                        array,
                        k,
                        g_threshold,
                        sorted,
                        t,
                        p1_k_way_parallel_results[i]
                    );
                    if (!p1_k_way_parallel_results[i].timed()) {
                        p1_k_way_parallel_results[i] = k_way_parallel_results[i];
                    }
                }

                std::cout << "sorting ranks parallel" << std::endl;
                const Result &ranks_parallel_result = benchmark(
                    parallel_ranks_mergesort,
                    array,
                    g_threshold,
                    sorted,
                    t,
                    p1_ranks_parallel_result
                );
                if (!p1_ranks_parallel_result.timed()) {
                    p1_ranks_parallel_result = ranks_parallel_result;
                }

                std::array<Result, ks.size()> ranks_k_way_parallel_results;
                for (int i = 0; i < ks.size(); i++) {
                    const int k = ks[i];
                    std::cout << "sorting ranks + k-way (" << k << ") parallel" << std::endl;
                    ranks_k_way_parallel_results[i] = benchmark(
                        parallel_ranks_k_way_mergesort,
                        array,
                        k,
                        g_threshold,
                        sorted,
                        t,
                        p1_ranks_k_way_parallel_results[i]
                    );
                    if (!p1_ranks_k_way_parallel_results[i].timed()) {
                        p1_ranks_k_way_parallel_results[i] = ranks_k_way_parallel_results[i];
                    }
                }

                std::cout << "\n"
                    "parallel results:\n"
                    "regular               " << regular_parallel_result << "\n";

                for (const auto &result: k_way_parallel_results) {
                    std::cout << "k-way (" << std::setw(2) << result.k << ")            " << result << "\n";
                }

                std::cout <<
                    "ranks                 " << ranks_parallel_result << "\n";

                for (const auto &result: ranks_k_way_parallel_results) {
                    std::cout << "ranks + k-way (" << std::setw(2) << result.k << ")    " << result << "\n";
                }

                std::cout << std::endl;
            }
        }
    }

    return 0;
}

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
    if (!r.correct) {
        out << "\033[1m\033[31m"; // bold red
    }

    out << "correct: " << std::setw(5) << r.correct << "  |  time: " << std::setw(PRECISION + 3) << r.time << " s";

    if (r.measured()) {
        out << "  |  speedup: " << std::setw(9) << r.speedup << "  |  efficiency: " << r.efficiency;
    }

    if (!r.correct) {
        out << "\033[0m"; // reset
    }

    return out;
}

Result benchmark(
    const std::function<void(std::vector<int> &)> &fn,
    const std::vector<int> &array,
    const std::vector<int> &sorted
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

Result benchmark(
    const std::function<void(std::vector<int> &, int)> &fn,
    const std::vector<int> &array,
    const int k,
    const std::vector<int> &sorted
) {
    std::vector copy(array);

    const double start_time = omp_get_wtime();
    fn(copy, k);
    const double end_time = omp_get_wtime();

    return {
        .time = end_time - start_time,
        .k = k,
        .correct = copy == sorted,
    };
}

Result benchmark(
    const std::function<void(std::vector<int> &, int)> &fn,
    const std::vector<int> &array,
    const int g_threshold,
    const std::vector<int> &sorted,
    const int p,
    const Result &ref
) {
    std::vector copy(array);

    const double start_time = omp_get_wtime();
    fn(copy, g_threshold);
    const double end_time = omp_get_wtime();

    Result result = {
        .time = end_time - start_time,
        .p = p,
        .g_threshold = g_threshold,
        .correct = copy == sorted,
    };
    result.calculate_measurements(ref);

    return result;
}

Result benchmark(
    const std::function<void(std::vector<int> &, int, int)> &fn,
    const std::vector<int> &array,
    const int k,
    const int g_threshold,
    const std::vector<int> &sorted,
    const int p,
    const Result &ref
) {
    std::vector copy(array);

    const double start_time = omp_get_wtime();
    fn(copy, k, g_threshold);
    const double end_time = omp_get_wtime();

    Result result = {
        .time = end_time - start_time,
        .p = p,
        .k = k,
        .g_threshold = g_threshold,
        .correct = copy == sorted,
    };
    result.calculate_measurements(ref);

    return result;
}
