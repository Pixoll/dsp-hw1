#include <algorithm>
#include <array>
#include <cstring>
#include <filesystem>
#include <fstream>
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

constexpr auto CSV_COLS = "type,correct,n,k,p,s,e,g_threshold,t_mean,t_stdev,t_q0,t_q1,t_q2,t_q3,t_q4";
constexpr int BENCH_SAMPLES = 4;
constexpr int PRECISION = 9;

struct Result {
    const char *type = "";
    std::array<double, BENCH_SAMPLES> times{};
    double time_mean = -1;
    double time_stdev = -1;
    std::array<double, 5> time_quartiles{};
    double speedup = -1;
    double efficiency = -1;
    int p = -1;
    int k = -1;
    int g_threshold = -1;
    bool correct = false;

    [[nodiscard]] bool timed() const {
        return time_mean != -1;
    }

    [[nodiscard]] bool measured() const {
        return speedup != -1 && efficiency != -1;
    }

    void calculate_measurements() {
        time_mean = 0;
        time_stdev = 0;

        for (const double &time: times) {
            time_mean += time;
        }
        time_mean /= static_cast<double>(times.size());

        for (const double &time: times) {
            const double dev = time - time_mean;
            time_stdev += dev * dev;
        }

        time_stdev /= static_cast<double>(times.size() - 1);
        time_stdev = std::sqrt(time_stdev);

        calculate_quartiles();
    }

    void calculate_measurements(const Result &ref) {
        calculate_measurements();

        if (p > 1 && timed() && ref.p == 1 && ref.timed()) {
            speedup = ref.time_mean / time_mean;
            efficiency = speedup / p;
        }
    }

private:
    void calculate_quartiles() {
        const size_t n = times.size();
        size_t part;

        std::ranges::sort(times);

        time_quartiles[0] = times.front();
        time_quartiles[4] = times.back();

        if (n % 2 == 1) {
            time_quartiles[2] = times[n / 2];
        } else {
            part = n / 2;
            time_quartiles[2] = (times[part - 1] + times[part]) / 2.0;
        }

        if (n % 4 >= 2) {
            time_quartiles[1] = times[n / 4];
            time_quartiles[3] = times[3 * n / 4];
        } else {
            part = n / 4;
            time_quartiles[1] = 0.25 * times[part - 1] + 0.75 * times[part];
            part = 3 * n / 4;
            time_quartiles[3] = 0.75 * times[part - 1] + 0.25 * times[part];
        }
    }
};

template<typename T>
std::ostream &operator<<(std::ostream &out, const std::vector<T> &v);

std::ostream &operator<<(std::ostream &out, const Result &r);

std::ofstream &operator<<(std::ofstream &out, const Result &r) {
    out << r.type << ","
        << r.correct << ","
        << BENCH_SAMPLES << ",";

    if (r.k != -1) out << r.k;
    out << ",";

    if (r.p != -1) out << r.p;
    out << ",";

    if (r.speedup != -1) out << r.speedup;
    out << ",";

    if (r.efficiency != -1) out << r.efficiency;
    out << ",";

    if (r.g_threshold != -1) out << r.g_threshold;
    out << ",";

    out << r.time_mean << ","
        << r.time_stdev << ","
        << r.time_quartiles[0] << ","
        << r.time_quartiles[1] << ","
        << r.time_quartiles[2] << ","
        << r.time_quartiles[3] << ","
        << r.time_quartiles[4] << "\n";

    if (!r.correct) {
        std::cout << r << "\n";
    }

    return out;
}

template<size_t N>
std::ofstream &operator<<(std::ofstream &out, const std::array<Result, N> &results) {
    for (const Result &r: results) {
        out << r;
    }
    return out;
}

Result benchmark(
    const char *type,
    const std::function<void(std::vector<int> &)> &fn,
    const std::vector<int> &array,
    const std::vector<int> &sorted
);

Result benchmark(
    const char *type,
    const std::function<void(std::vector<int> &, int)> &fn,
    const std::vector<int> &array,
    int k,
    const std::vector<int> &sorted
);

Result benchmark(
    const char *type,
    const std::function<void(std::vector<int> &, int)> &fn,
    const std::vector<int> &array,
    int g_threshold,
    const std::vector<int> &sorted,
    int p,
    const Result &ref
);

Result benchmark(
    const char *type,
    const std::function<void(std::vector<int> &, int, int)> &fn,
    const std::vector<int> &array,
    int k,
    int g_threshold,
    const std::vector<int> &sorted,
    int p,
    const Result &ref
);

int main(const int argc, const char *argv[]) {
    const int max_threads = std::min(omp_get_max_threads(), 8);

    if (argc > 1) {
        const char *type = argv[1];

        omp_set_num_threads(max_threads);
        constexpr int k = 16;
        constexpr int g_threshold = 1 << 12;
        constexpr int size = 26;

        std::vector<int> array(1 << size);
        int_generator<int> generate_int(std::numeric_limits<int>::min(), std::numeric_limits<int>::max());
        for (int &n: array) {
            n = generate_int();
        }

        if (strcmp(type, "sequential_mergesort") == 0) {
            sequential_mergesort(array);
            return 0;
        }

        if (strcmp(type, "sequential_k_way_mergesort") == 0) {
            sequential_k_way_mergesort(array, k);
            return 0;
        }

        if (strcmp(type, "parallel_mergesort") == 0) {
            parallel_mergesort(array, g_threshold);
            return 0;
        }

        if (strcmp(type, "parallel_k_way_mergesort") == 0) {
            parallel_k_way_mergesort(array, k, g_threshold);
            return 0;
        }

        if (strcmp(type, "parallel_ranks_mergesort") == 0) {
            parallel_ranks_mergesort(array, g_threshold);
            return 0;
        }

        if (strcmp(type, "parallel_ranks_k_way_mergesort") == 0) {
            parallel_ranks_k_way_mergesort(array, k, g_threshold);
            return 0;
        }

        std::cerr << "invalid test type " << type << std::endl;
        return 1;
    }

    std::vector<int> threads;

    for (int t = 1; t <= max_threads; t *= 2) {
        threads.emplace_back(t);
    }

    // ReSharper disable once CppTooWideScope
    constexpr std::array ks{4, 8};
    constexpr std::array g_thresholds{1 << 10, 1 << 12};

    if (!std::filesystem::exists("data")) {
        std::filesystem::create_directory("data");
    }

    std::ofstream time_data("data/measurements.csv");
    time_data << "type,correct,n,k,p,s,e,g_threshold,t_mean,t_stdev,t_q0,t_q1,t_q2,t_q3,t_q4\n";

    std::cout << std::boolalpha << std::fixed << std::setprecision(PRECISION);
    time_data << std::fixed << std::setprecision(PRECISION);

    for (int size = 20; size <= 26; size += 2) {
        std::cout << "\n"
            "==================================================\n"
            "\n"
            "size       2^" << size << "\n"
            << std::endl;

        std::vector<int> array(1 << size);
        int_generator<int> generate_int(std::numeric_limits<int>::min(), std::numeric_limits<int>::max());
        for (int &n: array) {
            n = generate_int();
        }

        std::vector sorted(array);
        std::ranges::sort(sorted);

        std::cout << "benchmarking regular sequential" << std::endl;
        const Result &regular_sequential_result = benchmark(
            "sequential_mergesort",
            sequential_mergesort,
            array,
            sorted
        );

        std::array<Result, ks.size()> k_way_sequential_results;
        for (int i = 0; i < ks.size(); i++) {
            const int k = ks[i];
            std::cout << "benchmarking k-way (" << k << ") sequential" << std::endl;
            k_way_sequential_results[i] = benchmark(
                "sequential_k_way_mergesort",
                sequential_k_way_mergesort,
                array,
                k,
                sorted
            );
        }

        time_data
            << regular_sequential_result
            << k_way_sequential_results;

        std::array<Result, g_thresholds.size()> p1_regular_parallel_results;
        std::array<std::array<Result, ks.size()>, g_thresholds.size()> p1_k_way_parallel_results;
        std::array<Result, g_thresholds.size()> p1_ranks_parallel_results;
        std::array<std::array<Result, ks.size()>, g_thresholds.size()> p1_ranks_k_way_parallel_results;

        for (const int &t: threads) {
            omp_set_num_threads(t);

            for (int g = 0; g < g_thresholds.size(); g++) {
                const int g_threshold = g_thresholds[g];

                std::cout << "\n"
                    "--+----+----+----+----+----+----+----+----+----+--\n"
                    "\n"
                    "size                     2^" << size << "\n"
                    "threads                  " << t << "\n"
                    "granularity threshold    " << g_threshold << "\n"
                    << std::endl;

                std::cout << "benchmarking regular parallel" << std::endl;
                const Result &regular_parallel_result = benchmark(
                    "parallel_mergesort",
                    parallel_mergesort,
                    array,
                    g_threshold,
                    sorted,
                    t,
                    p1_regular_parallel_results[g]
                );
                if (!p1_regular_parallel_results[g].timed()) {
                    p1_regular_parallel_results[g] = regular_parallel_result;
                }

                std::array<Result, ks.size()> k_way_parallel_results;
                for (int i = 0; i < ks.size(); i++) {
                    const int k = ks[i];
                    std::cout << "benchmarking k-way (" << k << ") parallel" << std::endl;
                    k_way_parallel_results[i] = benchmark(
                        "parallel_k_way_mergesort",
                        parallel_k_way_mergesort,
                        array,
                        k,
                        g_threshold,
                        sorted,
                        t,
                        p1_k_way_parallel_results[g][i]
                    );
                    if (!p1_k_way_parallel_results[g][i].timed()) {
                        p1_k_way_parallel_results[g][i] = k_way_parallel_results[i];
                    }
                }

                std::cout << "benchmarking ranks parallel" << std::endl;
                const Result &ranks_parallel_result = benchmark(
                    "parallel_ranks_mergesort",
                    parallel_ranks_mergesort,
                    array,
                    g_threshold,
                    sorted,
                    t,
                    p1_ranks_parallel_results[g]
                );
                if (!p1_ranks_parallel_results[g].timed()) {
                    p1_ranks_parallel_results[g] = ranks_parallel_result;
                }

                std::array<Result, ks.size()> ranks_k_way_parallel_results;
                for (int i = 0; i < ks.size(); i++) {
                    const int k = ks[i];
                    std::cout << "benchmarking ranks + k-way (" << k << ") parallel" << std::endl;
                    ranks_k_way_parallel_results[i] = benchmark(
                        "parallel_ranks_k_way_mergesort",
                        parallel_ranks_k_way_mergesort,
                        array,
                        k,
                        g_threshold,
                        sorted,
                        t,
                        p1_ranks_k_way_parallel_results[g][i]
                    );
                    if (!p1_ranks_k_way_parallel_results[g][i].timed()) {
                        p1_ranks_k_way_parallel_results[g][i] = ranks_k_way_parallel_results[i];
                    }
                }

                time_data
                    << regular_parallel_result
                    << k_way_parallel_results
                    << ranks_parallel_result
                    << ranks_k_way_parallel_results;
            }
        }
    }

    time_data.close();

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

    out << "type: " << std::setw(30) << r.type;

    if (r.k != -1) {
        out << "  |  k: " << std::setw(2) << r.k;
    }

    out << "  |  correct: " << std::setw(5) << r.correct
        << "  |  time: ~" << std::setw(PRECISION + 3) << r.time_mean << " s";

    if (r.measured()) {
        out << "  |  speedup: " << std::setw(9) << r.speedup << "  |  efficiency: " << r.efficiency;
    }

    if (!r.correct) {
        out << "\033[0m"; // reset
    }

    return out;
}

Result benchmark(
    const char *type,
    const std::function<void(std::vector<int> &)> &fn,
    const std::vector<int> &array,
    const std::vector<int> &sorted
) {
    bool first = true;
    Result result{
        .type = type,
    };

    for (double &time: result.times) {
        std::vector copy(array);

        const double start_time = omp_get_wtime();
        fn(copy);
        const double end_time = omp_get_wtime();
        time = end_time - start_time;

        if (first) {
            result.correct = copy == sorted;
            first = false;
        }
    }

    result.calculate_measurements();

    return result;
}

Result benchmark(
    const char *type,
    const std::function<void(std::vector<int> &, int)> &fn,
    const std::vector<int> &array,
    const int k,
    const std::vector<int> &sorted
) {
    bool first = true;
    Result result{
        .type = type,
        .k = k,
    };

    for (double &time: result.times) {
        std::vector copy(array);

        const double start_time = omp_get_wtime();
        fn(copy, k);
        const double end_time = omp_get_wtime();
        time = end_time - start_time;

        if (first) {
            result.correct = copy == sorted;
            first = false;
        }
    }

    result.calculate_measurements();

    return result;
}

Result benchmark(
    const char *type,
    const std::function<void(std::vector<int> &, int)> &fn,
    const std::vector<int> &array,
    const int g_threshold,
    const std::vector<int> &sorted,
    const int p,
    const Result &ref
) {
    bool first = true;
    Result result{
        .type = type,
        .p = p,
        .g_threshold = g_threshold,
    };

    for (double &time: result.times) {
        std::vector copy(array);

        const double start_time = omp_get_wtime();
        fn(copy, g_threshold);
        const double end_time = omp_get_wtime();
        time = end_time - start_time;

        if (first) {
            result.correct = copy == sorted;
            first = false;
        }
    }

    result.calculate_measurements(ref);

    return result;
}

Result benchmark(
    const char *type,
    const std::function<void(std::vector<int> &, int, int)> &fn,
    const std::vector<int> &array,
    const int k,
    const int g_threshold,
    const std::vector<int> &sorted,
    const int p,
    const Result &ref
) {
    bool first = true;
    Result result{
        .type = type,
        .p = p,
        .k = k,
        .g_threshold = g_threshold,
    };

    for (double &time: result.times) {
        std::vector copy(array);

        const double start_time = omp_get_wtime();
        fn(copy, k, g_threshold);
        const double end_time = omp_get_wtime();
        time = end_time - start_time;

        if (first) {
            result.correct = copy == sorted;
            first = false;
        }
    }

    result.calculate_measurements(ref);

    return result;
}
