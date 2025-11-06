// This is a test file for testing the interface
//  >>> virtual std::vector<std::pair<dist_t, labeltype>>
//  >>>    searchKnnCloserFirst(const void* query_data, size_t k) const;
// of class AlgorithmInterface

#include "../../hnswlib/hnswlib.h"

#include <assert.h>

#include <sstream>
#include <filesystem>
#include <vector>
#include <iostream>

namespace {

using idx_t = hnswlib::labeltype;


class StopW {
    std::chrono::steady_clock::time_point time_begin;

 public:
    StopW() {
        time_begin = std::chrono::steady_clock::now();
    }

    float getElapsedTimeMicro() {
        std::chrono::steady_clock::time_point time_end = std::chrono::steady_clock::now();
        return (std::chrono::duration_cast<std::chrono::microseconds>(time_end - time_begin).count());
    }

    void reset() {
        time_begin = std::chrono::steady_clock::now();
    }
};

void test(int d_i, int n_i, int nq_i, int k_i, 
          bool test_brute_force_i, bool test_hnsw_i) {
    int d = d_i; // dim 
    idx_t n = n_i; // data size (in count of float)
    idx_t nq = nq_i; // query size (in count of float)
    size_t k = k_i; // top k
    

    std::vector<float> data(n * d);
    std::vector<float> query(nq * d);

    std::mt19937 rng;
    rng.seed(47);
    std::uniform_real_distribution<> distrib;

    for (idx_t i = 0; i < n * d; ++i) {
        data[i] = distrib(rng);
    }
    for (idx_t i = 0; i < nq * d; ++i) {
        query[i] = distrib(rng);
    }

    hnswlib::L2Space space(d);
    hnswlib::AlgorithmInterface<float>* alg_brute;  
    hnswlib::AlgorithmInterface<float>* alg_hnsw; 

    std::stringstream ss_hnsw;
    ss_hnsw << "./hnsw_d" << d_i << "_n" << n_i;
    std::string filepath_hnsw = ss_hnsw.str();

    if (std::filesystem::exists(filepath_hnsw)) {
        std::cout << filepath_hnsw << " found, start loading" << std::endl;
        alg_hnsw = new hnswlib::HierarchicalNSW<float>(&space, filepath_hnsw, false, 2 * n);
    } else {
        std::cout << filepath_hnsw << " does not exists, starts building points " << std::endl;
        alg_hnsw = new hnswlib::HierarchicalNSW<float>(&space, 2 * n);
        for (size_t i = 0; i < n; ++i) {
            alg_hnsw->addPoint(data.data() + d * i, i);
        }
        std::cout << "saving to " << filepath_hnsw << std::endl;
        alg_hnsw->saveIndex(filepath_hnsw);
    }

    std::stringstream ss_brute;
    ss_brute << "./brute_d" << d_i << "_n" << n_i;
    std::string filepath_brute = ss_brute.str();

    if (std::filesystem::exists(filepath_brute)) {
        std::cout << filepath_brute << " found, start loading" << std::endl;
        alg_brute = new hnswlib::BruteforceSearch<float>(&space, filepath_brute);

    } else {
        std::cout << filepath_brute << " does not exists, starts building points " << std::endl;
        alg_brute = new hnswlib::BruteforceSearch<float>(&space, 2 * n);
        for (size_t i = 0; i < n; ++i) {
            alg_brute->addPoint(data.data() + d * i, i);
        }
        std::cout << "saving to " << filepath_brute << std::endl;
        alg_brute->saveIndex(filepath_brute);
    }


    std::cout << "done adding points, start testing ... " << std::endl;

    // test searchKnnCloserFirst of BruteforceSearch
    StopW stopw = StopW();
    float time_us_per_query;
    if (test_brute_force_i) {
        for (size_t j = 0; j < nq; ++j) {
            const void* p = query.data() + j * d;
            auto gd = alg_brute->searchKnn(p, k);
            auto res = alg_brute->searchKnnCloserFirst(p, k);
            assert(gd.size() == res.size());
            size_t t = gd.size();
            while (!gd.empty()) {
                assert(gd.top() == res[--t]);
                gd.pop();
            }
        }
        time_us_per_query = stopw.getElapsedTimeMicro() / nq;
        std::cout << "us per query: " << time_us_per_query << std::endl;
    }

    if (test_hnsw_i) {
        stopw.reset();
        for (size_t j = 0; j < nq; ++j) {
            const void* p = query.data() + j * d;
            auto gd = alg_hnsw->searchKnn(p, k);
            auto res = alg_hnsw->searchKnnCloserFirst(p, k);
            assert(gd.size() == res.size());
            size_t t = gd.size();
            while (!gd.empty()) {
                assert(gd.top() == res[--t]);
                gd.pop();
            }
        }
        time_us_per_query = stopw.getElapsedTimeMicro() / nq;
        std::cout << "us per query: " << time_us_per_query << std::endl;
    }

    delete alg_brute;
    delete alg_hnsw;
}

}  // namespace

static bool test_brute_force = false;
static bool test_hnsw = false;
static int d = 128; // dim 
static int n = 100000; // data size (in count of float)
static int nq = 10; // query size (in count of float)
static int k = 10; // top k
int parse_arg(int argc, char* argv[]) {

    test_brute_force = false;
    test_hnsw = false;
    d = 128; // dim 
    n = 100000; // data size (in count of float)
    nq = 10; // query size (in count of float)
    k = 10; // top k

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        if (arg == "-b") {
            test_brute_force = true;
        } else if (arg == "-w") {
            test_hnsw = true;
        } else if (arg == "-d" && i + 1 < argc) {
            d = std::atoi(argv[++i]);
        } else if (arg == "-n" && i + 1 < argc) {
            n = std::atoi(argv[++i]);
        } else if (arg == "-t" && i + 1 < argc) {
            nq = std::atoi(argv[++i]);
        } else if (arg == "-k" && i + 1 < argc) {
            k = std::atoi(argv[++i]);
        } else if (arg == "-h") {
            std::cout << "-d -- dim of vector" << std::endl;
            std::cout << "-n -- number of vector in the graph" << std::endl;
            std::cout << "-t -- number of test" << std::endl;
            std::cout << "-k -- top k" << std::endl;
            std::cout << "-b -- test brute force" << std::endl;
            std::cout << "-w -- test hnsw" << std::endl;
            return 1;
        } else {
            std::cerr << "Unknown or incomplete argument: " << arg << std::endl;
            return 1;
        }
        std::cout << arg << std::endl;
    }
    return 0;
}

int main(int argc, char* argv[]) {
    int ret = parse_arg(argc, argv);
    if (ret) {
        std::cout << "Parsing err ..." << std::endl;
        return 1;
    }

    //std::cout << "Testing ..." << std::endl;
    test(d, n, nq, k, test_brute_force, test_hnsw);
    std::cout << "Test ok" << std::endl;

    return 0;
}
