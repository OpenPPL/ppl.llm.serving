#include "utils/prefix_cache_manager.h"
#include "utils/utils.h"
#include "ppl/common/log.h"
#include "ppl/common/cuda/cuda_env.h"
#include <vector>
#include <iostream>
#include <stdint.h>
#include <cuda_runtime.h>
#include <string.h>

using namespace ppl::llm;
using namespace ppl::llm::utils;
using namespace ppl::common;
using namespace std;

template <class T>
static void PrintVector(vector<T> vec, const std::string& prefix = "") {
    stringstream ss;
    for (auto& ele : vec) {
        ss << ele << ", ";
    }
    std::cout << prefix << ": " << ss.str() << std::endl;
}

void test_hash() {
    uint64_t prev = 0;
    int32_t vec[] = {1, 2, 3, 4, 5};
    int32_t len = 5;
    uint64_t hash_val = HashCombine(prev, vec, len);
    std::cout << "hash_val: " << hash_val << std::endl;
}

void test_prefix_mgr() {
    PrefixCacheManager prefix_mgr;
    std::vector<uint64_t> hash_list = {0, 1, 2, 3};
    std::vector<int64_t> page_list = {11, 12, 13, 14};
    int nums = hash_list.size();

    for (size_t i = 0; i < hash_list.size(); ++i) {
        prefix_mgr.Insert(hash_list[i], page_list[i]);
    }

    std::vector<uint64_t> hash_list2 = {5, 6, 7, 8};
    std::vector<int64_t> page_list2 = {15, 16, 17, 18};

    for (size_t i = 0; i < hash_list.size(); ++i) {
        prefix_mgr.Insert(hash_list2[i], page_list2[i]);
    }

    prefix_mgr.DecRefCount(hash_list.data(), hash_list.size());
    std::cout << prefix_mgr.Size() << std::endl;

    prefix_mgr.DecRefCount(hash_list2.data(), hash_list2.size());

    std::vector<int64_t> evicted_page_list;
    prefix_mgr.Evict(nums, &evicted_page_list);
    std::cout << prefix_mgr.Size() << std::endl;
    evicted_page_list.clear();
    prefix_mgr.Evict(nums, &evicted_page_list);
}

int main(int argc, char const* argv[]) {
    test_hash();
    test_prefix_mgr();
    return 0;
}
