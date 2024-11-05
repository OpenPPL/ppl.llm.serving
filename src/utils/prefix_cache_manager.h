#ifndef __PPL_LLM_PREFIX_CACHE_MANAGER_H__
#define __PPL_LLM_PREFIX_CACHE_MANAGER_H__

#include "ppl/common/log.h"

#include <unordered_map>
#include <vector>
#include <algorithm>
#include <stdint.h>
#include <iostream>

namespace ppl { namespace llm { namespace utils {

class LRUCache {
public:
    struct Node {
        Node(uint64_t _hash_val, int64_t _page_id)
            : hash_val(_hash_val), page_id(_page_id), prev(nullptr), next(nullptr) {}

        uint64_t hash_val;
        int64_t page_id;
        Node* prev;
        Node* next;
    };

    LRUCache() {
        head_ = new Node(-1, -1);
        tail_ = new Node(-1, -1);
        head_->next = tail_;
        tail_->prev = head_;
    }

    ~LRUCache() {
        while (head_) {
            Node* tmp = head_;
            head_ = head_->next;
            delete tmp;
        }
    }

    bool Find(uint64_t hash_val) const {
        return cache_.find(hash_val) != cache_.end();
    }

    void Insert(uint64_t hash_val, int64_t page_id) {
        if (cache_.find(hash_val) != cache_.end()) { // almost impossible to reach
            LOG(WARNING) << "hash_val [" << hash_val << "] page_id [" << page_id << "] already exists in cache_";
            return;
        }
        Node* node = new Node(hash_val, page_id);
        cache_.insert({hash_val, node});
        AddToHead(node);
    }

    void EvictNode(uint64_t hash_val) {
        Node* node = cache_[hash_val];
        int success = cache_.erase(hash_val);
        if (success == 0) {
            LOG(WARNING) << "erase unexist hash [" << node->hash_val << "]";
        }
        DeleteNode(node);
    }

    void EvictList(int64_t nums, std::vector<int64_t>* page_list, std::vector<uint64_t>* hash_list) {
        int64_t erase_nums = (uint64_t)nums < cache_.size() ? nums : cache_.size();
        for (int i = 0; i < erase_nums; ++i) {
            Node* node = tail_->prev;
            page_list->push_back(node->page_id);
            hash_list->push_back(node->hash_val);
            int success = cache_.erase(node->hash_val);
            if (success == 0) {
                LOG(WARNING) << "erase unexist hash [" << node->hash_val << "]";
            }
            DeleteNode(node);
        }
    }

    int32_t Size() const {
        return cache_.size();
    }

    void Reset() {
        while (head_->next != tail_) {
            Node* node = head_->next;
            head_->next = node->next;
            delete node;
        }
        cache_.clear();
    }

private:
    void AddToHead(Node* node) {
        node->next = head_->next;
        node->prev = head_;
        head_->next->prev = node;
        head_->next = node;
    }

    void DeleteNode(Node* node) {
        node->prev->next = node->next;
        node->next->prev = node->prev;
        delete node;
    }

private:
    Node* head_;
    Node* tail_;
    std::unordered_map<uint64_t, Node*> cache_;
};

class PrefixCacheManager {
public:
    struct PrefixItem {
        PrefixItem(uint64_t _hash_val, int64_t _page_id) : hash_val(_hash_val), page_id(_page_id) {}
        uint64_t hash_val;
        int64_t page_id;
        int32_t ref_count = 1;
    };

    PrefixCacheManager() {}

    int64_t Find(uint64_t hash_val) const {
        auto iter = prefix_map_.find(hash_val);
        if (iter == prefix_map_.end()) {
            return -1;
        }
        return iter->second.page_id;
    }

    void Insert(uint64_t hash_val, int64_t page_id) {
        prefix_map_.insert({hash_val, PrefixItem(hash_val, page_id)});
    }

    void IncRefCount(const uint64_t* hash_list, int64_t nums) {
        for (int i = 0; i < nums; ++i) {
            uint64_t hash_val = hash_list[i];
            auto iter = prefix_map_.find(hash_val);
            if (iter == prefix_map_.end()) {
                LOG(WARNING) << "hash [" << hash_val << "] not found in prefix map";
                break;
            }

            iter->second.ref_count++;
            if (lru_cache_.Find(hash_val)) {
                lru_cache_.EvictNode(hash_val);
            }
        }
    }

    void DecRefCount(const uint64_t* hash_list, int64_t nums) {
        for (int i = 0; i < nums; ++i) {
            auto iter = prefix_map_.find(hash_list[i]);
            if (iter == prefix_map_.end()) {
                LOG(WARNING) << "hash [" << hash_list[i] << "] not found in prefix map";
                break;
            }
            iter->second.ref_count--;

            if (iter->second.ref_count == 0) {
                lru_cache_.Insert(iter->second.hash_val, iter->second.page_id);
            }
        }
    }

    void Evict(int64_t nums, std::vector<int64_t>* page_list) {
        std::vector<uint64_t> hash_list;
        lru_cache_.EvictList(nums, page_list, &hash_list);
        for (size_t i = 0; i < hash_list.size(); ++i) {
            uint64_t hash_val = hash_list[i];
            prefix_map_.erase(hash_val);
        }
    }

    int32_t Size() const {
        return prefix_map_.size();
    }

    void Reset() {
        prefix_map_.clear();
        lru_cache_.Reset();
    }

private:
    std::unordered_map<uint64_t, PrefixItem> prefix_map_;
    LRUCache lru_cache_;
};

}}} // namespace ppl::llm::utils

#endif