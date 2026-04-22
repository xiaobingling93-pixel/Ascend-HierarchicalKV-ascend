/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <ostream>
#include <shared_mutex>
#include <string>
#include <type_traits>
#include "aclrtlaunch_allocate_bucket_others_kernel.h"
#include "aclrtlaunch_allocate_bucket_vectors_kernel.h"
#include "aclrtlaunch_clear_kernel.h"
#include "aclrtlaunch_create_atomic_keys_kernel.h"
#include "aclrtlaunch_create_atomic_scores_kernel.h"
#include "aclrtlaunch_find_and_update_kernel.h"
#include "aclrtlaunch_find_and_update_kernel_with_filter.h"
#include "aclrtlaunch_find_or_insert_ptr_kernel.h"
#include "aclrtlaunch_find_or_insert_ptr_kernel_v2.h"
#include "aclrtlaunch_find_ptr_kernel.h"
#include "aclrtlaunch_find_ptr_with_digest_kernel.h"
#include "aclrtlaunch_get_bucket_others_address_kernel.h"
#include "aclrtlaunch_host_nano_kernel.h"
#include "aclrtlaunch_insert_or_assign_kernel.h"
#include "aclrtlaunch_insert_or_assign_kernel_with_thread_1024.h"
#ifndef USE_DUMP_KERNEL_ASC
#include "aclrtlaunch_dump_kernel.h"
#endif
#include "aclnn_helper.h"
#include "aclnnop/aclnn_reduce_sum.h"
#include "aclrtlaunch_assign_scores_kernel.h"
#include "aclrtlaunch_assign_scores_kernel_with_filter.h"
#include "aclrtlaunch_rehash_kernel.h"
#include "aclrtlaunch_insert_and_evict_kernel.h"
#include "bucket_memory_pool_manager.h"
#include "hashtable_options.h"
#include "memory_pool.h"
#include "table.h"
#include "tiling/platform/platform_ascendc.h"
#include "types.h"
#include "utils.h"

namespace npu {
namespace hkv {

/**
 * @brief The eviction strategies.
 *
 * @note The `Score` concept is introduced to define the importance of each key,
 * the larger, the more important, the less likely they will be evicted. On
 * `kLru` mode, the `scores` parameter of the APIs should keep `nullptr`, the
 * score for each key is assigned internally in LRU(Least Recently Used) policy.
 * On `kCustomized` mode, the `scores` should be provided by caller.
 *
 * @note Eviction occurs automatically when a bucket is full. The keys with the
 * minimum `score` value are evicted first.
 *
 * @note on `kLru`, Set the score to the Device clock in a nanosecond, which
 * could differ slightly from the host clock.
 *
 * @note For `kEpochLru` and `kEpochLfu`, the high 32bits would be set to
 * `global_epoch` while the low 32 bits is `timestamp` or `frequency`.
 *
 * @note on `kLfu`, Frequency increment provided by caller via the input
 * parameter of `scores` of `insert-like` APIs as the increment of frequency.
 * when the scores reaches to the max of `uint64_t`, it will not increase any
 * more.
 *
 * @note On `kEpochLru`, the high 32bits is the global epoch provided via the
 * input parameter of `global_epoch`, the low 32bits is equal to `(device_clock
 * >> 20) & 0xffffffff` with granularity close to 1 ms.
 *
 * @note On `kEpochLfu`, the high 32bits is the global epoch provided via the
 * input parameter of `global_epoch`, the low 32bits is the frequency, the
 * frequency will keep constant after reaching the max value of `0xffffffff`.
 *
 * @note On `kCustomized`, fully provided by the caller via the input parameter
 * of `scores` of `insert-like` APIs.
 *
 */
struct EvictStrategy {
  enum EvictStrategyEnum {
    kLru = 0,         ///< LRU mode.
    kLfu = 1,         ///< LFU mode.
    kEpochLru = 2,    ///< Epoch Lru mode.
    kEpochLfu = 3,    ///< Epoch Lfu mode.
    kCustomized = 4,  ///< Customized mode.
  };
};

struct ValueMoveOpt {
  uint32_t dim = 0;
  uint32_t size = 0;
  uint32_t cg_size = 0;
  bool is_large_size = false;
};

/**
 * @brief A customizable template function indicates which keys should be
 * erased from the hash table by returning `true`.
 *
 * @note The `erase_if` or `export_batch_if` API traverses all of the items by
 * this function and the items that return `true` are removed or exported.
 *
 *  Example for erase_if:
 *
 *    ```
 *    template <class K, class S>
 *    struct EraseIfPredFunctor {
 *      __forceinline__ __simt_callee__ bool operator()(const K& key,
 *                                                 S& score,
 *                                                 const K& pattern,
 *                                                 const S& threshold) {
 *        return ((key & 0xFFFF000000000000 == pattern) &&
 *                (score < threshold));
 *      }
 *    };
 *    ```
 *
 *  Example for export_batch_if:
 *    ```
 *    template <class K, class S>
 *    struct ExportIfPredFunctor {
 *      __forceinline__ __simt_callee__ bool operator()(const K& key,
 *                                                 S& score,
 *                                                 const K& pattern,
 *                                                 const S& threshold) {
 *        return score >= threshold;
 *      }
 *    };
 *    ```
 */
template <class K, class S>
using EraseIfPredict = bool (*)(
    const K& key,       ///< The traversed key in a hash table.
    S& score,           ///< The traversed score in a hash table.
    const K& pattern,   ///< The key pattern to compare with the `key` argument.
    const S& threshold  ///< The threshold to compare with the `score` argument.
);

template <typename K, typename V, typename S = uint64_t>
class HashTableBase {
 public:
  using size_type = size_t;
  using key_type = K;
  using value_type = V;
  using score_type = S;
  using allocator_type = BaseAllocator;

 public:
  virtual ~HashTableBase() {}

  /**
   * @brief Initialize a hkv::HashTable.
   *
   * @param options The configuration options.
   */
  virtual void init(const HashTableOptions& options,
                    allocator_type* allocator = nullptr) = 0;

  /**
   * @brief Insert new key-value-score tuples into the hash table.
   * If the key already exists, the values and scores are assigned new values.
   *
   * If the target bucket is full, the keys with minimum score will be
   * overwritten by new key unless the score of the new key is even less than
   * minimum score of the target bucket.
   *
   * @param n Number of key-value-score tuples to insert or assign.
   * @param keys The keys to insert on NPU-accessible memory with shape
   * (n).
   * @param values The values to insert on NPU-accessible memory with
   * shape (n, DIM).
   * @param scores The scores to insert on NPU-accessible memory with shape
   * (n).
   * @parblock
   * The scores should be a `uint64_t` value. You can specify a value that
   * such as the timestamp of the key insertion, number of the key
   * occurrences, or another value to perform a custom eviction strategy.
   *
   * The @p scores should be `nullptr`, when the LRU eviction strategy is
   * applied.
   * @endparblock
   *
   * @param stream The CANN stream that is used to execute the operation.
   * @param unique_key If all keys in the same batch are unique.
   *
   * @param ignore_evict_strategy A boolean option indicating whether if
   * the insert_or_assign ignores the evict strategy of table with current
   * scores anyway. If true, it does not check whether the scores conforms to
   * the evict strategy. If false, it requires the scores follow the evict
   * strategy of table.
   */
  virtual void insert_or_assign(const size_type n,
                                const key_type* keys,                // (n)
                                const value_type* values,            // (n, DIM)
                                const score_type* scores = nullptr,  // (n)
                                aclrtStream stream = 0, bool unique_key = true,
                                bool ignore_evict_strategy = false) = 0;

  /**
   * @brief Insert new key-value-score tuples into the hash table.
   * If the key already exists, the values and scores are assigned new values.
   *
   * If the target bucket is full, the keys with minimum score will be
   * overwritten by new key unless the score of the new key is even less than
   * minimum score of the target bucket. The overwritten key with minimum
   * score will be evicted, with its values and score, to evicted_keys,
   * evicted_values, evcted_scores seperately in compact format.
   *
   * @param n Number of key-value-score tuples to insert or assign.
   * @param keys The keys to insert on NPU-accessible memory with shape
   * (n).
   * @param values The values to insert on NPU-accessible memory with
   * shape (n, DIM).
   * @param scores The scores to insert on NPU-accessible memory with shape
   * (n).
   * @param scores The scores to insert on NPU-accessible memory with shape
   * (n).
   * @params evicted_keys The output of keys replaced with minimum score.
   * @params evicted_values The output of values replaced with minimum score on
   * keys.
   * @params evicted_scores The output of scores replaced with minimum score on
   * keys.
   * @parblock
   * The scores should be a `uint64_t` value. You can specify a value that
   * such as the timestamp of the key insertion, number of the key
   * occurrences, or another value to perform a custom eviction strategy.
   *
   * The @p scores should be `nullptr`, when the LRU eviction strategy is
   * applied.
   * @endparblock
   *
   * @param d_evicted_counter The number of elements evicted on NPU-accessible
   * memory. @notice The caller should guarantee it is set to `0` before
   * calling.
   * @param stream The CANN stream that is used to execute the operation.
   * @param unique_key If all keys in the same batch are unique.
   *
   * @param ignore_evict_strategy A boolean option indicating whether if
   * the insert_or_assign ignores the evict strategy of table with current
   * scores anyway. If true, it does not check whether the scores confroms to
   * the evict strategy. If false, it requires the scores follow the evict
   * strategy of table.
   */
  virtual void insert_and_evict(const size_type n,
                                const key_type* keys,          // (n)
                                const value_type* values,      // (n, DIM)
                                const score_type* scores,      // (n)
                                key_type* evicted_keys,        // (n)
                                value_type* evicted_values,    // (n, DIM)
                                score_type* evicted_scores,    // (n)
                                size_type* d_evicted_counter,  // (1)
                                aclrtStream stream = 0, bool unique_key = true,
                                bool ignore_evict_strategy = false) = 0;

  /**
   * @brief Insert new key-value-score tuples into the hash table.
   * If the key already exists, the values and scores are assigned new values.
   *
   * If the target bucket is full, the keys with minimum score will be
   * overwritten by new key unless the score of the new key is even less than
   * minimum score of the target bucket. The overwritten key with minimum
   * score will be evicted, with its values and score, to evicted_keys,
   * evicted_values, evcted_scores seperately in compact format.
   *
   * @param n Number of key-value-score tuples to insert or assign.
   * @param keys The keys to insert on NPU-accessible memory with shape
   * (n).
   * @param values The values to insert on NPU-accessible memory with
   * shape (n, DIM).
   * @param scores The scores to insert on NPU-accessible memory with shape
   * (n).
   * @param scores The scores to insert on NPU-accessible memory with shape
   * (n).
   * @params evicted_keys The output of keys replaced with minimum score.
   * @params evicted_values The output of values replaced with minimum score on
   * keys.
   * @params evicted_scores The output of scores replaced with minimum score on
   * keys.
   * @parblock
   * The scores should be a `uint64_t` value. You can specify a value that
   * such as the timestamp of the key insertion, number of the key
   * occurrences, or another value to perform a custom eviction strategy.
   *
   * The @p scores should be `nullptr`, when the LRU eviction strategy is
   * applied.
   * @endparblock
   *
   * @param stream The CANN stream that is used to execute the operation.
   * @param unique_key If all keys in the same batch are unique.
   *
   * @param ignore_evict_strategy A boolean option indicating whether if
   * the insert_or_assign ignores the evict strategy of table with current
   * scores anyway. If true, it does not check whether the scores confroms to
   * the evict strategy. If false, it requires the scores follow the evict
   * strategy of table.
   *
   * @return The number of elements evicted.
   */
  virtual size_type insert_and_evict(const size_type n,
                                     const key_type* keys,        // (n)
                                     const value_type* values,    // (n, DIM)
                                     const score_type* scores,    // (n)
                                     key_type* evicted_keys,      // (n)
                                     value_type* evicted_values,  // (n, DIM)
                                     score_type* evicted_scores,  // (n)
                                     aclrtStream stream = 0,
                                     bool unique_key = true,
                                     bool ignore_evict_strategy = false) = 0;

  /**
   * Searches for each key in @p keys in the hash table.
   * If the key is found and the corresponding value in @p accum_or_assigns is
   * `true`, the @p vectors_or_deltas is treated as a delta to the old
   * value, and the delta is added to the old value of the key.
   *
   * If the key is not found and the corresponding value in @p accum_or_assigns
   * is `false`, the @p vectors_or_deltas is treated as a new value and the
   * key-value pair is updated in the table directly.
   *
   * @note When the key is found and the value of @p accum_or_assigns is
   * `false`, or when the key is not found and the value of @p accum_or_assigns
   * is `true`, nothing is changed and this operation is ignored.
   * The algorithm assumes these situations occur while the key was modified or
   * removed by other processes just now.
   *
   * @param n The number of key-value-score tuples to process.
   * @param keys The keys to insert on NPU-accessible memory with shape (n).
   * @param value_or_deltas The values or deltas to insert on NPU-accessible
   * memory with shape (n, DIM).
   * @param accum_or_assigns The operation type with shape (n). A value of
   * `true` indicates to accum and `false` indicates to assign.
   * @param scores The scores to insert on NPU-accessible memory with shape (n).
   * @parblock
   * The scores should be a `uint64_t` value. You can specify a value that
   * such as the timestamp of the key insertion, number of the key
   * occurrences, or another value to perform a custom eviction strategy.
   *
   * The @p scores should be `nullptr`, when the LRU eviction strategy is
   * applied.
   * @endparblock
   *
   * @param stream The CANN stream that is used to execute the operation.
   *
   * @param ignore_evict_strategy A boolean option indicating whether if
   * the accum_or_assign ignores the evict strategy of table with current
   * scores anyway. If true, it does not check whether the scores confroms to
   * the evict strategy. If false, it requires the scores follow the evict
   * strategy of table.
   */
  virtual void accum_or_assign(const size_type n,
                               const key_type* keys,                // (n)
                               const value_type* value_or_deltas,   // (n, DIM)
                               const bool* accum_or_assigns,        // (n)
                               const score_type* scores = nullptr,  // (n)
                               aclrtStream stream = 0,
                               bool ignore_evict_strategy = false) = 0;

  /**
   * @brief Searches the hash table for the specified keys.
   * When a key is missing, the value in @p values and @p scores will be
   * inserted.
   *
   * @param n The number of key-value-score tuples to search or insert.
   * @param keys The keys to search on NPU-accessible memory with shape (n).
   * @param values The values to search on NPU-accessible memory with
   * shape (n, DIM).
   * @param scores The scores to search on NPU-accessible memory with shape (n).
   * @parblock
   * If @p scores is `nullptr`, the score for each key will not be returned.
   * @endparblock
   * @param stream The CANN stream that is used to execute the operation.
   * @param unique_key If all keys in the same batch are unique.
   *
   */
  virtual void find_or_insert(const size_type n, const key_type* keys,  // (n)
                              value_type* values,            // (n * DIM)
                              score_type* scores = nullptr,  // (n)
                              aclrtStream stream = 0, bool unique_key = true,
                              bool ignore_evict_strategy = false) = 0;

  /**
   * @brief Searches the hash table for the specified keys and returns address
   * of the values. When a key is missing, the value in @p values and @p scores
   * will be inserted.
   *
   * @warning This API returns internal addresses for high-performance but
   * thread-unsafe. The caller is responsible for guaranteeing data consistency.
   *
   * @param n The number of key-value-score tuples to search or insert.
   * @param keys The keys to search on NPU-accessible memory with shape (n).
   * @param values  The addresses of values to search on NPU-accessible memory
   * with shape (n).
   * @param founds The status that indicates if the keys are found on
   * @param scores The scores to search on NPU-accessible memory with shape (n).
   * @parblock
   * If @p scores is `nullptr`, the score for each key will not be returned.
   * @endparblock
   * @param stream The CANN stream that is used to execute the operation.
   * @param unique_key If all keys in the same batch are unique.
   * @param locked_key_ptrs If it isn't nullptr then the keys in the table will
   * be locked, and key's address will write to locked_key_ptrs. Using
   * unlock_keys to unlock these keys.
   *
   */
  virtual void find_or_insert(const size_type n, const key_type* keys,  // (n)
                              value_type** values,                      // (n)
                              bool* founds,                             // (n)
                              score_type* scores = nullptr,             // (n)
                              aclrtStream stream = 0, bool unique_key = true,
                              bool ignore_evict_strategy = false,
                              key_type** locked_key_ptrs = nullptr) = 0;

  /**
   * @brief
   * This function will lock the keys in the table and unexisted keys will be
   * ignored.
   *
   * @param n The number of keys in the table to be locked.
   * @param locked_key_ptrs The pointers of locked keys in the table with shape
   * (n).
   * @param keys The keys to search on NPU-accessible memory with shape (n).
   * @param succeededs The status that indicates if the lock operation is
   * succeed.
   * @param scores The scores of the input keys will set to scores if provided.
   * @param stream The CANN stream that is used to execute the operation.
   *
   */
  virtual void lock_keys(const size_type n,
                         key_type const* keys,        // (n)
                         key_type** locked_key_ptrs,  // (n)
                         bool* succeededs = nullptr,  // (n)
                         aclrtStream stream = 0,
                         score_type const* scores = nullptr) = 0;

  /**
   * @brief Using pointers to address the keys in the hash table and set them
   * to target keys.
   * This function will unlock the keys in the table which are locked by
   * the previous call to find_or_insert.
   *
   * @param n The number of keys in the table to be unlocked.
   * @param locked_key_ptrs The pointers of locked keys in the table with shape
   * (n).
   * @param keys The keys to search on NPU-accessible memory with shape (n).
   * @param succeededs The status that indicates if the unlock operation is
   * succeed.
   * @param stream The CANN stream that is used to execute the operation.
   *
   */
  virtual void unlock_keys(const size_type n,
                           key_type** locked_key_ptrs,  // (n)
                           const key_type* keys,        // (n)
                           bool* succeededs = nullptr,  // (n)
                           aclrtStream stream = 0) = 0;

  /**
   * @brief Assign new key-value-score tuples into the hash table.
   * If the key doesn't exist, the operation on the key will be ignored.
   *
   * @param n Number of key-value-score tuples to insert or assign.
   * @param keys The keys to insert on NPU-accessible memory with shape
   * (n).
   * @param values The values to insert on NPU-accessible memory with
   * shape (n, DIM).
   * @param scores The scores to insert on NPU-accessible memory with shape
   * (n).
   * @parblock
   * The scores should be a `uint64_t` value. You can specify a value that
   * such as the timestamp of the key insertion, number of the key
   * occurrences, or another value to perform a custom eviction strategy.
   *
   * The @p scores should be `nullptr`, when the LRU eviction strategy is
   * applied.
   * @endparblock
   *
   * @param stream The CANN stream that is used to execute the operation.
   *
   * @param unique_key If all keys in the same batch are unique.
   */
  virtual void assign(const size_type n,
                      const key_type* keys,                // (n)
                      const value_type* values,            // (n, DIM)
                      const score_type* scores = nullptr,  // (n)
                      aclrtStream stream = 0, bool unique_key = true) = 0;

  /**
   * @brief Assign new scores for keys.
   * If the key doesn't exist, the operation on the key will be ignored.
   *
   * @param n Number of key-score pairs to assign.
   * @param keys The keys to insert on NPU-accessible memory with shape
   * (n).
   * @parblock
   * The scores should be a `uint64_t` value. You can specify a value that
   * such as the timestamp of the key insertion, number of the key
   * occurrences, or another value to perform a custom eviction strategy.
   *
   * The @p scores should be `nullptr`, when the LRU eviction strategy is
   * applied.
   * @endparblock
   *
   * @param stream The CANN stream that is used to execute the operation.
   *
   * @param unique_key If all keys in the same batch are unique.
   */
  virtual void assign_scores(const size_type n,
                             const key_type* keys,                // (n)
                             const score_type* scores = nullptr,  // (n)
                             aclrtStream stream = 0,
                             bool unique_key = true) = 0;

  /**
   * @brief Alias of `assign_scores`.
   */
  virtual void assign(const size_type n,
                      const key_type* keys,                // (n)
                      const score_type* scores = nullptr,  // (n)
                      aclrtStream stream = 0, bool unique_key = true) = 0;

  /**
   * @brief Assign new values for each keys .
   * If the key doesn't exist, the operation on the key will be ignored.
   *
   * @param n Number of key-value pairs to assign.
   * @param keys The keys need to be operated, which must be on NPU-accessible
   * memory with shape (n).
   * @param values The values need to be updated, which must be on
   * NPU-accessible memory with shape (n, DIM).
   *
   * @param stream The CANN stream that is used to execute the operation.
   *
   * @param unique_key If all keys in the same batch are unique.
   */
  virtual void assign_values(const size_type n,
                             const key_type* keys,      // (n)
                             const value_type* values,  // (n, DIM)
                             aclrtStream stream = 0,
                             bool unique_key = true) = 0;
  /**
   * @brief Searches the hash table for the specified keys.
   *
   * @note When a key is missing, the value in @p values is not changed.
   *
   * @param n The number of key-value-score tuples to search.
   * @param keys The keys to search on NPU-accessible memory with shape (n).
   * @param values The values to search on NPU-accessible memory with
   * shape (n, DIM).
   * @param founds The status that indicates if the keys are found on
   * NPU-accessible memory with shape (n).
   * @param scores The scores to search on NPU-accessible memory with shape (n).
   * @parblock
   * If @p scores is `nullptr`, the score for each key will not be returned.
   * @endparblock
   * @param stream The CANN stream that is used to execute the operation.
   *
   */
  virtual void find(const size_type n, const key_type* keys,  // (n)
                    value_type* values,                       // (n, DIM)
                    bool* founds,                             // (n)
                    score_type* scores = nullptr,             // (n)
                    aclrtStream stream = 0) const = 0;

  /**
   * @brief Searches the hash table for the specified keys.
   *
   * @note When the searched keys are not hit, missed keys/indices/size can be
   * obtained.
   *
   * @param n The number of key-value-score tuples to search.
   * @param keys The keys to search on NPU-accessible memory with shape (n).
   * @param values The values to search on NPU-accessible memory with
   * shape (n, DIM).
   * @param missed_keys The missed keys to search on NPU-accessible memory with
   * shape (n).
   * @param missed_indices The missed indices to search on NPU-accessible memory
   * with shape (n).
   * @param missed_size The size of `missed_keys` and `missed_indices`.
   * @param scores The scores to search on NPU-accessible memory with shape (n).
   * @parblock
   * If @p scores is `nullptr`, the score for each key will not be returned.
   * @endparblock
   * @param stream The CANN stream that is used to execute the operation.
   */
  virtual void find(const size_type n, const key_type* keys,  // (n)
                    value_type* values,                       // (n, DIM)
                    key_type* missed_keys,                    // (n)
                    int* missed_indices,                      // (n)
                    int* missed_size,                         // scalar
                    score_type* scores = nullptr,             // (n)
                    aclrtStream stream = 0) const = 0;

  /**
   * @brief Searches the hash table for the specified keys and returns address
   * of the values.
   *
   * @note When a key is missing, the data in @p values won't change.
   * @warning This API returns internal addresses for high-performance but
   * thread-unsafe. The caller is responsible for guaranteeing data consistency.
   *
   * @param n The number of key-value-score tuples to search.
   * @param keys The keys to search on NPU-accessible memory with shape (n).
   * @param values The addresses of values to search on NPU-accessible memory
   * with shape (n).
   * @param founds The status that indicates if the keys are found on
   * NPU-accessible memory with shape (n).
   * @param scores The scores to search on NPU-accessible memory with shape (n).
   * @parblock
   * If @p scores is `nullptr`, the score for each key will not be returned.
   * @endparblock
   * @param stream The CANN stream that is used to execute the operation.
   * @param unique_key If all keys in the same batch are unique.
   *
   */
  virtual void find(const size_type n, const key_type* keys,  // (n)
                    value_type** values,                      // (n)
                    bool* founds,                             // (n)
                    score_type* scores = nullptr,             // (n)
                    aclrtStream stream = 0, bool unique_key = true) const = 0;

  /**
   * @brief Searches the hash table for the specified keys and returns address
   * of the values, and will update the scores.
   *
   * @note When a key is missing, the data in @p values won't change.
   * @warning This API returns internal addresses for high-performance but
   * thread-unsafe. The caller is responsible for guaranteeing data consistency.
   *
   * @param n The number of key-value-score tuples to search.
   * @param keys The keys to search on NPU-accessible memory with shape (n).
   * @param values The addresses of values to search on NPU-accessible memory
   * with shape (n).
   * @param founds The status that indicates if the keys are found on
   * NPU-accessible memory with shape (n).
   * @param scores The scores to search on NPU-accessible memory with shape (n).
   * @parblock
   * If @p scores is `nullptr`, the score for each key will not be returned.
   * @endparblock
   * @param stream The CANN stream that is used to execute the operation.
   * @param unique_key If all keys in the same batch are unique.
   *
   */
  virtual void find_and_update(const size_type n, const key_type* keys,  // (n)
                               value_type** values,                      // (n)
                               bool* founds,                             // (n)
                               score_type* scores = nullptr,             // (n)
                               aclrtStream stream = 0,
                               bool unique_key = true) = 0;

  /**
   * @brief Checks if there are elements with key equivalent to `keys` in the
   * table.
   *
   * @param n The number of `keys` to check.
   * @param keys The keys to search on NPU-accessible memory with shape (n).
   * @param founds The result that indicates if the keys are found, and should
   * be allocated by caller on NPU-accessible memory with shape (n).
   * @param stream The CANN stream that is used to execute the operation.
   *
   */
  virtual void contains(const size_type n, const key_type* keys,  // (n)
                        bool* founds,                             // (n)
                        aclrtStream stream = 0) const = 0;

  /**
   * @brief Removes specified elements from the hash table.
   *
   * @param n The number of keys to remove.
   * @param keys The keys to remove on NPU-accessible memory.
   * @param stream The CANN stream that is used to execute the operation.
   *
   */
  virtual void erase(const size_type n, const key_type* keys,
                     aclrtStream stream = 0) = 0;

  /**
   * @brief Removes all of the elements in the hash table with no release
   * object.
   */
  virtual void clear(aclrtStream stream = 0) = 0;

  /**
   * @brief Exports a certain number of the key-value-score tuples from the
   * hash table.
   *
   * @param n The maximum number of exported pairs.
   * @param offset The position of the key to search.
   * @param d_counter Accumulates amount of successfully exported values.
   * @param keys The keys to dump from NPU-accessible memory with shape (n).
   * @param values The values to dump from NPU-accessible memory with shape
   * (n, DIM).
   * @param scores The scores to search on NPU-accessible memory with shape (n).
   * @parblock
   * If @p scores is `nullptr`, the score for each key will not be returned.
   * @endparblock
   *
   * @param stream The CANN stream that is used to execute the operation.
   *
   * @return The number of elements dumped.
   *
   * @throw NpuException If the key-value size is too large for NPU
   * memory. Reducing the value for @p n is currently required if this exception
   * occurs.
   */
  virtual void export_batch(size_type n, const size_type offset,
                            size_type* d_counter,          // (1)
                            key_type* keys,                // (n)
                            value_type* values,            // (n, DIM)
                            score_type* scores = nullptr,  // (n)
                            aclrtStream stream = 0) const = 0;

  virtual size_type export_batch(const size_type n, const size_type offset,
                                 key_type* keys,                // (n)
                                 value_type* values,            // (n, DIM)
                                 score_type* scores = nullptr,  // (n)
                                 aclrtStream stream = 0) const = 0;

  /**
   * @brief Indicates if the hash table has no elements.
   *
   * @param stream The CANN stream that is used to execute the operation.
   * @return `true` if the table is empty and `false` otherwise.
   */
  virtual bool empty(aclrtStream stream = 0) const = 0;

  /**
   * @brief Returns the hash table size.
   *
   * @param stream The CANN stream that is used to execute the operation.
   * @return The table size.
   */
  virtual size_type size(aclrtStream stream = 0) const = 0;

  /**
   * @brief Returns the hash table capacity.
   *
   * @note The value that is returned might be less than the actual capacity of
   * the hash table because the hash table currently keeps the capacity to be
   * a power of 2 for performance considerations.
   *
   * @return The table capacity.
   */
  virtual size_type capacity() const = 0;

  /**
   * @brief Sets the number of buckets to the number that is needed to
   * accommodate at least @p new_capacity elements without exceeding the maximum
   * load factor. This method rehashes the hash table. Rehashing puts the
   * elements into the appropriate buckets considering that total number of
   * buckets has changed.
   *
   * @note If the value of @p new_capacity or double of @p new_capacity is
   * greater or equal than `options_.max_capacity`, the reserve does not perform
   * any change to the hash table.
   *
   * @param new_capacity The requested capacity for the hash table.
   * @param stream The CANN stream that is used to execute the operation.
   */
  virtual void reserve(const size_type new_capacity,
                       aclrtStream stream = 0) = 0;

  /**
   * @brief Returns the average number of elements per slot, that is, size()
   * divided by capacity().
   *
   * @param stream The CANN stream that is used to execute the operation.
   *
   * @return The load factor
   */
  virtual float load_factor(aclrtStream stream = 0) const = 0;

  /**
   * @brief Set max_capacity of the table.
   *
   * @param new_max_capacity The new expecting max_capacity. It must be power
   * of 2. Otherwise it will raise an error.
   */
  virtual void set_max_capacity(size_type new_max_capacity) = 0;

  /**
   * @brief Returns the dimension of the vectors.
   *
   * @return The dimension of the vectors.
   */
  virtual size_type dim() const noexcept = 0;

  /**
   * @brief Returns The length of each bucket.
   *
   * @return The length of each bucket.
   */
  virtual size_type max_bucket_size() const noexcept = 0;

  /**
   * @brief Returns the number of buckets in the table.
   *
   * @return The number of buckets in the table.
   */
  virtual size_type bucket_count() const noexcept = 0;

  /**
   * @brief Save keys, vectors, scores in table to file or files.
   *
   * @param file A BaseKVFile object defined the file format on host filesystem.
   * @param max_workspace_size Saving is conducted in chunks. This value denotes
   * the maximum amount of temporary memory to use when dumping the table.
   * Larger values *can* lead to higher performance.
   * @param stream The CANN stream used to execute the operation.
   *
   * @return Number of KV pairs saved to file.
   */
  virtual size_type save(BaseKVFile<K, V, S>* file,
                         const size_t max_workspace_size = 1L * 1024 * 1024,
                         aclrtStream stream = 0) const = 0;

  /**
   * @brief Load keys, vectors, scores from file to table.
   *
   * @param file An BaseKVFile defined the file format within filesystem.
   * @param max_workspace_size Loading is conducted in chunks. This value
   * denotes the maximum size of such chunks. Larger values *can* lead to higher
   * performance.
   * @param stream The CANN stream used to execute the operation.
   *
   * @return Number of keys loaded from file.
   */
  virtual size_type load(BaseKVFile<K, V, S>* file,
                         const size_t max_workspace_size = 1L * 1024 * 1024,
                         aclrtStream stream = 0) = 0;

  virtual void set_global_epoch(const uint64_t epoch) = 0;
};

/**
 * A HierarchicalKV hash table is a concurrent and hierarchical hash table that
 * is powered by NPUs and can use HBM and host memory as storage for key-value
 * pairs. Support for SSD storage is a future consideration.
 *
 * The `score` is introduced to define the importance of each key, the
 * larger, the more important, the less likely they will be evicted. Eviction
 * occurs automatically when a bucket is full. The keys with the minimum `score`
 * value are evicted first. In a customized eviction strategy, we recommend
 * using the timestamp or frequency of the key occurrence as the `score` value
 * for each key. You can also assign a special value to the `score` to
 * perform a customized eviction strategy.
 *
 * @note By default configuration, this class is thread-safe.
 *
 * @tparam K The data type of the key.
 * @tparam V The data type of the vector's item type.
 *         The item data type should be a basic data type of C++/CANN.
 * @tparam S The data type for `score`.
 *           The currently supported data type is only `uint64_t`.
 *
 */
template <typename K, typename V, typename S = uint64_t,
          int Strategy = EvictStrategy::kLru>
class HashTable : public HashTableBase<K, V, S> {
 public:
  using size_type = size_t;
  using key_type = K;
  using value_type = V;
  using score_type = S;
  static constexpr int evict_strategy = Strategy;

  using Pred = EraseIfPredict<key_type, score_type>;
  using allocator_type = BaseAllocator;

 private:
  using TableCore = npu::hkv::Table<key_type, value_type, score_type>;
  static constexpr unsigned int TILE_SIZE = 4;

  using DeviceMemoryPool = MemoryPool<DeviceAllocator<char>>;
  using HostMemoryPool = MemoryPool<HostAllocator<char>>;

 public:
  /**
   * @brief Default constructor for the hash table class.
   */
  HashTable() {
    static_assert((std::is_same<key_type, int64_t>::value ||
                   std::is_same<key_type, uint64_t>::value),
                  "The key_type must be int64_t or uint64_t.");

    static_assert(std::is_same<score_type, uint64_t>::value,
                  "The key_type must be uint64_t.");
  };

  /**
   * @brief Frees the resources used by the hash table and destroys the hash
   * table object.
   */
  ~HashTable() {
    if (initialized_) {
      initialized_ = false;
      destroy_table<key_type, value_type, score_type>(
          &table_, allocator_,
          bucket_memory_pool_manager_ != nullptr &&
              bucket_memory_pool_manager_->use_pool());
      allocator_->free(MemoryType::Device, d_table_);
      dev_mem_pool_.reset();
      host_mem_pool_.reset();
      bucket_memory_pool_manager_.reset();

      if (default_allocator_ && allocator_ != nullptr) {
        delete allocator_;
      }
    }
  }

 private:
  HashTable(const HashTable&) = delete;
  HashTable& operator=(const HashTable&) = delete;
  HashTable(HashTable&&) = delete;
  HashTable& operator=(HashTable&&) = delete;

 public:
  /**
   * @brief Initialize a hkv::HashTable.
   *
   * @param options The configuration options.
   */
  void init(const HashTableOptions& options,
            allocator_type* allocator = nullptr) {
    if (initialized_) {
      return;
    }
    auto platform = platform_ascendc::PlatformAscendCManager::GetInstance();
    HKV_CHECK(platform != nullptr, "get platform failed.");
    block_dim_ = platform->GetCoreNumAiv();
    options_ = options;
    HKV_CHECK(options.reserved_key_start_bit >= 0 &&
                  options.reserved_key_start_bit <= MAX_RESERVED_KEY_BIT,
              "options.reserved_key_start_bit should >= 0 and <= 62.");
    NPU_CHECK(init_reserved_keys(options.reserved_key_start_bit));

    default_allocator_ = (allocator == nullptr);
    allocator_ = (allocator == nullptr) ? (new DefaultAllocator()) : allocator;

    // thrust_allocator_.set_allocator(allocator_);

    if (options_.device_id >= 0) {
      NPU_CHECK(aclrtSetDevice(options_.device_id));
    } else {
      NPU_CHECK(aclrtGetDevice(&(options_.device_id)));
    }

    HKV_CHECK((options_.max_bucket_size >= 16 ),
              "Bucket size should be greater than or equal to 16");
    HKV_CHECK(ispow2(static_cast<uint32_t>(options_.max_bucket_size)),
              "Bucket size should be the pow of 2");
    HKV_CHECK(ispow2(static_cast<uint32_t>(options_.num_of_buckets_per_alloc)),
              "Then `num_of_buckets_per_alloc` should be the pow of 2");
    HKV_CHECK(options_.init_capacity >=
                  options_.num_of_buckets_per_alloc * options_.max_bucket_size,
              "Then `num_of_buckets_per_alloc` must be equal or less than "
              "initial required buckets number");

    options_.block_size = SAFE_GET_BLOCK_SIZE(options_.block_size);

    HKV_CHECK(
        (((options_.max_bucket_size * (sizeof(key_type) + sizeof(score_type))) %
          128) == 0),
        "Storage size of keys and scores in one bucket should be the mutiple "
        "of cache line size");

    // Initialize bucket memory pool manager and construct table.
    bucket_memory_pool_manager_ =
        std::make_unique<BucketMemoryPoolManager<key_type, value_type, score_type>>();
    bucket_memory_pool_manager_->initialize(options_);
    create_table<key_type, value_type, score_type>(
        &table_, allocator_, block_dim_, options_.dim, options_.init_capacity,
        options_.max_capacity, options_.max_hbm_for_vectors,
        options_.max_bucket_size, options_.num_of_buckets_per_alloc,
        /* tile_size */ 32, /* primary */ true,
        bucket_memory_pool_manager_.get());
    options_.block_size = SAFE_GET_BLOCK_SIZE(options_.block_size);
    reach_max_capacity_ = (options_.init_capacity * 2 > options_.max_capacity);
    HKV_CHECK((!(options_.io_by_cpu && options_.max_hbm_for_vectors != 0)),
              "[HierarchicalKV] `io_by_cpu` should not be true when "
              "`max_hbm_for_vectors` is not 0!");
    allocator_->alloc(MemoryType::Device, (void**)&(d_table_),
                      sizeof(TableCore));

    sync_table_configuration();

    uint32_t move_byte_per_value = table_->dim * sizeof(V);
    value_move_opt_ = GetValueMoveOpt(move_byte_per_value);

    // Create memory pools.
    dev_mem_pool_ = std::make_unique<MemoryPool<DeviceAllocator<char>>>(
        options_.device_memory_pool, allocator_);
    host_mem_pool_ = std::make_unique<MemoryPool<HostAllocator<char>>>(
        options_.host_memory_pool, allocator_);

    NPU_CHECK(aclrtSynchronizeDevice());

    initialized_ = true;
    NpuCheckError();
  }

  /**
   * @brief Insert new key-value-score tuples into the hash table.
   * If the key already exists, the values and scores are assigned new values.
   *
   * If the target bucket is full, the keys with minimum score will be
   * overwritten by new key unless the score of the new key is even less than
   * minimum score of the target bucket.
   *
   * @param n Number of key-value-score tuples to insert or assign.
   * @param keys The keys to insert on NPU-accessible memory with shape
   * (n).
   * @param values The values to insert on NPU-accessible memory with
   * shape (n, DIM).
   * @param scores The scores to insert on NPU-accessible memory with shape
   * (n).
   * @parblock
   * The scores should be a `uint64_t` value. You can specify a value that
   * such as the timestamp of the key insertion, number of the key
   * occurrences, or another value to perform a custom eviction strategy.
   *
   * The @p scores should be `nullptr`, when the LRU eviction strategy is
   * applied.
   * @endparblock
   *
   * @param stream The CANN stream that is used to execute the operation.
   * @param unique_key If all keys in the same batch are unique.
   *
   * @param ignore_evict_strategy A boolean option indicating whether if
   * the insert_or_assign ignores the evict strategy of table with current
   * scores anyway. If true, it does not check whether the scores conforms to
   * the evict strategy. If false, it requires the scores follow the evict
   * strategy of table.
   */
  void insert_or_assign(const size_type n,
                        const key_type* keys,                // (n)
                        const value_type* values,            // (n, DIM)
                        const score_type* scores = nullptr,  // (n)
                        aclrtStream stream = 0, bool unique_key = true,
                        bool ignore_evict_strategy = false) {
    if (ignore_evict_strategy) {
      insert_or_assign_impl<EvictStrategy::kCustomized>(
          n, keys, values, scores, stream, unique_key, ignore_evict_strategy);
    } else {
      insert_or_assign_impl<evict_strategy>(n, keys, values, scores, stream,
                                            unique_key, ignore_evict_strategy);
    }
  }

  template <int evict_strategy_>
  void insert_or_assign_impl(const size_type n,
                             const key_type* keys,      // (n)
                             const value_type* values,  // (n, DIM)
                             const score_type* scores,  // (n)
                             aclrtStream stream, bool unique_key,
                             bool ignore_evict_strategy) {
    if (n == 0) {
      return;
    }

    while (!reach_max_capacity_ &&
           fast_load_factor(n, stream) > options_.max_load_factor) {
      reserve(capacity() * 2, stream);
    }

    if (!ignore_evict_strategy) {
      check_evict_strategy(scores);
    }

    uint64_t n_align_warp = ((n + WARP_SIZE - 1) / WARP_SIZE) * WARP_SIZE;
    if (value_move_opt_.is_large_size) {
      ACLRT_LAUNCH_KERNEL(insert_or_assign_kernel_with_thread_1024)(
          block_dim_, stream, table_->buckets, table_->buckets_size,
          table_->capacity, table_->bucket_max_size, value_move_opt_.dim,
          const_cast<key_type*>(keys), const_cast<value_type*>(values),
          const_cast<score_type*>(scores), n, global_epoch_, evict_strategy_,
          value_move_opt_.size, table_->max_bucket_shift, table_->capacity_divisor_magic,
          table_->capacity_divisor_shift, n_align_warp, value_move_opt_.cg_size);
    } else {
      ACLRT_LAUNCH_KERNEL(insert_or_assign_kernel)(
          block_dim_, stream, table_->buckets, table_->buckets_size,
          table_->capacity, table_->bucket_max_size, value_move_opt_.dim,
          const_cast<key_type*>(keys), const_cast<value_type*>(values),
          const_cast<score_type*>(scores), n, global_epoch_, evict_strategy_,
          value_move_opt_.size, table_->max_bucket_shift, table_->capacity_divisor_magic,
          table_->capacity_divisor_shift, n_align_warp, value_move_opt_.cg_size);
    }

    NpuCheckError();
  }

  /**
   * @brief Insert new key-value-score tuples into the hash table.
   * If the key already exists, the values and scores are assigned new values.
   *
   * If the target bucket is full, the keys with minimum score will be
   * overwritten by new key unless the score of the new key is even less than
   * minimum score of the target bucket. The overwritten key with minimum
   * score will be evicted, with its values and score, to evicted_keys,
   * evicted_values, evcted_scores seperately in compact format.
   *
   * @param n Number of key-value-score tuples to insert or assign.
   * @param keys The keys to insert on NPU-accessible memory with shape
   * (n).
   * @param values The values to insert on NPU-accessible memory with
   * shape (n, DIM).
   * @param scores The scores to insert on NPU-accessible memory with shape
   * (n).
   * @params evicted_keys The output of keys replaced with minimum score.
   * @params evicted_values The output of values replaced with minimum score on
   * keys.
   * @params evicted_scores The output of scores replaced with minimum score on
   * keys.
   * @parblock
   * The scores should be a `uint64_t` value. You can specify a value that
   * such as the timestamp of the key insertion, number of the key
   * occurrences, or another value to perform a custom eviction strategy.
   *
   * The @p scores should be `nullptr`, when the LRU eviction strategy is
   * applied.
   * @endparblock
   *
   * @param d_evicted_counter The number of elements evicted on NPU-accessible
   * memory. @notice The caller should guarantee it is set to `0` before
   * calling.
   * @param stream The CANN stream that is used to execute the operation.
   * @param unique_key If all keys in the same batch are unique.
   *
   * @param ignore_evict_strategy A boolean option indicating whether if
   * the insert_or_assign ignores the evict strategy of table with current
   * scores anyway. If true, it does not check whether the scores confroms to
   * the evict strategy. If false, it requires the scores follow the evict
   * strategy of table.
   */
  void insert_and_evict(const size_type n,
                        const key_type* keys,          // (n)
                        const value_type* values,      // (n, DIM)
                        const score_type* scores,      // (n)
                        key_type* evicted_keys,        // (n)
                        value_type* evicted_values,    // (n, DIM)
                        score_type* evicted_scores,    // (n)
                        size_type* d_evicted_counter,  // (1)
                        aclrtStream stream = 0, bool unique_key = true,
                        bool ignore_evict_strategy = false) {
    if (n == 0) {
      return;
    }
    if (keys == nullptr || values == nullptr || evicted_keys == nullptr || 
        evicted_values == nullptr || d_evicted_counter == nullptr) {
 	    return;
 	  }

    while (!reach_max_capacity_ &&
           fast_load_factor(n, stream) > options_.max_load_factor) {
      reserve(capacity() * 2, stream);
    }

    if (!ignore_evict_strategy) {
      check_evict_strategy(scores);
    }

    // Currently only need eviction when using HashTable as HBM cache.
    if (!is_fast_mode()) {
      throw std::runtime_error("Only allow insert_and_evict in pure HBM mode.");
    }

    uint64_t n_align_warp = ((n + WARP_SIZE - 1) / WARP_SIZE) * WARP_SIZE;

    ACLRT_LAUNCH_KERNEL(insert_and_evict_kernel)(
      block_dim_, stream, table_->buckets, table_->buckets_size,
      table_->capacity, options_.max_bucket_size, value_move_opt_.dim,
      const_cast<key_type*>(keys), const_cast<value_type*>(values),
      const_cast<score_type*>(scores), evicted_keys, evicted_values,
      evicted_scores, d_evicted_counter, n, global_epoch_, evict_strategy,
      value_move_opt_.size, table_->max_bucket_shift, table_->capacity_divisor_magic,
      table_->capacity_divisor_shift, n_align_warp, value_move_opt_.cg_size);

    NpuCheckError();
    return;
  }

  /**
   * @brief Insert new key-value-score tuples into the hash table.
   * If the key already exists, the values and scores are assigned new values.
   *
   * If the target bucket is full, the keys with minimum score will be
   * overwritten by new key unless the score of the new key is even less than
   * minimum score of the target bucket. The overwritten key with minimum
   * score will be evicted, with its values and score, to evicted_keys,
   * evicted_values, evcted_scores seperately in compact format.
   *
   * @param n Number of key-value-score tuples to insert or assign.
   * @param keys The keys to insert on NPU-accessible memory with shape
   * (n).
   * @param values The values to insert on NPU-accessible memory with
   * shape (n, DIM).
   * @param scores The scores to insert on NPU-accessible memory with shape
   * (n).
   * @params evicted_keys The output of keys replaced with minimum score.
   * @params evicted_values The output of values replaced with minimum score on
   * keys.
   * @params evicted_scores The output of scores replaced with minimum score on
   * keys.
   * @parblock
   * The scores should be a `uint64_t` value. You can specify a value that
   * such as the timestamp of the key insertion, number of the key
   * occurrences, or another value to perform a custom eviction strategy.
   *
   * The @p scores should be `nullptr`, when the LRU eviction strategy is
   * applied.
   * @endparblock
   *
   * @param stream The CANN stream that is used to execute the operation.
   * @param unique_key If all keys in the same batch are unique.
   *
   * @param ignore_evict_strategy A boolean option indicating whether if
   * the insert_or_assign ignores the evict strategy of table with current
   * scores anyway. If true, it does not check whether the scores confroms to
   * the evict strategy. If false, it requires the scores follow the evict
   * strategy of table.
   *
   * @return The number of elements evicted.
   */
  size_type insert_and_evict(const size_type n,
                             const key_type* keys,        // (n)
                             const value_type* values,    // (n, DIM)
                             const score_type* scores,    // (n)
                             key_type* evicted_keys,      // (n)
                             value_type* evicted_values,  // (n, DIM)
                             score_type* evicted_scores,  // (n)
                             aclrtStream stream = 0, bool unique_key = true,
                             bool ignore_evict_strategy = false) {
    if (n == 0) {
      return 0;
    }
    auto dev_ws{dev_mem_pool_->get_workspace<1>(sizeof(size_type), stream)};
    auto d_evicted_counter{dev_ws.get<size_type*>(0)};
    NPU_CHECK(aclrtMemset(d_evicted_counter, sizeof(size_type), 0, sizeof(size_type)));

    insert_and_evict(n, keys, values, scores, evicted_keys, evicted_values, evicted_scores, d_evicted_counter,
                     stream, unique_key, ignore_evict_strategy);

    size_type h_evicted_counter = 0;
    NPU_CHECK(aclrtMemcpyAsync(&h_evicted_counter, sizeof(size_type),
              d_evicted_counter, sizeof(size_type),
              ACL_MEMCPY_DEVICE_TO_HOST, stream));
    NPU_CHECK(aclrtSynchronizeStream(stream));
    NpuCheckError();
    return h_evicted_counter;
  }

  /**
   * Searches for each key in @p keys in the hash table.
   * If the key is found and the corresponding value in @p accum_or_assigns is
   * `true`, the @p vectors_or_deltas is treated as a delta to the old
   * value, and the delta is added to the old value of the key.
   *
   * If the key is not found and the corresponding value in @p accum_or_assigns
   * is `false`, the @p vectors_or_deltas is treated as a new value and the
   * key-value pair is updated in the table directly.
   *
   * @note When the key is found and the value of @p accum_or_assigns is
   * `false`, or when the key is not found and the value of @p accum_or_assigns
   * is `true`, nothing is changed and this operation is ignored.
   * The algorithm assumes these situations occur while the key was modified or
   * removed by other processes just now.
   *
   * @param n The number of key-value-score tuples to process.
   * @param keys The keys to insert on NPU-accessible memory with shape (n).
   * @param value_or_deltas The values or deltas to insert on NPU-accessible
   * memory with shape (n, DIM).
   * @param accum_or_assigns The operation type with shape (n). A value of
   * `true` indicates to accum and `false` indicates to assign.
   * @param scores The scores to insert on NPU-accessible memory with shape (n).
   * @parblock
   * The scores should be a `uint64_t` value. You can specify a value that
   * such as the timestamp of the key insertion, number of the key
   * occurrences, or another value to perform a custom eviction strategy.
   *
   * The @p scores should be `nullptr`, when the LRU eviction strategy is
   * applied.
   * @endparblock
   *
   * @param stream The CANN stream that is used to execute the operation.
   *
   * @param ignore_evict_strategy A boolean option indicating whether if
   * the accum_or_assign ignores the evict strategy of table with current
   * scores anyway. If true, it does not check whether the scores confroms to
   * the evict strategy. If false, it requires the scores follow the evict
   * strategy of table.
   */
  void accum_or_assign(const size_type n,
                       const key_type* keys,                // (n)
                       const value_type* value_or_deltas,   // (n, DIM)
                       const bool* accum_or_assigns,        // (n)
                       const score_type* scores = nullptr,  // (n)
                       aclrtStream stream = 0,
                       bool ignore_evict_strategy = false) {
    if (n == 0) {
      return;
    }

    std::cout << "[Unsupport accum_or_assign yet]\n";
    NpuCheckError();
  }

  /**
   * @brief Searches the hash table for the specified keys.
   * When a key is missing, the value in @p values and @p scores will be
   * inserted.
   *
   * @param n The number of key-value-score tuples to search or insert.
   * @param keys The keys to search on NPU-accessible memory with shape (n).
   * @param values The values to search on NPU-accessible memory with
   * shape (n, DIM).
   * @param scores The scores to search on NPU-accessible memory with shape (n).
   * @parblock
   * If @p scores is `nullptr`, the score for each key will not be returned.
   * @endparblock
   * @param stream The CANN stream that is used to execute the operation.
   * @param unique_key If all keys in the same batch are unique.
   *
   */
  void find_or_insert(const size_type n, const key_type* keys,  // (n)
                      value_type* values,                       // (n * DIM)
                      score_type* scores = nullptr,             // (n)
                      aclrtStream stream = 0, bool unique_key = true,
                      bool ignore_evict_strategy = false) {
    if (n == 0) {
      return;
    }

    std::cout << "[Unsupport find_or_insert yet]\n";

    NpuCheckError();
  }

  /**
   * @brief Searches the hash table for the specified keys and returns address
   * of the values. When a key is missing, the value in @p values and @p scores
   * will be inserted.
   *
   * @warning This API returns internal addresses for high-performance but
   * thread-unsafe. The caller is responsible for guaranteeing data consistency.
   *
   * @param n The number of key-value-score tuples to search or insert.
   * @param keys The keys to search on NPU-accessible memory with shape (n).
   * @param values  The addresses of values to search on NPU-accessible memory
   * with shape (n).
   * @param founds The status that indicates if the keys are found on
   * @param scores The scores to search on NPU-accessible memory with shape (n).
   * @parblock
   * If @p scores is `nullptr`, the score for each key will not be returned.
   * @endparblock
   * @param stream The CANN stream that is used to execute the operation.
   * @param unique_key If all keys in the same batch are unique.
   * @param locked_key_ptrs If it isn't nullptr then the keys in the table will
   * be locked, and key's address will write to locked_key_ptrs. Using
   * unlock_keys to unlock these keys.
   *
   */
  void find_or_insert(const size_type n, const key_type* keys,  // (n)
                      value_type** values,                      // (n)
                      bool* founds,                             // (n)
                      score_type* scores = nullptr,             // (n)
                      aclrtStream stream = 0, bool unique_key = true,
                      bool ignore_evict_strategy = false,
                      key_type** locked_key_ptrs = nullptr) {
    if (n == 0) {
      return;
    }

    while (!reach_max_capacity_ &&
           fast_load_factor(n, stream) > options_.max_load_factor) {
      reserve(capacity() * 2, stream);
    }

    if (!ignore_evict_strategy) {
      check_evict_strategy(scores);
    }

    uint64_t n_align_warp = ((n + WARP_SIZE - 1) / WARP_SIZE) * WARP_SIZE;
    ACLRT_LAUNCH_KERNEL(find_or_insert_ptr_kernel_v2)
    (block_dim_, stream, table_->buckets, table_->buckets_size,
     table_->buckets_num, options_.max_bucket_size, value_move_opt_.dim,
     (void*)keys, values, scores, locked_key_ptrs, n, founds, global_epoch_,
     evict_strategy, value_move_opt_.size, table_->max_bucket_shift,
     table_->capacity_divisor_magic, table_->capacity_divisor_shift,
     n_align_warp, table_->capacity);

    NpuCheckError();
  }

  /**
   * @brief
   * This function will lock the keys in the table and unexisted keys will be
   * ignored.
   *
   * @param n The number of keys in the table to be locked.
   * @param locked_key_ptrs The pointers of locked keys in the table with shape
   * (n).
   * @param keys The keys to search on NPU-accessible memory with shape (n).
   * @param success The status that indicates if the lock operation is
   * succeed.
   * @param stream The CANN stream that is used to execute the operation.
   * @param scores The scores of the input keys will set to scores if provided.
   *
   */
  void lock_keys(const size_type n,
                 key_type const* keys,        // (n)
                 key_type** locked_key_ptrs,  // (n)
                 bool* success = nullptr,     // (n)
                 aclrtStream stream = 0, score_type const* scores = nullptr) {
    if (n == 0) {
      return;
    }

    std::cout << "[Unsupport lock_keys yet]\n";
    NpuCheckError();
  }

  /**
   * @brief Using pointers to address the keys in the hash table and set them
   * to target keys.
   * This function will unlock the keys in the table which are locked by
   * the previous call to find_or_insert.
   *
   * @param n The number of keys in the table to be unlocked.
   * @param locked_key_ptrs The pointers of locked keys in the table with shape
   * (n).
   * @param keys The keys to search on NPU-accessible memory with shape (n).
   * @param success The status that indicates if the unlock operation is
   * succeed.
   * @param stream The CANN stream that is used to execute the operation.
   *
   */
  void unlock_keys(const size_type n, key_type** locked_key_ptrs,  // (n)
                   const key_type* keys,                           // (n)
                   bool* success = nullptr,                        // (n)
                   aclrtStream stream = 0) {
    if (n == 0) {
      return;
    }

    std::cout << "[Unsupport unlock_keys yet]\n";
  }

  /**
   * @brief Assign new key-value-score tuples into the hash table.
   * If the key doesn't exist, the operation on the key will be ignored.
   *
   * @param n Number of key-value-score tuples to insert or assign.
   * @param keys The keys to insert on NPU-accessible memory with shape
   * (n).
   * @param values The values to insert on NPU-accessible memory with
   * shape (n, DIM).
   * @param scores The scores to insert on NPU-accessible memory with shape
   * (n).
   * @parblock
   * The scores should be a `uint64_t` value. You can specify a value that
   * such as the timestamp of the key insertion, number of the key
   * occurrences, or another value to perform a custom eviction strategy.
   *
   * The @p scores should be `nullptr`, when the LRU eviction strategy is
   * applied.
   * @endparblock
   *
   * @param stream The CANN stream that is used to execute the operation.
   *
   * @param unique_key If all keys in the same batch are unique.
   */
  void assign(const size_type n,
              const key_type* keys,                // (n)
              const value_type* values,            // (n, DIM)
              const score_type* scores = nullptr,  // (n)
              aclrtStream stream = 0, bool unique_key = true) {
    if (n == 0) {
      return;
    }

    check_evict_strategy(scores);

    std::cout << "[Unsupport assign yet]\n";

    NpuCheckError();
  }

  /**
   * @brief Assign new scores for keys.
   * If the key doesn't exist, the operation on the key will be ignored.
   *
   * @param n Number of key-score pairs to assign.
   * @param keys The keys to insert on NPU-accessible memory with shape
   * (n).
   * @parblock
   * The scores should be a `uint64_t` value. You can specify a value that
   * such as the timestamp of the key insertion, number of the key
   * occurrences, or another value to perform a custom eviction strategy.
   *
   * The @p scores should be `nullptr`, when the LRU eviction strategy is
   * applied.
   * @endparblock
   *
   * @param stream The CANN stream that is used to execute the operation.
   *
   * @param unique_key If all keys in the same batch are unique.
   */
  void assign_scores(const size_type n,
    const key_type* keys,                // (n)
    const score_type* scores = nullptr,  // (n)
    aclrtStream stream = 0, bool unique_key = true) {
    if (n == 0) {
      return;
    }

    check_evict_strategy(scores);

    constexpr uint32_t MinBucketCapacityFilter = sizeof(VecD_Load) / sizeof(D);
    if (unique_key && options_.max_bucket_size >= MinBucketCapacityFilter) {
      ACLRT_LAUNCH_KERNEL(assign_scores_kernel_with_filter)
      (block_dim_, stream, table_->buckets, table_->capacity, options_.max_bucket_size,
      options_.dim, (void*)keys, (void*)scores, n, global_epoch_, evict_strategy, value_size_,
      table_->max_bucket_shift, table_->capacity_divisor_magic, table_->capacity_divisor_shift);
    } else {
      throw std::runtime_error(
        "Not support update score when keys are not unique or bucket "
        "capacity is smaller than " + std::to_string(MinBucketCapacityFilter) + ".");
  }
    NpuCheckError();
  }

  /**
   * @brief Alias of `assign_scores`.
   */
  void assign(const size_type n,
              const key_type* keys,                // (n)
              const score_type* scores = nullptr,  // (n)
              aclrtStream stream = 0, bool unique_key = true) {
    assign_scores(n, keys, scores, stream, unique_key);
  }

  /**
   * @brief Assign new values for each keys .
   * If the key doesn't exist, the operation on the key will be ignored.
   *
   * @param n Number of key-value pairs to assign.
   * @param keys The keys need to be operated, which must be on NPU-accessible
   * memory with shape (n).
   * @param values The values need to be updated, which must be on
   * NPU-accessible memory with shape (n, DIM).
   *
   * @param stream The CANN stream that is used to execute the operation.
   *
   * @param unique_key If all keys in the same batch are unique.
   */
  void assign_values(const size_type n,
                     const key_type* keys,      // (n)
                     const value_type* values,  // (n, DIM)
                     aclrtStream stream = 0, bool unique_key = true) {
    if (n == 0) {
      return;
    }

    std::cout << "[Unsupport assign_values yet]\n";

    NpuCheckError();
  }

  /**
   * @brief Searches the hash table for the specified keys.
   *
   * @note When a key is missing, the value in @p values is not changed.
   *
   * @param n The number of key-value-score tuples to search.
   * @param keys The keys to search on NPU-accessible memory with shape (n).
   * @param values The values to search on NPU-accessible memory with
   * shape (n, DIM).
   * @param founds The status that indicates if the keys are found on
   * NPU-accessible memory with shape (n).
   * @param scores The scores to search on NPU-accessible memory with shape (n).
   * @parblock
   * If @p scores is `nullptr`, the score for each key will not be returned.
   * @endparblock
   * @param stream The CANN stream that is used to execute the operation.
   *
   */
  void find(const size_type n, const key_type* keys,  // (n)
            value_type* values,                       // (n, DIM)
            bool* founds,                             // (n)
            score_type* scores = nullptr,             // (n)
            aclrtStream stream = 0) const {
    if (n == 0) {
      return;
    }

    std::cout << "[Unsupport find yet]\n";

    NpuCheckError();
  }

  /**
   * @brief Searches the hash table for the specified keys.
   *
   * @note When the searched keys are not hit, missed keys/indices/size can be
   * obtained.
   *
   * @param n The number of key-value-score tuples to search.
   * @param keys The keys to search on NPU-accessible memory with shape (n).
   * @param values The values to search on NPU-accessible memory with
   * shape (n, DIM).
   * @param missed_keys The missed keys to search on NPU-accessible memory with
   * shape (n).
   * @param missed_indices The missed indices to search on NPU-accessible memory
   * with shape (n).
   * @param missed_size The size of `missed_keys` and `missed_indices`.
   * @param scores The scores to search on NPU-accessible memory with shape (n).
   * @parblock
   * If @p scores is `nullptr`, the score for each key will not be returned.
   * @endparblock
   * @param stream The CANN stream that is used to execute the operation.
   */
  void find(const size_type n, const key_type* keys,  // (n)
            value_type* values,                       // (n, DIM)
            key_type* missed_keys,                    // (n)
            int* missed_indices,                      // (n)
            int* missed_size,                         // scalar
            score_type* scores = nullptr,             // (n)
            aclrtStream stream = 0) const {
    if (n == 0) {
      return;
    }

    std::cout << "[Unsupport find yet]\n";

    NpuCheckError();
  }

  /**
   * @brief Searches the hash table for the specified keys and returns address
   * of the values.
   *
   * @note When a key is missing, the data in @p values won't change.
   * @warning This API returns internal addresses for high-performance but
   * thread-unsafe. The caller is responsible for guaranteeing data consistency.
   *
   * @param n The number of key-value-score tuples to search.
   * @param keys The keys to search on NPU-accessible memory with shape (n).
   * @param values The addresses of values to search on NPU-accessible memory
   * with shape (n).
   * @param founds The status that indicates if the keys are found on
   * NPU-accessible memory with shape (n).
   * @param scores The scores to search on NPU-accessible memory with shape (n).
   * @parblock
   * If @p scores is `nullptr`, the score for each key will not be returned.
   * @endparblock
   * @param stream The CANN stream that is used to execute the operation.
   * @param unique_key If all keys in the same batch are unique.
   *
   */
  void find(const size_type n, const key_type* keys,  // (n)
            value_type** values,                      // (n)
            bool* founds,                             // (n)
            score_type* scores = nullptr,             // (n)
            aclrtStream stream = 0, bool unique_key = true) const {
    if (n == 0) {
      return;
    }
    ACLRT_LAUNCH_KERNEL(find_ptr_with_digest_kernel)(
      block_dim_, stream, table_->buckets, table_->capacity, table_->buckets_num, options_.max_bucket_size,
      options_.dim, const_cast<key_type*>(keys), values, scores, founds, n, global_epoch_, value_size_,
      table_->max_bucket_shift, table_->capacity_divisor_magic, table_->capacity_divisor_shift);
    NpuCheckError();
  }

  /**
   * @brief Searches the hash table for the specified keys and returns address
   * of the values, and will update the scores.
   *
   * @note When a key is missing, the data in @p values won't change.
   * @warning This API returns internal addresses for high-performance but
   * thread-unsafe. The caller is responsible for guaranteeing data consistency.
   *
   * @param n The number of key-value-score tuples to search.
   * @param keys The keys to search on NPU-accessible memory with shape (n).
   * @param values The addresses of values to search on NPU-accessible memory
   * with shape (n).
   * @param founds The status that indicates if the keys are found on
   * NPU-accessible memory with shape (n).
   * @param scores The scores to search on NPU-accessible memory with shape (n).
   * @parblock
   * If @p scores is `nullptr`, the score for each key will not be returned.
   * @endparblock
   * @param stream The CANN stream that is used to execute the operation.
   * @param unique_key If all keys in the same batch are unique.
   *
   */
  void find_and_update(const size_type n, const key_type* keys,  // (n)
                       value_type** values,                      // (n)
                       bool* founds,                             // (n)
                       score_type* scores = nullptr,             // (n)
                       aclrtStream stream = 0, bool unique_key = true) {
    if (n == 0) {
      return;
    }

    /* api_lock暂不支持，先注释标注下，后续支持后放开
    std::unique_ptr<read_shared_lock> lock_ptr;
    if (options_.api_lock) {
      lock_ptr = std::make_unique<read_shared_lock>(mutex_, stream);
    }
    */
    check_evict_strategy(scores);
    constexpr uint32_t MinBucketCapacityFilter = sizeof(VecD_Load) / sizeof(D);
    if (unique_key && options_.max_bucket_size >= MinBucketCapacityFilter) {
      ACLRT_LAUNCH_KERNEL(find_and_update_kernel_with_filter)
      (block_dim_, stream, table_->buckets, table_->capacity, options_.max_bucket_size,
        options_.dim, (void*)keys, values, scores, founds, n, true, global_epoch_, evict_strategy, value_size_,
        table_->max_bucket_shift, table_->capacity_divisor_magic, table_->capacity_divisor_shift);
    } else {
      throw std::runtime_error(
          "Not support update score when keys are not unique or bucket "
          "capacity is small.");
    }
    NpuCheckError();
  }

  /**
   * @brief Checks if there are elements with key equivalent to `keys` in the
   * table.
   *
   * @param n The number of `keys` to check.
   * @param keys The keys to search on NPU-accessible memory with shape (n).
   * @param founds The result that indicates if the keys are found, and should
   * be allocated by caller on NPU-accessible memory with shape (n).
   * @param stream The CANN stream that is used to execute the operation.
   *
   */
  void contains(const size_type n, const key_type* keys,  // (n)
                bool* founds,                             // (n)
                aclrtStream stream = 0) const {
    if (n == 0) {
      return;
    }

    std::cout << "[Unsupport contains yet]\n";
    NpuCheckError();
  }

  /**
   * @brief Removes specified elements from the hash table.
   *
   * @param n The number of keys to remove.
   * @param keys The keys to remove on NPU-accessible memory.
   * @param stream The CANN stream that is used to execute the operation.
   *
   */
  void erase(const size_type n, const key_type* keys, aclrtStream stream = 0) {
    if (n == 0) {
      return;
    }

    std::cout << "[Unsupport erase yet]\n";

    NpuCheckError();
    return;
  }

  /**
   * @brief Erases all elements that satisfy the predicate @p pred from the
   * hash table.
   *
   * @tparam PredFunctor The predicate template <typename K, typename S>
   * function with operator signature (bool*)(const K&, const S&, const K&,
   * const threshold) that returns `true` if the element should be erased. The
   * value for @p pred should be a function with type `Pred` defined like the
   * following example:
   *
   *    ```
   *    template <class K, class S>
   *    struct EraseIfPredFunctor {
   *      __forceinline__ __simt_callee__ bool operator()(const K& key,
   *                                                 S& score,
   *                                                 const K& pattern,
   *                                                 const S& threshold) {
   *        return ((key & 0x1 == pattern) && (score < threshold));
   *      }
   *    };
   *    ```
   *
   * @param pattern The third user-defined argument to @p pred with key_type
   * type.
   * @param threshold The fourth user-defined argument to @p pred with
   * score_type type.
   * @param stream The CANN stream that is used to execute the operation.
   *
   * @return The number of elements removed.
   *
   */
  template <template <typename, typename> class PredFunctor>
  size_type erase_if(const key_type& pattern, const score_type& threshold,
                     aclrtStream stream = 0) {
    std::cout << "[Unsupport erase_if yet]\n";
    NpuCheckError();
    return 0;
  }

  /**
   * @brief Erase the key-value-score tuples which match @tparam PredFunctor.
   * @param pred A functor with template <K, V, S> defined an operator with
   * signature:  __simt_callee__ (bool*)(const K&, const V*, const S&, const
   * cg::thread_block_tile<GroupSize>&).
   *  @param stream The CANN stream that is used to execute the operation.
   *
   * @return The number of elements removed.
   */

  template <typename PredFunctor>
  size_type erase_if_v2(PredFunctor& pred, aclrtStream stream = 0) {
    std::cout << "[Unsupport erase_if_v2 yet]\n";
    NpuCheckError();
    return 0;
  }

  /**
   * @brief Removes all of the elements in the hash table with no release
   * object.
   */
  void clear(aclrtStream stream = 0) {
    ACLRT_LAUNCH_KERNEL(clear_kernel)(
        block_dim_, stream, table_->buckets, table_->buckets_size,
        options_.max_bucket_size, table_->capacity, value_size_);

    NpuCheckError();
  }

 public:
  /**
   * @brief Exports a certain number of the key-value-score tuples from the
   * hash table.
   *
   * @param n The maximum number of exported pairs.
   * @param offset The position of the key to search.
   * @param d_counter Accumulates amount of successfully exported values.
   * @param keys The keys to dump from NPU-accessible memory with shape (n).
   * @param values The values to dump from NPU-accessible memory with shape
   * (n, DIM).
   * @param scores The scores to search on NPU-accessible memory with shape (n).
   * @parblock
   * If @p scores is `nullptr`, the score for each key will not be returned.
   * @endparblock
   *
   * @param stream The CANN stream that is used to execute the operation.
   *
   * @return The number of elements dumped.
   *
   * @throw NpuException If the key-value size is too large for NPU
   * memory. Reducing the value for @p n is currently required if this exception
   * occurs.
   */
  void export_batch(size_type n, const size_type offset,
                    size_type* d_counter,          // (1)
                    key_type* keys,                // (n)
                    value_type* values,            // (n, DIM)
                    score_type* scores = nullptr,  // (n)
                    aclrtStream stream = 0) const {
    NPU_CHECK(aclrtMemset(d_counter, sizeof(size_type), 0, sizeof(size_type)));
    if (offset >= table_->capacity) {
      return;
    }
    n = std::min(table_->capacity - offset, n);
#ifdef USE_DUMP_KERNEL_ASC
    extern void dump_kernel_do(uint32_t , void*, void*, void*, void*, void*, void*,
      const size_t ,const size_t ,void*, uint32_t, int32_t, uint32_t);
    dump_kernel_do(block_dim_, stream, d_table_, table_->buckets, keys, values,
                   scores, offset, n, d_counter, value_move_opt_.size,
                   value_move_opt_.cg_size, value_move_opt_.dim);
#else
    ACLRT_LAUNCH_KERNEL(dump_kernel)(
        block_dim_, stream, d_table_, table_->buckets, keys, values, scores,
        offset, n, d_counter, value_move_opt_.size, value_move_opt_.cg_size,
        value_move_opt_.dim);
#endif
    NpuCheckError();
  }

  size_type export_batch(const size_type n, const size_type offset,
                         key_type* keys,                // (n)
                         value_type* values,            // (n, DIM)
                         score_type* scores = nullptr,  // (n)
                         aclrtStream stream = 0) const {
    auto dev_ws{dev_mem_pool_->get_workspace<1>(sizeof(size_type), stream)};
    auto d_counter{dev_ws.get<size_type*>(0)};

    NPU_CHECK(aclrtMemset(d_counter, sizeof(size_type), 0, sizeof(size_type)));
    export_batch(n, offset, d_counter, keys, values, scores, stream);

    // 同步stream以确保内核函数执行完成
    NPU_CHECK(aclrtSynchronizeStream(stream));

    size_type counter = 0;
    NPU_CHECK(aclrtMemcpy(&counter, sizeof(size_type), d_counter,
                          sizeof(size_type), ACL_MEMCPY_DEVICE_TO_HOST));
    return counter;
  }

  /**
   * @brief Exports a certain number of the key-value-score tuples which match
   *
   * @tparam PredFunctor A functor with template <K, S> defined an operator
   * with signature:  __simt_callee__ (bool*)(const K&, S&, const K&, const S&).
   * specified condition from the hash table.
   *
   * @param n The maximum number of exported pairs.
   * The value for @p pred should be a function with type `Pred` defined like
   * the following example:
   *
   *    ```
   *    template <class K, class S>
   *    struct ExportIfPredFunctor {
   *      __forceinline__ __simt_callee__ bool operator()(const K& key,
   *                                                 S& score,
   *                                                 const K& pattern,
   *                                                 const S& threshold) {
   *        return score >= threshold;
   *      }
   *    };
   *    ```
   *
   * @param pattern The third user-defined argument to @p pred with key_type
   * type.
   * @param threshold The fourth user-defined argument to @p pred with
   * score_type type.
   * @param offset The position of the key to search.
   * @param keys The keys to dump from NPU-accessible memory with shape (n).
   * @param values The values to dump from NPU-accessible memory with shape
   * (n, DIM).
   * @param scores The scores to search on NPU-accessible memory with shape (n).
   * @parblock
   * If @p scores is `nullptr`, the score for each key will not be returned.
   * @endparblock
   *
   * @param stream The CANN stream that is used to execute the operation.
   *
   * @return The number of elements dumped.
   *
   * @throw NpuException If the key-value size is too large for NPU
   * memory. Reducing the value for @p n is currently required if this exception
   * occurs.
   */
  template <template <typename, typename> class PredFunctor>
  void export_batch_if(const key_type& pattern, const score_type& threshold,
                       size_type n, const size_type offset,
                       size_type* d_counter,
                       key_type* keys,                // (n)
                       value_type* values,            // (n, DIM)
                       score_type* scores = nullptr,  // (n)
                       aclrtStream stream = 0) const {
    std::cout << "[Unsupport export_batch_if yet]\n";

    NpuCheckError();
  }

  /**
   * @brief Exports a certain number of key-value-score tuples that match a
   * given predicate.
   *
   * @tparam PredFunctor A functor type with a template signature `<K, V, S>`.
   * It should define an operator with the signature:
   * `__simt_callee__ bool operator()(const K&, const V*, const S&,
   * cg::thread_block_tile<GroupSize>&)`.
   *
   * @param pred A functor of type `PredFunctor` that defines the predicate for
   * filtering tuples.
   * @param n The maximum number of exported pairs.
   * @param offset The position of the key to search.
   * @param d_counter The number of elements dumped which is on device.
   * @param keys The keys to dump from NPU-accessible memory with shape (n).
   * @param values The values to dump from NPU-accessible memory with shape (n,
   * DIM).
   * @param scores The scores to search on NPU-accessible memory with shape (n).
   * @parblock
   * If @p scores is `nullptr`, the score for each key will not be returned.
   * @endparblock
   *
   * @param stream The CANN stream that is used to execute the operation.
   *
   * @return void
   *
   */

  template <typename PredFunctor>
  void export_batch_if_v2(PredFunctor& pred, size_type n,
                          const size_type offset, size_type* d_counter,
                          key_type* keys,                // (n)
                          value_type* values,            // (n, DIM)
                          score_type* scores = nullptr,  // (n)
                          aclrtStream stream = 0) const {
    std::cout << "[Unsupport export_batch_if_v2 yet]\n";

    NpuCheckError();
  }

  /**
   * @brief Applies the given function to items in the range [first, last) in
   * the table.
   *
   * @tparam ExecutionFunc A functor type with a template signature `<K, V, S>`.
   * It should define an operator with the signature:
   * `__simt_callee__ void operator()(const K&, V*, S*,
   * cg::thread_block_tile<GroupSize>&)`.
   *
   * @param first The first element to which the function object will be
   * applied.
   * @param last The last element(excluding) to which the function object will
   * be applied.
   * @param f A functor of type `ExecutionFunc` that defines the predicate for
   * filtering tuples. signature:  __simt_callee__ (bool*)(const K&, const V*, const
   * S&, const cg::tiled_partition<GroupSize>&).
   * @param stream The CANN stream that is used to execute the operation.
   *
   * @return void
   *
   */

  template <typename ExecutionFunc>
  void for_each(const size_type first, const size_type last, ExecutionFunc& f,
                aclrtStream stream = 0) {
    std::cout << "[Unsupport for_each yet]\n";

    NpuCheckError();
  }

 public:
  /**
   * @brief Indicates if the hash table has no elements.
   *
   * @param stream The CANN stream that is used to execute the operation.
   * @return `true` if the table is empty and `false` otherwise.
   */
  bool empty(aclrtStream stream = 0) const { return size(stream) == 0; }

  /**
   * @brief Returns the hash table size.
   *
   * @param input_stream The CANN stream that is used to execute the operation.
   * @return The table size.
   */
  size_type size(aclrtStream input_stream = 0) const {
    ScopedStream scoped_stream(input_stream);
    aclrtStream stream = scoped_stream.get();
    DeviceTensor input;
    input.init(table_->buckets_size, aclDataType::ACL_INT32,
               {static_cast<int64_t>(table_->buckets_num)});
    std::vector<int64_t> dims = {0};
    bool keep_dims = false;
    auto out_data_type = aclDataType::ACL_INT64;
    DeviceTensor out;
    out.init(out_data_type, {1});

    EXEC_ACLNN_OP(aclnnReduceSum, input, dims, keep_dims, out_data_type, out);
    auto ret = aclrtSynchronizeStream(stream);
    NPU_CHECK(ret);

    int64_t h_total_size = 0;
    ret = aclrtMemcpy(&h_total_size, sizeof(h_total_size), out.get_data(),
                      out.get_data_size(), ACL_MEMCPY_DEVICE_TO_HOST);
    NPU_CHECK(ret);
    return h_total_size;
  }

  /**
   * @brief Returns the number of keys if meet PredFunctor.
   *
   * @param stream The CANN stream that is used to execute the operation.
   * @return The table size match condiction of PredFunctor.
   */
  template <template <typename, typename> class PredFunctor>
  void size_if(const key_type& pattern, const score_type& threshold,
               size_type* d_counter, aclrtStream stream = 0) const {
    std::cout << "[Unsupport size_if yet]\n";
    NpuCheckError();
  }

  /**
   * @brief Returns the hash table capacity.
   *
   * @note The value that is returned might be less than the actual capacity of
   * the hash table because the hash table currently keeps the capacity to be
   * a power of 2 for performance considerations.
   *
   * @return The table capacity.
   */
  size_type capacity() const { return table_->capacity; }

  /**
   * @brief Sets the number of buckets to the number that is needed to
   * accommodate at least @p new_capacity elements without exceeding the maximum
   * load factor. This method rehashes the hash table. Rehashing puts the
   * elements into the appropriate buckets considering that total number of
   * buckets has changed.
   *
   * @note If the value of @p new_capacity or double of @p new_capacity is
   * greater or equal than `options_.max_capacity`, the reserve does not perform
   * any change to the hash table.
   *
   * @param new_capacity The requested capacity for the hash table.
   * @param stream The CANN stream that is used to execute the operation.
   */
  void reserve(const size_type new_capacity, aclrtStream stream = 0) {
    if (reach_max_capacity_ || new_capacity > options_.max_capacity) {
      reach_max_capacity_ = (capacity() * 2 > options_.max_capacity);
      return;
    }

    NPU_CHECK(aclrtSynchronizeDevice());

    while (capacity() < new_capacity &&
           capacity() * 2 <= options_.max_capacity) {
      if (bucket_memory_pool_manager_ != nullptr &&
          bucket_memory_pool_manager_->use_pool()) {
        bucket_memory_pool_manager_->ensure_capacity(table_->buckets_num * 2);
      }
      double_capacity(
          &table_, allocator_, block_dim_,
          bucket_memory_pool_manager_ != nullptr
              ? bucket_memory_pool_manager_.get()
              : nullptr);
      sync_table_configuration();

      ACLRT_LAUNCH_KERNEL(rehash_kernel)(block_dim_, stream, d_table_,
                                         table_->buckets_num / 2, value_size_);
      NPU_CHECK(aclrtSynchronizeDevice());
    }

    NPU_CHECK(aclrtSynchronizeDevice());
    reach_max_capacity_ = (capacity() * 2 > options_.max_capacity);

    NpuCheckError();
  }

  /**
   * @brief Returns the average number of elements per slot, that is, size()
   * divided by capacity().
   *
   * @param stream The CANN stream that is used to execute the operation.
   *
   * @return The load factor
   */
  float load_factor(aclrtStream stream = 0) const {
    return static_cast<float>((size(stream) * 1.0) / (capacity() * 1.0));
  }

  /**
   * @brief Set max_capacity of the table.
   *
   * @param new_max_capacity The new expecting max_capacity. It must be power
   * of 2. Otherwise it will raise an error.
   */
  void set_max_capacity(size_type new_max_capacity) {
    if (!is_power(2, new_max_capacity)) {
      throw std::invalid_argument(
          "None power-of-2 new_max_capacity is not supported.");
    }

    if (new_max_capacity < capacity()) {
      return;
    }
    if (reach_max_capacity_) {
      reach_max_capacity_ = false;
    }
    options_.max_capacity = new_max_capacity;
  }

  /**
   * @brief Returns the dimension of the vectors.
   *
   * @return The dimension of the vectors.
   */
  size_type dim() const noexcept { return options_.dim; }

  /**
   * @brief Returns The length of each bucket.
   *
   * @return The length of each bucket.
   */
  size_type max_bucket_size() const noexcept {
    return options_.max_bucket_size;
  }

  /**
   * @brief Returns the number of buckets in the table.
   *
   * @return The number of buckets in the table.
   */
  size_type bucket_count() const noexcept { return table_->buckets_num; }

  /**
   * @brief Save keys, vectors, scores in table to file or files.
   *
   * @param file A BaseKVFile object defined the file format on host filesystem.
   * @param max_workspace_size Saving is conducted in chunks. This value denotes
   * the maximum amount of temporary memory to use when dumping the table.
   * Larger values *can* lead to higher performance.
   * @param stream The CANN stream used to execute the operation.
   *
   * @return Number of KV pairs saved to file.
   */
  size_type save(BaseKVFile<K, V, S>* file,
                 const size_t max_workspace_size = 1L * 1024 * 1024,
                 aclrtStream stream = 0) const {
    // Calculate the size of each tuple (key + score + values)
    const size_type tuple_size{sizeof(key_type) + sizeof(score_type) +
                               sizeof(value_type) * dim()};
    HKV_CHECK(max_workspace_size >= tuple_size,
              "[HierarchicalKV] max_workspace_size is smaller than a single "
              "`key + score + value` tuple! Please set a larger value!");

    const size_type total_size{capacity()};
    // Calculate how many tuples can fit in the workspace
    const size_type n{std::min(max_workspace_size / tuple_size, total_size)};

    // Calculate host workspace size for keys, scores, and values
    const size_type host_ws_size{n * tuple_size};
    auto host_ws{host_mem_pool_->get_workspace<1>(host_ws_size, stream)};
    auto h_keys{host_ws.get<key_type*>(0)};
    auto h_scores{reinterpret_cast<score_type*>(h_keys + n)};
    auto h_values{reinterpret_cast<value_type*>(h_scores + n)};

    // Calculate device workspace size: counter + keys + scores + values
    const size_type dev_ws_size{sizeof(size_type) + host_ws_size};
    auto dev_ws{dev_mem_pool_->get_workspace<1>(dev_ws_size, stream)};
    auto d_count{dev_ws.get<size_type*>(0)};
    auto d_keys{reinterpret_cast<key_type*>(d_count + 1)};
    auto d_scores{reinterpret_cast<score_type*>(d_keys + n)};
    auto d_values{reinterpret_cast<value_type*>(d_scores + n)};

    // Step through table, dumping contents in batches
    size_type total_count{0};
    for (size_type i{0}; i < total_size; i += n) {
      // Reset the counter
      NPU_CHECK(aclrtMemset(d_count, sizeof(size_type), 0, sizeof(size_type)));

      // Calculate the batch size for this iteration
      const size_type batch_size = std::min(total_size - i, n);

      // Launch dump_kernel to export data to device memory
#ifdef USE_DUMP_KERNEL_ASC
      extern void dump_kernel_do(uint32_t, void*, void*, void*, void*, void*,
                                 void*, const size_t, const size_t, void*,
                                 uint32_t, int32_t, uint32_t);
      dump_kernel_do(block_dim_, stream, d_table_, table_->buckets, d_keys,
                     d_values, d_scores, i, batch_size, d_count,
                     value_move_opt_.size, value_move_opt_.cg_size,
                     value_move_opt_.dim);
#else
      ACLRT_LAUNCH_KERNEL(dump_kernel)
      (block_dim_, stream, d_table_, table_->buckets, d_keys, d_values,
       d_scores, i, batch_size, d_count, value_move_opt_.size,
       value_move_opt_.cg_size, value_move_opt_.dim);
#endif
      NpuCheckError();
      // Copy counter from device to host
      size_type count;
      NPU_CHECK(aclrtMemcpyAsync(&count, sizeof(size_type), d_count,
                                 sizeof(size_type), ACL_MEMCPY_DEVICE_TO_HOST,
                                 stream));
      NPU_CHECK(aclrtSynchronizeStream(stream));

      if (count > 0) {
        // Copy data from device to host
        if (count == n) {
          // Full batch: can copy all at once since memory layout matches
          NPU_CHECK(aclrtMemcpyAsync(h_keys, host_ws_size, d_keys, host_ws_size,
                                     ACL_MEMCPY_DEVICE_TO_HOST, stream));
        } else {
          // Partial batch: copy each array separately
          NPU_CHECK(aclrtMemcpyAsync(h_keys, sizeof(key_type) * count, d_keys,
                                     sizeof(key_type) * count,
                                     ACL_MEMCPY_DEVICE_TO_HOST, stream));
          NPU_CHECK(aclrtMemcpyAsync(h_scores, sizeof(score_type) * count,
                                     d_scores, sizeof(score_type) * count,
                                     ACL_MEMCPY_DEVICE_TO_HOST, stream));
          NPU_CHECK(aclrtMemcpyAsync(h_values, sizeof(value_type) * dim() * count,
                                     d_values, sizeof(value_type) * dim() * count,
                                     ACL_MEMCPY_DEVICE_TO_HOST, stream));
        }
        NPU_CHECK(aclrtSynchronizeStream(stream));

        // Write to file
        file->write(count, dim(), h_keys, h_values, h_scores);
        total_count += count;
      }
    }

    return total_count;
  }

  /**
   * @brief Load keys, vectors, scores from file to table.
   *
   * @param file An BaseKVFile defined the file format within filesystem.
   * @param max_workspace_size Loading is conducted in chunks. This value
   * denotes the maximum size of such chunks. Larger values *can* lead to higher
   * performance.
   * @param stream The CANN stream used to execute the operation.
   *
   * @return Number of keys loaded from file.
   */
  size_type load(BaseKVFile<K, V, S>* file,
                 const size_t max_workspace_size = 1L * 1024 * 1024,
                 aclrtStream stream = 0) {
    // Calculate the size of each tuple (key + score + values)
    const size_type tuple_size{sizeof(key_type) + sizeof(score_type) +
                               sizeof(value_type) * dim()};
    HKV_CHECK(max_workspace_size >= tuple_size,
              "[HierarchicalKV] max_workspace_size is smaller than a single "
              "`key + score + value` tuple! Please set a larger value!");

    // Calculate how many tuples can fit in the workspace
    const size_type n{max_workspace_size / tuple_size};
    const size_type ws_size{n * tuple_size};

    // Grab enough host memory to hold batch data
    auto host_ws{host_mem_pool_->get_workspace<1>(ws_size, stream)};
    auto h_keys{host_ws.get<key_type*>(0)};
    auto h_scores{reinterpret_cast<score_type*>(h_keys + n)};
    auto h_values{reinterpret_cast<value_type*>(h_scores + n)};

    // Attempt a first read
    size_type count{file->read(n, dim(), h_keys, h_values, h_scores)};
    if (count == 0) {
      return 0;
    }

    // Grab equal amount of device memory as temporary storage
    auto dev_ws{dev_mem_pool_->get_workspace<1>(ws_size, stream)};
    auto d_keys{dev_ws.get<key_type*>(0)};
    auto d_scores{reinterpret_cast<score_type*>(d_keys + n)};
    auto d_values{reinterpret_cast<value_type*>(d_scores + n)};

    size_type total_count{0};
    do {
      if (count == n) {
        // Full batch: can copy all at once since memory layout matches
        NPU_CHECK(aclrtMemcpyAsync(d_keys, ws_size, h_keys, ws_size,
                                   ACL_MEMCPY_HOST_TO_DEVICE, stream));
      } else {
        // Partial batch: copy each array separately
        NPU_CHECK(aclrtMemcpyAsync(d_keys, sizeof(key_type) * count, h_keys,
                                   sizeof(key_type) * count,
                                   ACL_MEMCPY_HOST_TO_DEVICE, stream));
        NPU_CHECK(aclrtMemcpyAsync(d_scores, sizeof(score_type) * count,
                                   h_scores, sizeof(score_type) * count,
                                   ACL_MEMCPY_HOST_TO_DEVICE, stream));
        NPU_CHECK(aclrtMemcpyAsync(d_values, sizeof(value_type) * dim() * count,
                                   h_values, sizeof(value_type) * dim() * count,
                                   ACL_MEMCPY_HOST_TO_DEVICE, stream));
      }

      // Insert data into hash table with ignored global epoch
      set_global_epoch(static_cast<S>(IGNORED_GLOBAL_EPOCH));
      insert_or_assign(count, d_keys, d_values, d_scores, stream, true, true);
      total_count += count;

      // Read next batch
      NPU_CHECK(aclrtSynchronizeStream(stream));
      count = file->read(n, dim(), h_keys, h_values, h_scores);
    } while (count > 0);

    return total_count;
  }

  void set_global_epoch(const uint64_t epoch) { global_epoch_ = epoch; }

 private:
  bool is_power(size_t base, size_t n) {
    if (base < 2) {
      throw std::invalid_argument("is_power with zero base.");
    }
    while (n > 1) {
      if (n % base != 0) {
        return false;
      }
      n /= base;
    }
    return true;
  }

 private:
  inline bool is_fast_mode() const noexcept { return table_->is_pure_hbm; }

  /**
   * @brief Returns the load factor by sampling up to 1024 buckets.
   *
   * @note For performance consideration, the returned load factor is
   * inaccurate but within an error in 1% empirically which is enough for
   * capacity control. But it's not suitable for end-users.
   *
   * @param delta A hypothetical upcoming change on table size.
   * @param input_stream The CANN stream used to execute the operation.
   * @param need_lock If lock is needed.
   *
   * @return The evaluated load factor
   */
  inline float fast_load_factor(const size_type delta = 0,
                                aclrtStream input_stream = 0,
                                const bool need_lock = true) const {
    ScopedStream scoped_stream(input_stream);
    aclrtStream stream = scoped_stream.get();

    size_t N = std::min(table_->buckets_num, 1024UL);

    DeviceTensor input;
    input.init(table_->buckets_size, aclDataType::ACL_INT32,
               {static_cast<int64_t>(N)});
    std::vector<int64_t> dims = {0};
    bool keep_dims = false;
    auto out_data_type = aclDataType::ACL_INT64;
    DeviceTensor out;
    out.init(out_data_type, {1});

    EXEC_ACLNN_OP(aclnnReduceSum, input, dims, keep_dims, out_data_type, out);

    int64_t h_total_size = 0;
    NPU_CHECK(aclrtMemcpyAsync(&h_total_size, sizeof(h_total_size),
                               out.get_data(), out.get_data_size(),
                               ACL_MEMCPY_DEVICE_TO_HOST, stream));
    NPU_CHECK(aclrtSynchronizeStream(stream));
    NpuCheckError();
    return static_cast<float>((delta * 1.0) / (capacity() * 1.0) +
                              (h_total_size * 1.0) /
                                  (options_.max_bucket_size * N * 1.0));
  }

  inline void check_evict_strategy(const score_type* scores) {
    if (evict_strategy == EvictStrategy::kLru ||
        evict_strategy == EvictStrategy::kEpochLru) {
      HKV_CHECK(scores == nullptr,
                "the scores should not be specified when running on "
                "LRU or Epoch LRU mode.");
    }

    if (evict_strategy == EvictStrategy::kLfu ||
        evict_strategy == EvictStrategy::kEpochLfu) {
      HKV_CHECK(scores != nullptr,
                "the scores should be specified when running on "
                "LFU or Epoch LFU mode.");
    }

    if (evict_strategy == EvictStrategy::kCustomized) {
      HKV_CHECK(scores != nullptr,
                "the scores should be specified when running on "
                "customized mode.");
    }

    if ((evict_strategy == EvictStrategy::kEpochLru ||
         evict_strategy == EvictStrategy::kEpochLfu)) {
      HKV_CHECK(global_epoch_ != static_cast<S>(IGNORED_GLOBAL_EPOCH),
                "the global_epoch is invalid and should be assigned by calling "
                "`set_global_epoch` when running on "
                "Epoch LRU or Epoch LFU mode.");
    }
  }

  /**
   * @brief Synchronize the TableCore struct to replicas.
   *
   * @note For performance consideration, synchronize the TableCore struct to
   * its replicas in constant memory and device memory when it's changed.
   */
  inline void sync_table_configuration() {
    NPU_CHECK(aclrtMemcpy(d_table_, sizeof(TableCore), table_,
                          sizeof(TableCore), ACL_MEMCPY_HOST_TO_DEVICE));
  }

 private:
  /**
   * @note The function is provided to get the best value move params.
   * 
   * @note On `move_byte_per_value`, get best params from the move bytes.
   * 
   * @note On `ValueMoveOpt`, size is 8 or 16, which is more suitable for NPUs.
   */
  ValueMoveOpt GetValueMoveOpt(uint32_t move_byte_per_value) {
    ValueMoveOpt opt;
    opt.size = 1;
    opt.cg_size = 1;
    opt.dim = move_byte_per_value;
    // based on real data, when move bytes in [128, 4096), we chose 16 as the best size, otherwise, 8 is the best. 
    uint32_t value_move_size_best = (move_byte_per_value >= 128 && move_byte_per_value < 4096) ? 16 : 8;
    // try to use the best size 8 or 16
    while (opt.dim % 2 == 0 && opt.size < value_move_size_best) {
      opt.size *= 2;
      opt.dim /= 2;
    }
    // best cg_size for diff dim, group_size can only be one of [1, 2, 4, 8, 16, 32].
    uint32_t log_value = static_cast<uint32_t>(std::log2(static_cast<double>(opt.dim)));
    opt.cg_size <<= log_value;
    // based on real data, cg_size is not necessarily better then it is larger.
    uint32_t value_move_group_size_best = opt.size == 8 ? 32 : 16;
    opt.cg_size = std::min(value_move_group_size_best, opt.cg_size);
    opt.cg_size = std::max(2u, opt.cg_size);
    // based on real data, when size is large enough, we can get better performance by increasing threads num of cores.
    opt.is_large_size = move_byte_per_value >= 4096;

    return opt;
  }

 private:
  HashTableOptions options_;
  TableCore* table_ = nullptr;
  TableCore* d_table_ = nullptr;
  size_t shared_mem_size_ = 0;
  uint32_t block_dim_ = 0;
  int sm_cnt_ = 0;
  int max_threads_per_block_ = 0;
  std::atomic_bool reach_max_capacity_{false};
  bool initialized_ = false;
  const unsigned int kernel_select_interval_ = 7;
  std::unique_ptr<DeviceMemoryPool> dev_mem_pool_;
  std::unique_ptr<HostMemoryPool> host_mem_pool_;
  allocator_type* allocator_;
  bool default_allocator_ = true;
  std::atomic<uint64_t> global_epoch_{
      static_cast<uint64_t>(IGNORED_GLOBAL_EPOCH)};
  const uint32_t value_size_ = sizeof(V);
  ValueMoveOpt value_move_opt_;
  std::unique_ptr<BucketMemoryPoolManager<key_type, value_type, score_type>>
      bucket_memory_pool_manager_;
};

}  // namespace hkv
}  // namespace npu
