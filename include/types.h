/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
 * Copyright (C) 2025. Huawei Technologies Co., Ltd. All rights reserved.
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

#include <stddef.h>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <string>
#include "cuda2npu.h"
#include "utils.h"
namespace npu {
namespace hkv {

/**
 * Shorthand for a Key-Value-score tuple.
 */
template <class K, class V, class S>
struct KVM {
  K key;
  uint64_t value;
  S score;
};

#if (!defined(__CCE__))
struct uint4 {
  uint32_t x, y, z, w;
};

struct uint2 {
  uint32_t x, y;
};
#endif

// Storage size.
using byte16 = uint4;
using byte8 = uint2;
using byte4 = uint32_t;
using byte2 = uint16_t;
using byte = uint8_t;

// Digest.
using D = byte;
// Vector Type of digests for memory access.
using VecD_Load = byte16;
// Vector Type of digests for computation.
using VecD_Comp = byte4;

constexpr uint64_t DEFAULT_EMPTY_KEY = UINT64_C(0xFFFFFFFFFFFFFFFF);
constexpr uint64_t DEFAULT_RECLAIM_KEY = UINT64_C(0xFFFFFFFFFFFFFFFE);
constexpr uint64_t DEFAULT_LOCKED_KEY = UINT64_C(0xFFFFFFFFFFFFFFFD);

constexpr uint64_t DEFAULT_RESERVED_KEY_MASK = UINT64_C(0xFFFFFFFFFFFFFFFC);
constexpr uint64_t DEFAULT_VACANT_KEY_MASK = UINT64_C(0xFFFFFFFFFFFFFFFE);

constexpr uint64_t MAX_SCORE = UINT64_C(0xFFFFFFFFFFFFFFFF);
constexpr uint64_t EMPTY_SCORE = UINT64_C(0);
constexpr uint64_t IGNORED_GLOBAL_EPOCH = UINT64_C(0xFFFFFFFFFFFFFFFF);

constexpr uint64_t EMPTY_KEY_CPU = DEFAULT_EMPTY_KEY;
constexpr uint64_t EMPTY_KEY = DEFAULT_EMPTY_KEY;
constexpr uint64_t RECLAIM_KEY = DEFAULT_RECLAIM_KEY;
constexpr uint64_t LOCKED_KEY = DEFAULT_LOCKED_KEY;

constexpr uint64_t RESERVED_KEY_MASK_1 = DEFAULT_RESERVED_KEY_MASK;
constexpr uint64_t RESERVED_KEY_MASK_2 = DEFAULT_RESERVED_KEY_MASK;
constexpr uint64_t VACANT_KEY_MASK_1 = DEFAULT_VACANT_KEY_MASK;
constexpr uint64_t VACANT_KEY_MASK_2 = DEFAULT_VACANT_KEY_MASK;

constexpr uint32_t INVALID_KEY_POS = std::numeric_limits<uint32_t>::max();

constexpr int MAX_RESERVED_KEY_BIT = 62;

constexpr uint32_t WARP_SIZE = 32;

template <typename K>
__forceinline__ __simt_callee__ D empty_digest() {
  const K hashed_key = Murmur3HashDevice(static_cast<K>(EMPTY_KEY));
  return static_cast<D>(hashed_key >> 32);
}

__forceinline__ __simt_callee__ uint32_t vcmpeq4(uint32_t a, uint32_t b) {
  uint32_t eq = ~(a ^ b);
  uint32_t mask0 = ((eq >> 0) & 0xFF) == 0xFF ? 0x000000FF : 0;
  uint32_t mask1 = ((eq >> 8) & 0xFF) == 0xFF ? 0x0000FF00 : 0;
  uint32_t mask2 = ((eq >> 16) & 0xFF) == 0xFF ? 0x00FF0000 : 0;
  uint32_t mask3 = ((eq >> 24) & 0xFF) == 0xFF ? 0xFF000000 : 0;

  return mask0 | mask1 | mask2 | mask3;
}

template <typename K>
__forceinline__ __simt_callee__ D get_digest(const K& key) {
  const K hashed_key = Murmur3HashDevice(key);
  return static_cast<D>(hashed_key >> 32);
}

template <typename K>
__forceinline__ __simt_callee__ VecD_Comp digests_from_hashed(const K& hashed_key) {
  D digest = static_cast<D>(hashed_key >> 32);
  // Set every byte in VecD_Comp to `digest`.
  return static_cast<VecD_Comp>(digest) * 0x01010101U;
}

template <typename K>
__forceinline__ __simt_callee__ VecD_Comp empty_digests() {
  D digest = empty_digest<K>();
  // Set every byte in VecD_Comp to `digest`.
  return static_cast<VecD_Comp>(digest) * 0x01010101U;
}

// Position alignment.
template <uint32_t ALIGN_SIZE>
__forceinline__ __simt_callee__ uint32_t align_to(uint32_t& pos) {
  constexpr uint32_t MASK = 0xffffffffU - (ALIGN_SIZE - 1);
  return pos & MASK;
}

__forceinline__ __simt_callee__ uint32_t get_start_position(
    const uint64_t& global_idx, const uint64_t& bucket_capacity) {
  uint32_t start_idx = global_idx & (bucket_capacity - 1);
  start_idx -= start_idx % 4;
  return start_idx;
}

#if defined(__CCE__)
template <class K>
__simt_callee__ inline bool IS_RESERVED_KEY(K key) {
  return (RESERVED_KEY_MASK_1 & key) == RESERVED_KEY_MASK_2;
}
#endif

template <class K>
bool IS_VACANT_KEY(K key) {
  return (VACANT_KEY_MASK_1 & key) == VACANT_KEY_MASK_2;
}

#if !defined(__CCE__)
__attribute__((unused)) static aclError init_reserved_keys(int index) {
  if (index < 1 || index > MAX_RESERVED_KEY_BIT) {
    // index = 0 is the default,
    // index = 62 is the maximum index can be set for reserved keys.
    return ACL_SUCCESS;
  }
  return ACL_SUCCESS;
}
#endif

template <class K, class V, class S>
struct Bucket {
  __gm__ K* keys_;
  __gm__ S* scores_;
  /// @brief not visible to users
  __gm__ D* digests_;
  __gm__ V* vectors;  // Pinned memory or HBM

  D* digests(int index) const { return digests_ + index; }

  __gm__ __forceinline__ __simt_callee__ K* keys(int index) const {
    return keys_ + index;
  }

  __gm__ __forceinline__ __simt_callee__ S* scores(int index) const {
    return scores_ + index;
  }

  K** keys_addr() { return reinterpret_cast<K**>(&keys_); }

  static __forceinline__ __simt_callee__ __gm__ K* keys(__gm__ K* keys,
                                                   uint32_t offset) {
    return reinterpret_cast<__gm__ K*>(keys) + offset;
  }

  static __forceinline__ __simt_callee__ __gm__ D* digests(__gm__ K* keys,
                                                      uint32_t bucket_capacity,
                                                      uint32_t offset) {
    bucket_capacity = bucket_capacity > 128U ? bucket_capacity : 128U;
    return reinterpret_cast<__gm__ D*>(keys) - bucket_capacity + offset;
  }

  static __forceinline__ __simt_callee__ __gm__ S* scores(__gm__ K* keys,
                                                     uint32_t bucket_capacity,
                                                     uint32_t offset) {
    return reinterpret_cast<__gm__ S*>(keys + bucket_capacity) + offset;
  }
};

template <class T = int>
class Lock {
  mutable T _lock;

 public:
  __simt_callee__ Lock() : _lock{1} {}
};

using Mutex = Lock<>;

template <class K, class V, class S>
struct Table {
  __gm__ Bucket<K, V, S>* buckets;
  __gm__ Mutex* locks;                   // mutex for write buckets
  __gm__ int* buckets_size;              // size of each buckets.
  __gm__ V** slices;                     // Handles of the HBM/ HMEM slices.
  size_t dim;                            // Dimension of the `vectors`.
  size_t bytes_per_slice;                // Size by byte of one slice.
  size_t num_of_memory_slices;           // Number of vectors memory slices.
  size_t capacity = 134217728;           // Initial capacity.
  size_t max_size = 0xFFFFFFFFFFFFFFFF;  // Up limit of the table capacity.
  size_t buckets_num;                    // Number of the buckets.
  size_t bucket_max_size = 128;          // Volume of each buckets.
  size_t max_hbm_for_vectors = 0;        // Max HBM allocated for vectors
  size_t remaining_hbm_for_vectors = 0;  // Remaining HBM allocated for vectors
  size_t num_of_buckets_per_alloc = 1;   // Number of buckets allocated in each
                                         // HBM allocation, must be power of 2.
  bool is_pure_hbm = true;               // unused
  bool primary = true;                   // unused
  int slots_offset = 0;                  // unused
  int slots_number = 0;                  // unused
  int device_id = 0;                     // Device id
  int tile_size;
  uint32_t max_bucket_shift = 0;        // log2(bucket_max_size) for kernel
  uint64_t capacity_divisor_magic = 0;  // quick div magic for kernel
  uint64_t capacity_divisor_shift = 0;  // quick div shift for kernel
};

enum OccupyResult {
  INITIAL,         ///< Initial status
  CONTINUE,        ///< Insert did not succeed, continue trying to insert
  OCCUPIED_EMPTY,  ///< New pair inserted successfully
  OCCUPIED_RECLAIMED,
  DUPLICATE,  ///< Insert did not succeed, key is already present
  EVICT,      ///< Insert succeeded by evicting one key with minimum score.
  REFUSED,    ///< Insert did not succeed, insert score is too low.
  ILLEGAL,    ///< Illegal state, and don't need to do anything.
};

enum class OverrideResult {
  INITIAL,   ///< Initial status
  CONTINUE,  ///< Override did not succeed, continue trying to override
  SUCCESS,   ///< Override successfully
  REFUSED,   ///< Override is refused.
};

/* This struct is mainly for keeping the code readable, it should be strictly
 * consistent with `EvictStrategy::EvictStrategyEnum`.
 */
struct EvictStrategyInternal {
  constexpr static int kLru = 0;         ///< LRU mode.
  constexpr static int kLfu = 1;         ///< LFU mode.
  constexpr static int kEpochLru = 2;    ///< Epoch + LRU mode.
  constexpr static int kEpochLfu = 3;    ///< Epoch + LFU mode.
  constexpr static int kCustomized = 4;  ///< Customized mode.
};

/**
 * An abstract class provides interface between the npu::hkv::HashTable
 * and a file, which enables the table to save to the file or load from
 * the file, by overriding the `read` and `write` method.
 *
 * @tparam K The data type of the key.
 * @tparam V The data type of the vector's elements.
 *         The item data type should be a basic data type of C++/NPU.
 * @tparam S The data type for `score`.
 *           The currently supported data type is only `uint64_t`.
 *
 */
template <class K, class V, class S>
class BaseKVFile {
 public:
  virtual ~BaseKVFile() {}

  /**
   * Read from file and fill into the keys, values, and scores buffer.
   * When calling save/load method from table, it can assume that the
   * received buffer of keys, vectors, and scores are automatically
   * pre-allocated.
   *
   * @param n The number of KV pairs expect to read. `int64_t` was used
   *          here to adapt to various filesytem and formats.
   * @param dim The dimension of the `vectors`.
   * @param keys The pointer to received buffer for keys.
   * @param vectors The pointer to received buffer for vectors.
   * @param scores The pointer to received buffer for scores.
   *
   * @return Number of KV pairs have been successfully read.
   */
  virtual size_t read(const size_t n, const size_t dim, K* keys, V* vectors,
                      S* scores) = 0;

  /**
   * Write keys, values, scores from table to the file. It defines
   * an abstract method to get batch of KV pairs and write them into
   * file.
   *
   * @param n The number of KV pairs to be written. `int64_t` was used
   *          here to adapt to various filesytem and formats.
   * @param dim The dimension of the `vectors`.
   * @param keys The keys will be written to file.
   * @param vectors The vectors of values will be written to file.
   * @param scores The scores will be written to file.
   *
   * @return Number of KV pairs have been successfully written.
   */
  virtual size_t write(const size_t n, const size_t dim, const K* keys,
                       const V* vectors, const S* scores) = 0;
};

/**
 * The KV file on local file system. It only save/load keys and vectors
 * between table and file. `scores` are ignored in it since absolute
 * values of scores are commonly time-variant, while the time interval
 * between save/load calling is not deterministic, in default case. If
 * other specified rules are required, the BaseKVFile could be inherited
 * to implement customized read/write rules. The LocalKVFile uses compact,
 * consecutive binary format, where keys, values, and scores are stored in
 * seperated paths.
 *
 * @tparam K The data type of the key.
 * @tparam V The data type of the vector's elements.
 *         The item data type should be a basic data type of C++/NPU.
 * @tparam S The data type for `score`.
 *           The currently supported data type is only `uint64_t`.
 *
 */
template <class K, class V, class S>
class LocalKVFile : public BaseKVFile<K, V, S> {
 public:
  LocalKVFile() : keys_fp_(nullptr), values_fp_(nullptr), scores_fp_(nullptr) {}

  ~LocalKVFile() { close(); }

  /**
   * @brief Open the file from local path. A LocalKVFile can only be
   * read or written when it stays opened.
   *
   * @param keys_path Path to file to store keys.
   * @param values_path Path to file to store values.
   * @param scores_path Path to file to store scores.
   * @params mode The mode to the file. The mode follows glibc style
   *              and behavior like fopen.
   */
  bool open(const std::string& keys_path, const std::string& values_path,
            const std::string& scores_path, const char* mode) {
    close();
    keys_fp_ = fopen(keys_path.c_str(), mode);
    if (!keys_fp_) {
      return false;
    }
    values_fp_ = fopen(values_path.c_str(), mode);
    if (!values_fp_) {
      close();
      return false;
    }
    scores_fp_ = fopen(scores_path.c_str(), mode);
    if (!scores_fp_) {
      close();
      return false;
    }
    return true;
  }

  /**
   * @brief Close the file from open status and release fd(s) on files
   * of keys, values, and scores.
   */
  void close() noexcept {
    if (keys_fp_) {
      fclose(keys_fp_);
      keys_fp_ = nullptr;
    }
    if (values_fp_) {
      fclose(values_fp_);
      values_fp_ = nullptr;
    }
    if (scores_fp_) {
      fclose(scores_fp_);
      scores_fp_ = nullptr;
    }
  }

  /**
   * Read from file and fill into the keys, values, and scores buffer.
   * When calling save/load method from table, it can assume that the
   * received buffer of keys, vectors, and scores are automatically
   * pre-allocated.
   *
   * @param n The number of KV pairs expect to read. `int64_t` was used
   *          here to adapt to various filesytem and formats.
   * @param dim The dimension of the `vectors`.
   * @param keys The pointer to received buffer for keys.
   * @param vectors The pointer to received buffer for vectors.
   * @param scores The pointer to received buffer for scores.
   *
   * @return Number of KV pairs have been successfully read.
   */
  size_t read(const size_t n, const size_t dim, K* keys, V* vectors,
              S* scores) override {
    size_t nread_keys =
        fread(keys, sizeof(K), static_cast<size_t>(n), keys_fp_);
    size_t nread_vecs =
        fread(vectors, sizeof(V) * dim, static_cast<size_t>(n), values_fp_);
    size_t nread_scores =
        fread(scores, sizeof(S), static_cast<size_t>(n), scores_fp_);
    if (nread_keys != nread_vecs || nread_keys != nread_scores) {
      return 0;
    }
    return nread_keys;
  }

  /**
   * Write keys, values, scores from table to the file.
   *
   * @param n The number of KV pairs to be written. `int64_t` was used
   *          here to adapt to various filesytem and formats.
   * @param dim The dimension of the `vectors`.
   * @param keys The keys will be written to file.
   * @param vectors The vectors of values will be written to file.
   * @param scores The scores will be written to file.
   *
   * @return Number of KV pairs have been successfully written.
   */
  size_t write(const size_t n, const size_t dim, const K* keys,
               const V* vectors, const S* scores) override {
    size_t nwritten_keys =
        fwrite(keys, sizeof(K), static_cast<size_t>(n), keys_fp_);
    size_t nwritten_vecs =
        fwrite(vectors, sizeof(V) * dim, static_cast<size_t>(n), values_fp_);
    size_t nwritten_scores =
        fwrite(scores, sizeof(S), static_cast<size_t>(n), scores_fp_);
    if (nwritten_keys != nwritten_vecs || nwritten_keys != nwritten_scores) {
      return 0;
    }
    return nwritten_keys;
  }

 private:
  FILE* keys_fp_;
  FILE* values_fp_;
  FILE* scores_fp_;
};
}  // namespace hkv
}  // namespace npu
