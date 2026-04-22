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

#include <kernel_operator.h>
#include <cstdint>
#include "cuda2npu.h"
#include "types.h"
#include "utils.h"

namespace npu {
namespace hkv {
using namespace AscendC;

constexpr int32_t EPOCH_BITS = 32;
constexpr uint64_t EPOCH_BITS_MASK = UINT64_C(0xFFFFFFFF00000000);
constexpr uint64_t SCORE_BITS_MASK = UINT64_C(0xFFFFFFFF);
constexpr uint64_t SCORE_32BIT_MAX = UINT64_C(0xFFFFFFFF);
static constexpr int32_t RSHIFT_ON_NANO = 20;

template <class S>
__forceinline__ __simt_callee__ S make_epoch(const S& epoch) {
  return epoch << EPOCH_BITS;
}

template <class S>
__forceinline__ __simt_callee__ S make_nano(const S& cycle) {
  return (SCORE_BITS_MASK & (cycle >> RSHIFT_ON_NANO));
}

template <class K, class V, class S, int Strategy>
struct ScoreFunctor;

template <class K, class V, class S>
struct ScoreFunctor<K, V, S, EvictStrategyInternal::kLru> {
  using BUCKET = Bucket<K, V, S>;

  __forceinline__ __simt_callee__ static S desired_when_missed(
      __gm__ const S* __restrict const input_scores, const int32_t key_idx,
      const S& epoch, const S& cur_cycle) {
    return cur_cycle;
  }

  __forceinline__ __simt_callee__ static void update(
      __gm__ BUCKET* __restrict bucket, const int32_t key_pos,
      __gm__ const S* __restrict const input_scores, const int32_t key_idx,
      const S& desired_score_when_missed, const bool new_insert) {
    auto scores_ptr = (bucket->scores_) + key_pos;
    (void)Simt::AtomicExch(scores_ptr, desired_score_when_missed);
  }

  __forceinline__ __simt_callee__ static void update_with_digest(
      __gm__ K* __restrict bucket_key_ptr, const uint32_t& key_pos,
      __gm__ const S* __restrict const input_scores, const uint32_t& key_idx,
      const S& desired_score_when_missed, const uint32_t& bucket_capacity,
      const D& digest, const bool new_insert) {
    __gm__ S* dst_score_ptr =
        BUCKET::scores(bucket_key_ptr, bucket_capacity, key_pos);
    __gm__ D* dst_digest_ptr =
        BUCKET::digests(bucket_key_ptr, bucket_capacity, key_pos);
    // Cache in L2 cache, bypass L1 Cache.
    __stg<ST_L2CacheType::L2_CACHE_HINT_NORMAL_FV, L1CacheType::NON_CACHEABLE>(
        dst_digest_ptr, digest);
    __stg<ST_L2CacheType::L2_CACHE_HINT_NORMAL_FV, L1CacheType::NON_CACHEABLE>(
        dst_score_ptr, desired_score_when_missed);
  }

  __forceinline__ __simt_callee__ static void update_score_only(
      __gm__ K* __restrict bucket_key_ptr, const uint32_t& key_pos,
      __gm__ const S* __restrict const input_scores, const uint32_t& key_idx,
      const S& desired_score_when_missed, const uint32_t& bucket_capacity,
      const bool new_insert) {
      __gm__ S* dst_score_ptr =
        BUCKET::scores(bucket_key_ptr, bucket_capacity, key_pos);
        // Cache in L2 cache, bypass L1 Cache.
        __stg<ST_L2CacheType::L2_CACHE_HINT_NORMAL_FV, L1CacheType::NON_CACHEABLE>(
          dst_score_ptr, desired_score_when_missed);
  }

  __forceinline__ __simt_callee__ static void update_without_missed(
      __gm__ BUCKET* __restrict bucket, const int32_t key_pos,
      __gm__ const S* __restrict const input_scores, const int32_t key_idx,
      const S& epoch, const S& cur_cycle) {
    auto scores_ptr = (bucket->scores_) + key_pos;
    (void)Simt::AtomicExch(scores_ptr, cur_cycle);
  }

  __forceinline__ __simt_callee__ static void update_without_missed(
      __gm__ K* bucket_keys_ptr, const uint32_t bucket_capacity,
      const uint32_t key_pos, __gm__ const S* __restrict const input_scores,
      const int32_t key_idx, const S& epoch, const S& cur_cycle) {
    __gm__ S* dst_score_ptr =
        BUCKET::scores(bucket_keys_ptr, bucket_capacity, key_pos);
    // Cache in L2 cache, bypass L1 Cache.
    __stg<ST_L2CacheType::L2_CACHE_HINT_NORMAL_FV, L1CacheType::NON_CACHEABLE>(
        dst_score_ptr, cur_cycle);
  }
};

template <class K, class V, class S>
struct ScoreFunctor<K, V, S, EvictStrategyInternal::kLfu> {
  using BUCKET = Bucket<K, V, S>;

  __forceinline__ __simt_callee__ static S desired_when_missed(
      __gm__ const S* __restrict const input_scores, const int32_t key_idx,
      const S& epoch, const S& cur_cycle) {
    return static_cast<S>(MAX_SCORE);
  }

  __forceinline__ __simt_callee__ static void update(
      __gm__ BUCKET* __restrict bucket, const int32_t key_pos,
      __gm__ const S* __restrict const input_scores, const int32_t key_idx,
      const S& desired_score_when_missed, const bool new_insert) {
    if (input_scores == nullptr) {
      return;
    }
    auto scores_ptr = (bucket->scores_) + key_pos;
    if (new_insert) {
      (void)Simt::AtomicExch(scores_ptr, input_scores[key_idx]);
    } else {
      (void)Simt::AtomicAdd(scores_ptr, input_scores[key_idx]);
    }
  }

  __forceinline__ __simt_callee__ static void update_with_digest(
      __gm__ K* __restrict bucket_key_ptr, const uint32_t& key_pos,
      __gm__ const S* __restrict const input_scores, const uint32_t& key_idx,
      const S& desired_score_when_missed, const uint32_t& bucket_capacity,
      const D& digest, const bool new_insert) {
    if (input_scores == nullptr) {
      return;
    }

    __gm__ S* dst_score_ptr =
        BUCKET::scores(bucket_key_ptr, bucket_capacity, key_pos);
    __gm__ D* dst_digest_ptr =
        BUCKET::digests(bucket_key_ptr, bucket_capacity, key_pos);
    // Cache in L2 cache, bypass L1 Cache.
    __stg<ST_L2CacheType::L2_CACHE_HINT_NORMAL_FV, L1CacheType::NON_CACHEABLE>(
        dst_digest_ptr, digest);
    if (new_insert) {
      __stg<ST_L2CacheType::L2_CACHE_HINT_NORMAL_FV,
            L1CacheType::NON_CACHEABLE>(dst_score_ptr, input_scores[key_idx]);
    } else {
      __stg<ST_L2CacheType::L2_CACHE_HINT_NORMAL_FV,
            L1CacheType::NON_CACHEABLE>(dst_score_ptr,
                                        input_scores[key_idx] + *dst_score_ptr);
    }
  }

  __forceinline__ __simt_callee__ static void update_score_only(
      __gm__ K* __restrict bucket_key_ptr, const uint32_t& key_pos,
      __gm__ const S* __restrict const input_scores, const uint32_t& key_idx,
      const S& desired_score_when_missed, const uint32_t& bucket_capacity,
      const bool new_insert) {
    if (input_scores == nullptr) {
      return;
    }

    __gm__ S* dst_score_ptr =
        BUCKET::scores(bucket_key_ptr, bucket_capacity, key_pos);
    // Cache in L2 cache, bypass L1 Cache.
    if (new_insert) {
      __stg<ST_L2CacheType::L2_CACHE_HINT_NORMAL_FV,
            L1CacheType::NON_CACHEABLE>(dst_score_ptr, input_scores[key_idx]);
    } else {
      __stg<ST_L2CacheType::L2_CACHE_HINT_NORMAL_FV,
            L1CacheType::NON_CACHEABLE>(dst_score_ptr,
                                        input_scores[key_idx] + *dst_score_ptr);
    }
  }

  __forceinline__ __simt_callee__ static void update_without_missed(
      __gm__ BUCKET* __restrict bucket, const int32_t key_pos,
      __gm__ const S* __restrict const input_scores, const int32_t key_idx,
      const S& epoch, const S& cur_cycle) {
    if (input_scores == nullptr) {
      return;
    }
    auto scores_ptr = (bucket->scores_) + key_pos;
    (void)Simt::AtomicAdd(scores_ptr, input_scores[key_idx]);
  }

  __forceinline__ __simt_callee__ static void update_without_missed(
      __gm__ K* bucket_keys_ptr, const uint32_t bucket_capacity,
      const uint32_t key_pos, __gm__ const S* __restrict const input_scores,
      const int32_t key_idx, const S& epoch, const S& cur_cycle) {
    if (input_scores == nullptr) {
      return;
    }
    __gm__ S* dst_score_ptr =
        BUCKET::scores(bucket_keys_ptr, bucket_capacity, key_pos);
    // Cache in L2 cache, bypass L1 Cache.
    __stg<ST_L2CacheType::L2_CACHE_HINT_NORMAL_FV, L1CacheType::NON_CACHEABLE>(
        dst_score_ptr, input_scores[key_idx] + *dst_score_ptr);
  }
};

template <class K, class V, class S>
struct ScoreFunctor<K, V, S, EvictStrategyInternal::kEpochLru> {
  using BUCKET = Bucket<K, V, S>;

  __forceinline__ __simt_callee__ static S desired_when_missed(
      __gm__ const S* __restrict const input_scores, const int32_t key_idx,
      const S& epoch, const S& cur_cycle) {
    if (epoch == static_cast<S>(IGNORED_GLOBAL_EPOCH) &&
        input_scores != nullptr) {
      return input_scores[key_idx];
    }
    return make_epoch<S>(epoch) | make_nano<S>(cur_cycle);
  }

  __forceinline__ __simt_callee__ static void update(
      __gm__ BUCKET* __restrict bucket, const int32_t key_pos,
      __gm__ const S* __restrict const input_scores, const int32_t key_idx,
      const S& desired_score_when_missed, const bool new_insert) {
    auto scores_ptr = (bucket->scores_) + key_pos;
    (void)Simt::AtomicExch(scores_ptr, desired_score_when_missed);
  }

  __forceinline__ __simt_callee__ static void update_with_digest(
      __gm__ K* __restrict bucket_key_ptr, const uint32_t& key_pos,
      __gm__ const S* __restrict const input_scores, const uint32_t& key_idx,
      const S& desired_score_when_missed, const uint32_t& bucket_capacity,
      const D& digest, const bool new_insert) {
    __gm__ S* dst_score_ptr =
        BUCKET::scores(bucket_key_ptr, bucket_capacity, key_pos);
    __gm__ D* dst_digest_ptr =
        BUCKET::digests(bucket_key_ptr, bucket_capacity, key_pos);
    // Cache in L2 cache, bypass L1 Cache.
    __stg<ST_L2CacheType::L2_CACHE_HINT_NORMAL_FV, L1CacheType::NON_CACHEABLE>(
        dst_digest_ptr, digest);
    __stg<ST_L2CacheType::L2_CACHE_HINT_NORMAL_FV, L1CacheType::NON_CACHEABLE>(
        dst_score_ptr, desired_score_when_missed);
  }

  __forceinline__ __simt_callee__ static void update_score_only(
      __gm__ K* __restrict bucket_key_ptr, const uint32_t& key_pos,
      __gm__ const S* __restrict const input_scores, const uint32_t& key_idx,
      const S& desired_score_when_missed, const uint32_t& bucket_capacity,
      const bool new_insert) {
    __gm__ S* dst_score_ptr =
        BUCKET::scores(bucket_key_ptr, bucket_capacity, key_pos);
    // Cache in L2 cache, bypass L1 Cache.
    __stg<ST_L2CacheType::L2_CACHE_HINT_NORMAL_FV, L1CacheType::NON_CACHEABLE>(
        dst_score_ptr, desired_score_when_missed);
  }

  __forceinline__ __simt_callee__ static void update_without_missed(
      __gm__ BUCKET* __restrict bucket, const int32_t key_pos,
      __gm__ const S* __restrict const input_scores, const int32_t key_idx,
      const S& epoch, const S& cur_cycle) {
    auto scores_ptr = (bucket->scores_) + key_pos;
    (void)Simt::AtomicExch(scores_ptr,
                           make_epoch<S>(epoch) | make_nano<S>(cur_cycle));
  }

  __forceinline__ __simt_callee__ static void update_without_missed(
      __gm__ K* bucket_keys_ptr, const uint32_t bucket_capacity,
      const uint32_t key_pos, __gm__ const S* __restrict const input_scores,
      const int32_t key_idx, const S& epoch, const S& cur_cycle) {
    __gm__ S* dst_score_ptr =
        BUCKET::scores(bucket_keys_ptr, bucket_capacity, key_pos);
    // Cache in L2 cache, bypass L1 Cache.
    __stg<ST_L2CacheType::L2_CACHE_HINT_NORMAL_FV, L1CacheType::NON_CACHEABLE>(
        dst_score_ptr, make_epoch<S>(epoch) | make_nano<S>(cur_cycle));
  }
};

template <class K, class V, class S>
struct ScoreFunctor<K, V, S, EvictStrategyInternal::kEpochLfu> {
  using BUCKET = Bucket<K, V, S>;

  __forceinline__ __simt_callee__ static S desired_when_missed(
      __gm__ const S* __restrict const input_scores, const int32_t key_idx,
      const S& epoch, const S& cur_cycle) {
    if (epoch == static_cast<S>(IGNORED_GLOBAL_EPOCH)) {
      return input_scores[key_idx];
    }
    return make_epoch<S>(epoch) | (input_scores[key_idx] & SCORE_BITS_MASK);
  }

  __forceinline__ __simt_callee__ static void update(
      __gm__ BUCKET* __restrict bucket, const int32_t key_pos,
      __gm__ const S* __restrict const input_scores, const int32_t key_idx,
      const S& desired_score_when_missed, const bool new_insert) {
    S new_score = desired_score_when_missed;
    auto scores_ptr = (bucket->scores_) + key_pos;
    if (!new_insert) {
      new_score = (*scores_ptr & SCORE_BITS_MASK);
      if (SCORE_32BIT_MAX - new_score >
          (desired_score_when_missed & SCORE_BITS_MASK)) {
        new_score += desired_score_when_missed;
      } else {
        new_score =
            (desired_score_when_missed & EPOCH_BITS_MASK) | SCORE_32BIT_MAX;
      }
    }
    (void)Simt::AtomicExch(scores_ptr, new_score);
  }

  __forceinline__ __simt_callee__ static void update_with_digest(
      __gm__ K* __restrict bucket_key_ptr, const uint32_t& key_pos,
      __gm__ const S* __restrict const input_scores, const uint32_t& key_idx,
      const S& desired_score_when_missed, const uint32_t& bucket_capacity,
      const D& digest, const bool new_insert) {
    S new_score = desired_score_when_missed;
    __gm__ S* dst_score_ptr =
        BUCKET::scores(bucket_key_ptr, bucket_capacity, key_pos);
    __gm__ D* dst_digest_ptr =
        BUCKET::digests(bucket_key_ptr, bucket_capacity, key_pos);
    if (!new_insert) {
      new_score = (*dst_score_ptr & SCORE_BITS_MASK);
      if (SCORE_32BIT_MAX - new_score >
          (desired_score_when_missed & SCORE_BITS_MASK)) {
        new_score += desired_score_when_missed;
      } else {
        new_score =
            (desired_score_when_missed & EPOCH_BITS_MASK) | SCORE_32BIT_MAX;
      }
    }
    // Cache in L2 cache, bypass L1 Cache.
    __stg<ST_L2CacheType::L2_CACHE_HINT_NORMAL_FV, L1CacheType::NON_CACHEABLE>(
        dst_digest_ptr, digest);
    __stg<ST_L2CacheType::L2_CACHE_HINT_NORMAL_FV, L1CacheType::NON_CACHEABLE>(
        dst_score_ptr, new_score);
  }

  __forceinline__ __simt_callee__ static void update_score_only(
      __gm__ K* __restrict bucket_key_ptr, const uint32_t& key_pos,
      __gm__ const S* __restrict const input_scores, const uint32_t& key_idx,
      const S& desired_score_when_missed, const uint32_t& bucket_capacity,
      const bool new_insert) {
    S new_score = desired_score_when_missed;
    __gm__ S* dst_score_ptr =
        BUCKET::scores(bucket_key_ptr, bucket_capacity, key_pos);
    if (!new_insert) {
      new_score = (*dst_score_ptr & SCORE_BITS_MASK);
      if (SCORE_32BIT_MAX - new_score >
          (desired_score_when_missed & SCORE_BITS_MASK)) {
        new_score += desired_score_when_missed;
      } else {
        new_score =
            (desired_score_when_missed & EPOCH_BITS_MASK) | SCORE_32BIT_MAX;
      }
    }
    // Cache in L2 cache, bypass L1 Cache.
    __stg<ST_L2CacheType::L2_CACHE_HINT_NORMAL_FV, L1CacheType::NON_CACHEABLE>(
        dst_score_ptr, new_score);
  }

  __forceinline__ __simt_callee__ static void update_without_missed(
      __gm__ BUCKET* __restrict bucket, const int32_t key_pos,
      __gm__ const S* __restrict const input_scores, const int32_t key_idx,
      const S& epoch, const S& cur_cycle) {
    if (input_scores == nullptr) {
      return;
    }
    auto scores_ptr = (bucket->scores_) + key_pos;
    S new_score = (*scores_ptr & SCORE_BITS_MASK);
    if (SCORE_32BIT_MAX - new_score >
        (input_scores[key_idx] & SCORE_BITS_MASK)) {
      new_score +=
          (make_epoch<S>(epoch) | (input_scores[key_idx] & SCORE_BITS_MASK));
    } else {
      new_score = make_epoch<S>(epoch) | SCORE_32BIT_MAX;
    }
    (void)Simt::AtomicExch(scores_ptr, new_score);
  }

  __forceinline__ __simt_callee__ static void update_without_missed(
      __gm__ K* bucket_keys_ptr, const uint32_t bucket_capacity,
      const uint32_t key_pos, __gm__ const S* __restrict const input_scores,
      const int32_t key_idx, const S& epoch, const S& cur_cycle) {
    if (input_scores == nullptr) {
      return;
    }
    __gm__ S* dst_score_ptr =
        BUCKET::scores(bucket_keys_ptr, bucket_capacity, key_pos);
    S new_score = *dst_score_ptr & SCORE_BITS_MASK;
    if (SCORE_32BIT_MAX - new_score >
        (input_scores[key_idx] & SCORE_BITS_MASK)) {
      new_score +=
          (make_epoch<S>(epoch) | (input_scores[key_idx] & SCORE_BITS_MASK));
    } else {
      new_score = make_epoch<S>(epoch) | SCORE_32BIT_MAX;
    }
    // Cache in L2 cache, bypass L1 Cache.
    __stg<ST_L2CacheType::L2_CACHE_HINT_NORMAL_FV, L1CacheType::NON_CACHEABLE>(
        dst_score_ptr, new_score);
  }
};

template <class K, class V, class S>
struct ScoreFunctor<K, V, S, EvictStrategyInternal::kCustomized> {
  using BUCKET = Bucket<K, V, S>;

  __forceinline__ __simt_callee__ static S desired_when_missed(
      __gm__ const S* __restrict const input_scores, const int32_t key_idx,
      const S& epoch, const S& cur_cycle) {
    return input_scores[key_idx];
  }

  __forceinline__ __simt_callee__ static void update(
      __gm__ BUCKET* __restrict bucket, const int32_t key_pos,
      __gm__ const S* __restrict const input_scores, const int32_t key_idx,
      const S& desired_score_when_missed, const bool new_insert) {
    auto scores_ptr = (bucket->scores_) + key_pos;
    (void)Simt::AtomicExch(scores_ptr, desired_score_when_missed);
  }

  __forceinline__ __simt_callee__ static void update_with_digest(
      __gm__ K* __restrict bucket_key_ptr, const uint32_t& key_pos,
      __gm__ const S* __restrict const input_scores, const uint32_t& key_idx,
      const S& desired_score_when_missed, const uint32_t& bucket_capacity,
      const D& digest, const bool new_insert) {
    __gm__ S* dst_score_ptr =
        BUCKET::scores(bucket_key_ptr, bucket_capacity, key_pos);
    __gm__ D* dst_digest_ptr =
        BUCKET::digests(bucket_key_ptr, bucket_capacity, key_pos);
    // Cache in L2 cache, bypass L1 Cache.
    __stg<ST_L2CacheType::L2_CACHE_HINT_NORMAL_FV, L1CacheType::NON_CACHEABLE>(
        dst_digest_ptr, digest);
    __stg<ST_L2CacheType::L2_CACHE_HINT_NORMAL_FV, L1CacheType::NON_CACHEABLE>(
        dst_score_ptr, desired_score_when_missed);
  }

  __forceinline__ __simt_callee__ static void update_score_only(
      __gm__ K* __restrict bucket_key_ptr, const uint32_t& key_pos,
      __gm__ const S* __restrict const input_scores, const uint32_t& key_idx,
      const S& desired_score_when_missed, const uint32_t& bucket_capacity,
      const bool new_insert) {
    __gm__ S* dst_score_ptr =
        BUCKET::scores(bucket_key_ptr, bucket_capacity, key_pos);
    // Cache in L2 cache, bypass L1 Cache.
    __stg<ST_L2CacheType::L2_CACHE_HINT_NORMAL_FV, L1CacheType::NON_CACHEABLE>(
        dst_score_ptr, desired_score_when_missed);
  }

  __forceinline__ __simt_callee__ static void update_without_missed(
      __gm__ BUCKET* __restrict bucket, const int32_t key_pos,
      __gm__ const S* __restrict const input_scores, const int32_t key_idx,
      const S& epoch, const S& cur_cycle) {
    if (input_scores == nullptr) {
      return;
    }
    auto scores_ptr = (bucket->scores_) + key_pos;
    (void)Simt::AtomicExch(scores_ptr, input_scores[key_idx]);
  }

  __forceinline__ __simt_callee__ static void update_without_missed(
      __gm__ K* bucket_keys_ptr, const uint32_t bucket_capacity,
      const uint32_t key_pos, __gm__ const S* __restrict const input_scores,
      const int32_t key_idx, const S& epoch, const S& cur_cycle) {
    if (input_scores == nullptr) {
      return;
    }
    __gm__ S* dst_score_ptr =
        BUCKET::scores(bucket_keys_ptr, bucket_capacity, key_pos);
    // Cache in L2 cache, bypass L1 Cache.
    __stg<ST_L2CacheType::L2_CACHE_HINT_NORMAL_FV, L1CacheType::NON_CACHEABLE>(
        dst_score_ptr, input_scores[key_idx]);
  }
};

}  // namespace hkv
}  // namespace npu
