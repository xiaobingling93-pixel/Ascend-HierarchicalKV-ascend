/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef ASCENDC_REHASH_KERNEL_H_
#define ASCENDC_REHASH_KERNEL_H_

#include "kernel_operator.h"
#include "../../../include/score_functor.h"
#include "../../../include/types.h"
#include "../../../include/utils.h"

namespace npu {
namespace hkv {
using namespace AscendC;

template <class K, class V, class S>
__inline__ __simt_callee__ void copy_key_to_new_bucket(
    const K& key, const S& score, __gm__ const V* __restrict__ vector,
    __gm__ Bucket<K, V, S>* __restrict__ new_bucket,
    __gm__ int32_t* __restrict__ new_bucket_size, const uint32_t& new_key_pos,
    const size_t& bucket_max_size, const size_t& dim) {
  // 1. 遍历新桶，找到空key
  for (uint32_t offset = 0; offset < bucket_max_size; offset++) {
    uint32_t cur_pos = (new_key_pos + offset) % bucket_max_size;
    // 2. 复制旧桶key
    if (new_bucket->keys_[cur_pos] == EMPTY_KEY) {
      new_bucket->keys_[cur_pos] = key;
      new_bucket->scores_[cur_pos] = score;
      *Bucket<K, V, S>::digests(new_bucket->keys_, bucket_max_size, cur_pos) = get_digest<K>(key);
      (*new_bucket_size)++;

      size_t vector_start_pos = cur_pos * dim;
      for (size_t i = 0; i < dim; i++) {
        new_bucket->vectors[vector_start_pos + i] = vector[i];
      }
      break;
    }
  }
}

/* 压缩的目的是让真实存储位置key_pos更靠近理想位置，即find过程中的offset更少
 * 因此，如下3中场景符合压缩条件（其中，*为理想位置，[]为空位，|为当前位置）
 * 1. -----*-----[]-----|-----
 * 2. -----|-----*-----[]-----
 * 3. -----[]-----|-----*-----
 * []与|位置替换后，由*->|的距离缩小
 */
template <class K, class V, class S>
__inline__ __simt_callee__ void defragmentation_for_rehash(
    __gm__ Bucket<K, V, S>* __restrict__ cur_bucket, uint32_t remove_pos,
    const size_t& bucket_max_size, const size_t& old_buckets_num,
    const size_t& dim) {
  // 1. 从空位置的后一个key开始遍历整个桶
  size_t offset = 1;
  while (offset < bucket_max_size) {
    size_t cur_pos = (remove_pos + offset) % bucket_max_size;
    K cur_key = cur_bucket->keys_[cur_pos];
    // 2. 理想位置要在空位置前面，则key一定是连续有效的，不能为空key
    if (cur_key == EMPTY_KEY) {
      break;
    }

    // 3. 计算理想位置
    K hashed_key = Murmur3HashDevice(cur_key);
    uint64_t global_idx =
        static_cast<uint64_t>(hashed_key % (old_buckets_num * bucket_max_size));
    size_t start_pos = global_idx % bucket_max_size;
    if ((start_pos <= remove_pos && remove_pos < cur_pos) ||
        (cur_pos < start_pos && start_pos <= remove_pos) ||
        (remove_pos < cur_pos && cur_pos < start_pos)) {
      // 4. 找到符合场景，将目标key前移，后将目标key位置置空
      cur_bucket->keys_[remove_pos] = cur_key;
      cur_bucket->scores_[remove_pos] = cur_bucket->scores_[cur_pos];
      *Bucket<K, V, S>::digests(cur_bucket->keys_, bucket_max_size, remove_pos) = get_digest<K>(cur_key);
      for (size_t i = 0; i < dim; i++) {
        cur_bucket->vectors[remove_pos * dim + i] =
            cur_bucket->vectors[cur_pos * dim + i];
      }

      cur_bucket->keys_[cur_pos] = EMPTY_KEY;
      *Bucket<K, V, S>::digests(cur_bucket->keys_, bucket_max_size, cur_pos) = empty_digest<K>();

      // 5. 当前位置变为新的空位置，重新循环
      remove_pos = cur_pos;
      offset = 1;
    } else {
      offset++;
    }
  }
}

constexpr uint32_t THREAD_NUM = 512;
template <typename K = uint64_t, typename V = float, typename S = uint64_t>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_NUM) inline void rehash_kernel_vf(
    __gm__ Table<uint64_t, float, uint64_t>* __restrict__ table,
    const size_t old_buckets_num, const uint64_t thread_all,
    const uint32_t block_index) {
  const size_t new_buckets_num = table->buckets_num;
  const size_t bucket_max_size = table->bucket_max_size;
  const size_t dim = table->dim;

  K cur_key = 0;
  S cur_score = 0;
  K hashed_key = 0;
  uint64_t new_global_idx = 0;
  uint64_t bkt_idx = block_index * blockDim.x + threadIdx.x;
  for (; bkt_idx < old_buckets_num; bkt_idx += thread_all) {
    // 1. 每个线程处理一个桶
    auto cur_bucket = table->buckets + bkt_idx;
    // 2. 遍历桶内key，进行rehash
    uint32_t key_pos = 0;
    while (key_pos < bucket_max_size) {
      cur_key = cur_bucket->keys_[key_pos];
      cur_score = cur_bucket->scores_[key_pos];
      if ((cur_key == EMPTY_KEY) || (cur_key == RECLAIM_KEY)) {
        key_pos++;
        continue;
      }

      // 3. rehash非空key值
      hashed_key = Murmur3HashDevice(cur_key);
      new_global_idx = static_cast<uint64_t>(
          hashed_key % (new_buckets_num * bucket_max_size));
      uint64_t new_bkt_idx = new_global_idx / bucket_max_size;
      if (new_bkt_idx == bkt_idx) {
        key_pos++;
        continue;
      }

      // 4. 搬运key
      uint32_t new_key_pos = new_global_idx % bucket_max_size;
      copy_key_to_new_bucket(
          cur_key, cur_score, (cur_bucket->vectors + key_pos * dim),
          table->buckets + new_bkt_idx, table->buckets_size + new_bkt_idx,
          new_key_pos, bucket_max_size, dim);
      cur_bucket->keys_[key_pos] = EMPTY_KEY;
      *Bucket<K, V, S>::digests(cur_bucket->keys_, bucket_max_size, key_pos) = empty_digest<K>();
      table->buckets_size[bkt_idx]--;

      // 5. 压缩碎片化分布的key。由于key被重排列，因此重新遍历桶。
      defragmentation_for_rehash(cur_bucket, key_pos, bucket_max_size,
                                 old_buckets_num, dim);
      key_pos = 0;
    }
  }
}
}  // namespace hkv
}  // namespace npu

#endif  // ASCENDC_REHASH_KERNEL_H_

