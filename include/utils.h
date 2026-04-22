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

#include <stdarg.h>
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <limits>
#include <string>
#include "cuda2npu.h"
#if (!defined(__CCE__))
#include <acl/acl.h>
#include "aclrtlaunch_host_nano_kernel.h"
#include "debug.h"
#endif

namespace npu {
namespace hkv {

#if (!defined(__CCE__))
static inline size_t SAFE_GET_GRID_SIZE(size_t N, int block_size) {
  return ((N) > std::numeric_limits<size_t>::max())
             ? (((1 << 30) - 1) / block_size + 1)
             : (((N)-1) / block_size + 1);
}
#endif

static inline int SAFE_GET_BLOCK_SIZE(int block_size, int device = -1) {
  return std::min(2048, block_size);
}

inline uint64_t Murmur3HashHost(const uint64_t& key) {
  uint64_t k = key;
  k ^= k >> 33;
  k *= UINT64_C(0xff51afd7ed558ccd);
  k ^= k >> 33;
  k *= UINT64_C(0xc4ceb9fe1a85ec53);
  k ^= k >> 33;
  return k;
}

#if defined(__CCE__)
__inline__ __simt_callee__ uint64_t Murmur3HashDevice(uint64_t const& key) {
  uint64_t k = key;
  k ^= k >> 33;
  k *= UINT64_C(0xff51afd7ed558ccd);
  k ^= k >> 33;
  k *= UINT64_C(0xc4ceb9fe1a85ec53);
  k ^= k >> 33;
  return k;
}
#endif

#if defined(__CCE__)
/* 本函数原公式为：global_idx = hashed_key % capacity
 * 现由快除法计算出div_res = hashed_key / capacity，再由global_idx = hashed_key - div_res * capacity
 * uint64_t快除法：
 * r = x / y，如果y的值固定，则除法可以等效替换为如下公式
 * 预计算部分（host）：
 * 1. shift = ceil(log2(y))
 * 2. magic = ceil(2 ^ (64 + shift) / y)
 * 运行时计算部分（本函数内）：
 * 3. q = (x * magic) >> 64 由__umul64hi完成
 * 4. t = ((x - q) >> 1) + q
 * 5. r = t >> (shift - 1)
 */
__inline__ __simt_callee__ uint64_t get_global_idx(
    const uint64_t& hashed_key, const uint64_t& capacity_divisor_magic,
    const uint64_t& capacity_divisor_shift, const uint64_t& capacity) {
#ifdef FORBID_QUICK_DIV
  return hashed_key % capacity;
#else
  uint64_t div_tmp = __umul64hi(hashed_key, capacity_divisor_magic);
  div_tmp = ((hashed_key - div_tmp) >> 1) + div_tmp;
  uint64_t div_result = div_tmp >> capacity_divisor_shift;
  return hashed_key - div_result * capacity;
#endif  // FORBID_QUICK_DIV
}
#endif  // __CCE__

static inline size_t GB(size_t n) { return n << 30; }

static inline size_t MB(size_t n) { return n << 20; }

static inline size_t KB(size_t n) { return n << 10; }

constexpr inline bool ispow2(unsigned x) { return x && (!(x & (x - 1))); }

#if (!defined(__CCE__))
template <class S>
S host_nano(aclrtStream stream = 0) {
  static_assert(
      (std::is_same<S, int64_t>::value || std::is_same<S, uint64_t>::value),
      "The S must be int64_t or uint64_t.");
  S h_clk = 0;
  S* d_clk;

  NPU_CHECK(aclrtMalloc((void**)&d_clk, sizeof(S), ACL_MEM_MALLOC_HUGE_FIRST));

  ACLRT_LAUNCH_KERNEL(host_nano_kernel)
  (1, 0, d_clk);

  NPU_CHECK(aclrtSynchronizeStream(stream));

  NPU_CHECK(aclrtMemcpy(&h_clk, sizeof(S), d_clk, sizeof(S),
                        ACL_MEMCPY_DEVICE_TO_HOST));
  NPU_CHECK(aclrtFree(d_clk));
  return h_clk;
}

static inline bool is_on_device(const void* ptr) {
  aclrtPtrAttributes attr;
  NPU_CHECK(aclrtPointerGetAttributes(ptr, &attr));

  return (attr.location.type == ACL_MEM_LOCATION_TYPE_DEVICE);
}
#endif

#if defined(__CCE__)
template <typename T>
__forceinline__ __simt_callee__ T ldg_l2nc_l1c(__gm__ T* ptr) {
  return __ldg<LD_L2CacheType::L2_CACHE_HINT_NOTALLOC_CLEAN, L1CacheType::CACHEABLE>(ptr);
}
#endif

}  // namespace hkv
}  // namespace npu
