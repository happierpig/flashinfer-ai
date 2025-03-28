#ifndef FLASHINFER_POD_CUH_
#define FLASHINFER_POD_CUH_

#include <cooperative_groups.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include "../cp_async.cuh"
#include "../fastdiv.cuh"
#include "../frag_layout_swizzle.cuh"
#include "../layout.cuh"
#include "../math.cuh"
#include "../mma.cuh"
#include "../page.cuh"
#include "../permuted_smem.cuh"
#include "../pos_enc.cuh"
#include "../utils.cuh"
#include "cascade.cuh"
#include "decode.cuh"
#include "mask.cuh"
#include "prefill.cuh"
#include "variants.cuh"

namespace flashinfer {

namespace PoDOp {
constexpr int PREFILL = 0;
constexpr int DECODE = 1;
constexpr int NUM_OPS = 2;
}  // namespace PoDOp

template <typename KTraits_P, typename KTraits_D, typename PrefillParams, typename DecodeParams>
__global__ __launch_bounds__(std::max(
    KTraits_P::NUM_THREADS,
    KTraits_D::NUM_THREADS)) void PODWithPagedKVCacheKernel(const __grid_constant__ PrefillParams
                                                                prefill_params,
                                                            const __grid_constant__ DecodeParams
                                                                decode_params) {
  extern __shared__ uint8_t smem[];

  const uint32_t padded_bsz_p = prefill_params.padded_batch_size;
  const uint32_t num_kv_heads_p = prefill_params.paged_kv.num_heads;
  const uint32_t padded_bsz_d = decode_params.padded_batch_size;
  const uint32_t num_kv_heads_d = decode_params.paged_kv.num_heads;

  const uint32_t num_blk_p = padded_bsz_p * num_kv_heads_p;
  const uint32_t num_blk_d = padded_bsz_d * num_kv_heads_d;
  const uint32_t physical_bid = blockIdx.x;

  if (physical_bid < num_blk_p) {
    /* OP == PREFILL */
    auto& smem_storage = reinterpret_cast<typename KTraits_P::SharedStorage&>(smem);

    const uint32_t logical_bid = physical_bid % padded_bsz_p;
    const uint32_t kv_head_idx = physical_bid / padded_bsz_p;
    const uint32_t physical_tid = threadIdx.x;

    // Return if threadId exceeds number of threads for this op
    if (physical_tid >= WARP_SIZE * KTraits_P::NUM_WARPS_Q * KTraits_P::NUM_WARPS_KV) return;

    const dim3 tid =
        dim3(physical_tid % WARP_SIZE, (physical_tid / WARP_SIZE) % KTraits_P::NUM_WARPS_Q,
             (physical_tid / WARP_SIZE) / KTraits_P::NUM_WARPS_Q);

    BatchPrefillWithPagedKVCacheDevice<KTraits_P>(prefill_params, smem_storage, tid, logical_bid,
                                                  kv_head_idx, num_kv_heads_p);
  } else /* OP == DECODE */ {
    auto& smem_storage = reinterpret_cast<typename KTraits_D::SharedStorage&>(smem);

    const uint32_t logical_bid = (physical_bid - num_blk_p) % padded_bsz_d;
    const uint32_t kv_head_idx = (physical_bid - num_blk_p) / padded_bsz_d;
    const uint32_t physical_tid = threadIdx.x;

    // Return if threadId exceeds number of threads for this op
    if (physical_tid >= WARP_SIZE * KTraits_D::NUM_WARPS_Q * KTraits_D::NUM_WARPS_KV) return;

    const dim3 tid =
        dim3(physical_tid % WARP_SIZE, (physical_tid / WARP_SIZE) % KTraits_D::NUM_WARPS_Q,
             (physical_tid / WARP_SIZE) / KTraits_D::NUM_WARPS_Q);

    BatchPrefillWithPagedKVCacheDevice<KTraits_D>(decode_params, smem_storage, tid, logical_bid,
                                                  kv_head_idx, num_kv_heads_d);
  }
}

template <uint32_t CTA_TILE_Q_P, uint32_t CTA_TILE_Q_D, uint32_t HEAD_DIM_QK, uint32_t HEAD_DIM_VO,
          PosEncodingMode POS_ENCODING_MODE, bool USE_FP16_QK_REDUCTION, MaskMode MASK_MODE,
          typename AttentionVariant, typename PrefillParams, typename DecodeParams>
cudaError_t PODWithPagedKVCacheDispatched(PrefillParams prefill_params, DecodeParams decode_params,
                                          typename DecodeParams::DTypeO* tmp_v, float* tmp_s,
                                          cudaStream_t stream) {
  using DTypeQ = typename PrefillParams::DTypeQ;
  using DTypeKV = typename PrefillParams::DTypeKV;
  using DTypeO = typename PrefillParams::DTypeO;
  const uint32_t num_qo_heads = prefill_params.num_qo_heads;
  const uint32_t num_kv_heads = prefill_params.paged_kv.num_heads;

  static_assert(std::is_same<DTypeQ, typename DecodeParams::DTypeQ>::value);
  static_assert(std::is_same<DTypeKV, typename DecodeParams::DTypeKV>::value);
  static_assert(std::is_same<DTypeO, typename DecodeParams::DTypeO>::value);
  assert(num_qo_heads == decode_params.num_qo_heads);
  assert(num_kv_heads == decode_params.paged_kv.num_heads);

  const uint32_t padded_bsz_p = prefill_params.padded_batch_size;
  const uint32_t padded_bsz_d = decode_params.padded_batch_size;

  if (padded_bsz_p == 0 && padded_bsz_d == 0) {
    // No request, skip
    return cudaSuccess;
  }

  constexpr uint32_t NUM_MMA_D_QK = HEAD_DIM_QK / 16;
  constexpr uint32_t NUM_MMA_D_VO = HEAD_DIM_VO / 16;
  using DTypeQKAccum =
      typename std::conditional<USE_FP16_QK_REDUCTION && std::is_same_v<DTypeQ, half>, half,
                                float>::type;

  // Prefill metadata setups
  constexpr uint32_t NUM_MMA_Q_P = get_num_mma_q(CTA_TILE_Q_P);
  constexpr uint32_t NUM_WARPS_Q_P = get_num_warps_q(CTA_TILE_Q_P);
  constexpr uint32_t NUM_WARPS_KV_P = get_num_warps_kv(CTA_TILE_Q_P);

  // Decode metadata setups
  constexpr uint32_t NUM_MMA_Q_D = get_num_mma_q(CTA_TILE_Q_D);
  constexpr uint32_t NUM_WARPS_Q_D = get_num_warps_q(CTA_TILE_Q_D);
  constexpr uint32_t NUM_WARPS_KV_D = get_num_warps_kv(CTA_TILE_Q_D);

  int nblks_p(padded_bsz_p * 1 * num_kv_heads);
  int nthrs_p(32 * NUM_WARPS_Q_P * NUM_WARPS_KV_P);
  int nblks_d(padded_bsz_d * 1 * num_kv_heads);
  int nthrs_d(32 * NUM_WARPS_Q_D * NUM_WARPS_KV_D);

  int nblks = nblks_p + nblks_d;
  int nthrs = max(nthrs_p, nthrs_d);

  // Calculate occupancy
  // we expect each sm execute two threadblocks
  int dev_id = 0;
  FLASHINFER_CUDA_CALL(cudaGetDevice(&dev_id));
  int max_smem_per_sm = 0;
  FLASHINFER_CUDA_CALL(cudaDeviceGetAttribute(&max_smem_per_sm,
                                              cudaDevAttrMaxSharedMemoryPerMultiprocessor, dev_id));
  int num_sm = 0;
  FLASHINFER_CUDA_CALL(cudaDeviceGetAttribute(&num_sm, cudaDevAttrMultiProcessorCount, dev_id));

  // Prefill occupancy
  const int num_ctas_per_sm_p =
      max_smem_per_sm >= 2 * (CTA_TILE_Q_P * HEAD_DIM_QK * sizeof(DTypeQ) +
                              (HEAD_DIM_QK + HEAD_DIM_VO) * 16 * NUM_WARPS_KV_P * sizeof(DTypeKV))
          ? 2
          : 1;
  const int max_smem_per_threadblock_p = max_smem_per_sm / num_ctas_per_sm_p;
  const uint32_t max_num_mma_kv_smem_p =
      (max_smem_per_threadblock_p - CTA_TILE_Q_P * HEAD_DIM_QK * sizeof(DTypeQ)) /
      ((HEAD_DIM_QK + HEAD_DIM_VO) * 16 * NUM_WARPS_KV_P * sizeof(DTypeKV));
  const uint32_t max_num_mma_kv_reg_p =
      (HEAD_DIM_VO >= 128 && NUM_MMA_Q_P == 2 && POS_ENCODING_MODE == PosEncodingMode::kRoPELlama &&
       !USE_FP16_QK_REDUCTION)
          ? 2
          : (8 / NUM_MMA_Q_P);

  // Decode occupancy
  const int num_ctas_per_sm_d =
      max_smem_per_sm >= 2 * (CTA_TILE_Q_D * HEAD_DIM_QK * sizeof(DTypeQ) +
                              (HEAD_DIM_QK + HEAD_DIM_VO) * 16 * NUM_WARPS_KV_D * sizeof(DTypeKV))
          ? 2
          : 1;
  const int max_smem_per_threadblock_d = max_smem_per_sm / num_ctas_per_sm_d;
  const uint32_t max_num_mma_kv_smem_d =
      (max_smem_per_threadblock_d - CTA_TILE_Q_D * HEAD_DIM_QK * sizeof(DTypeQ)) /
      ((HEAD_DIM_QK + HEAD_DIM_VO) * 16 * NUM_WARPS_KV_D * sizeof(DTypeKV));
  const uint32_t max_num_mma_kv_reg_d =
      (HEAD_DIM_VO >= 128 && NUM_MMA_Q_D == 2 && POS_ENCODING_MODE == PosEncodingMode::kRoPELlama &&
       !USE_FP16_QK_REDUCTION)
          ? 2
          : (8 / NUM_MMA_Q_D);

  DISPATCH_NUM_MMA_KV(min(max_num_mma_kv_smem_p, max_num_mma_kv_reg_p), NUM_MMA_KV_P, {
    DISPATCH_NUM_MMA_KV(min(max_num_mma_kv_smem_d, max_num_mma_kv_reg_d), NUM_MMA_KV_D, {
      using KTraits_P = KernelTraits<MASK_MODE, CTA_TILE_Q_P, NUM_MMA_Q_P, NUM_MMA_KV_P,
                                     NUM_MMA_D_QK, NUM_MMA_D_VO, NUM_WARPS_Q_P, NUM_WARPS_KV_P,
                                     POS_ENCODING_MODE, DTypeQ, DTypeKV, DTypeO, DTypeQKAccum,
                                     typename PrefillParams::IdType, AttentionVariant>;
      using KTraits_D = KernelTraits<MASK_MODE, CTA_TILE_Q_D, NUM_MMA_Q_D, NUM_MMA_KV_D,
                                     NUM_MMA_D_QK, NUM_MMA_D_VO, NUM_WARPS_Q_D, NUM_WARPS_KV_D,
                                     POS_ENCODING_MODE, DTypeQ, DTypeKV, DTypeO, DTypeQKAccum,
                                     typename DecodeParams::IdType, AttentionVariant>;
      if constexpr (KTraits_D::IsInvalid() || KTraits_P::IsInvalid()) {
        // Invalid configuration, skip
        std::ostringstream err_msg;
        err_msg << "FlashInfer Internal Error: Invalid configuration : NUM_MMA_Q_P=" << NUM_MMA_Q_P
                << " NUM_MMA_D_QK=" << NUM_MMA_D_QK << " NUM_MMA_D_VO=" << NUM_MMA_D_VO
                << " NUM_MMA_KV_P=" << NUM_MMA_KV_P << " NUM_WARPS_Q_P=" << NUM_WARPS_Q_P
                << " NUM_WARPS_KV_P=" << NUM_WARPS_KV_P << std::endl;
        err_msg << "FlashInfer Internal Error: Invalid configuration : NUM_MMA_Q_D=" << NUM_MMA_Q_D
                << " NUM_MMA_D_QK=" << NUM_MMA_D_QK << " NUM_MMA_D_VO=" << NUM_MMA_D_VO
                << " NUM_MMA_KV_D=" << NUM_MMA_KV_D << " NUM_WARPS_Q_D=" << NUM_WARPS_Q_D
                << " NUM_WARPS_KV_D=" << NUM_WARPS_KV_D
                << " please create an issue (https://github.com/flashinfer-ai/flashinfer/issues)"
                   " and report the issue to the developers.";
        FLASHINFER_ERROR(err_msg.str());
      } else {
        size_t smem_size_p = sizeof(typename KTraits_P::SharedStorage);
        size_t smem_size_d = sizeof(typename KTraits_D::SharedStorage);
        size_t smem_size = max(smem_size_p, smem_size_d);
        auto kernel = PODWithPagedKVCacheKernel<KTraits_P, KTraits_D, PrefillParams, DecodeParams>;
        FLASHINFER_CUDA_CALL(
            cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));

        // setup partition_kv metadata
        auto o = prefill_params.o;
        auto lse = prefill_params.lse;
        assert(o == decode_params.o);
        assert(lse == decode_params.lse);
        if (prefill_params.partition_kv || decode_params.partition_kv) {
          assert(tmp_v != nullptr && tmp_s != nullptr);
          // either is partitioned will lead to additional merge kernel
          prefill_params.o = tmp_v;
          prefill_params.lse = tmp_s;
          decode_params.o = tmp_v;
          decode_params.lse = tmp_s;
        }

        // Launch kernel
        void* args[] = {(void*)&prefill_params, (void*)&decode_params};
        FLASHINFER_CUDA_CALL(
            cudaLaunchKernel((void*)kernel, nblks, nthrs, args, smem_size, stream));

        // Post-kernel stuff for split-kv
        if (prefill_params.partition_kv || decode_params.partition_kv) {
          assert(prefill_params.merge_indptr == decode_params.merge_indptr);
          if constexpr (AttentionVariant::use_softmax) {
            FLASHINFER_CUDA_CALL(VariableLengthMergeStates(
                tmp_v, tmp_s, prefill_params.merge_indptr, o, lse,
                (prefill_params.max_total_num_rows + decode_params.max_total_num_rows), nullptr,
                num_qo_heads, HEAD_DIM_VO, stream));
          } else {
            FLASHINFER_CUDA_CALL(VariableLengthAttentionSum(
                tmp_v, prefill_params.merge_indptr, o,
                (prefill_params.max_total_num_rows + decode_params.max_total_num_rows), nullptr,
                num_qo_heads, HEAD_DIM_VO, stream));
          }
        }
      }
    });
  });
  return cudaSuccess;
}

}  // namespace flashinfer

#endif  // FLASHINFER_PREFILL_CUH_
