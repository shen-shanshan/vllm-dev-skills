# ATOM Trace Digest — atom-minimax-m3-tp4-8k1k-conc8.json

## 1. Trace basics

| Field | Value |
|---|---|
| Engine | atom |
| roctracer_version | 4.1 |
| hip_runtime_version | 70253211 |
| hip_driver_version | 70253211 |
| schemaVersion | 1 |
| deviceProperties | 8 GPU(s): AMD Radeon Graphics(sm9.5×256), AMD Radeon Graphics(sm9.5×256), AMD Radeon Graphics(sm9.5×256), AMD Radeon Graphics(sm9.5×256), AMD Radeon Graphics(sm9.5×256), AMD Radeon Graphics(sm9.5×256), AMD Radeon Graphics(sm9.5×256), AMD Radeon Graphics(sm9.5×256) |
| distributedInfo | {"backend": "nccl", "rank": 0, "world_size": 4, "pg_count": 39, "pg_config": [{"pg_name": "0", "pg_desc": "default_pg", "backend_config": "cuda:nccl", "pg_size": 4, "ranks": [0, 1, 2, 3]}, {"pg_name": "1", "pg_desc": "undefined", "backend_config": "cuda:nccl", "pg_size": 4, "ranks": [0, 1, 2, 3]}, { |
| baseTimeNanoseconds | 1782967788000000000 |
| Events (ph) | {'M': 60, 'X': 7256, 'f': 6019, 'i': 2, 's': 162} |
| Events (cat) | {None: 62, 'Trace': 1, 'ac2g': 6181, 'cpu_op': 1062, 'cuda_runtime': 277, 'gpu_memcpy': 72, 'gpu_user_annotation': 6, 'kernel': 5832, 'user_annotation': 6} |
| Kernel events (all) | 5832 on pid 2 |

## 2. Step structure

- user_annotation annotation names: [('execute_context_0(0)_generation_8(8)', 6)]
- gpu_user_annotation annotation names: [('execute_context_0(0)_generation_8(8)', 6)]

| Win | start_rel_us | end_rel_us | wall_us | kernels | note |
|---|---|---|---|---|---|
| 0 | 0.0 | 9047.0 | 9047.0 | 972 |  |
| 1 | 9047.0 | 18290.4 | 9243.4 | 972 |  |
| 2 | 18290.4 | 27473.6 | 9183.2 | 972 **<-- CHOSEN** |  |
| 3 | 27473.6 | 36795.8 | 9322.2 | 972 |  |
| 4 | 36795.8 | 45931.6 | 9135.8 | 972 |  |
| 5 | 45931.6 | 55114.8 | 9183.2 | 972 | trailing |

- windows: 6 | chosen step: 2 | bracket mode: gpu-annot
- replay_stable: True
- occurrence rule: middle (per kernel name, occurrence index in ts order)

## 3. Chosen window summary

- wall_us: 9183.2 | kernels: 972 | distinct names: 39 | busy_us: 9096.2 | busy/wall: 99.1%

| stream | events | busy_us | % of busy |
|---|---|---|---|
| 3 | 972 | 9096.2 | 100.0% |

## 4. Per-kernel roll-up (chosen window)

| id | kernel | C | sum_us | mean_us | median_us | min_us | max_us | pct_busy | streams | first_rel_us | last_rel_us |
|---|---|---|---|---|---|---|---|---|---|---|
| k1 | void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<int>, std::array<char*,... | 2 | 8.9 | 4.4 | 4.4 | 4.2 | 4.6 | 0.10 | 3 | 13.0 | 59.0 |
| k2 | void at::native::index_elementwise_kernel<128, 4, at::native::gpu_index_kernel<at::native::index_... | 1 | 4.0 | 4.0 | 4.0 | 4.0 | 4.0 | 0.04 | 3 | 34.0 | 34.0 |
| k3 | void at::native::elementwise_kernel_manual_unroll<128, 4, at::native::gpu_kernel_impl<at::native:... | 2 | 8.8 | 4.4 | 4.4 | 4.2 | 4.6 | 0.10 | 3 | 38.0 | 8873.7 |
| k4 | void at::native::vectorized_elementwise_kernel<4, at::native::CUDAFunctor_add<long>, std::array<c... | 1 | 4.4 | 4.4 | 4.4 | 4.4 | 4.4 | 0.05 | 3 | 42.2 | 42.2 |
| k5 | void at::native::vectorized_elementwise_kernel<4, at::native::CUDAFunctor_add<int>, std::array<ch... | 1 | 4.3 | 4.3 | 4.3 | 4.3 | 4.3 | 0.05 | 3 | 50.7 | 50.7 |
| k6 | _compute_slot_mapping_kernel | 1 | 4.4 | 4.4 | 4.4 | 4.4 | 4.4 | 0.05 | 3 | 63.6 | 63.6 |
| k7 | void at::native::unrolled_elementwise_kernel<at::native::CUDAFunctorOnSelf_add<int>, std::array<c... | 1 | 3.4 | 3.4 | 3.4 | 3.4 | 3.4 | 0.04 | 3 | 72.1 | 72.1 |
| k8 | _masked_embedding_kernel | 1 | 6.4 | 6.4 | 6.4 | 6.4 | 6.4 | 0.07 | 3 | 81.2 | 81.2 |
| k9 | void aiter::cross_device_reduce_1stage<std::bfloat16_t, 4, false>(aiter::RankData*, aiter::RankDa... | 1 | 8.9 | 8.9 | 8.9 | 8.9 | 8.9 | 0.10 | 3 | 87.7 | 87.7 |
| k10 | _ZN5aiter35fused_qk_rmsnorm_group_quant_kernelIDF16bDB8_Li512ELi16ELi8ELb0ELb1ELb1ELb1ELb0ELb0EEE... | 1 | 4.5 | 4.5 | 4.5 | 4.5 | 4.5 | 0.05 | 3 | 96.6 | 96.6 |
| k11 | hgemm_bf16_16x64x64x8_SPK6_W1x2x1_BLDS1_TN_AS1_0 | 3 | 27.7 | 9.2 | 9.2 | 9.1 | 9.4 | 0.30 | 3 | 101.1 | 348.0 |
| k12 | void aiter::fused_qknorm_idxrqknorm_ops::fusedQKNormIdxrQKNormKernel<hip_bfloat16, hip_bfloat16, ... | 3 | 12.9 | 4.3 | 4.2 | 4.2 | 4.5 | 0.14 | 3 | 110.2 | 357.2 |
| k13 | void at::native::elementwise_kernel_manual_unroll<128, 8, at::native::gpu_kernel_impl_nocast<at::... | 9 | 37.6 | 4.2 | 4.3 | 3.9 | 4.4 | 0.41 | 3 | 114.7 | 369.8 |
| k14 | reshape_and_cache_kernel_flash | 3 | 13.1 | 4.4 | 4.4 | 4.3 | 4.4 | 0.14 | 3 | 127.3 | 374.1 |
| k15 | kernel_unified_attention | 3 | 81.7 | 27.2 | 27.2 | 27.1 | 27.4 | 0.90 | 3 | 131.6 | 378.4 |
| k16 | reduce_segments | 3 | 13.8 | 4.6 | 4.6 | 4.5 | 4.7 | 0.15 | 3 | 158.7 | 405.8 |
| k17 | _gemm_a16_w16_kernel_BLOCK_SIZE_M_16_BLOCK_SIZE_N_16_BLOCK_SIZE_K_256_GROUP_SIZE_M_1_NUM_KSPLIT_1... | 60 | 504.2 | 8.4 | 8.3 | 8.0 | 10.6 | 5.54 | 3 | 163.2 | 8773.5 |
| k18 | void aiter::allreduce_fusion_kernel_1stage<std::bfloat16_t, std::bfloat16_t, 4, true>(aiter::Rank... | 120 | 1196.4 | 10.0 | 9.6 | 8.3 | 15.6 | 13.15 | 3 | 173.8 | 8857.0 |
| k19 | hgemm_bf16_16x64x128x5_SPK2_W1x2x1_BLDS1_TN_AS1_0 | 6 | 86.8 | 14.5 | 14.4 | 10.2 | 18.9 | 0.95 | 3 | 182.8 | 450.4 |
| k20 | _swiglu_oai_kernel | 3 | 12.0 | 4.0 | 4.0 | 4.0 | 4.0 | 0.13 | 3 | 200.9 | 446.3 |
| k21 | hgemm_bf16_16x64x64x7_SPK6_W1x2x1_BLDS1_TN_AS1_0 | 57 | 582.8 | 10.2 | 10.0 | 9.5 | 13.8 | 6.41 | 3 | 469.2 | 8724.2 |
| k22 | void aiter::fused_qknorm_idxrqknorm_ops::fusedQKNormIdxrQKNormKernel<hip_bfloat16, hip_bfloat16, ... | 57 | 287.3 | 5.0 | 5.0 | 4.6 | 7.1 | 3.16 | 3 | 479.3 | 8734.7 |
| k23 | _decode_index_score_topk_partial_kernel | 57 | 630.3 | 11.1 | 11.0 | 10.7 | 11.9 | 6.93 | 3 | 485.2 | 8739.6 |
| k24 | _topk_index_merge_kernel | 57 | 377.1 | 6.6 | 6.5 | 6.2 | 7.7 | 4.15 | 3 | 496.6 | 8750.8 |
| k25 | paged_attention_decode_sliding_window_head_1 | 57 | 659.8 | 11.6 | 11.5 | 11.2 | 13.0 | 7.25 | 3 | 504.2 | 8757.2 |
| k26 | void aiter::pa_decode_ps_reduce_hip_kernel<__hip_bfloat16, __hip_bfloat16, __hip_bfloat16, false,... | 57 | 258.4 | 4.5 | 4.5 | 4.3 | 4.8 | 2.84 | 3 | 515.6 | 8768.8 |
| k27 | hgemm_bf16_16x64x64x6_SPK12_W1x2x1_BLDS1_TN_AS1_0 | 57 | 296.5 | 5.2 | 5.2 | 4.9 | 5.6 | 3.26 | 3 | 538.3 | 8791.3 |
| k28 | void aiter::topk_gating_kernel_opt<hip_bfloat16, float, 128, true, 1>(hip_bfloat16 const*, float ... | 57 | 237.9 | 4.2 | 4.2 | 4.0 | 4.4 | 2.62 | 3 | 543.8 | 8796.4 |
| k29 | void aiter::opus_moe_sorting_entry<aiter::MoeSortingKernel<aiter::MoeSortingProblemEx<int, float,... | 57 | 379.2 | 6.7 | 6.6 | 6.1 | 7.8 | 4.17 | 3 | 548.0 | 8800.7 |
| k30 | void at::native::vectorized_elementwise_kernel<8, at::native::FillFunctor<c10::BFloat16>, std::ar... | 57 | 247.4 | 4.3 | 4.3 | 4.2 | 4.6 | 2.72 | 3 | 555.8 | 8807.7 |
| k31 | void ck_tile::kentry<2, ck_tile::MoeFlatmmKernel<ck_tile::GemmSpatiallyLocalTilePartitioner<ck_ti... | 57 | 1675.3 | 29.4 | 28.6 | 21.5 | 35.1 | 18.42 | 3 | 560.2 | 8812.3 |
| k32 | void aiter::swiglu_act_and_mul_kernel<std::bfloat16_t, std::bfloat16_t, 8>(std::bfloat16_t*, std:... | 57 | 240.3 | 4.2 | 4.2 | 4.0 | 4.6 | 2.64 | 3 | 593.4 | 8837.8 |
| k33 | void ck_tile::kentry<2, ck_tile::MoeFlatmmKernel<ck_tile::GemmSpatiallyLocalTilePartitioner<ck_ti... | 57 | 904.8 | 15.9 | 15.7 | 11.3 | 19.1 | 9.95 | 3 | 597.5 | 8842.2 |
| k34 | void at::native::vectorized_gather_kernel<16, long>(char*, char*, long*, int, long, long, long, l... | 1 | 4.6 | 4.6 | 4.6 | 4.6 | 4.6 | 0.05 | 3 | 8878.4 | 8878.4 |
| k35 | _gemm_a16_w16_kernel_BLOCK_SIZE_M_16_BLOCK_SIZE_N_16_BLOCK_SIZE_K_256_GROUP_SIZE_M_1_NUM_KSPLIT_1... | 1 | 164.5 | 164.5 | 164.5 | 164.5 | 164.5 | 1.81 | 3 | 8882.9 | 8882.9 |
| k36 | void aiter::allgather_lastdim<std::bfloat16_t, 4>(aiter::RankData*, aiter::RankSignals, aiter::Si... | 1 | 33.2 | 33.2 | 33.2 | 33.2 | 33.2 | 0.36 | 3 | 9052.2 | 9052.2 |
| k37 | void at::native::vectorized_elementwise_kernel<4, at::native::bfloat16tofloat32_copy_kernel_cuda(... | 1 | 5.2 | 5.2 | 5.2 | 5.2 | 5.2 | 0.06 | 3 | 9085.3 | 9085.3 |
| k38 | void at::native::reduce_kernel<512, 1, at::native::ReduceOp<float, at::native::ArgMaxOps<float>, ... | 1 | 63.2 | 63.2 | 63.2 | 63.2 | 63.2 | 0.70 | 3 | 9090.5 | 9090.5 |
| k39 | void at::native::elementwise_kernel_manual_unroll<128, 4, at::native::gpu_kernel_impl<at::native:... | 1 | 4.2 | 4.2 | 4.2 | 4.2 | 4.2 | 0.05 | 3 | 9153.7 | 9153.7 |

## 5. Middle-occurrence slice (chosen window, sorted by ts_rel)

| id | kernel | C | occ | ts_rel_us | dur_us | dur_median_across_steps_us | stream |
|---|---|---|---|---|---|---|---|
| k1 | void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<int>, std::array<char*,... | 2 | 1 | 59.0 | 4.6 | 4.7 | 3 |
| k3 | void at::native::elementwise_kernel_manual_unroll<128, 4, at::native::gpu_kernel_impl<at::native:... | 2 | 1 | 8873.7 | 4.6 | 4.7 | 3 |
| k11 | hgemm_bf16_16x64x64x8_SPK6_W1x2x1_BLDS1_TN_AS1_0 | 3 | 1 | 225.4 | 9.4 | 9.4 | 3 |
| k12 | void aiter::fused_qknorm_idxrqknorm_ops::fusedQKNormIdxrQKNormKernel<hip_bfloat16, hip_bfloat16, ... | 3 | 1 | 234.8 | 4.2 | 4.2 | 3 |
| k13 | void at::native::elementwise_kernel_manual_unroll<128, 8, at::native::gpu_kernel_impl_nocast<at::... | 9 | 4 | 243.0 | 3.9 | 3.9 | 3 |
| k14 | reshape_and_cache_kernel_flash | 3 | 1 | 251.3 | 4.4 | 4.4 | 3 |
| k15 | kernel_unified_attention | 3 | 1 | 255.8 | 27.2 | 27.3 | 3 |
| k16 | reduce_segments | 3 | 1 | 283.0 | 4.7 | 4.6 | 3 |
| k17 | _gemm_a16_w16_kernel_BLOCK_SIZE_M_16_BLOCK_SIZE_N_16_BLOCK_SIZE_K_256_GROUP_SIZE_M_1_NUM_KSPLIT_1... | 60 | 30 | 4514.8 | 8.4 | 8.4 | 3 |
| k18 | void aiter::allreduce_fusion_kernel_1stage<std::bfloat16_t, std::bfloat16_t, 4, true>(aiter::Rank... | 120 | 60 | 4523.3 | 12.1 | 9.8 | 3 |
| k19 | hgemm_bf16_16x64x128x5_SPK2_W1x2x1_BLDS1_TN_AS1_0 | 6 | 3 | 328.2 | 10.5 | 10.4 | 3 |
| k20 | _swiglu_oai_kernel | 3 | 1 | 324.2 | 4.0 | 4.0 | 3 |
| k21 | hgemm_bf16_16x64x64x7_SPK6_W1x2x1_BLDS1_TN_AS1_0 | 57 | 28 | 4612.8 | 9.8 | 9.8 | 3 |
| k22 | void aiter::fused_qknorm_idxrqknorm_ops::fusedQKNormIdxrQKNormKernel<hip_bfloat16, hip_bfloat16, ... | 57 | 28 | 4622.5 | 4.8 | 4.8 | 3 |
| k23 | _decode_index_score_topk_partial_kernel | 57 | 28 | 4627.4 | 11.3 | 11.1 | 3 |
| k24 | _topk_index_merge_kernel | 57 | 28 | 4638.6 | 6.5 | 6.5 | 3 |
| k25 | paged_attention_decode_sliding_window_head_1 | 57 | 28 | 4645.1 | 11.5 | 11.5 | 3 |
| k26 | void aiter::pa_decode_ps_reduce_hip_kernel<__hip_bfloat16, __hip_bfloat16, __hip_bfloat16, false,... | 57 | 28 | 4656.6 | 4.6 | 4.6 | 3 |
| k27 | hgemm_bf16_16x64x64x6_SPK12_W1x2x1_BLDS1_TN_AS1_0 | 57 | 28 | 4679.0 | 5.0 | 5.0 | 3 |
| k28 | void aiter::topk_gating_kernel_opt<hip_bfloat16, float, 128, true, 1>(hip_bfloat16 const*, float ... | 57 | 28 | 4684.0 | 4.2 | 4.2 | 3 |
| k29 | void aiter::opus_moe_sorting_entry<aiter::MoeSortingKernel<aiter::MoeSortingProblemEx<int, float,... | 57 | 28 | 4688.2 | 6.6 | 6.6 | 3 |
| k30 | void at::native::vectorized_elementwise_kernel<8, at::native::FillFunctor<c10::BFloat16>, std::ar... | 57 | 28 | 4694.9 | 4.4 | 4.4 | 3 |
| k31 | void ck_tile::kentry<2, ck_tile::MoeFlatmmKernel<ck_tile::GemmSpatiallyLocalTilePartitioner<ck_ti... | 57 | 28 | 4699.3 | 23.5 | 23.0 | 3 |
| k32 | void aiter::swiglu_act_and_mul_kernel<std::bfloat16_t, std::bfloat16_t, 8>(std::bfloat16_t*, std:... | 57 | 28 | 4722.8 | 4.4 | 4.4 | 3 |
| k33 | void ck_tile::kentry<2, ck_tile::MoeFlatmmKernel<ck_tile::GemmSpatiallyLocalTilePartitioner<ck_ti... | 57 | 28 | 4727.2 | 14.1 | 13.7 | 3 |

## 6. Non-repeating kernels (C==1 in chosen window)

| name | ts_rel_us | dur_us | stream |
|---|---|---|---|
| void at::native::index_elementwise_kernel<128, 4, at::native::gpu_index_kernel<at::native::index_... | 34.0 | 4.0 | 3 |
| void at::native::vectorized_elementwise_kernel<4, at::native::CUDAFunctor_add<long>, std::array<c... | 42.2 | 4.4 | 3 |
| void at::native::vectorized_elementwise_kernel<4, at::native::CUDAFunctor_add<int>, std::array<ch... | 50.7 | 4.3 | 3 |
| _compute_slot_mapping_kernel | 63.6 | 4.4 | 3 |
| void at::native::unrolled_elementwise_kernel<at::native::CUDAFunctorOnSelf_add<int>, std::array<c... | 72.1 | 3.4 | 3 |
| _masked_embedding_kernel | 81.2 | 6.4 | 3 |
| void aiter::cross_device_reduce_1stage<std::bfloat16_t, 4, false>(aiter::RankData*, aiter::RankDa... | 87.7 | 8.9 | 3 |
| _ZN5aiter35fused_qk_rmsnorm_group_quant_kernelIDF16bDB8_Li512ELi16ELi8ELb0ELb1ELb1ELb1ELb0ELb0EEE... | 96.6 | 4.5 | 3 |
| void at::native::vectorized_gather_kernel<16, long>(char*, char*, long*, int, long, long, long, l... | 8878.4 | 4.6 | 3 |
| _gemm_a16_w16_kernel_BLOCK_SIZE_M_16_BLOCK_SIZE_N_16_BLOCK_SIZE_K_256_GROUP_SIZE_M_1_NUM_KSPLIT_1... | 8882.9 | 164.5 | 3 |
| void aiter::allgather_lastdim<std::bfloat16_t, 4>(aiter::RankData*, aiter::RankSignals, aiter::Si... | 9052.2 | 33.2 | 3 |
| void at::native::vectorized_elementwise_kernel<4, at::native::bfloat16tofloat32_copy_kernel_cuda(... | 9085.3 | 5.2 | 3 |
| void at::native::reduce_kernel<512, 1, at::native::ReduceOp<float, at::native::ArgMaxOps<float>, ... | 9090.5 | 63.2 | 3 |
| void at::native::elementwise_kernel_manual_unroll<128, 4, at::native::gpu_kernel_impl<at::native:... | 9153.7 | 4.2 | 3 |

## 7. Name dictionary (id -> full kernel name)

```
k1 = void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<int>, std::array<char*, 1ul> >(int, at::native::FillFunctor<int>, std::array<char*, 1ul>)
k2 = void at::native::index_elementwise_kernel<128, 4, at::native::gpu_index_kernel<at::native::index_kernel_impl<at::native::OpaqueType<4> >(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>)::{lambda(char*, char const*, long)#1}>(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>, at::native::index_kernel_impl<at::native::OpaqueType<4> >(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>)::{lambda(char*, char const*, long)#1} const&, bool)::{lambda(int)#1}>(long, at::native::gpu_index_kernel<at::native::index_kernel_impl<at::native::OpaqueType<4> >(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>)::{lambda(char*, char const*, long)#1}>(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>, at::native::index_kernel_impl<at::native::OpaqueType<4> >(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>)::{lambda(char*, char const*, long)#1} const&, bool)::{lambda(int)#1})
k3 = void at::native::elementwise_kernel_manual_unroll<128, 4, at::native::gpu_kernel_impl<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#4}::operator()() const::{lambda(long)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#4}::operator()() const::{lambda(long)#1} const&)::{lambda(int, bool)#1}>(int, at::native::gpu_kernel_impl<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#4}::operator()() const::{lambda(long)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#4}::operator()() const::{lambda(long)#1} const&)::{lambda(int, bool)#1})
k4 = void at::native::vectorized_elementwise_kernel<4, at::native::CUDAFunctor_add<long>, std::array<char*, 3ul> >(int, at::native::CUDAFunctor_add<long>, std::array<char*, 3ul>)
k5 = void at::native::vectorized_elementwise_kernel<4, at::native::CUDAFunctor_add<int>, std::array<char*, 3ul> >(int, at::native::CUDAFunctor_add<int>, std::array<char*, 3ul>)
k6 = _compute_slot_mapping_kernel
k7 = void at::native::unrolled_elementwise_kernel<at::native::CUDAFunctorOnSelf_add<int>, std::array<char*, 2ul>, 4, TrivialOffsetCalculator<1, unsigned int>, TrivialOffsetCalculator<1, unsigned int>, at::native::memory::LoadWithoutCast, at::native::memory::StoreWithoutCast>(int, at::native::CUDAFunctorOnSelf_add<int>, std::array<char*, 2ul>, TrivialOffsetCalculator<1, unsigned int>, TrivialOffsetCalculator<1, unsigned int>, at::native::memory::LoadWithoutCast, at::native::memory::StoreWithoutCast)
k8 = _masked_embedding_kernel
k9 = void aiter::cross_device_reduce_1stage<std::bfloat16_t, 4, false>(aiter::RankData*, aiter::RankData*, aiter::RankSignals, aiter::Signal*, std::bfloat16_t*, int, int)
k10 = _ZN5aiter35fused_qk_rmsnorm_group_quant_kernelIDF16bDB8_Li512ELi16ELi8ELb0ELb1ELb1ELb1ELb0ELb0EEEvPT0_PvPT_S6_S6_PKS5_S8_S8_S8_S8_ffiiiiiiiiiiiii
k11 = hgemm_bf16_16x64x64x8_SPK6_W1x2x1_BLDS1_TN_AS1_0
k12 = void aiter::fused_qknorm_idxrqknorm_ops::fusedQKNormIdxrQKNormKernel<hip_bfloat16, hip_bfloat16, hip_bfloat16, (vllm::Fp8KVCacheDataType)0, (vllm::Fp8KVCacheDataType)0, false, false, false, false>(hip_bfloat16*, hip_bfloat16*, hip_bfloat16*, hip_bfloat16 const*, hip_bfloat16 const*, hip_bfloat16 const*, hip_bfloat16 const*, hip_bfloat16 const*, long const*, long const*, long const*, hip_bfloat16*, hip_bfloat16*, hip_bfloat16*, float*, float*, float, int, int, int, int, int, int, int, int, long, long, long, long, long, long)
k13 = void at::native::elementwise_kernel_manual_unroll<128, 8, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1} const&)::{lambda(int, bool)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1} const&)::{lambda(int, bool)#1})
k14 = reshape_and_cache_kernel_flash
k15 = kernel_unified_attention
k16 = reduce_segments
k17 = _gemm_a16_w16_kernel_BLOCK_SIZE_M_16_BLOCK_SIZE_N_16_BLOCK_SIZE_K_256_GROUP_SIZE_M_1_NUM_KSPLIT_1_SPLITK_BLOCK_SIZE_2048_EVEN_K_1_EVEN_MN_0_cache_modifier_CG_activation_NONE_use_activation_0_ADD_BIAS_0_SKIP_REDUCE_0
k18 = void aiter::allreduce_fusion_kernel_1stage<std::bfloat16_t, std::bfloat16_t, 4, true>(aiter::RankData*, aiter::RankSignals, aiter::Signal*, int, std::bfloat16_t*, std::bfloat16_t*, std::bfloat16_t*, std::bfloat16_t*, float*, int, int, int, int, float, std::bfloat16_t*)
k19 = hgemm_bf16_16x64x128x5_SPK2_W1x2x1_BLDS1_TN_AS1_0
k20 = _swiglu_oai_kernel
k21 = hgemm_bf16_16x64x64x7_SPK6_W1x2x1_BLDS1_TN_AS1_0
k22 = void aiter::fused_qknorm_idxrqknorm_ops::fusedQKNormIdxrQKNormKernel<hip_bfloat16, hip_bfloat16, hip_bfloat16, (vllm::Fp8KVCacheDataType)0, (vllm::Fp8KVCacheDataType)0, false, true, true, true>(hip_bfloat16*, hip_bfloat16*, hip_bfloat16*, hip_bfloat16 const*, hip_bfloat16 const*, hip_bfloat16 const*, hip_bfloat16 const*, hip_bfloat16 const*, long const*, long const*, long const*, hip_bfloat16*, hip_bfloat16*, hip_bfloat16*, float*, float*, float, int, int, int, int, int, int, int, int, long, long, long, long, long, long)
k23 = _decode_index_score_topk_partial_kernel
k24 = _topk_index_merge_kernel
k25 = paged_attention_decode_sliding_window_head_1
k26 = void aiter::pa_decode_ps_reduce_hip_kernel<__hip_bfloat16, __hip_bfloat16, __hip_bfloat16, false, 128, 16, 8>(__hip_bfloat16*, float const*, float const*, __hip_bfloat16 const*, __hip_bfloat16 const*, int, int, int, int, int, int, int, int, int, int, int)
k27 = hgemm_bf16_16x64x64x6_SPK12_W1x2x1_BLDS1_TN_AS1_0
k28 = void aiter::topk_gating_kernel_opt<hip_bfloat16, float, 128, true, 1>(hip_bfloat16 const*, float const*, float*, int*, unsigned long, int, int, float)
k29 = void aiter::opus_moe_sorting_entry<aiter::MoeSortingKernel<aiter::MoeSortingProblemEx<int, float, 1, true, false, false, true, 0> >, aiter::MoeSortingKernel<aiter::MoeSortingProblemEx<int, float, 1, true, false, false, true, 0> >::Kargs>(aiter::MoeSortingKernel<aiter::MoeSortingProblemEx<int, float, 1, true, false, false, true, 0> >::Kargs)
k30 = void at::native::vectorized_elementwise_kernel<8, at::native::FillFunctor<c10::BFloat16>, std::array<char*, 1ul> >(int, at::native::FillFunctor<c10::BFloat16>, std::array<char*, 1ul>)
k31 = void ck_tile::kentry<2, ck_tile::MoeFlatmmKernel<ck_tile::GemmSpatiallyLocalTilePartitioner<ck_tile::TileGemmShape<ck_tile::sequence<16, 128, 256>, ck_tile::sequence<1, 4, 1>, ck_tile::sequence<16, 16, 32>, false, false>, 1, 1>, ck_tile::F16xMXF4FlatmmPipelineAGmemBGmemCRegV1<ck_tile::F16xMXF4FlatmmPipelineProblem<std::bfloat16_t, ck_tile::pk_float4_e2m1_t, float, ck_tile::TileGemmShape<ck_tile::sequence<16, 128, 256>, ck_tile::sequence<1, 4, 1>, ck_tile::sequence<16, 16, 32>, false, false>, ck_tile::TileGemmUniversalTraits<false, false, false, false, ck_tile::tensor_layout::gemm::RowMajor, ck_tile::tensor_layout::gemm::ColumnMajor, ck_tile::tensor_layout::gemm::RowMajor, false, false, false, 1, true, 16, (ck_tile::DataCachePrefetchKind)0, (ck_tile::DataCachePrefetchKind)0, false, false>, (ck_tile::GemmPipelineScheduler)0, true, (ck_tile::TailNumber)1, (ck_tile::amd_buffer_coherence_enum)2, false, std::bfloat16_t>, ck_tile::F16xMXF4FlatmmPipelineAgBgCrPolicy>, ck_tile::CShuffleEpilogue<ck_tile::CShuffleEpilogueProblem<std::bfloat16_t, std::bfloat16_t, ck_tile::tuple<>, float, std::bfloat16_t, ck_tile::tuple<>, ck_tile::tensor_layout::gemm::RowMajor, ck_tile::element_wise::PassThrough, 16, 128, 1, 4, 16, 16, 32, false, 1, false, 1, 2, false, void, void, false, float, std::bfloat16_t>, void>, (ck_tile::MoeFlatmmKind)3, ck_tile::moe::MoeSilu>, ck_tile::MoeFlatmmKernel<ck_tile::GemmSpatiallyLocalTilePartitioner<ck_tile::TileGemmShape<ck_tile::sequence<16, 128, 256>, ck_tile::sequence<1, 4, 1>, ck_tile::sequence<16, 16, 32>, false, false>, 1, 1>, ck_tile::F16xMXF4FlatmmPipelineAGmemBGmemCRegV1<ck_tile::F16xMXF4FlatmmPipelineProblem<std::bfloat16_t, ck_tile::pk_float4_e2m1_t, float, ck_tile::TileGemmShape<ck_tile::sequence<16, 128, 256>, ck_tile::sequence<1, 4, 1>, ck_tile::sequence<16, 16, 32>, false, false>, ck_tile::TileGemmUniversalTraits<false, false, false, false, ck_tile::tensor_layout::gemm::RowMajor, ck_tile::tensor_layout::gemm::ColumnMajor, ck_tile::tensor_layout::gemm::RowMajor, false, false, false, 1, true, 16, (ck_tile::DataCachePrefetchKind)0, (ck_tile::DataCachePrefetchKind)0, false, false>, (ck_tile::GemmPipelineScheduler)0, true, (ck_tile::TailNumber)1, (ck_tile::amd_buffer_coherence_enum)2, false, std::bfloat16_t>, ck_tile::F16xMXF4FlatmmPipelineAgBgCrPolicy>, ck_tile::CShuffleEpilogue<ck_tile::CShuffleEpilogueProblem<std::bfloat16_t, std::bfloat16_t, ck_tile::tuple<>, float, std::bfloat16_t, ck_tile::tuple<>, ck_tile::tensor_layout::gemm::RowMajor, ck_tile::element_wise::PassThrough, 16, 128, 1, 4, 16, 16, 32, false, 1, false, 1, 2, false, void, void, false, float, std::bfloat16_t>, void>, (ck_tile::MoeFlatmmKind)3, ck_tile::moe::MoeSilu>::MoeFlatmmKernelArgs<ck_tile::FlatmmScalePointer<1, 32, ck_tile::e8m0_bexp_t>, ck_tile::FlatmmScalePointer<1, 32, ck_tile::e8m0_bexp_t>, ck_tile::FlatmmScalePointer<-1, 0, float> > >(ck_tile::MoeFlatmmKernel<ck_tile::GemmSpatiallyLocalTilePartitioner<ck_tile::TileGemmShape<ck_tile::sequence<16, 128, 256>, ck_tile::sequence<1, 4, 1>, ck_tile::sequence<16, 16, 32>, false, false>, 1, 1>, ck_tile::F16xMXF4FlatmmPipelineAGmemBGmemCRegV1<ck_tile::F16xMXF4FlatmmPipelineProblem<std::bfloat16_t, ck_tile::pk_float4_e2m1_t, float, ck_tile::TileGemmShape<ck_tile::sequence<16, 128, 256>, ck_tile::sequence<1, 4, 1>, ck_tile::sequence<16, 16, 32>, false, false>, ck_tile::TileGemmUniversalTraits<false, false, false, false, ck_tile::tensor_layout::gemm::RowMajor, ck_tile::tensor_layout::gemm::ColumnMajor, ck_tile::tensor_layout::gemm::RowMajor, false, false, false, 1, true, 16, (ck_tile::DataCachePrefetchKind)0, (ck_tile::DataCachePrefetchKind)0, false, false>, (ck_tile::GemmPipelineScheduler)0, true, (ck_tile::TailNumber)1, (ck_tile::amd_buffer_coherence_enum)2, false, std::bfloat16_t>, ck_tile::F16xMXF4FlatmmPipelineAgBgCrPolicy>, ck_tile::CShuffleEpilogue<ck_tile::CShuffleEpilogueProblem<std::bfloat16_t, std::bfloat16_t, ck_tile::tuple<>, float, std::bfloat16_t, ck_tile::tuple<>, ck_tile::tensor_layout::gemm::RowMajor, ck_tile::element_wise::PassThrough, 16, 128, 1, 4, 16, 16, 32, false, 1, false, 1, 2, false, void, void, false, float, std::bfloat16_t>, void>, (ck_tile::MoeFlatmmKind)3, ck_tile::moe::MoeSilu>::MoeFlatmmKernelArgs<ck_tile::FlatmmScalePointer<1, 32, ck_tile::e8m0_bexp_t>, ck_tile::FlatmmScalePointer<1, 32, ck_tile::e8m0_bexp_t>, ck_tile::FlatmmScalePointer<-1, 0, float> >)
k32 = void aiter::swiglu_act_and_mul_kernel<std::bfloat16_t, std::bfloat16_t, 8>(std::bfloat16_t*, std::bfloat16_t const*, int)
k33 = void ck_tile::kentry<2, ck_tile::MoeFlatmmKernel<ck_tile::GemmSpatiallyLocalTilePartitioner<ck_tile::TileGemmShape<ck_tile::sequence<16, 128, 256>, ck_tile::sequence<1, 4, 1>, ck_tile::sequence<16, 16, 32>, false, false>, 1, 1>, ck_tile::F16xMXF4FlatmmPipelineAGmemBGmemCRegV1<ck_tile::F16xMXF4FlatmmPipelineProblem<std::bfloat16_t, ck_tile::pk_float4_e2m1_t, float, ck_tile::TileGemmShape<ck_tile::sequence<16, 128, 256>, ck_tile::sequence<1, 4, 1>, ck_tile::sequence<16, 16, 32>, false, false>, ck_tile::TileGemmUniversalTraits<false, false, false, false, ck_tile::tensor_layout::gemm::RowMajor, ck_tile::tensor_layout::gemm::ColumnMajor, ck_tile::tensor_layout::gemm::RowMajor, false, false, false, 1, true, 16, (ck_tile::DataCachePrefetchKind)0, (ck_tile::DataCachePrefetchKind)0, false, false>, (ck_tile::GemmPipelineScheduler)0, true, (ck_tile::TailNumber)0, (ck_tile::amd_buffer_coherence_enum)2, false, std::bfloat16_t>, ck_tile::F16xMXF4FlatmmPipelineAgBgCrPolicy>, ck_tile::CShuffleEpilogue<ck_tile::CShuffleEpilogueProblem<std::bfloat16_t, std::bfloat16_t, ck_tile::tuple<>, float, std::bfloat16_t, ck_tile::tuple<>, ck_tile::tensor_layout::gemm::RowMajor, ck_tile::element_wise::PassThrough, 16, 128, 1, 4, 16, 16, 32, false, 1, false, 1, 2, false, void, void, false, float, std::bfloat16_t>, void>, (ck_tile::MoeFlatmmKind)2, ck_tile::moe::MoeSilu>, ck_tile::MoeFlatmmKernel<ck_tile::GemmSpatiallyLocalTilePartitioner<ck_tile::TileGemmShape<ck_tile::sequence<16, 128, 256>, ck_tile::sequence<1, 4, 1>, ck_tile::sequence<16, 16, 32>, false, false>, 1, 1>, ck_tile::F16xMXF4FlatmmPipelineAGmemBGmemCRegV1<ck_tile::F16xMXF4FlatmmPipelineProblem<std::bfloat16_t, ck_tile::pk_float4_e2m1_t, float, ck_tile::TileGemmShape<ck_tile::sequence<16, 128, 256>, ck_tile::sequence<1, 4, 1>, ck_tile::sequence<16, 16, 32>, false, false>, ck_tile::TileGemmUniversalTraits<false, false, false, false, ck_tile::tensor_layout::gemm::RowMajor, ck_tile::tensor_layout::gemm::ColumnMajor, ck_tile::tensor_layout::gemm::RowMajor, false, false, false, 1, true, 16, (ck_tile::DataCachePrefetchKind)0, (ck_tile::DataCachePrefetchKind)0, false, false>, (ck_tile::GemmPipelineScheduler)0, true, (ck_tile::TailNumber)0, (ck_tile::amd_buffer_coherence_enum)2, false, std::bfloat16_t>, ck_tile::F16xMXF4FlatmmPipelineAgBgCrPolicy>, ck_tile::CShuffleEpilogue<ck_tile::CShuffleEpilogueProblem<std::bfloat16_t, std::bfloat16_t, ck_tile::tuple<>, float, std::bfloat16_t, ck_tile::tuple<>, ck_tile::tensor_layout::gemm::RowMajor, ck_tile::element_wise::PassThrough, 16, 128, 1, 4, 16, 16, 32, false, 1, false, 1, 2, false, void, void, false, float, std::bfloat16_t>, void>, (ck_tile::MoeFlatmmKind)2, ck_tile::moe::MoeSilu>::MoeFlatmmKernelArgs<ck_tile::FlatmmScalePointer<1, 32, ck_tile::e8m0_bexp_t>, ck_tile::FlatmmScalePointer<1, 32, ck_tile::e8m0_bexp_t>, ck_tile::FlatmmScalePointer<-1, 0, float> > >(ck_tile::MoeFlatmmKernel<ck_tile::GemmSpatiallyLocalTilePartitioner<ck_tile::TileGemmShape<ck_tile::sequence<16, 128, 256>, ck_tile::sequence<1, 4, 1>, ck_tile::sequence<16, 16, 32>, false, false>, 1, 1>, ck_tile::F16xMXF4FlatmmPipelineAGmemBGmemCRegV1<ck_tile::F16xMXF4FlatmmPipelineProblem<std::bfloat16_t, ck_tile::pk_float4_e2m1_t, float, ck_tile::TileGemmShape<ck_tile::sequence<16, 128, 256>, ck_tile::sequence<1, 4, 1>, ck_tile::sequence<16, 16, 32>, false, false>, ck_tile::TileGemmUniversalTraits<false, false, false, false, ck_tile::tensor_layout::gemm::RowMajor, ck_tile::tensor_layout::gemm::ColumnMajor, ck_tile::tensor_layout::gemm::RowMajor, false, false, false, 1, true, 16, (ck_tile::DataCachePrefetchKind)0, (ck_tile::DataCachePrefetchKind)0, false, false>, (ck_tile::GemmPipelineScheduler)0, true, (ck_tile::TailNumber)0, (ck_tile::amd_buffer_coherence_enum)2, false, std::bfloat16_t>, ck_tile::F16xMXF4FlatmmPipelineAgBgCrPolicy>, ck_tile::CShuffleEpilogue<ck_tile::CShuffleEpilogueProblem<std::bfloat16_t, std::bfloat16_t, ck_tile::tuple<>, float, std::bfloat16_t, ck_tile::tuple<>, ck_tile::tensor_layout::gemm::RowMajor, ck_tile::element_wise::PassThrough, 16, 128, 1, 4, 16, 16, 32, false, 1, false, 1, 2, false, void, void, false, float, std::bfloat16_t>, void>, (ck_tile::MoeFlatmmKind)2, ck_tile::moe::MoeSilu>::MoeFlatmmKernelArgs<ck_tile::FlatmmScalePointer<1, 32, ck_tile::e8m0_bexp_t>, ck_tile::FlatmmScalePointer<1, 32, ck_tile::e8m0_bexp_t>, ck_tile::FlatmmScalePointer<-1, 0, float> >)
k34 = void at::native::vectorized_gather_kernel<16, long>(char*, char*, long*, int, long, long, long, long, bool)
k35 = _gemm_a16_w16_kernel_BLOCK_SIZE_M_16_BLOCK_SIZE_N_16_BLOCK_SIZE_K_256_GROUP_SIZE_M_1_NUM_KSPLIT_1_SPLITK_BLOCK_SIZE_6144_EVEN_K_1_EVEN_MN_0_cache_modifier_CG_activation_NONE_use_activation_0_ADD_BIAS_0_SKIP_REDUCE_0
k36 = void aiter::allgather_lastdim<std::bfloat16_t, 4>(aiter::RankData*, aiter::RankSignals, aiter::Signal*, std::bfloat16_t*, int, int, int)
k37 = void at::native::vectorized_elementwise_kernel<4, at::native::bfloat16tofloat32_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda(c10::BFloat16)#1}, std::array<char*, 2ul> >(int, at::native::bfloat16tofloat32_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda(c10::BFloat16)#1}, std::array<char*, 2ul>)
k38 = void at::native::reduce_kernel<512, 1, at::native::ReduceOp<float, at::native::ArgMaxOps<float>, unsigned int, long, 4, 4> >(at::native::ReduceOp<float, at::native::ArgMaxOps<float>, unsigned int, long, 4, 4>)
k39 = void at::native::elementwise_kernel_manual_unroll<128, 4, at::native::gpu_kernel_impl<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#3}::operator()() const::{lambda(int)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#3}::operator()() const::{lambda(int)#1} const&)::{lambda(int, bool)#1}>(int, at::native::gpu_kernel_impl<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#3}::operator()() const::{lambda(int)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#3}::operator()() const::{lambda(int)#1} const&)::{lambda(int, bool)#1})
```

## 8. Notes / warnings

(none)

Full data: `atom-minimax-m3-tp4-8k1k-conc8_slice.json` (rollup / middle_slice / once_kernels / windows / streams / annotations)