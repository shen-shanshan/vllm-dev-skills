# VLLM Trace Digest — vllm-minimax-m3-tp4-8k1k-conc8.json

## 1. Trace basics

| Field | Value |
|---|---|
| Engine | vllm |
| vllm_version | 0.28.1rc1.dev199+g7c5dc571c |
| vllm_version_tuple | ['0', '28', '1', 'rc1', 'dev199', 'g7c5dc571c'] |
| rocprofiler-sdk_version | 1.1 |
| hip_runtime_version | 70253211 |
| hip_driver_version | 70253211 |
| schemaVersion | 1 |
| deviceProperties | 8 GPU(s): (sm9.5×256), (sm9.5×256), (sm9.5×256), (sm9.5×256), (sm9.5×256), (sm9.5×256), (sm9.5×256), (sm9.5×256) |
| distributedInfo | {"backend": "nccl", "rank": 0, "world_size": 4, "pg_count": 39, "pg_config": [{"pg_name": "0", "pg_desc": "default_pg", "backend_config": "cuda:nccl", "pg_size": 4, "ranks": [0, 1, 2, 3]}, {"pg_name": "1", "pg_desc": "undefined", "backend_config": "cuda:nccl", "pg_size": 4, "ranks": [0, 1, 2, 3]}, { |
| baseTimeNanoseconds | 1782967788000000000 |
| Events (ph) | {'M': 60, 'X': 8807, 'f': 7134, 'i': 2, 's': 170} |
| Events (cat) | {None: 62, 'Trace': 1, 'ac2g': 7304, 'cpu_op': 1478, 'cuda_runtime': 714, 'gpu_user_annotation': 12, 'kernel': 6590, 'user_annotation': 12} |
| Kernel events (all) | 6590 on pid 0 |

## 2. Step structure

- user_annotation annotation names: [('execute_context_0(0)_generation_8(8)', 6), ('nccl:_all_gather_base', 6)]
- gpu_user_annotation annotation names: [('execute_context_0(0)_generation_8(8)', 6), ('nccl:_all_gather_base', 6)]

| Win | start_rel_us | end_rel_us | wall_us | kernels | note |
|---|---|---|---|---|---|
| 0 | 0.0 | 9180.0 | 9180.0 | 1099 |  |
| 1 | 9180.0 | 18076.4 | 8896.3 | 1098 |  |
| 2 | 18076.4 | 27090.2 | 9013.8 | 1098 **<-- CHOSEN** |  |
| 3 | 27090.2 | 36248.3 | 9158.2 | 1098 |  |
| 4 | 36248.3 | 45269.7 | 9021.4 | 1098 |  |
| 5 | 45269.7 | 54291.1 | 9021.4 | 1088 | trailing |

- windows: 6 | chosen step: 2 | bracket mode: gpu-annot
- replay_stable: True
- occurrence rule: middle (per kernel name, occurrence index in ts order)

## 3. Chosen window summary

- wall_us: 9013.8 | kernels: 1098 | distinct names: 48 | busy_us: 8968.4 | busy/wall: 99.5%

| stream | events | busy_us | % of busy |
|---|---|---|---|
| 2 | 2 | 9.4 | 0.1% |
| 4 | 1096 | 8959.0 | 99.9% |

## 4. Per-kernel roll-up (chosen window)

| id | kernel | C | sum_us | mean_us | median_us | min_us | max_us | pct_busy | streams | first_rel_us | last_rel_us |
|---|---|---|---|---|---|---|---|---|---|---|
| k1 | void at::native::vectorized_elementwise_kernel<16, at::native::FillFunctor<bool>, std::array<char... | 1 | 4.4 | 4.4 | 4.4 | 4.4 | 4.4 | 0.05 | 4 | 0.0 | 0.0 |
| k2 | __amd_rocclr_copyBuffer.kd | 7 | 29.7 | 4.2 | 3.8 | 3.5 | 5.9 | 0.33 | 2,4 | 2.7 | 9012.1 |
| k3 | void (anonymous namespace)::elementwise_kernel_with_index<int, at::native::arange_cuda_out(c10::S... | 1 | 2.7 | 2.7 | 2.7 | 2.7 | 2.7 | 0.03 | 4 | 17.9 | 17.9 |
| k4 | void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<int>, std::array<char*,... | 2 | 8.9 | 4.5 | 4.5 | 4.4 | 4.5 | 0.10 | 4 | 20.6 | 8992.5 |
| k5 | _prepare_pos_seq_lens_kernel.kd | 1 | 3.1 | 3.1 | 3.1 | 3.1 | 3.1 | 0.03 | 4 | 38.1 | 38.1 |
| k6 | _combine_sampled_and_draft_tokens_kernel.kd | 1 | 4.6 | 4.6 | 4.6 | 4.6 | 4.6 | 0.05 | 4 | 41.2 | 41.2 |
| k7 | _gather_block_tables_kernel.kd | 1 | 4.7 | 4.7 | 4.7 | 4.7 | 4.7 | 0.05 | 4 | 45.8 | 45.8 |
| k8 | _compute_slot_mappings_kernel.kd | 1 | 4.4 | 4.4 | 4.4 | 4.4 | 4.4 | 0.05 | 4 | 50.2 | 50.2 |
| k9 | void at::native::unrolled_elementwise_kernel<at::native::CUDAFunctor_add<int>, std::array<char*, ... | 1 | 4.2 | 4.2 | 4.2 | 4.2 | 4.2 | 0.05 | 4 | 54.9 | 54.9 |
| k10 | void at::native::vectorized_elementwise_kernel<4, at::native::CUDAFunctor_add<int>, std::array<ch... | 1 | 4.3 | 4.3 | 4.3 | 4.3 | 4.3 | 0.05 | 4 | 59.0 | 59.0 |
| k11 | triton_poi_fused_add_bitwise_and_bitwise_not_bitwise_or_ge_lt_mul_sub_0.kd | 1 | 4.5 | 4.5 | 4.5 | 4.5 | 4.5 | 0.05 | 4 | 72.6 | 72.6 |
| k12 | void at::native::(anonymous namespace)::indexSelectSmallIndex<c10::BFloat16, long, unsigned int, ... | 1 | 7.6 | 7.6 | 7.6 | 7.6 | 7.6 | 0.08 | 4 | 77.1 | 77.1 |
| k13 | void at::native::elementwise_kernel_manual_unroll<128, 8, at::native::gpu_kernel_impl_nocast<at::... | 1 | 5.2 | 5.2 | 5.2 | 5.2 | 5.2 | 0.06 | 4 | 84.7 | 84.7 |
| k14 | void aiter::cross_device_reduce_1stage<std::bfloat16_t, 4, false>(aiter::RankData*, aiter::RankDa... | 121 | 1052.9 | 8.7 | 8.5 | 7.3 | 16.1 | 11.74 | 4 | 89.8 | 8727.8 |
| k15 | _gemma_rmsnorm_kernel.kd | 1 | 4.4 | 4.4 | 4.4 | 4.4 | 4.4 | 0.05 | 4 | 100.3 | 100.3 |
| k16 | hgemm_bf16_16x64x64x8_SPK6_W1x2x1_BLDS1_TN_AS1_0.kd | 3 | 26.9 | 9.0 | 8.8 | 8.7 | 9.4 | 0.30 | 4 | 104.7 | 345.6 |
| k17 | void vllm::minimax_m3_fused_ops::fusedMiniMaxM3QNormRopeKVInsertKernel<c10::BFloat16, __hip_bfloa... | 3 | 13.8 | 4.6 | 4.6 | 4.4 | 4.9 | 0.15 | 4 | 114.1 | 354.4 |
| k18 | reshape_and_cache_kernel_flash.kd | 3 | 13.8 | 4.6 | 4.7 | 4.3 | 4.8 | 0.15 | 4 | 118.7 | 358.8 |
| k19 | kernel_unified_attention.kd | 3 | 82.1 | 27.4 | 27.2 | 27.0 | 27.9 | 0.92 | 4 | 123.4 | 363.1 |
| k20 | reduce_segments.kd | 3 | 14.0 | 4.7 | 4.6 | 4.4 | 4.9 | 0.16 | 4 | 150.6 | 391.0 |
| k21 | _gemm_a16_w16_kernel_BLOCK_SIZE_M_16_BLOCK_SIZE_N_32_BLOCK_SIZE_K_256_GROUP_SIZE_M_1_NUM_KSPLI....kd | 60 | 711.3 | 11.9 | 11.8 | 11.4 | 13.4 | 7.93 | 4 | 155.2 | 8637.6 |
| k22 | _gemma_fused_add_rmsnorm_kernel.kd | 120 | 572.9 | 4.8 | 4.7 | 4.2 | 5.3 | 6.39 | 4 | 174.7 | 8735.6 |
| k23 | hgemm_bf16_16x64x128x5_SPK2_W1x2x1_BLDS1_TN_AS1_0.kd | 6 | 84.0 | 14.0 | 14.1 | 10.2 | 17.8 | 0.94 | 4 | 179.0 | 442.0 |
| k24 | _swiglu_oai_kernel.kd | 3 | 12.5 | 4.2 | 4.2 | 4.1 | 4.2 | 0.14 | 4 | 196.6 | 438.0 |
| k25 | hgemm_bf16_16x64x64x7_SPK6_W1x2x1_BLDS1_TN_AS1_0.kd | 57 | 542.3 | 9.5 | 9.4 | 9.0 | 11.2 | 6.05 | 4 | 467.4 | 8597.8 |
| k26 | void vllm::minimax_m3_fused_ops::fusedMiniMaxM3QNormRopeKVInsertKernel<c10::BFloat16, unsigned ch... | 57 | 278.1 | 4.9 | 4.9 | 4.6 | 5.1 | 3.10 | 4 | 477.2 | 8607.3 |
| k27 | _decode_index_score_balanced_kernel.kd | 57 | 438.6 | 7.7 | 7.7 | 7.3 | 8.1 | 4.89 | 4 | 481.8 | 8612.2 |
| k28 | _decode_topk_fused_kernel.kd | 57 | 268.6 | 4.7 | 4.7 | 4.4 | 5.2 | 2.99 | 4 | 489.7 | 8619.8 |
| k29 | _gqa_sparse_decode_kernel.kd | 57 | 449.3 | 7.9 | 7.9 | 7.6 | 8.6 | 5.01 | 4 | 494.8 | 8624.5 |
| k30 | _merge_topk_attn_out_kernel.kd | 57 | 274.7 | 4.8 | 4.8 | 4.6 | 5.1 | 3.06 | 4 | 503.4 | 8632.7 |
| k31 | _rocm_fp32_router_gemm_kernel.kd | 57 | 307.3 | 5.4 | 5.4 | 5.1 | 7.0 | 3.43 | 4 | 534.6 | 8662.6 |
| k32 | void aiter::grouped_topk_kernel<float, float __vector(4), 1, true, true, false>(float*, float con... | 57 | 258.0 | 4.5 | 4.5 | 4.3 | 4.8 | 2.88 | 4 | 540.1 | 8667.8 |
| k33 | void aiter::opus_moe_sorting_entry<aiter::MoeSortingKernel<aiter::MoeSortingProblemEx<int, float,... | 57 | 391.7 | 6.9 | 6.8 | 6.5 | 7.7 | 4.37 | 4 | 544.8 | 8672.4 |
| k34 | void at::native::vectorized_elementwise_kernel<8, at::native::FillFunctor<c10::BFloat16>, std::ar... | 57 | 252.9 | 4.4 | 4.4 | 4.1 | 4.6 | 2.82 | 4 | 552.5 | 8679.4 |
| k35 | void ck_tile::kentry<2, ck_tile::MoeFlatmmKernel<ck_tile::GemmSpatiallyLocalTilePartitioner<ck_ti... | 57 | 1505.9 | 26.4 | 25.9 | 21.6 | 34.2 | 16.79 | 4 | 557.0 | 8683.9 |
| k36 | void aiter::swiglu_act_and_mul_kernel<std::bfloat16_t, std::bfloat16_t, 8>(std::bfloat16_t*, std:... | 57 | 259.2 | 4.5 | 4.6 | 4.3 | 4.7 | 2.89 | 4 | 591.2 | 8709.7 |
| k37 | void ck_tile::kentry<2, ck_tile::MoeFlatmmKernel<ck_tile::GemmSpatiallyLocalTilePartitioner<ck_ti... | 57 | 834.3 | 14.6 | 14.6 | 11.4 | 19.0 | 9.30 | 4 | 595.9 | 8714.0 |
| k38 | void at::native::vectorized_gather_kernel<16, long>(char*, char*, long*, int, long, long, long, l... | 1 | 4.5 | 4.5 | 4.5 | 4.5 | 4.5 | 0.05 | 4 | 8751.6 | 8751.6 |
| k39 | _gemm_a16_w16_kernel_BLOCK_SIZE_M_16_BLOCK_SIZE_N_32_BLOCK_SIZE_K_256_GROUP_SIZE_M_1_NUM_KSPLI....kd | 1 | 129.9 | 129.9 | 129.9 | 129.9 | 129.9 | 1.45 | 4 | 8756.1 | 8756.1 |
| k40 | ncclDevKernel_Generic_1(ncclDevKernelArgsStorage<4096ul>) [clone .kd] | 1 | 53.8 | 53.8 | 53.8 | 53.8 | 53.8 | 0.60 | 4 | 8886.0 | 8886.0 |
| k41 | void at::native::elementwise_kernel_manual_unroll<128, 8, at::native::gpu_kernel_impl_nocast<at::... | 1 | 7.4 | 7.4 | 7.4 | 7.4 | 7.4 | 0.08 | 4 | 8958.1 | 8958.1 |
| k42 | void at::native::index_elementwise_kernel<128, 4, at::native::gpu_index_kernel<at::native::index_... | 1 | 4.2 | 4.2 | 4.2 | 4.2 | 4.2 | 0.05 | 4 | 8965.6 | 8965.6 |
| k43 | void at::native::index_elementwise_kernel<128, 4, at::native::gpu_index_kernel<at::native::index_... | 1 | 4.6 | 4.6 | 4.6 | 4.6 | 4.6 | 0.05 | 4 | 8969.7 | 8969.7 |
| k44 | _gumbel_sample_kernel.kd | 1 | 6.4 | 6.4 | 6.4 | 6.4 | 6.4 | 0.07 | 4 | 8974.4 | 8974.4 |
| k45 | void at::native::reduce_kernel<512, 1, at::native::ReduceOp<float, at::native::ArgMaxOps<float>, ... | 1 | 7.4 | 7.4 | 7.4 | 7.4 | 7.4 | 0.08 | 4 | 8980.9 | 8980.9 |
| k46 | void at::native::_scatter_gather_elementwise_kernel<256, 4, at::native::_cuda_scatter_gather_inte... | 1 | 4.2 | 4.2 | 4.2 | 4.2 | 4.2 | 0.05 | 4 | 8988.1 | 8988.1 |
| k47 | _get_num_sampled_and_rejected_kernel.kd | 1 | 3.5 | 3.5 | 3.5 | 3.5 | 3.5 | 0.04 | 4 | 8996.7 | 8996.7 |
| k48 | _post_update_kernel.kd | 1 | 4.6 | 4.6 | 4.6 | 4.6 | 4.6 | 0.05 | 4 | 9009.2 | 9009.2 |

## 5. Middle-occurrence slice (chosen window, sorted by ts_rel)

| id | kernel | C | occ | ts_rel_us | dur_us | dur_median_across_steps_us | stream |
|---|---|---|---|---|---|---|---|
| k2 | __amd_rocclr_copyBuffer.kd | 7 | 3 | 63.4 | 4.4 | 4.5 | 4 |
| k4 | void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<int>, std::array<char*,... | 2 | 1 | 8992.5 | 4.4 | 4.4 | 4 |
| k14 | void aiter::cross_device_reduce_1stage<std::bfloat16_t, 4, false>(aiter::RankData*, aiter::RankDa... | 121 | 60 | 4327.3 | 7.9 | 8.5 | 4 |
| k16 | hgemm_bf16_16x64x64x8_SPK6_W1x2x1_BLDS1_TN_AS1_0.kd | 3 | 1 | 224.2 | 8.7 | 8.7 | 4 |
| k17 | void vllm::minimax_m3_fused_ops::fusedMiniMaxM3QNormRopeKVInsertKernel<c10::BFloat16, __hip_bfloa... | 3 | 1 | 232.8 | 4.9 | 4.6 | 4 |
| k18 | reshape_and_cache_kernel_flash.kd | 3 | 1 | 237.7 | 4.8 | 4.6 | 4 |
| k19 | kernel_unified_attention.kd | 3 | 1 | 242.5 | 27.0 | 26.8 | 4 |
| k20 | reduce_segments.kd | 3 | 1 | 269.5 | 4.9 | 4.8 | 4 |
| k21 | _gemm_a16_w16_kernel_BLOCK_SIZE_M_16_BLOCK_SIZE_N_32_BLOCK_SIZE_K_256_GROUP_SIZE_M_1_NUM_KSPLI....kd | 60 | 30 | 4379.1 | 11.9 | 11.7 | 4 |
| k22 | _gemma_fused_add_rmsnorm_kernel.kd | 120 | 60 | 4400.8 | 5.0 | 5.0 | 4 |
| k23 | hgemm_bf16_16x64x128x5_SPK2_W1x2x1_BLDS1_TN_AS1_0.kd | 6 | 3 | 320.1 | 10.7 | 10.8 | 4 |
| k24 | _swiglu_oai_kernel.kd | 3 | 1 | 315.8 | 4.2 | 4.2 | 4 |
| k25 | hgemm_bf16_16x64x64x7_SPK6_W1x2x1_BLDS1_TN_AS1_0.kd | 57 | 28 | 4485.4 | 9.4 | 9.5 | 4 |
| k26 | void vllm::minimax_m3_fused_ops::fusedMiniMaxM3QNormRopeKVInsertKernel<c10::BFloat16, unsigned ch... | 57 | 28 | 4494.8 | 4.9 | 4.8 | 4 |
| k27 | _decode_index_score_balanced_kernel.kd | 57 | 28 | 4499.7 | 7.7 | 7.6 | 4 |
| k28 | _decode_topk_fused_kernel.kd | 57 | 28 | 4507.4 | 4.7 | 4.7 | 4 |
| k29 | _gqa_sparse_decode_kernel.kd | 57 | 28 | 4512.1 | 7.7 | 7.7 | 4 |
| k30 | _merge_topk_attn_out_kernel.kd | 57 | 28 | 4519.8 | 4.8 | 4.7 | 4 |
| k31 | _rocm_fp32_router_gemm_kernel.kd | 57 | 28 | 4550.1 | 5.3 | 5.3 | 4 |
| k32 | void aiter::grouped_topk_kernel<float, float __vector(4), 1, true, true, false>(float*, float con... | 57 | 28 | 4555.4 | 4.5 | 4.4 | 4 |
| k33 | void aiter::opus_moe_sorting_entry<aiter::MoeSortingKernel<aiter::MoeSortingProblemEx<int, float,... | 57 | 28 | 4559.9 | 6.7 | 6.8 | 4 |
| k34 | void at::native::vectorized_elementwise_kernel<8, at::native::FillFunctor<c10::BFloat16>, std::ar... | 57 | 28 | 4566.6 | 4.5 | 4.4 | 4 |
| k35 | void ck_tile::kentry<2, ck_tile::MoeFlatmmKernel<ck_tile::GemmSpatiallyLocalTilePartitioner<ck_ti... | 57 | 28 | 4571.0 | 25.8 | 25.9 | 4 |
| k36 | void aiter::swiglu_act_and_mul_kernel<std::bfloat16_t, std::bfloat16_t, 8>(std::bfloat16_t*, std:... | 57 | 28 | 4596.9 | 4.5 | 4.6 | 4 |
| k37 | void ck_tile::kentry<2, ck_tile::MoeFlatmmKernel<ck_tile::GemmSpatiallyLocalTilePartitioner<ck_ti... | 57 | 28 | 4601.4 | 14.8 | 14.3 | 4 |

## 6. Non-repeating kernels (C==1 in chosen window)

| name | ts_rel_us | dur_us | stream |
|---|---|---|---|
| void at::native::vectorized_elementwise_kernel<16, at::native::FillFunctor<bool>, std::array<char... | 0.0 | 4.4 | 4 |
| void (anonymous namespace)::elementwise_kernel_with_index<int, at::native::arange_cuda_out(c10::S... | 17.9 | 2.7 | 4 |
| _prepare_pos_seq_lens_kernel.kd | 38.1 | 3.1 | 4 |
| _combine_sampled_and_draft_tokens_kernel.kd | 41.2 | 4.6 | 4 |
| _gather_block_tables_kernel.kd | 45.8 | 4.7 | 4 |
| _compute_slot_mappings_kernel.kd | 50.2 | 4.4 | 4 |
| void at::native::unrolled_elementwise_kernel<at::native::CUDAFunctor_add<int>, std::array<char*, ... | 54.9 | 4.2 | 4 |
| void at::native::vectorized_elementwise_kernel<4, at::native::CUDAFunctor_add<int>, std::array<ch... | 59.0 | 4.3 | 4 |
| triton_poi_fused_add_bitwise_and_bitwise_not_bitwise_or_ge_lt_mul_sub_0.kd | 72.6 | 4.5 | 4 |
| void at::native::(anonymous namespace)::indexSelectSmallIndex<c10::BFloat16, long, unsigned int, ... | 77.1 | 7.6 | 4 |
| void at::native::elementwise_kernel_manual_unroll<128, 8, at::native::gpu_kernel_impl_nocast<at::... | 84.7 | 5.2 | 4 |
| _gemma_rmsnorm_kernel.kd | 100.3 | 4.4 | 4 |
| void at::native::vectorized_gather_kernel<16, long>(char*, char*, long*, int, long, long, long, l... | 8751.6 | 4.5 | 4 |
| _gemm_a16_w16_kernel_BLOCK_SIZE_M_16_BLOCK_SIZE_N_32_BLOCK_SIZE_K_256_GROUP_SIZE_M_1_NUM_KSPLI....kd | 8756.1 | 129.9 | 4 |
| ncclDevKernel_Generic_1(ncclDevKernelArgsStorage<4096ul>) [clone .kd] | 8886.0 | 53.8 | 4 |
| void at::native::elementwise_kernel_manual_unroll<128, 8, at::native::gpu_kernel_impl_nocast<at::... | 8958.1 | 7.4 | 4 |
| void at::native::index_elementwise_kernel<128, 4, at::native::gpu_index_kernel<at::native::index_... | 8965.6 | 4.2 | 4 |
| void at::native::index_elementwise_kernel<128, 4, at::native::gpu_index_kernel<at::native::index_... | 8969.7 | 4.6 | 4 |
| _gumbel_sample_kernel.kd | 8974.4 | 6.4 | 4 |
| void at::native::reduce_kernel<512, 1, at::native::ReduceOp<float, at::native::ArgMaxOps<float>, ... | 8980.9 | 7.4 | 4 |
| void at::native::_scatter_gather_elementwise_kernel<256, 4, at::native::_cuda_scatter_gather_inte... | 8988.1 | 4.2 | 4 |
| _get_num_sampled_and_rejected_kernel.kd | 8996.7 | 3.5 | 4 |
| _post_update_kernel.kd | 9009.2 | 4.6 | 4 |

## 7. Name dictionary (id -> full kernel name)

```
k1 = void at::native::vectorized_elementwise_kernel<16, at::native::FillFunctor<bool>, std::array<char*, 1ul> >(int, at::native::FillFunctor<bool>, std::array<char*, 1ul>) [clone .kd]
k2 = __amd_rocclr_copyBuffer.kd
k3 = void (anonymous namespace)::elementwise_kernel_with_index<int, at::native::arange_cuda_out(c10::Scalar const&, c10::Scalar const&, c10::Scalar const&, at::Tensor&)::{lambda()#1}::operator()() const::{lambda()#3}::operator()() const::{lambda(long)#1}>(int, at::native::arange_cuda_out(c10::Scalar const&, c10::Scalar const&, c10::Scalar const&, at::Tensor&)::{lambda()#1}::operator()() const::{lambda()#3}::operator()() const::{lambda(long)#1}, function_traits<at::native::arange_cuda_out(c10::Scalar const&, c10::Scalar const&, c10::Scalar const&, at::Tensor&)::{lambda()#1}::operator()() const::{lambda()#3}::operator()() const::{lambda(long)#1}>::result_type*) [clone .kd]
k4 = void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<int>, std::array<char*, 1ul> >(int, at::native::FillFunctor<int>, std::array<char*, 1ul>) [clone .kd]
k5 = _prepare_pos_seq_lens_kernel.kd
k6 = _combine_sampled_and_draft_tokens_kernel.kd
k7 = _gather_block_tables_kernel.kd
k8 = _compute_slot_mappings_kernel.kd
k9 = void at::native::unrolled_elementwise_kernel<at::native::CUDAFunctor_add<int>, std::array<char*, 3ul>, 4, TrivialOffsetCalculator<2, unsigned int>, TrivialOffsetCalculator<1, unsigned int>, at::native::memory::LoadWithoutCast, at::native::memory::StoreWithoutCast>(int, at::native::CUDAFunctor_add<int>, std::array<char*, 3ul>, TrivialOffsetCalculator<2, unsigned int>, TrivialOffsetCalculator<1, unsigned int>, at::native::memory::LoadWithoutCast, at::native::memory::StoreWithoutCast) [clone .kd]
k10 = void at::native::vectorized_elementwise_kernel<4, at::native::CUDAFunctor_add<int>, std::array<char*, 3ul> >(int, at::native::CUDAFunctor_add<int>, std::array<char*, 3ul>) [clone .kd]
k11 = triton_poi_fused_add_bitwise_and_bitwise_not_bitwise_or_ge_lt_mul_sub_0.kd
k12 = void at::native::(anonymous namespace)::indexSelectSmallIndex<c10::BFloat16, long, unsigned int, 2, 2, -2>(at::cuda::detail::TensorInfo<c10::BFloat16, unsigned int>, at::cuda::detail::TensorInfo<c10::BFloat16 const, unsigned int>, at::cuda::detail::TensorInfo<long const, unsigned int>, int, int, unsigned int, long) [clone .kd]
k13 = void at::native::elementwise_kernel_manual_unroll<128, 8, at::native::gpu_kernel_impl_nocast<at::native::(anonymous namespace)::masked_fill_kernel(at::TensorIterator&, c10::Scalar const&)::{lambda()#1}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16, bool)#1}>(at::TensorIteratorBase&, at::native::(anonymous namespace)::masked_fill_kernel(at::TensorIterator&, c10::Scalar const&)::{lambda()#1}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16, bool)#1} const&)::{lambda(int, bool)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::(anonymous namespace)::masked_fill_kernel(at::TensorIterator&, c10::Scalar const&)::{lambda()#1}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16, bool)#1}>(at::TensorIteratorBase&, at::native::(anonymous namespace)::masked_fill_kernel(at::TensorIterator&, c10::Scalar const&)::{lambda()#1}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16, bool)#1} const&)::{lambda(int, bool)#1}) [clone .kd]
k14 = void aiter::cross_device_reduce_1stage<std::bfloat16_t, 4, false>(aiter::RankData*, aiter::RankData*, aiter::RankSignals, aiter::Signal*, std::bfloat16_t*, int, int) [clone .kd]
k15 = _gemma_rmsnorm_kernel.kd
k16 = hgemm_bf16_16x64x64x8_SPK6_W1x2x1_BLDS1_TN_AS1_0.kd
k17 = void vllm::minimax_m3_fused_ops::fusedMiniMaxM3QNormRopeKVInsertKernel<c10::BFloat16, __hip_bfloat16, (vllm::Fp8KVCacheDataType)0, c10::BFloat16, false, false, false, false>(c10::BFloat16*, c10::BFloat16*, unsigned char*, c10::BFloat16*, c10::BFloat16 const*, c10::BFloat16 const*, c10::BFloat16 const*, c10::BFloat16 const*, c10::BFloat16 const*, long const*, long const*, long const*, __hip_bfloat16*, c10::BFloat16*, float, float, int, int, int, int, int, int, long, long, long, long) [clone .kd]
k18 = reshape_and_cache_kernel_flash.kd
k19 = kernel_unified_attention.kd
k20 = reduce_segments.kd
k21 = _gemm_a16_w16_kernel_BLOCK_SIZE_M_16_BLOCK_SIZE_N_32_BLOCK_SIZE_K_256_GROUP_SIZE_M_1_NUM_KSPLIT_1_SPLITK_BLOCK_SIZE_2048_EVEN_K_1_EVEN_MN_0_cache_modifier_CG_activation_NONE_use_activation_0_ADD_BIAS_0_SKIP_REDUCE_0.kd
k22 = _gemma_fused_add_rmsnorm_kernel.kd
k23 = hgemm_bf16_16x64x128x5_SPK2_W1x2x1_BLDS1_TN_AS1_0.kd
k24 = _swiglu_oai_kernel.kd
k25 = hgemm_bf16_16x64x64x7_SPK6_W1x2x1_BLDS1_TN_AS1_0.kd
k26 = void vllm::minimax_m3_fused_ops::fusedMiniMaxM3QNormRopeKVInsertKernel<c10::BFloat16, unsigned char, (vllm::Fp8KVCacheDataType)1, c10::BFloat16, true, true, true, false>(c10::BFloat16*, c10::BFloat16*, unsigned char*, c10::BFloat16*, c10::BFloat16 const*, c10::BFloat16 const*, c10::BFloat16 const*, c10::BFloat16 const*, c10::BFloat16 const*, long const*, long const*, long const*, unsigned char*, c10::BFloat16*, float, float, int, int, int, int, int, int, long, long, long, long) [clone .kd]
k27 = _decode_index_score_balanced_kernel.kd
k28 = _decode_topk_fused_kernel.kd
k29 = _gqa_sparse_decode_kernel.kd
k30 = _merge_topk_attn_out_kernel.kd
k31 = _rocm_fp32_router_gemm_kernel.kd
k32 = void aiter::grouped_topk_kernel<float, float __vector(4), 1, true, true, false>(float*, float const*, float*, int*, unsigned long, unsigned long, int, int, int, int, float) [clone .kd]
k33 = void aiter::opus_moe_sorting_entry<aiter::MoeSortingKernel<aiter::MoeSortingProblemEx<int, float, 1, true, false, false, true, 0> >, aiter::MoeSortingKernel<aiter::MoeSortingProblemEx<int, float, 1, true, false, false, true, 0> >::Kargs>(aiter::MoeSortingKernel<aiter::MoeSortingProblemEx<int, float, 1, true, false, false, true, 0> >::Kargs) [clone .kd]
k34 = void at::native::vectorized_elementwise_kernel<8, at::native::FillFunctor<c10::BFloat16>, std::array<char*, 1ul> >(int, at::native::FillFunctor<c10::BFloat16>, std::array<char*, 1ul>) [clone .kd]
k35 = void ck_tile::kentry<2, ck_tile::MoeFlatmmKernel<ck_tile::GemmSpatiallyLocalTilePartitioner<ck_tile::TileGemmShape<ck_tile::sequence<16, 128, 256>, ck_tile::sequence<1, 4, 1>, ck_tile::sequence<16, 16, 32>, false, false>, 1, 1>, ck_tile::F16xMXF4FlatmmPipelineAGmemBGmemCRegV1<ck_tile::F16xMXF4FlatmmPipelineProblem<std::bfloat16_t, ck_tile::pk_float4_e2m1_t, float, ck_tile::TileGemmShape<ck_tile::sequence<16, 128, 256>, ck_tile::sequence<1, 4, 1>, ck_tile::sequence<16, 16, 32>, false, false>, ck_tile::TileGemmUniversalTraits<false, false, false, false, ck_tile::tensor_layout::gemm::RowMajor, ck_tile::tensor_layout::gemm::ColumnMajor, ck_tile::tensor_layout::gemm::RowMajor, false, false, false, 1, true, 16, (ck_tile::DataCachePrefetchKind)0, (ck_tile::DataCachePrefetchKind)0, false, false>, (ck_tile::GemmPipelineScheduler)0, true, (ck_tile::TailNumber)1, (ck_tile::amd_buffer_coherence_enum)2, false, std::bfloat16_t>, ck_tile::F16xMXF4FlatmmPipelineAgBgCrPolicy>, ck_tile::CShuffleEpilogue<ck_tile::CShuffleEpilogueProblem<std::bfloat16_t, std::bfloat16_t, ck_tile::tuple<>, float, std::bfloat16_t, ck_tile::tuple<>, ck_tile::tensor_layout::gemm::RowMajor, ck_tile::element_wise::PassThrough, 16, 128, 1, 4, 16, 16, 32, false, 1, false, 1, 2, false, void, void, false, float, std::bfloat16_t>, void>, (ck_tile::MoeFlatmmKind)3, ck_tile::moe::MoeSilu>, ck_tile::MoeFlatmmKernel<ck_tile::GemmSpatiallyLocalTilePartitioner<ck_tile::TileGemmShape<ck_tile::sequence<16, 128, 256>, ck_tile::sequence<1, 4, 1>, ck_tile::sequence<16, 16, 32>, false, false>, 1, 1>, ck_tile::F16xMXF4FlatmmPipelineAGmemBGmemCRegV1<ck_tile::F16xMXF4FlatmmPipelineProblem<std::bfloat16_t, ck_tile::pk_float4_e2m1_t, float, ck_tile::TileGemmShape<ck_tile::sequence<16, 128, 256>, ck_tile::sequence<1, 4, 1>, ck_tile::sequence<16, 16, 32>, false, false>, ck_tile::TileGemmUniversalTraits<false, false, false, false, ck_tile::tensor_layout::gemm::RowMajor, ck_tile::tensor_layout::gemm::ColumnMajor, ck_tile::tensor_layout::gemm::RowMajor, false, false, false, 1, true, 16, (ck_tile::DataCachePrefetchKind)0, (ck_tile::DataCachePrefetchKind)0, false, false>, (ck_tile::GemmPipelineScheduler)0, true, (ck_tile::TailNumber)1, (ck_tile::amd_buffer_coherence_enum)2, false, std::bfloat16_t>, ck_tile::F16xMXF4FlatmmPipelineAgBgCrPolicy>, ck_tile::CShuffleEpilogue<ck_tile::CShuffleEpilogueProblem<std::bfloat16_t, std::bfloat16_t, ck_tile::tuple<>, float, std::bfloat16_t, ck_tile::tuple<>, ck_tile::tensor_layout::gemm::RowMajor, ck_tile::element_wise::PassThrough, 16, 128, 1, 4, 16, 16, 32, false, 1, false, 1, 2, false, void, void, false, float, std::bfloat16_t>, void>, (ck_tile::MoeFlatmmKind)3, ck_tile::moe::MoeSilu>::MoeFlatmmKernelArgs<ck_tile::FlatmmScalePointer<1, 32, ck_tile::e8m0_bexp_t>, ck_tile::FlatmmScalePointer<1, 32, ck_tile::e8m0_bexp_t>, ck_tile::FlatmmScalePointer<-1, 0, float> > >(ck_tile::MoeFlatmmKernel<ck_tile::GemmSpatiallyLocalTilePartitioner<ck_tile::TileGemmShape<ck_tile::sequence<16, 128, 256>, ck_tile::sequence<1, 4, 1>, ck_tile::sequence<16, 16, 32>, false, false>, 1, 1>, ck_tile::F16xMXF4FlatmmPipelineAGmemBGmemCRegV1<ck_tile::F16xMXF4FlatmmPipelineProblem<std::bfloat16_t, ck_tile::pk_float4_e2m1_t, float, ck_tile::TileGemmShape<ck_tile::sequence<16, 128, 256>, ck_tile::sequence<1, 4, 1>, ck_tile::sequence<16, 16, 32>, false, false>, ck_tile::TileGemmUniversalTraits<false, false, false, false, ck_tile::tensor_layout::gemm::RowMajor, ck_tile::tensor_layout::gemm::ColumnMajor, ck_tile::tensor_layout::gemm::RowMajor, false, false, false, 1, true, 16, (ck_tile::DataCachePrefetchKind)0, (ck_tile::DataCachePrefetchKind)0, false, false>, (ck_tile::GemmPipelineScheduler)0, true, (ck_tile::TailNumber)1, (ck_tile::amd_buffer_coherence_enum)2, false, std::bfloat16_t>, ck_tile::F16xMXF4FlatmmPipelineAgBgCrPolicy>, ck_tile::CShuffleEpilogue<ck_tile::CShuffleEpilogueProblem<std::bfloat16_t, std::bfloat16_t, ck_tile::tuple<>, float, std::bfloat16_t, ck_tile::tuple<>, ck_tile::tensor_layout::gemm::RowMajor, ck_tile::element_wise::PassThrough, 16, 128, 1, 4, 16, 16, 32, false, 1, false, 1, 2, false, void, void, false, float, std::bfloat16_t>, void>, (ck_tile::MoeFlatmmKind)3, ck_tile::moe::MoeSilu>::MoeFlatmmKernelArgs<ck_tile::FlatmmScalePointer<1, 32, ck_tile::e8m0_bexp_t>, ck_tile::FlatmmScalePointer<1, 32, ck_tile::e8m0_bexp_t>, ck_tile::FlatmmScalePointer<-1, 0, float> >) [clone .kd]
k36 = void aiter::swiglu_act_and_mul_kernel<std::bfloat16_t, std::bfloat16_t, 8>(std::bfloat16_t*, std::bfloat16_t const*, int) [clone .kd]
k37 = void ck_tile::kentry<2, ck_tile::MoeFlatmmKernel<ck_tile::GemmSpatiallyLocalTilePartitioner<ck_tile::TileGemmShape<ck_tile::sequence<16, 128, 256>, ck_tile::sequence<1, 4, 1>, ck_tile::sequence<16, 16, 32>, false, false>, 1, 1>, ck_tile::F16xMXF4FlatmmPipelineAGmemBGmemCRegV1<ck_tile::F16xMXF4FlatmmPipelineProblem<std::bfloat16_t, ck_tile::pk_float4_e2m1_t, float, ck_tile::TileGemmShape<ck_tile::sequence<16, 128, 256>, ck_tile::sequence<1, 4, 1>, ck_tile::sequence<16, 16, 32>, false, false>, ck_tile::TileGemmUniversalTraits<false, false, false, false, ck_tile::tensor_layout::gemm::RowMajor, ck_tile::tensor_layout::gemm::ColumnMajor, ck_tile::tensor_layout::gemm::RowMajor, false, false, false, 1, true, 16, (ck_tile::DataCachePrefetchKind)0, (ck_tile::DataCachePrefetchKind)0, false, false>, (ck_tile::GemmPipelineScheduler)0, true, (ck_tile::TailNumber)0, (ck_tile::amd_buffer_coherence_enum)2, false, std::bfloat16_t>, ck_tile::F16xMXF4FlatmmPipelineAgBgCrPolicy>, ck_tile::CShuffleEpilogue<ck_tile::CShuffleEpilogueProblem<std::bfloat16_t, std::bfloat16_t, ck_tile::tuple<>, float, std::bfloat16_t, ck_tile::tuple<>, ck_tile::tensor_layout::gemm::RowMajor, ck_tile::element_wise::PassThrough, 16, 128, 1, 4, 16, 16, 32, false, 1, false, 1, 2, false, void, void, false, float, std::bfloat16_t>, void>, (ck_tile::MoeFlatmmKind)2, ck_tile::moe::MoeSilu>, ck_tile::MoeFlatmmKernel<ck_tile::GemmSpatiallyLocalTilePartitioner<ck_tile::TileGemmShape<ck_tile::sequence<16, 128, 256>, ck_tile::sequence<1, 4, 1>, ck_tile::sequence<16, 16, 32>, false, false>, 1, 1>, ck_tile::F16xMXF4FlatmmPipelineAGmemBGmemCRegV1<ck_tile::F16xMXF4FlatmmPipelineProblem<std::bfloat16_t, ck_tile::pk_float4_e2m1_t, float, ck_tile::TileGemmShape<ck_tile::sequence<16, 128, 256>, ck_tile::sequence<1, 4, 1>, ck_tile::sequence<16, 16, 32>, false, false>, ck_tile::TileGemmUniversalTraits<false, false, false, false, ck_tile::tensor_layout::gemm::RowMajor, ck_tile::tensor_layout::gemm::ColumnMajor, ck_tile::tensor_layout::gemm::RowMajor, false, false, false, 1, true, 16, (ck_tile::DataCachePrefetchKind)0, (ck_tile::DataCachePrefetchKind)0, false, false>, (ck_tile::GemmPipelineScheduler)0, true, (ck_tile::TailNumber)0, (ck_tile::amd_buffer_coherence_enum)2, false, std::bfloat16_t>, ck_tile::F16xMXF4FlatmmPipelineAgBgCrPolicy>, ck_tile::CShuffleEpilogue<ck_tile::CShuffleEpilogueProblem<std::bfloat16_t, std::bfloat16_t, ck_tile::tuple<>, float, std::bfloat16_t, ck_tile::tuple<>, ck_tile::tensor_layout::gemm::RowMajor, ck_tile::element_wise::PassThrough, 16, 128, 1, 4, 16, 16, 32, false, 1, false, 1, 2, false, void, void, false, float, std::bfloat16_t>, void>, (ck_tile::MoeFlatmmKind)2, ck_tile::moe::MoeSilu>::MoeFlatmmKernelArgs<ck_tile::FlatmmScalePointer<1, 32, ck_tile::e8m0_bexp_t>, ck_tile::FlatmmScalePointer<1, 32, ck_tile::e8m0_bexp_t>, ck_tile::FlatmmScalePointer<-1, 0, float> > >(ck_tile::MoeFlatmmKernel<ck_tile::GemmSpatiallyLocalTilePartitioner<ck_tile::TileGemmShape<ck_tile::sequence<16, 128, 256>, ck_tile::sequence<1, 4, 1>, ck_tile::sequence<16, 16, 32>, false, false>, 1, 1>, ck_tile::F16xMXF4FlatmmPipelineAGmemBGmemCRegV1<ck_tile::F16xMXF4FlatmmPipelineProblem<std::bfloat16_t, ck_tile::pk_float4_e2m1_t, float, ck_tile::TileGemmShape<ck_tile::sequence<16, 128, 256>, ck_tile::sequence<1, 4, 1>, ck_tile::sequence<16, 16, 32>, false, false>, ck_tile::TileGemmUniversalTraits<false, false, false, false, ck_tile::tensor_layout::gemm::RowMajor, ck_tile::tensor_layout::gemm::ColumnMajor, ck_tile::tensor_layout::gemm::RowMajor, false, false, false, 1, true, 16, (ck_tile::DataCachePrefetchKind)0, (ck_tile::DataCachePrefetchKind)0, false, false>, (ck_tile::GemmPipelineScheduler)0, true, (ck_tile::TailNumber)0, (ck_tile::amd_buffer_coherence_enum)2, false, std::bfloat16_t>, ck_tile::F16xMXF4FlatmmPipelineAgBgCrPolicy>, ck_tile::CShuffleEpilogue<ck_tile::CShuffleEpilogueProblem<std::bfloat16_t, std::bfloat16_t, ck_tile::tuple<>, float, std::bfloat16_t, ck_tile::tuple<>, ck_tile::tensor_layout::gemm::RowMajor, ck_tile::element_wise::PassThrough, 16, 128, 1, 4, 16, 16, 32, false, 1, false, 1, 2, false, void, void, false, float, std::bfloat16_t>, void>, (ck_tile::MoeFlatmmKind)2, ck_tile::moe::MoeSilu>::MoeFlatmmKernelArgs<ck_tile::FlatmmScalePointer<1, 32, ck_tile::e8m0_bexp_t>, ck_tile::FlatmmScalePointer<1, 32, ck_tile::e8m0_bexp_t>, ck_tile::FlatmmScalePointer<-1, 0, float> >) [clone .kd]
k38 = void at::native::vectorized_gather_kernel<16, long>(char*, char*, long*, int, long, long, long, long, bool) [clone .kd]
k39 = _gemm_a16_w16_kernel_BLOCK_SIZE_M_16_BLOCK_SIZE_N_32_BLOCK_SIZE_K_256_GROUP_SIZE_M_1_NUM_KSPLIT_1_SPLITK_BLOCK_SIZE_6144_EVEN_K_1_EVEN_MN_0_cache_modifier_CG_activation_NONE_use_activation_0_ADD_BIAS_0_SKIP_REDUCE_0.kd
k40 = ncclDevKernel_Generic_1(ncclDevKernelArgsStorage<4096ul>) [clone .kd]
k41 = void at::native::elementwise_kernel_manual_unroll<128, 8, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1} const&)::{lambda(int, bool)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1}>(at::TensorIteratorBase&, at::native::direct_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda()#3}::operator()() const::{lambda()#12}::operator()() const::{lambda(c10::BFloat16)#1} const&)::{lambda(int, bool)#1}) [clone .kd]
k42 = void at::native::index_elementwise_kernel<128, 4, at::native::gpu_index_kernel<at::native::index_kernel_impl<at::native::OpaqueType<8> >(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>)::{lambda(char*, char const*, long)#1}>(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>, at::native::index_kernel_impl<at::native::OpaqueType<8> >(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>)::{lambda(char*, char const*, long)#1} const&, bool)::{lambda(int)#1}>(long, at::native::gpu_index_kernel<at::native::index_kernel_impl<at::native::OpaqueType<8> >(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>)::{lambda(char*, char const*, long)#1}>(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>, at::native::index_kernel_impl<at::native::OpaqueType<8> >(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>)::{lambda(char*, char const*, long)#1} const&, bool)::{lambda(int)#1}) [clone .kd]
k43 = void at::native::index_elementwise_kernel<128, 4, at::native::gpu_index_kernel<at::native::index_kernel_impl<at::native::OpaqueType<4> >(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>)::{lambda(char*, char const*, long)#1}>(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>, at::native::index_kernel_impl<at::native::OpaqueType<4> >(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>)::{lambda(char*, char const*, long)#1} const&, bool)::{lambda(int)#1}>(long, at::native::gpu_index_kernel<at::native::index_kernel_impl<at::native::OpaqueType<4> >(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>)::{lambda(char*, char const*, long)#1}>(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>, at::native::index_kernel_impl<at::native::OpaqueType<4> >(at::TensorIteratorBase&, c10::ArrayRef<long>, c10::ArrayRef<long>)::{lambda(char*, char const*, long)#1} const&, bool)::{lambda(int)#1}) [clone .kd]
k44 = _gumbel_sample_kernel.kd
k45 = void at::native::reduce_kernel<512, 1, at::native::ReduceOp<float, at::native::ArgMaxOps<float>, unsigned int, long, 4, 4> >(at::native::ReduceOp<float, at::native::ArgMaxOps<float>, unsigned int, long, 4, 4>) [clone .kd]
k46 = void at::native::_scatter_gather_elementwise_kernel<256, 4, at::native::_cuda_scatter_gather_internal_kernel<false, at::native::OpaqueType<8>, long>::operator()<at::native::TensorAssign>(at::TensorIterator&, long, long, long, at::native::TensorAssign const&)::{lambda(int)#1}>(int, at::native::_cuda_scatter_gather_internal_kernel<false, at::native::OpaqueType<8>, long>::operator()<at::native::TensorAssign>(at::TensorIterator&, long, long, long, at::native::TensorAssign const&)::{lambda(int)#1}) [clone .kd]
k47 = _get_num_sampled_and_rejected_kernel.kd
k48 = _post_update_kernel.kd
```

## 8. Notes / warnings

- WARNING: window 0 differs from steady-state windows (warmup/init kernels); excluded from replay check.

Full data: `vllm-minimax-m3-tp4-8k1k-conc8_slice.json` (rollup / middle_slice / once_kernels / windows / streams / annotations)