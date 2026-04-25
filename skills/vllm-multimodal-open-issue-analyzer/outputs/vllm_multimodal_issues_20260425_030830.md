# vllm-project/vllm 多模态相关 Open Issues

> 仓库: [vllm-project/vllm](https://github.com/vllm-project/vllm)  
> 生成时间: 2026-04-25 03:08 UTC  
> 共 **389** 个 open issue

## 一、Issue 总览

| # | 标题 | 作者 | 创建日期 | Labels |
|---|------|------|----------|--------|
| [#40856](https://github.com/vllm-project/vllm/issues/40856) | [Bug]: VLLM running qwen3.6 for image inference occasionally reports 500 Internal Server Error | @JasonChao-yue | 2026-04-25 | `bug` |
| [#40831](https://github.com/vllm-project/vllm/issues/40831) | [Bug]: TurboQuant KV × any speculative decoding (MTP or ngram) produces degenerate token loops — confirmed across dense and hybrid attention | @noonghunna | 2026-04-24 | - |
| [#40821](https://github.com/vllm-project/vllm/issues/40821) | [Bug]: Deepseek V4 failed to load on RTX PRO 6000 | @jerin-metrum-ai | 2026-04-24 | `bug` |
| [#40807](https://github.com/vllm-project/vllm/issues/40807) | [Bug]: TurboQuant KV + spec-decode + chunked-prefill crashes CUDA graph capture at query_start_loc.tolist() in continuation-prefill path (Qwen3-Next hybrid dense) | @noonghunna | 2026-04-24 | `bug` |
| [#40802](https://github.com/vllm-project/vllm/issues/40802) | [Feature]: Deepseek V4 cannot run ,Please support SM120 GPU,example rtx5090  rtxpro6000 | @wuwenthink | 2026-04-24 | `feature request` |
| [#40781](https://github.com/vllm-project/vllm/issues/40781) | [Feature]: vllm support audio otel tracing? | @laik | 2026-04-24 | `feature request` |
| [#40765](https://github.com/vllm-project/vllm/issues/40765) | [Bug]: runai_streamer loads both Ministral consolidated and HF sharded safetensors | @dhayanesh | 2026-04-24 | `bug` |
| [#40758](https://github.com/vllm-project/vllm/issues/40758) | [CI Failure]: `Qwen3.6-35B-A3B-FP8` fails on `NVIDIA GB10` with `cutlass_scaled_mm` / `cutlass_gemm_caller Error Internal` under vLLM nightly + CUDA 13.0 | @amuin-2hz | 2026-04-24 | `ci-failure` |
| [#40742](https://github.com/vllm-project/vllm/issues/40742) | [Bug]: CUDA graph capture crashes during startup due to Inductor autotuning torch.cuda.synchronize() inside graph capture (FULL_DECODE_ONLY + MLA + FP8) when PDL is enabled | @kyleliang-nv | 2026-04-23 | `torch.compile` |
| [#40741](https://github.com/vllm-project/vllm/issues/40741) | [Feature]: Make opencv-python-headless an optional dependency for FIPS compliance | @rdwj | 2026-04-23 | `feature request` |
| [#40707](https://github.com/vllm-project/vllm/issues/40707) | [Bug]: Scheduling deadlock in _mamba_block_aligned_split with multiple large multimodal inputs on hybrid Mamba models | @fanghao566 | 2026-04-23 | `bug` |
| [#40661](https://github.com/vllm-project/vllm/issues/40661) | [Bug]: CUBLAS_STATUS_EXECUTION_FAILED during CUDA graph compilation of BF16 vision encoder on NVIDIA Jetson AGX Thor (vLLM 0.19.0 regression) | @hvaniya5 | 2026-04-23 | - |
| [#40652](https://github.com/vllm-project/vllm/issues/40652) | [Bug]: Kimi 2.6 on 8x A100 SMX4 leads to NVLink Crash Coredump | @inheaden-admin | 2026-04-22 | `bug` |
| [#40649](https://github.com/vllm-project/vllm/issues/40649) | [Bug]: KeyError on model.layers.N.self_attn.attn during initialize_attn_backend with pipeline_parallel_size=4 (V1 engine + Ray) | @andersonlunz | 2026-04-22 | `bug` |
| [#40642](https://github.com/vllm-project/vllm/issues/40642) | KimiK25ForConditionalGeneration failed to be inspected — SIGSEGV in registry subprocess during process exit
  (GB200) | @c2w-sea | 2026-04-22 | - |
| [#40620](https://github.com/vllm-project/vllm/issues/40620) | [RFC]: Unified Device Capability Abstraction for Cross-Platform Feature Detection | @jikunshang | 2026-04-22 | `rocm`, `RFC` |
| [#40616](https://github.com/vllm-project/vllm/issues/40616) | [Bug]:  `/v2/embed` with `input_type` returns misleading 400 on nemotron-embed-vl | @oliverholworthy | 2026-04-22 | `bug` |
| [#40608](https://github.com/vllm-project/vllm/issues/40608) | [Kimi] Track Kimi K2.5/K2.6 MLA + EAGLE serving on Blackwell (DCP4/DCP8, FP8 KV, draft backend split) | @voipmonitor | 2026-04-22 | - |
| [#40591](https://github.com/vllm-project/vllm/issues/40591) | [Bug]: Regression in 0.19.1 - Gemma 4 26B MoE fails to load packed experts (KeyError: down_proj_packed). Worked in dev6. | @ghazal-bh | 2026-04-22 | `bug` |
| [#40585](https://github.com/vllm-project/vllm/issues/40585) | [Bug]: qwen3.5 can not use --decode-context-parallel-size with --enable-prefix-caching | @crystalww | 2026-04-22 | `bug` |
| [#40543](https://github.com/vllm-project/vllm/issues/40543) | [Feature]: Dynamic PDL Enablement | @benchislett | 2026-04-21 | `feature request` |
| [#40536](https://github.com/vllm-project/vllm/issues/40536) | [Feature]: Support iterative in-place weight editing on TP workers (online RLHF / steering / abliteration) | @wuwangzhang1216 | 2026-04-21 | - |
| [#40444](https://github.com/vllm-project/vllm/issues/40444) | [Bug]: Per-attention-head quantization is currently available only with the Flash Attention backend and requires the calibration pathway provided by llm-compressor. | @cuihangbin | 2026-04-21 | `bug` |
| [#40443](https://github.com/vllm-project/vllm/issues/40443) | [BUG] Port-allocation race between ApiServer processes in hybrid-LB mode (ZMQError: Address already in use) | @jing-4369 | 2026-04-21 | - |
| [#40420](https://github.com/vllm-project/vllm/issues/40420) | [Bug]: TurboQuant `_continuation_prefill` OOMs and kills engine at long-context prefill (~185K actual tokens) | @jhsmith409 | 2026-04-21 | - |
| [#40417](https://github.com/vllm-project/vllm/issues/40417) | [Bug]: `token_capacity_kv_cache_groups` (#40384) should also exclude `SlidingWindowSpec` / `ChunkedLocalAttentionSpec` | @Sandermage | 2026-04-21 | - |
| [#40381](https://github.com/vllm-project/vllm/issues/40381) | [Bug]: Buffer overflow when allocating memory error on Qwen3.5-122B-A10B-GPTQ-Int4 and NVFP4 | @ECMGit | 2026-04-20 | `bug` |
| [#40343](https://github.com/vllm-project/vllm/issues/40343) | [Installation]: 有提供cuda 12.6+python3.12的vllm预编译的whl包吗？ 以开发者模型需要本地安装下，发布的都是cuda13版本的，不适配cuda12.6的本机的版本型号 | @cuihangbin | 2026-04-20 | `installation` |
| [#40318](https://github.com/vllm-project/vllm/issues/40318) | [Bug]: Mistral3 text-only startup fails when text_config.architectures is None | @thromel | 2026-04-20 | - |
| [#40290](https://github.com/vllm-project/vllm/issues/40290) | [Bug]: Gemma 4 (31B/26B-A4B) vision outputs only <pad> under fp16 — vision_tower standardize overflows | @wenqiangire-commits | 2026-04-19 | - |
| [#40165](https://github.com/vllm-project/vllm/issues/40165) | [Bug]: HunyuanOCR crashes with "query and key must have the same dtype" during inference (vLLM 0.19.0, RTX 3050) | @hungthikcode | 2026-04-17 | `bug` |
| [#40121](https://github.com/vllm-project/vllm/issues/40121) | [Bug]: CUDA graph replay triggers Xid 13 illegal memory access on Qwen3-32B-AWQ with TP=2 on dual RTX 3090 | @kevinb361 | 2026-04-17 | `bug` |
| [#40113](https://github.com/vllm-project/vllm/issues/40113) | [CI Failure]: Multi-Modal Processor (CPU) | @ElizaWszola | 2026-04-17 | `ci-failure` |
| [#40106](https://github.com/vllm-project/vllm/issues/40106) | [Bug]: Gemma4 multimodal: missing vision-aware bidirectional attention mask for use_bidirectional_attention="vision" models | @jQizhang | 2026-04-17 | `bug` |
| [#40095](https://github.com/vllm-project/vllm/issues/40095) | [Bug]: Gemma4MultimodalEmbedder normalization order different from Transformers, causing bad audio inference | @jzheng636 | 2026-04-17 | `bug` |
| [#40080](https://github.com/vllm-project/vllm/issues/40080) | [Bug]: Gemma 4 (31B / 26B-A4B) generates infinite repetition loops, especially with structured output (JSON schema) | @Foreist | 2026-04-17 | `bug` |
| [#40047](https://github.com/vllm-project/vllm/issues/40047) | [Bug][Tracking Issue]: NaNs in CUDA Graph padding regions corrupt activations in some per-token kernels | @tlrmchlsmth | 2026-04-16 | - |
| [#40038](https://github.com/vllm-project/vllm/issues/40038) | [Bug]: cudaErrorIllegalAddress during PIECEWISE CUDA graph replay with MoE LoRA: stale buffer addresses in `moe_lora_align_block_size` | @TheDuyIT | 2026-04-16 | `bug` |
| [#40018](https://github.com/vllm-project/vllm/issues/40018) | [Bug]: `ROCM_AITER_MLA_SPARSE` prefill produces garbage for prompt_len > ~20K tokens on gfx950 (GLM-5.1-FP8) | @ghpu | 2026-04-16 | `bug`, `rocm` |
| [#40002](https://github.com/vllm-project/vllm/issues/40002) | [Bug]: Inconsistent KV Cache reporting and system hang on long context requests (Gemma-4 26B AWQ Int4) | @GitEventhandler | 2026-04-16 | `bug` |
| [#39996](https://github.com/vllm-project/vllm/issues/39996) | [Bug] Fatal AssertionError: Encoder KV cache fails to evict tokens, exceeding max_model_len in long-lived WebSocket sessions | @BioAGI-Moretti | 2026-04-16 | `bug` |
| [#39993](https://github.com/vllm-project/vllm/issues/39993) | [Usage]: does vllm support Qwen3_5ForCausalLM architecture inference? not just Qwen3_5ForConditionalGeneration? | @zejunwang1 | 2026-04-16 | `usage` |
| [#39979](https://github.com/vllm-project/vllm/issues/39979) | [RFC]: Ultimate Better Observability. | @noooop | 2026-04-16 | `RFC` |
| [#39928](https://github.com/vllm-project/vllm/issues/39928) | [Bug]: Qwen3.5 DFlash gives strange responses on SM90 | @mgoin | 2026-04-15 | `bug` |
| [#39919](https://github.com/vllm-project/vllm/issues/39919) | [Bug]: DeepSeek OCR doesn't work on vllm 0.19 | @PatrycyD | 2026-04-15 | `bug` |
| [#39915](https://github.com/vllm-project/vllm/issues/39915) | [Bug]: Engine core initialization failed (Parent process exited) on 2xH100 with Llama-3.3-70B-FP8 (TP/PP=2) | @MigueXl | 2026-04-15 | `bug` |
| [#39903](https://github.com/vllm-project/vllm/issues/39903) | [Bug]: Significant Cross-Instance Inference Variance in vLLM v0.18.0 on H20 (~10-point gap) Qwen3.5-35B-A3B | @yszhli | 2026-04-15 | `bug` |
| [#39839](https://github.com/vllm-project/vllm/issues/39839) | [Bug]:  MTP DeepSeek and Eagle Flash Attention Failures in Spec Decode Unit Tests | @puririshi98 | 2026-04-14 | `bug` |
| [#39814](https://github.com/vllm-project/vllm/issues/39814) | [Bug]: FlashInferFP8ScaledMMLinearKernel segfaults on Blackwell (sm100) | @ZhanqiuHu | 2026-04-14 | `bug` |
| [#39788](https://github.com/vllm-project/vllm/issues/39788) | [Bug]: CUDA OOM with Kimi-K2.5 NVFP4 on both TP4 and TP8 | @msanft | 2026-04-14 | `bug` |
| [#39766](https://github.com/vllm-project/vllm/issues/39766) | [RFC]: Support Mooncake Based ECConnector for EPD | @shen-shanshan | 2026-04-14 | `RFC` |
| [#39761](https://github.com/vllm-project/vllm/issues/39761) | [Bug]:CUDA illegal instruction during decode (V1 Engine + NVFP4) on aarch64 (NVIDIA GB10) | @Xenon0220 | 2026-04-14 | `bug` |
| [#39749](https://github.com/vllm-project/vllm/issues/39749) | [Roadmap] [Draft] vLLM Roadmap Q2 2026 | @simon-mo | 2026-04-13 | `rocm` |
| [#39735](https://github.com/vllm-project/vllm/issues/39735) | [Feature]: Expose Word-Level Timestamps in `/v1/realtime` API for Voxtral Realtime | @sh1man | 2026-04-13 | `feature request` |
| [#39722](https://github.com/vllm-project/vllm/issues/39722) | Gibberish with flashinfer_nvlink_two_sided on GB200/arm64 | @S1ro1 | 2026-04-13 | - |
| [#39708](https://github.com/vllm-project/vllm/issues/39708) | [Feature]: Pre-ViT visual token pruning for VLMs (PixelPrune) | @nhsjgczryf | 2026-04-13 | `feature request` |
| [#39701](https://github.com/vllm-project/vllm/issues/39701) | [RFC] Replace routing replay with CUDA-graph-compatible device cache approach | @TomerBN-Nvidia | 2026-04-13 | `performance`, `RFC`, `rl` |
| [#39687](https://github.com/vllm-project/vllm/issues/39687) | [Bug]: vllm(g0e39202ca) vllm serve: error: argument --limit-mm-per-prompt: Value image=4,audio=1 cannot be converted to <function | @Honghe | 2026-04-13 | `bug` |
| [#39682](https://github.com/vllm-project/vllm/issues/39682) | [BUGS] vLLM V1 Engine Hangs After Weight Loading on Blackwell (sm_121) Multi-Node Ray Setup (TP=2) | @Haroliao | 2026-04-13 | `bug` |
| [#39681](https://github.com/vllm-project/vllm/issues/39681) | [Bug]: Gemma4 multimodal crashes with "pixel_values contains inconsistent shapes" when concurrent image requests have different resolutions | @art3na | 2026-04-13 | `bug` |
| [#39680](https://github.com/vllm-project/vllm/issues/39680) | [Performance]: Qwen3.5 with mtp is slower than without | @weishu20 | 2026-04-13 | `performance` |
| [#39631](https://github.com/vllm-project/vllm/issues/39631) | [Bug]: Abnormal Scores in Batch Processing of Image-Text Pairs with qwen3-VL-reranker Model | @xl2014 | 2026-04-12 | `bug` |
| [#39583](https://github.com/vllm-project/vllm/issues/39583) | [RFC]: Deprecate bitsandbytes and GGUF quantization support | @mgoin | 2026-04-11 | `RFC`, `quantization` |
| [#39504](https://github.com/vllm-project/vllm/issues/39504) | [RFC]: Enable prompt_embeds content parts in Chat Completions API | @LuisRobaina | 2026-04-10 | `RFC` |
| [#39485](https://github.com/vllm-project/vllm/issues/39485) | [Bug]: Runtime error on ROCm platform serving Deepseek-R1 using VLLM_ROCM_USE_AITER=1 | @vllmellm | 2026-04-10 | `bug`, `rocm` |
| [#39474](https://github.com/vllm-project/vllm/issues/39474) | [Bug] Regression: GPTQ models fail to load on Intel XPU in v0.19.0 (missing XPU branches in gptq.py) | @bryanvine | 2026-04-10 | - |
| [#39408](https://github.com/vllm-project/vllm/issues/39408) | [Usage]: qwen3-asr-1.7b pre-allocated encoder cache size limit | @xi1212 | 2026-04-09 | `usage` |
| [#39407](https://github.com/vllm-project/vllm/issues/39407) | [Bug]: Gemma 4 31B FP8_BLOCK checkpoint produces garbage repetitive output — logit saturation at softcap wall due to absorbed activation scales being double-applied | @cferra | 2026-04-09 | - |
| [#39371](https://github.com/vllm-project/vllm/issues/39371) | DSA module construction corrupts CUDA RNG state (Offset increment outside graph capture) | @simone-chen | 2026-04-09 | - |
| [#39348](https://github.com/vllm-project/vllm/issues/39348) | [Bug]: Qwen3.5-9B-AWQ on ROCm/vLLM 0.19.0 can get stuck generating endless "!" inside JSON schema output | @Saturnix | 2026-04-08 | `bug`, `rocm` |
| [#39319](https://github.com/vllm-project/vllm/issues/39319) | [Bug]: vLLM docker container with Qwen3.5 - Connection error | @MatteoRiva95 | 2026-04-08 | `bug` |
| [#39288](https://github.com/vllm-project/vllm/issues/39288) | [Bug]: FlashInfer CUTLASS MoE backend causes CUDA illegal memory access on H100 during CUDA graph capture (Qwen3-Next-80B BF16) | @janbernloehr | 2026-04-08 | - |
| [#39231](https://github.com/vllm-project/vllm/issues/39231) | [Bug]: Qwen3.5 Text Only Model (Qwen3_5ForCausalLM) | @jefcoder | 2026-04-07 | `bug` |
| [#39210](https://github.com/vllm-project/vllm/issues/39210) | [Bug] Embedding/pooling models crash on B200 (SM 10.0) — encoder attention hardcodes FA2 which lacks SM100 support | @praateekmahajan | 2026-04-07 | `bug` |
| [#39202](https://github.com/vllm-project/vllm/issues/39202) | [Bug]: Crash on Transcription (size for tensor a must match the size of tensor b) with reproduce | @DefinitlyEvil | 2026-04-07 | `bug` |
| [#39158](https://github.com/vllm-project/vllm/issues/39158) | [RFC][Test]: Unified Platform-Aware Test Skip Mechanism | @jikunshang | 2026-04-07 | `RFC` |
| [#39149](https://github.com/vllm-project/vllm/issues/39149) | [Bug]: Segfault in Triton LLVM (MachineCSE / translateLLVMIRToASM) when serving Qwen3.5-4B on RTX 4090 (WSL2) with vLLM 0.19.0 | @1220856302 | 2026-04-07 | `bug` |
| [#39137](https://github.com/vllm-project/vllm/issues/39137) | [Bug]: fp8_e5m2 kv-cache gate in _init_kv_cache_quant fires on any quantized checkpoint, not only fp8 checkpoints | @ormandj | 2026-04-07 | - |
| [#39133](https://github.com/vllm-project/vllm/issues/39133) | [Bug]: Gemma 4 31B INT4 on 2×24GB GPUs (TP=2): GPU KV cache size is 25,200 tokens at max_model_len=131072, gpu_memory_utilization=0.96, BF16 KV | @ormandj | 2026-04-07 | - |
| [#39104](https://github.com/vllm-project/vllm/issues/39104) | [Usage]: The qwen3.5 model generates a random stream of words in thought mode. | @nagashik | 2026-04-06 | `usage` |
| [#39096](https://github.com/vllm-project/vllm/issues/39096) | [Bug]: Batch invariance breaks with torch.compile and/or CUDA graphs on SM<90 | @Monishver11 | 2026-04-06 | `bug` |
| [#39076](https://github.com/vllm-project/vllm/issues/39076) | [RFC]: Entropy-Gated Online KV Block Expiration During Active Decode | @groot-code24 | 2026-04-06 | `RFC` |
| [#39061](https://github.com/vllm-project/vllm/issues/39061) | [Bug]: Gemma4 vision encoder crashes with ValueError: Expected hidden_size to be 5376, but found: 72 | @ohsono | 2026-04-06 | - |
| [#39057](https://github.com/vllm-project/vllm/issues/39057) | [Bug]: Deepseek v3.2 RuntimeError: Worker failed with error "Assertion error" | @jxdn | 2026-04-06 | `bug` |
| [#39049](https://github.com/vllm-project/vllm/issues/39049) | [Bug]: Gemma 4 FP8 dynamic quantization = gibberish output | @frenzybiscuit | 2026-04-05 | `bug` |
| [#39015](https://github.com/vllm-project/vllm/issues/39015) | [Bug]: torch.distributed.DistNetworkError: The server socket has failed to listen on any local network address. port: 29500, useIpv6: false, code: -98, name: EADDRINUSE, message: address already in use | @qy0720 | 2026-04-05 | `bug` |
| [#39010](https://github.com/vllm-project/vllm/issues/39010) | [Bug]: Hang During CUDA Graph Capture on ROCM in 0.19 | @depuhitv | 2026-04-05 | `bug`, `rocm` |
| [#38999](https://github.com/vllm-project/vllm/issues/38999) | [Bug]: Gemma 4 MoE (26B-A4B) crashes with `--data-parallel-size > 1` — AssertionError in cuda_communicator all_gather | @leuski | 2026-04-04 | `bug` |
| [#38994](https://github.com/vllm-project/vllm/issues/38994) | Qwen-3.5 9B often producing repetitive/garbled output with Intel Backend | @AlexanderValentini | 2026-04-04 | `bug` |
| [#38976](https://github.com/vllm-project/vllm/issues/38976) | [Bug]:TimeoutError: RPC call to sample_tokens timed out. when pp is on under xpu env | @zwh20081 | 2026-04-04 | `bug` |
| [#38967](https://github.com/vllm-project/vllm/issues/38967) | [Bug] vLLM >= 0.18.0 NCCL segfault (cuMemCreate) with TP>1 on RTX 4090 (SM 89) | @zhouliang5266 | 2026-04-04 | - |
| [#38925](https://github.com/vllm-project/vllm/issues/38925) | [Feature]: Support lightweight import of vllm protocol types without torch dependency | @hexfusion | 2026-04-03 | - |
| [#38918](https://github.com/vllm-project/vllm/issues/38918) | [Usage]: Gemma4 on Turing GPUs (SM 7.5): all attention backends hit shared memory limits | @lisp19 | 2026-04-03 | `usage` |
| [#38912](https://github.com/vllm-project/vllm/issues/38912) | Gemma 4 MoE NVFP4: expert_params_mapping doesn't handle scale key suffixes | @marioiseli89 | 2026-04-03 | - |
| [#38903](https://github.com/vllm-project/vllm/issues/38903) | [Bug]: Cross-request context contamination with async scheduling + pipeline parallelism on multi-node | @agis09 | 2026-04-03 | `bug` |
| [#38886](https://github.com/vllm-project/vllm/issues/38886) | [Bug]: Gemma 4 E4B weight loading fails `Gemma4ClippableLinear` parameter `input_max` not recognized | @CunXin1 | 2026-04-03 | `bug` |
| [#38884](https://github.com/vllm-project/vllm/issues/38884) | [Bug]: Gemma 4 torch._dynamo.exc.TorchRuntimeError: Dynamo failed to run FX node with fake tensors | @NilsHellwig | 2026-04-03 | `bug` |
| [#38811](https://github.com/vllm-project/vllm/issues/38811) | [Usage]: Qwen3-VL inference on video complains of lack of metadata | @carlos-havier | 2026-04-02 | `usage` |
| [#38809](https://github.com/vllm-project/vllm/issues/38809) | [Feature]: How to disable chat template when using vllm serve | @sleepwalker2017 | 2026-04-02 | `feature request` |
| [#38760](https://github.com/vllm-project/vllm/issues/38760) | [RFC]: Per-iteration forward pass metrics with accurate engine-level timing | @tedzhouhk | 2026-04-01 | `RFC` |
| [#38754](https://github.com/vllm-project/vllm/issues/38754) | [Bug]: GPT OSS Router GEMM Causing NaNs | @benchislett | 2026-04-01 | `bug` |
| [#38740](https://github.com/vllm-project/vllm/issues/38740) | [Transformers v5] NemotronParseForConditionalGeneration | @hmellor | 2026-04-01 | `help wanted`, `good first issue` |
| [#38713](https://github.com/vllm-project/vllm/issues/38713) | [Bug]: Error when trying to serve MiniMax 2.5 on 4 H100 nodes with 4 GPUS | @F-Michelon | 2026-04-01 | `bug` |
| [#38710](https://github.com/vllm-project/vllm/issues/38710) | [Bug]: heterogeneous disaggregated serving XPU (Prefill) + CPU (Decode) accuracy issue | @Spycsh | 2026-04-01 | `bug` |
| [#38660](https://github.com/vllm-project/vllm/issues/38660) | [Bug]: CUDA assert in triton attention for MolmoWeb models (Molmo2 architecture with different max_position_embeddings) | @2imi9 | 2026-03-31 | - |
| [#38658](https://github.com/vllm-project/vllm/issues/38658) | [Bug]: MLA attention casts activations to int32 when using Marlin FP8 on GPUs without native FP8 support (sm < 89) | @marcusm117 | 2026-03-31 | `bug` |
| [#38656](https://github.com/vllm-project/vllm/issues/38656) | [Bug]: qwen 3.5 model launch get stuck for quite a long time | @yanan1116 | 2026-03-31 | `bug` |
| [#38603](https://github.com/vllm-project/vllm/issues/38603) | [Bug]: Streaming last chunk contains non-empty tool_calls with empty fields "type" causing type validation error  | @guohuanliang1 | 2026-03-31 | `bug` |
| [#38591](https://github.com/vllm-project/vllm/issues/38591) | Bug: ValueError: too many values to unpack in dispatch_cpu_unquantized_gemm when loading Qwen3.5-4B | @miguel-flowstate | 2026-03-30 | - |
| [#38586](https://github.com/vllm-project/vllm/issues/38586) | [Bug]: Whisper online benchmark with profiling error: TypeError: multi_modal_content must be a dict containing 'audio' | @AdityaKulshrestha | 2026-03-30 | `bug` |
| [#38560](https://github.com/vllm-project/vllm/issues/38560) | [Bug]: reasoning_effort passed to MistralCommonTokenizer.apply_chat_template breaks Mistral Small 4 chat completions on vLLM 0.18.0 | @BenjaminFuentesEviden | 2026-03-30 | `bug` |
| [#38551](https://github.com/vllm-project/vllm/issues/38551) | [Bug]: AssertionError: Encoder cache miss crashes engine with MTP + multimodal under high concurrency | @kaiktl | 2026-03-30 | - |
| [#38531](https://github.com/vllm-project/vllm/issues/38531) | [Feature]: Support direct binary/multipart file upload for video and image in OpenAI-compatible API | @harshvb20 | 2026-03-30 | `feature request`, `multi-modality` |
| [#38486](https://github.com/vllm-project/vllm/issues/38486) | [Bug]: cuda graph takes too much memory for qwen 3.5 | @ErykCh | 2026-03-29 | `bug` |
| [#38472](https://github.com/vllm-project/vllm/issues/38472) | [Bug]: [xPyD]Potential OOM when using v1 P2pNcclConnector as KV cache transport: KV cache accumulation on decode instance. | @cxywhite | 2026-03-29 | `bug` |
| [#38470](https://github.com/vllm-project/vllm/issues/38470) | [Bug]: When using the Sonnet dataset for benchmark testing, if the input length is too small, the CPU usage becomes abnormally high with no error logs, making it impossible to run the benchmark properly. | @frankie-ys | 2026-03-29 | `bug` |
| [#38459](https://github.com/vllm-project/vllm/issues/38459) | [Bug]: `limit_mm_per_prompt` is ineffective for Qwen3-VL | @Disapole-Xiao | 2026-03-29 | `bug` |
| [#38428](https://github.com/vllm-project/vllm/issues/38428) | [Bug]: V1 Engine: EngineDeadError (AssertionError) on max_model_len overflow during realtime audio streaming | @sh1man | 2026-03-28 | `bug` |
| [#38425](https://github.com/vllm-project/vllm/issues/38425) | [Transformers v5] InternVL2 | @hmellor | 2026-03-28 | `help wanted`, `good first issue` |
| [#38411](https://github.com/vllm-project/vllm/issues/38411) | [Bug]: Vision encoder crashes on SM100 (Jetson Thor) — `_vllm_fa2_C` compiled for SM80-only, no override available for vision encoder | @mrjbj | 2026-03-28 | `bug` |
| [#38389](https://github.com/vllm-project/vllm/issues/38389) | [Transformers v5] IsaacForConditionalGeneration | @hmellor | 2026-03-27 | `help wanted`, `good first issue` |
| [#38379](https://github.com/vllm-project/vllm/issues/38379) | Upgrade to Transformers v5 | @hmellor | 2026-03-27 | `help wanted` |
| [#38351](https://github.com/vllm-project/vllm/issues/38351) | [Bug]: When use_audio_in_video is enabled in qwen3-omni, the output may exhibit issues such as empty or repetitive output. | @wwxs123W | 2026-03-27 | `bug` |
| [#38297](https://github.com/vllm-project/vllm/issues/38297) | [Bug]: Gemma3n concurrent audio requests crash EngineCore — missing dynamic_dims on audio sequence dimension | @RushRed | 2026-03-27 | - |
| [#38256](https://github.com/vllm-project/vllm/issues/38256) | [RFC]: Incremental MoE Expert Offloading — GPU Cache + Async Pipeline | @e1n00r | 2026-03-26 | - |
| [#38235](https://github.com/vllm-project/vllm/issues/38235) | [Feature]: Quantization support (AWQ / GPTQ / FP8) for mistralai/Voxtral-Mini-4B-Realtime-2602 | @sh1man | 2026-03-26 | `feature request` |
| [#38233](https://github.com/vllm-project/vllm/issues/38233) | [Bug]: Voxtral-Mini-4B-Realtime hangs/crashes on multiple sessions due to encoder_cache_usage saturation on 16GB GPU | @sh1man | 2026-03-26 | `bug` |
| [#38208](https://github.com/vllm-project/vllm/issues/38208) | [Bug]: CUDA Illegal Instruction during CUDA Graph capture with Nemotron-3-Nano NVFP4 on sm_121 | @dennis-lynch | 2026-03-26 | `bug` |
| [#38203](https://github.com/vllm-project/vllm/issues/38203) | [Bug]: M2.5 tool call result is badcase, deploy 1p1d with nixl connector, P and D use DP8-EP-TP1 | @ssbandjl | 2026-03-26 | `bug` |
| [#38196](https://github.com/vllm-project/vllm/issues/38196) | GDN attention backend crashes with ngram speculative decoding on mixed decode batches | @bhaktatejas922 | 2026-03-26 | - |
| [#38176](https://github.com/vllm-project/vllm/issues/38176) | [Bug]: qwen3 235B model with latest vllm is going to generate only 1 token. | @jiangtaozh | 2026-03-26 | `bug` |
| [#38175](https://github.com/vllm-project/vllm/issues/38175) | [RFC]: Support ViT Full CUDA Graph (Tracker) | @shen-shanshan | 2026-03-26 | `help wanted`, `RFC`, `multi-modality` |
| [#38106](https://github.com/vllm-project/vllm/issues/38106) | [Bug]: tool_choice="required" + speculative decoding with lukealonso/Qwen3.5-397B-A17B-NVFP4 leads to failed tool calls. | @SvenLorenz | 2026-03-25 | `bug` |
| [#38079](https://github.com/vllm-project/vllm/issues/38079) | [RFC] Redesign enable_return_routed_experts to avoid blocking EngineCore event loop | @junjzhang | 2026-03-25 | - |
| [#38077](https://github.com/vllm-project/vllm/issues/38077) | [Bug]: Qwen3.5-9B answer !!!!!!!!! | @hoshinoyouyou | 2026-03-25 | `bug` |
| [#38056](https://github.com/vllm-project/vllm/issues/38056) | [Bug]: ImportError: flash_attn.ops.triton.rotary not found on older versions (< v2.1.2) | @xiaoajie738 | 2026-03-25 | `bug` |
| [#38022](https://github.com/vllm-project/vllm/issues/38022) | [Bug]: Marlin MoE kernel fails with MXFP4-quantized GPT-OSS 20B - Invalid thread config for non-aligned dimensions (K=2880, N=2880) | @zweack | 2026-03-24 | `bug` |
| [#38004](https://github.com/vllm-project/vllm/issues/38004) | [Bug]: Speech-to-Text endpoint may return 501 but not documented in OpenAPI | @Sebastian-dong | 2026-03-24 | `bug` |
| [#37996](https://github.com/vllm-project/vllm/issues/37996) | [Bug]: Qwen3.5 397B GPTQ model outputs all exclamation points on ROCM | @hnhyzz | 2026-03-24 | `bug`, `rocm`, `quantization` |
| [#37992](https://github.com/vllm-project/vllm/issues/37992) | [Bug]: RuntimeError triton error during profile_run with Qwen3.5-MoE vision encoder on ROCm | @xuebwang-amd | 2026-03-24 | `bug`, `rocm` |
| [#37981](https://github.com/vllm-project/vllm/issues/37981) | [Bug]: v0.18.0 fails to run MiniCPM-o-4.5 | @devalun | 2026-03-24 | `bug` |
| [#37977](https://github.com/vllm-project/vllm/issues/37977) | [Bug][Model] Eagle2.5-VL applies ImageNet normalization instead of SigLIP2 | @edwingao28 | 2026-03-24 | - |
| [#37928](https://github.com/vllm-project/vllm/issues/37928) | [Usage]: v0.18.0 nvidia/nemotron-colembed-vl-4b-v2 /embeddings 404 | @brandonbiggs | 2026-03-23 | `usage` |
| [#37890](https://github.com/vllm-project/vllm/issues/37890) | [Bug]: NaNs in vLLM using DeepSeek-R1-0528-NVFP4-v2 | @tlrmchlsmth | 2026-03-23 | `bug` |
| [#37858](https://github.com/vllm-project/vllm/issues/37858) | [Bug]: does not have the attribute 'FakeTensorMode' | @ACCEKLL | 2026-03-23 | `bug` |
| [#37828](https://github.com/vllm-project/vllm/issues/37828) | [Bug]: Intel ARC 140v not supported as XE2 cutlass kernel | @PterosDiacos | 2026-03-22 | `bug`, `intel-gpu` |
| [#37746](https://github.com/vllm-project/vllm/issues/37746) | [Bug] prompt_logprobs causes livelock with IsHybrid models (Qwen3.5) in DP mode | @dengoswei | 2026-03-21 | - |
| [#37737](https://github.com/vllm-project/vllm/issues/37737) | [Bug]: Missing logprobs for `<tool_call>` in streaming chat completions | @sdpkjc | 2026-03-21 | `bug` |
| [#37736](https://github.com/vllm-project/vllm/issues/37736) | [CI Failure]:  Gemma3 OOMs with transformers backend | @AndreasKaratzas | 2026-03-21 | `rocm`, `ci-failure` |
| [#37729](https://github.com/vllm-project/vllm/issues/37729) | [Bug]: V1 engine core deadlocks under concurrent load (fp8 + prefix caching + Qwen3.5) | @rahul003 | 2026-03-21 | `bug` |
| [#37675](https://github.com/vllm-project/vllm/issues/37675) | [Bug]: deepgemm compile error | @Speclkle | 2026-03-20 | `bug` |
| [#37674](https://github.com/vllm-project/vllm/issues/37674) | [Usage]: serve部署模型后，调用chat.completions输入给模型的text中image_pad token被提到了prompt开头 | @NormanWhc | 2026-03-20 | `usage` |
| [#37659](https://github.com/vllm-project/vllm/issues/37659) | [Bug]: RuntimeError: 基于megatron grpo Qwen3-Omni模型时，出现RuntimeError: Event device index  does not match recording stream's device index | @dlnn123 | 2026-03-20 | `bug` |
| [#37602](https://github.com/vllm-project/vllm/issues/37602) | [Bug]: Qwen3.5-122B-A10B-FP8 EngineCore crash on concurrent image requests | @serdarildercaglar | 2026-03-19 | - |
| [#37581](https://github.com/vllm-project/vllm/issues/37581) | [Bug]: /v1/chat/completions/render` crashes for Qwen/Qwen3-ASR-0.6B multimodal audio, and chat audio returns empty/junk | @peregilk | 2026-03-19 | `bug` |
| [#37563](https://github.com/vllm-project/vllm/issues/37563) | mm_fp4 trtllm backend leaks padding scales into real rows (use_8x4_sf_layout=True) | @elvircrn | 2026-03-19 | - |
| [#37553](https://github.com/vllm-project/vllm/issues/37553) | [Bug]: Mistral-Small-4-119B-2603 fails on 8x RTX 3090 (SM 8.6) with vLLM v0.17.1: no valid MLA attention backend | @Martossien | 2026-03-19 | `bug` |
| [#37551](https://github.com/vllm-project/vllm/issues/37551) | [Bug] vLLM 0.17.1: `zai-org/GLM-OCR` has `mtp_graph < no_mtp_graph` despite high acceptance | @AlejandroBaron | 2026-03-19 | `bug` |
| [#37472](https://github.com/vllm-project/vllm/issues/37472) | [Bug] V1 engine hangs on encoder cache profiling on AMD gfx1151 (MIOpen missing solver DB) | @3spky5u-oss | 2026-03-18 | `rocm` |
| [#37459](https://github.com/vllm-project/vllm/issues/37459) | [CI Failure]: MultiModal Models Extended 2 - isaac test case OOMs | @varun-sundar-rabindranath | 2026-03-18 | `ci-failure` |
| [#37451](https://github.com/vllm-project/vllm/issues/37451) | [Bug]: 0.17.1 - vllm serve deepseek-ai/DeepSeek-OCR-2 on H100 crashes during Capturing CUDA graphs (decode, FULL) | @jraby | 2026-03-18 | `bug` |
| [#37431](https://github.com/vllm-project/vllm/issues/37431) | Mamba-2 Triton kernels crash with illegal instruction on SM121 (DGX Spark) without CUDA_LAUNCH_BLOCKING=1 | @ced509msn | 2026-03-18 | - |
| [#37423](https://github.com/vllm-project/vllm/issues/37423) | [Feature]: Allow passing `images` to CompletionRequest | @patrickvonplaten | 2026-03-18 | `feature request` |
| [#37392](https://github.com/vllm-project/vllm/issues/37392) | [Bug]:推理时报错，模型关闭了。部署的Qwen3.5-122B-A10B-FP8模型 | @watch-Ultra | 2026-03-18 | `bug` |
| [#37367](https://github.com/vllm-project/vllm/issues/37367) | [Bug]: gcc: internal compiler error: Segmentation fault signal terminated program cc1 | @154461013 | 2026-03-18 | `bug` |
| [#37363](https://github.com/vllm-project/vllm/issues/37363) | fix(compilation): fix piecewise CUDA graph bugs with splitting_ops | @Complexity-ML | 2026-03-18 | - |
| [#37325](https://github.com/vllm-project/vllm/issues/37325) | [Bug][ARM CPU] Build/Runtime error: no matching function for call to ‘at::vec::CPU_CAPABILITY::VecMask<long int, 4>::VecMask(int&)’ when serving Qwen3-VL-8B-Instruct | @micyan01 | 2026-03-17 | `bug`, `cpu` |
| [#37255](https://github.com/vllm-project/vllm/issues/37255) | [Bug]: Failed to run distributed inference due to error list index out of range in omp_cpuids_list | @wenlujon | 2026-03-17 | `bug` |
| [#37242](https://github.com/vllm-project/vllm/issues/37242) | [Community] RTX 5090 (Blackwell sm_120) + WSL2 2.7.0: CUDA graphs work — benchmarks + full config | @Kyzcreig | 2026-03-17 | - |
| [#37185](https://github.com/vllm-project/vllm/issues/37185) | [Bug]: The 2-bit model repeatedly outputs !!!!, while Transformers works correctly. | @wenhuach21 | 2026-03-16 | `bug` |
| [#37175](https://github.com/vllm-project/vllm/issues/37175) | [Usage]:wen3.5-35B-A3B (FP8) with vLLM 0.17.1 , the first request takes significantly longer than subsequent requests | @robinyqd | 2026-03-16 | `usage` |
| [#37168](https://github.com/vllm-project/vllm/issues/37168) | [RFC]:  Active Coordination and Two-Zone Scheduling Mechanism for KV Cache in Long-Running Agents | @xinrunxue | 2026-03-16 | `RFC` |
| [#37159](https://github.com/vllm-project/vllm/issues/37159) | [Bug]: vLLM crashed with V100 by running zai-org/GLM-OCR | @sweihub | 2026-03-16 | `bug` |
| [#37140](https://github.com/vllm-project/vllm/issues/37140) | [Usage]: CUDA error: the provided PTX was compiled with an unsupported toolchain. | @870223666 | 2026-03-16 | `usage` |
| [#37113](https://github.com/vllm-project/vllm/issues/37113) | [SM120][GLM-5.1] NVFP4 DCP/MTP stack tracker | @voipmonitor | 2026-03-15 | - |
| [#37096](https://github.com/vllm-project/vllm/issues/37096) | [Bug]: v0.17.0-aarch64 onwards will run out of CUDA memory for gpt-oss-120b on GH200 144GB | @xuancong84 | 2026-03-15 | `bug` |
| [#37075](https://github.com/vllm-project/vllm/issues/37075) | [RFC]: Opt-in Media URL Cache for `MediaConnector` | @AndreasKaratzas | 2026-03-14 | `RFC` |
| [#37060](https://github.com/vllm-project/vllm/issues/37060) | [Bug]: sm110: torch.AcceleratorError: CUDA error: an illegal instruction was encountered | @shahizat | 2026-03-14 | `bug` |
| [#37035](https://github.com/vllm-project/vllm/issues/37035) | [Bug]: cudaErrorIllegalAddress in gdn_attn.py:237 when using qwen3_next_mtp with num_speculative_tokens=5 under load | @Quentin-M | 2026-03-14 | `bug` |
| [#36998](https://github.com/vllm-project/vllm/issues/36998) | [RFC]: Observation Plugin for Intercepting & Routing on Activations | @DDDDarrenWB | 2026-03-13 | `RFC` |
| [#36906](https://github.com/vllm-project/vllm/issues/36906) | [Bug]: EAGLE3 speculative decoding + multimodal crash under high concurrency | @staghado | 2026-03-12 | `bug` |
| [#36872](https://github.com/vllm-project/vllm/issues/36872) | [Bug]: Gibberish output and collapsing generation throughput with Qwen3.5-35B-A3B-FP8 and speculative decoding enabled | @marionchadal | 2026-03-12 | `bug` |
| [#36861](https://github.com/vllm-project/vllm/issues/36861) | [Bug]: Why does setting `--pipeline-parallel-size > 1` result in an OOM error, but `--tensor-parallel-size> 1` does not? | @VirgilG72 | 2026-03-12 | `bug` |
| [#36852](https://github.com/vllm-project/vllm/issues/36852) | [Bug]: GPU failure during repeated model loading when using --enable-prefix-caching with KV transfer (LMCacheConnectorV1) | @lavanyabollepalli | 2026-03-12 | `bug` |
| [#36843](https://github.com/vllm-project/vllm/issues/36843) | [Bug]: VLLM 0.17.1 initial mtp with FLASH_ATTN randomly crash | @flutist | 2026-03-12 | `bug` |
| [#36811](https://github.com/vllm-project/vllm/issues/36811) | [Bug]: CUDA illegal memory access on GPTQ Marlin | @tonibofarull | 2026-03-11 | `bug` |
| [#36796](https://github.com/vllm-project/vllm/issues/36796) | [Bug]: CPU offload errors on nightly with NVIDIA GH200 Unified Memory (UMA) | @ehfd | 2026-03-11 | `bug`, `torch.compile` |
| [#36793](https://github.com/vllm-project/vllm/issues/36793) | [Bug]: TP=2 DP=2 Broken for Qwen3-Next W4A16 | @Chibukach | 2026-03-11 | `bug` |
| [#36781](https://github.com/vllm-project/vllm/issues/36781) | [Bug]: vLLM 0.17.0 failed to serve Qwen3-30B-A3B-Instruct-2507 after adding `--enable_lora` | @Junxiao-Zhao | 2026-03-11 | `bug` |
| [#36780](https://github.com/vllm-project/vllm/issues/36780) | [RFC][NixlConnector]: Add support for hybrid SSM-FA models | @NickLucche | 2026-03-11 | `RFC` |
| [#36748](https://github.com/vllm-project/vllm/issues/36748) | [Bug]: In DP mode, waiting request stack in a few DP ranks. | @xiaoxiaosuaxuan | 2026-03-11 | `bug` |
| [#36627](https://github.com/vllm-project/vllm/issues/36627) | [Performance]: qwen3.5 vs qwen3 | @fangbaolei | 2026-03-10 | `performance` |
| [#36589](https://github.com/vllm-project/vllm/issues/36589) | [Bug]: SM 7.5 extreme slowness hangs indefinitely on T4 (vllm 0.17.0 with Qwen3.5-27B) | @billysyt | 2026-03-10 | `bug` |
| [#36585](https://github.com/vllm-project/vllm/issues/36585) | [Bug]: qwen3.5-27b-gptq deploy fail | @xiaotianns | 2026-03-10 | `bug` |
| [#36566](https://github.com/vllm-project/vllm/issues/36566) | [Bug]:Qwen3.5-35B-A3B vllm v0.17.0 ERROR 03-10 00:52:24 [multiproc_executor.py:261] Worker proc VllmWorker-0 died unexpectedly, shutting down executor. | @jieguolove | 2026-03-10 | `bug` |
| [#36476](https://github.com/vllm-project/vllm/issues/36476) | [Bug]: vllm 0.17.0 启动Qwen3.5-122B-A10B失败 | @Lee-xeo | 2026-03-09 | `bug` |
| [#36450](https://github.com/vllm-project/vllm/issues/36450) | [Bug]: Qwen3.5 AWQ models crash during inference on RTX 5090 (Blackwell) with Triton OOM in solve_tril despite successful model load | @NEWbie0709 | 2026-03-09 | `bug` |
| [#36407](https://github.com/vllm-project/vllm/issues/36407) | [Bug]: KeyError in get_layers_from_vllm_config with pipeline parallelism (vLLM 0.16.0) | @tusharshetty61 | 2026-03-08 | `bug` |
| [#36350](https://github.com/vllm-project/vllm/issues/36350) | [Bug]: Qwen 3.5 4B fail on first request on Intel XPU (Arc Graphics B580) | @Nero10578 | 2026-03-07 | `bug` |
| [#36328](https://github.com/vllm-project/vllm/issues/36328) | [Feature]: Include mm_hash and mm_transfer_params in Disaggregated Encoder response to prevent redundant data fetching | @roytman | 2026-03-07 | `feature request` |
| [#36275](https://github.com/vllm-project/vllm/issues/36275) | [Bug]: Qwen3.5 4b incompatibility | @aminsamir45 | 2026-03-06 | `bug` |
| [#36245](https://github.com/vllm-project/vllm/issues/36245) | [Bug]: Gemma3 mmproj-*.gguf is not downloaded in 'download_gguf' | @lmjantsch | 2026-03-06 | `bug` |
| [#36193](https://github.com/vllm-project/vllm/issues/36193) | [Bug]: Unsupported Activation Function for Step-3.5-Flash | @ColinZ22 | 2026-03-06 | `bug`, `rocm` |
| [#36180](https://github.com/vllm-project/vllm/issues/36180) | [Bug]: meta-llama/Llama-3.2-1B-Instruct Fails With ROCM_ATTN Due To Seg Fault | @micah-wil | 2026-03-05 | `bug`, `rocm` |
| [#36148](https://github.com/vllm-project/vllm/issues/36148) | [Bug]: Qwen3-VL-Reranker-8B fails with shape mismatch error when loading with --quantization bitsandbytes | @xl2014 | 2026-03-05 | `bug` |
| [#36116](https://github.com/vllm-project/vllm/issues/36116) | [Bug]: Pseudo-Streaming Issue When Using Tool Parser with Qwen3-Coder and MiniMax-M2 | @delwen123 | 2026-03-05 | `bug` |
| [#36103](https://github.com/vllm-project/vllm/issues/36103) | [Bug]: torch._inductor.exc.InductorError: | @crimoc-lgtm | 2026-03-05 | `bug` |
| [#36073](https://github.com/vllm-project/vllm/issues/36073) | [Bug]: Prolonged Latency in Some Streaming Responses in Function Call Mode（MiniMax model） | @delwen123 | 2026-03-05 | `bug` |
| [#36015](https://github.com/vllm-project/vllm/issues/36015) | [Bug]: Realtime audio transcription (Voxtral) silently hangs after ~10 minutes due to unhandled TimeoutError in background task | @sh1man | 2026-03-04 | `bug` |
| [#35980](https://github.com/vllm-project/vllm/issues/35980) | [Bug]: Why does deploying Qwen3-32B-AWQ via vllm:v0.10.1.1 result in different outputs for the same input? | @zhushuo5 | 2026-03-04 | `bug` |
| [#35950](https://github.com/vllm-project/vllm/issues/35950) | [Bug]: ValueError: too many values to unpack (expected 2) | @LuWei6896 | 2026-03-04 | `bug` |
| [#35944](https://github.com/vllm-project/vllm/issues/35944) | [Bug]: jetson agx thor报错 | @jouewer | 2026-03-04 | `bug` |
| [#35908](https://github.com/vllm-project/vllm/issues/35908) | [RFC]: Model-specific realtime streaming abstraction | @TheCodeWrangler | 2026-03-03 | - |
| [#35863](https://github.com/vllm-project/vllm/issues/35863) | [Bug]: Voxtral-Realtime stops returning transcribed text starting from the 3rd concurrent session | @sh1man999 | 2026-03-03 | `bug` |
| [#35848](https://github.com/vllm-project/vllm/issues/35848) | [RFC]: Revamp Ray Distributed Executor Backend (from Ray team) | @jeffreywang-anyscale | 2026-03-03 | `RFC` |
| [#35820](https://github.com/vllm-project/vllm/issues/35820) | [Bug]: deploy Qwen3.5-27B error | @ZTurboX | 2026-03-03 | `bug` |
| [#35778](https://github.com/vllm-project/vllm/issues/35778) | [Bug]: Regression: terrible mixed prefill-decode performance with CUDA graphs enabled. | @Zidrewndacht | 2026-03-02 | `bug` |
| [#35771](https://github.com/vllm-project/vllm/issues/35771) | [RFC][torch.compile]: Disable Sequence Parallelism (SP) for piecewise compilation | @ProExpertProg | 2026-03-02 | `RFC`, `torch.compile` |
| [#35767](https://github.com/vllm-project/vllm/issues/35767) | [Enhancement]: Qwen3-ASR realtime endpoint produces degraded output — stateless segments, no cross-segment context, raw format leaks | @TheCodeWrangler | 2026-03-02 | - |
| [#35755](https://github.com/vllm-project/vllm/issues/35755) | [Bug]: AsyncScheduler crashes with AssertionError during Realtime ASR streaming (num_output_placeholders underflow) | @TheCodeWrangler | 2026-03-02 | - |
| [#35743](https://github.com/vllm-project/vllm/issues/35743) | [Bug]: Qwen 3.5 27B AWQ 4bit capturing CUDA graph fails | @dule1322 | 2026-03-02 | `bug` |
| [#35659](https://github.com/vllm-project/vllm/issues/35659) | [Bug]: cudaErrorIllegalAddress under sustained parallel load with CUDA Graphs on Blackwell SM120 (NVFP4 MoE) | @munakaya | 2026-03-01 | - |
| [#35642](https://github.com/vllm-project/vllm/issues/35642) | [Bug]: HIP build in Docker: offload-arch stderr contaminates compiler flags via cmake/utils.cmake and CMAKE_HIP_FLAGS | @npathak13 | 2026-03-01 | `bug`, `rocm` |
| [#35624](https://github.com/vllm-project/vllm/issues/35624) | [Bug]: Qwen3-Omni Model Fails when try to l | @Blaze-DSP | 2026-02-28 | `bug` |
| [#35608](https://github.com/vllm-project/vllm/issues/35608) | [Bug]: vllm 0.16.0+image encountered CUDA error: CUBLAS_STATUS_INVALID_VALUE when calling `cublasGemmEx | @adenzhou1350 | 2026-02-28 | `bug` |
| [#35569](https://github.com/vllm-project/vllm/issues/35569) | [Bug]: [ROCm] ROCM_ATTN backend shows ~8.5% systematic score deviation on Qwen3-VL-Reranker pooling | @AndreasKaratzas | 2026-02-28 | `bug`, `rocm` |
| [#35566](https://github.com/vllm-project/vllm/issues/35566) | CUDA illegal memory access in MoE layer with MiniMax-M2.5 NVFP4 on Blackwell (SM120) | @jhsmith409 | 2026-02-28 | - |
| [#35550](https://github.com/vllm-project/vllm/issues/35550) | [RFC]: Batch-Aware Expert Pruning for MoE Decode (XShare) | @hai-meh-cs | 2026-02-27 | - |
| [#35521](https://github.com/vllm-project/vllm/issues/35521) | [Bug]: Suffix decoding crashes with assert total_num_scheduled_tokens > 0 | @rosstang | 2026-02-27 | `bug` |
| [#35519](https://github.com/vllm-project/vllm/issues/35519) | [Bug]: Qwen3.5 NVFP4 models crash on ARM64 GB10 DGX Spark (CUDA illegal instruction during generation) | @EmilHaase | 2026-02-27 | `bug` |
| [#35467](https://github.com/vllm-project/vllm/issues/35467) | [Performance]: non-optimal performance of `linear` for medium batches | @vadiklyutiy | 2026-02-27 | `performance` |
| [#35407](https://github.com/vllm-project/vllm/issues/35407) | [CI] DBO with DP+EP accuracy regression on GSM8K evaluation | @LucasWilkinson | 2026-02-26 | `ci-failure` |
| [#35303](https://github.com/vllm-project/vllm/issues/35303) | [Bug] CompressedTensorsWNA16MarlinMoEMethod registers g_idx params unconditionally, crashes with actorder=null AWQ MoE models | @jhsmith409 | 2026-02-25 | - |
| [#35285](https://github.com/vllm-project/vllm/issues/35285) | [Bug]: Large Video Request cause vLLM Progress Core Dump | @rhluo | 2026-02-25 | `bug` |
| [#35276](https://github.com/vllm-project/vllm/issues/35276) | [Bug]: OpenAI transcribe prompt parameter with whisper return hallucinated transcription | @Matht-j | 2026-02-25 | `bug` |
| [#35272](https://github.com/vllm-project/vllm/issues/35272) | [Feature]: Make the Qwen3-ASR use `request_prompt` from user input | @minh-nguyenhoang | 2026-02-25 | `feature request` |
| [#35254](https://github.com/vllm-project/vllm/issues/35254) | [Performance]: curious new kernels from vllm 0.11.1 | @fataswellassad | 2026-02-25 | `performance` |
| [#35169](https://github.com/vllm-project/vllm/issues/35169) | [Bug]: Memory Access Fault during Step-3.5-Flash inference (ROCM) | @ColinZ22 | 2026-02-24 | `bug`, `rocm` |
| [#35141](https://github.com/vllm-project/vllm/issues/35141) | [Feature]: Sequence Parallel Support for Model Runner V2 | @yewentao256 | 2026-02-23 | `feature request` |
| [#35091](https://github.com/vllm-project/vllm/issues/35091) | [Bug]: Triton CompilationError in speculative decoding (draft_model) | @rse173 | 2026-02-23 | `bug` |
| [#35089](https://github.com/vllm-project/vllm/issues/35089) | [RFC]: In-Tree AMD Zen CPU Backend via zentorch | @amd-lalithnc | 2026-02-23 | `rocm`, `RFC`, `cpu` |
| [#35087](https://github.com/vllm-project/vllm/issues/35087) | [Bug]: DeepSeek 3.2 P/D Disaggregation Support | @yanminjia | 2026-02-23 | `bug` |
| [#35028](https://github.com/vllm-project/vllm/issues/35028) | [Bug]: RuntimeError: CUDA error: CUBLAS_STATUS_INVALID_VALUE when calling `cublasGemmEx | @shahizat | 2026-02-21 | `bug` |
| [#34994](https://github.com/vllm-project/vllm/issues/34994) | [Feature]: Infrastructure Improvements for ROCm CI | @AndreasKaratzas | 2026-02-20 | `feature request`, `rocm` |
| [#34759](https://github.com/vllm-project/vllm/issues/34759) | [Bug]: nvidia/Llama-3.3-70B-Instruct-NVFP4 Degraded / Gibberish Output with TRITON_ATTN | @frankwang28 | 2026-02-17 | `bug` |
| [#34545](https://github.com/vllm-project/vllm/issues/34545) | [Installation]: unrecognized arguments: --omni | @HenryBao91 | 2026-02-14 | `installation` |
| [#34536](https://github.com/vllm-project/vllm/issues/34536) | [Feature]: Reasoning output for offline inference | @BartekKruczek | 2026-02-13 | `feature request` |
| [#34518](https://github.com/vllm-project/vllm/issues/34518) | [Feature]: [Whisper] Support for decoder prefix and custom task tokens in transcription API | @LouisChirol | 2026-02-13 | `feature request` |
| [#34505](https://github.com/vllm-project/vllm/issues/34505) | [Bug]: Qwen 2.5 Omni cuda graph has error | @tzhouam | 2026-02-13 | `bug` |
| [#34406](https://github.com/vllm-project/vllm/issues/34406) | [Bug]: Instruction following capability is deteriorating：Output introduces parameter  defined in functioncall incorrectly | @Patrickpan9 | 2026-02-12 | `bug` |
| [#34323](https://github.com/vllm-project/vllm/issues/34323) | [CI Failure]: Spawned tests can fail silently | @almayne | 2026-02-11 | `ci-failure` |
| [#34295](https://github.com/vllm-project/vllm/issues/34295) | [Bug]: | @dpankros | 2026-02-11 | `bug` |
| [#34250](https://github.com/vllm-project/vllm/issues/34250) | [Bug]: using vllm on Qwen3-Omni-30B-A3B-Instruct: Failed to apply prompt replacement for mm_items['audio'][0]. | @katie312 | 2026-02-10 | `bug` |
| [#34212](https://github.com/vllm-project/vllm/issues/34212) | [Performance]: W4Afp8 is slower than FP8-W8A8 | @liudl85 | 2026-02-10 | `performance` |
| [#34210](https://github.com/vllm-project/vllm/issues/34210) | [Bug]: Enable DBO on Qwen3-VL-235B-A22B raise TypeError: 'NoneType' object is not subscriptable | @Ethan-yt | 2026-02-10 | `bug` |
| [#34209](https://github.com/vllm-project/vllm/issues/34209) | [Usage]: How to decrease Encoder cache budget from 16384 tokens to something lower? | @DmitryFX | 2026-02-10 | `usage` |
| [#34201](https://github.com/vllm-project/vllm/issues/34201) | [Bug]: AttributeError: 'Parameter' object has no attribute 'weight_loader' | @bingoct | 2026-02-10 | `bug` |
| [#34154](https://github.com/vllm-project/vllm/issues/34154) | [Bug]: Failing to run DeepSeek-OCR on Radeon GPUs (memory fault) | @umarinkovic | 2026-02-09 | `bug`, `rocm` |
| [#33986](https://github.com/vllm-project/vllm/issues/33986) | [Tracking issue]: Issue Tracker for Qwen/Qwen3-VL-Embedding & Qwen/Qwen3-VL-Reranker | @noooop | 2026-02-06 | `usage` |
| [#33980](https://github.com/vllm-project/vllm/issues/33980) | [RFC]: Sparse attention KV cache offloading to support longer sequence length | @zhangsicheng5 | 2026-02-06 | `RFC` |
| [#33920](https://github.com/vllm-project/vllm/issues/33920) | [Bug]: GLM-4.7-Flash OOM during sampler warmup with tensor parallelism on RTX 4090 | @VecherVhatuX | 2026-02-05 | `bug` |
| [#33916](https://github.com/vllm-project/vllm/issues/33916) | [Bug] IndexError: list index out of range in chat_completion_stream_generator with --tool-call-parser=mistral during streaming tool calls | @dfischer-mw | 2026-02-05 | `bug` |
| [#33913](https://github.com/vllm-project/vllm/issues/33913) | [Usage]: Compile with `TORCH_USE_CUDA_DSA` to enable device-side assertions. | @870223666 | 2026-02-05 | `usage` |
| [#33871](https://github.com/vllm-project/vllm/issues/33871) | [Bug]: The local deployment achieves about 30% higher accuracy compared to the server deployment. | @charliess123 | 2026-02-05 | `bug` |
| [#33865](https://github.com/vllm-project/vllm/issues/33865) | [Bug]: OpenAI-compatible Embeddings API intermittently crashes with multimodal cache assertion (`Expected a cached item for mm_hash`) on Qwen3-VL-Embedding-8B | @ojipadeson | 2026-02-05 | `bug` |
| [#33828](https://github.com/vllm-project/vllm/issues/33828) | [Bug]: mistral3 offline multimodal inference example failing with prompt placeholder error | @skavulya | 2026-02-04 | `bug` |
| [#33672](https://github.com/vllm-project/vllm/issues/33672) | [Bug]: HTTP API multimodal embedding causes image_pad token duplication, producing incorrect results | @ojipadeson | 2026-02-03 | `bug` |
| [#33654](https://github.com/vllm-project/vllm/issues/33654) | [Bug]: The content of response from Kimi-K2.5 is empty. | @WangTuoxytt | 2026-02-03 | `bug` |
| [#33628](https://github.com/vllm-project/vllm/issues/33628) | [Bug]: Failed to run distributed inference with Gloo backend on aarch64 | @wenlujon | 2026-02-03 | `bug`, `cpu` |
| [#33534](https://github.com/vllm-project/vllm/issues/33534) | [Installation]: vllm with ray in AWS cluster | @haozhuang0000 | 2026-02-02 | `installation` |
| [#33458](https://github.com/vllm-project/vllm/issues/33458) | [Feature][Speculative Decoding]: Multi Modal Draft Model Support | @benchislett | 2026-01-31 | `feature request` |
| [#33424](https://github.com/vllm-project/vllm/issues/33424) | [Bug]: LoRA MoE Not Matching HF Output | @Jonahcb | 2026-01-30 | `bug` |
| [#33398](https://github.com/vllm-project/vllm/issues/33398) | [RFC]: Layerwise KV cache offloading to support longer sequence length | @zhangsicheng5 | 2026-01-30 | `RFC` |
| [#33338](https://github.com/vllm-project/vllm/issues/33338) | [Bug]: Crash when using presence_penalty with Qwen3-VL in v0.11.0 | @Lrcx | 2026-01-29 | `bug` |
| [#33336](https://github.com/vllm-project/vllm/issues/33336) | [Bug]: AttributeError: 'Step3VLProcessor' object has no attribute '_get_num_multimodal_tokens' | @Dineshkumar-Anandan-ZS0367 | 2026-01-29 | `bug` |
| [#33311](https://github.com/vllm-project/vllm/issues/33311) | [Feature]: support pixel_values_videos input for VLM | @wuxibin89 | 2026-01-29 | `feature request` |
| [#33307](https://github.com/vllm-project/vllm/issues/33307) | [Bug]: Latency spikes at input_len=1024 with batch_size=16 (TP1 & TP2) | @mingi001025 | 2026-01-29 | `bug` |
| [#33301](https://github.com/vllm-project/vllm/issues/33301) | [RFC]: Support FP8 LoRA inference | @jeejeelee | 2026-01-29 | `RFC` |
| [#33204](https://github.com/vllm-project/vllm/issues/33204) | [Bug]: Qwen3-VL-Embedding model produces different embeddings than official qwen_vl_utils implementation | @namburiamit | 2026-01-27 | `bug` |
| [#33107](https://github.com/vllm-project/vllm/issues/33107) | [Bug]: Whisper large-v3 accuracy degradation in vLLM 0.14.1 (134.56% WER) on L40S - works fine in 0.12.0 | @sayalibhavsar | 2026-01-26 | `bug` |
| [#32925](https://github.com/vllm-project/vllm/issues/32925) | [Bug]: tensorize_vllm_model double gpu | @lyc1995452 | 2026-01-23 | `bug`, `stale` |
| [#32921](https://github.com/vllm-project/vllm/issues/32921) | [Bug]: gpt-oss-20b streaming last reasoning content part into content | @DeoLeung | 2026-01-23 | `bug` |
| [#32897](https://github.com/vllm-project/vllm/issues/32897) | [Bug]: PaddleOCR-VL crash | @NiuBlibing | 2026-01-23 | `bug` |
| [#32864](https://github.com/vllm-project/vllm/issues/32864) | [Bug] [ROCm]: Loading Qwen3-MoE-MXFP4 Weights in v0.14. | @tjtanaa | 2026-01-22 | `bug`, `rocm` |
| [#32858](https://github.com/vllm-project/vllm/issues/32858) | [Bug]: Use vllm to obtain Qwen3VL last token hidden status | @JasonLeeUT | 2026-01-22 | `bug` |
| [#32826](https://github.com/vllm-project/vllm/issues/32826) | [Bug]: MiniMax-M2.1 NVFP4 fails on RTX PRO 6000 Blackwell (SM120) with expert parallel | @gittb | 2026-01-22 | `bug` |
| [#32733](https://github.com/vllm-project/vllm/issues/32733) | [RFC]: [P/D] Prefill compute optimizations with bi-directional KV cache transfers between P and D nodes | @snadampal | 2026-01-20 | `RFC` |
| [#32732](https://github.com/vllm-project/vllm/issues/32732) | [Bug]: Regression in v0.14.0: "No valid attention backend found" for nvidia/DeepSeek-R1-0528-NVFP4 on RTX Pro 6000 (Blackwell) | @heiderich | 2026-01-20 | `bug` |
| [#32685](https://github.com/vllm-project/vllm/issues/32685) | [Feature]: Support multi-modal inputs for OpenAI Response API | @FlynnOwen | 2026-01-20 | `feature request` |
| [#32659](https://github.com/vllm-project/vllm/issues/32659) | [RFC]: Tracking follow-up progress on Encode–Prefill–Decode Disaggregation | @fake0fan | 2026-01-20 | `RFC`, `stale` |
| [#32636](https://github.com/vllm-project/vllm/issues/32636) | [Bug]: Invalid base64-encoded string for audio input | @IamMegatron2025 | 2026-01-20 | `bug` |
| [#32604](https://github.com/vllm-project/vllm/issues/32604) | [Bug]: LMCache CPU kv offload cause decode speed degrade | @aabbccddwasd | 2026-01-19 | `bug`, `stale` |
| [#32600](https://github.com/vllm-project/vllm/issues/32600) | [Bug]: | @Scaramir | 2026-01-19 | `bug` |
| [#32588](https://github.com/vllm-project/vllm/issues/32588) | [Bug]: Wrong timestamps if audio > 30s | @dr75 | 2026-01-19 | `bug`, `help wanted`, `good first issue` |
| [#32541](https://github.com/vllm-project/vllm/issues/32541) | [Feature]: LoRa adapter support for Qwen3VLForConditionalGeneration | @giangvu-ai | 2026-01-18 | `feature request` |
| [#32468](https://github.com/vllm-project/vllm/issues/32468) | [Bug]: Engine core proc EngineCore_DP0 died unexpectedly, shutting down client. | @simplew2011 | 2026-01-16 | `bug` |
| [#32410](https://github.com/vllm-project/vllm/issues/32410) | [Bug][XPU]: Failed to serve w4a16 Qwen3 VL MoE on Intel Arc Pro B60 | @noobHappylife | 2026-01-15 | `bug`, `intel-gpu`, `stale` |
| [#32338](https://github.com/vllm-project/vllm/issues/32338) | [Feature]: support for LlavaQwenForCausalLM | @SH9959 | 2026-01-14 | `feature request`, `stale` |
| [#32320](https://github.com/vllm-project/vllm/issues/32320) | [Performance]: Issue in serving concurrent streams | @Adithya-Sakaray | 2026-01-14 | `performance`, `stale` |
| [#32259](https://github.com/vllm-project/vllm/issues/32259) | [Bug]: offline infer of mm model cache | @L-hongbin | 2026-01-13 | `bug`, `stale` |
| [#32218](https://github.com/vllm-project/vllm/issues/32218) | [RFC]: Consolidate Multimodal Related Info | @charlotte12l | 2026-01-12 | `RFC`, `stale` |
| [#32193](https://github.com/vllm-project/vllm/issues/32193) | [Bug]: vLLM engine crash under burst load despite expected request queuing (72 concurrent API calls) | @Dineshkumar-Anandan-ZS0367 | 2026-01-12 | `bug` |
| [#32176](https://github.com/vllm-project/vllm/issues/32176) | [Bug]: deepseekv3.2 core dumped with cpu_offload_gb | @mengniwang95 | 2026-01-12 | `bug`, `stale` |
| [#32151](https://github.com/vllm-project/vllm/issues/32151) | [Bug]: jina-reranker-m0 infer error | @chen03191108-lab | 2026-01-12 | `bug`, `stale` |
| [#32113](https://github.com/vllm-project/vllm/issues/32113) | [Bug]: Qwen3-VL-8B inference with video | @justgogorunrun | 2026-01-11 | `bug`, `stale` |
| [#32068](https://github.com/vllm-project/vllm/issues/32068) | [Bug]: Recompile in LLama model | @Lucaskabela | 2026-01-10 | `bug`, `stale` |
| [#32046](https://github.com/vllm-project/vllm/issues/32046) | Google MedASR | @Jeevi10 | 2026-01-09 | `stale` |
| [#32012](https://github.com/vllm-project/vllm/issues/32012) | [Bug]: Qwen3-VL-MoE fails loading when using bitsandbytes quantization | @Datta0 | 2026-01-09 | `bug`, `stale` |
| [#31961](https://github.com/vllm-project/vllm/issues/31961) | [Performance]: EPD Disaggregation Performance Testing Scripts | @Adenialzz | 2026-01-08 | `performance` |
| [#31936](https://github.com/vllm-project/vllm/issues/31936) | [Bug]: run deepseek v3.2 failed，not support RTX PRO 6000 * 8？ | @ljwps | 2026-01-08 | `bug`, `unstale` |
| [#31884](https://github.com/vllm-project/vllm/issues/31884) | [Bug]: run Qwen3-30B-A3B on 8*A800 2  nodes with DP failed zmq.error.ZMQError | @baoqian426 | 2026-01-07 | `bug`, `stale` |
| [#31871](https://github.com/vllm-project/vllm/issues/31871) | [Bug]: Streaming mode with --tool-call-parser hermes returns raw text instead of parsed tool_calls | @Xingsandesu | 2026-01-07 | `bug` |
| [#31803](https://github.com/vllm-project/vllm/issues/31803) | [RFC]: Video Frame Sparsification via Pixel-Level Similarity for Efficient Multimodal Long-Video Inference | @rayleeku | 2026-01-06 | `RFC`, `stale` |
| [#31782](https://github.com/vllm-project/vllm/issues/31782) | [Feature]: Support compressed-tensors NVFP4 quantization for MoE models (Nemotron-H non-gated MoE) | @Firworksyt | 2026-01-06 | `feature request`, `stale` |
| [#31709](https://github.com/vllm-project/vllm/issues/31709) | [Bug]: After upgrade to 0.11.2, vllm crashs with Qwen3. | @tonyaw | 2026-01-05 | `bug` |
| [#31661](https://github.com/vllm-project/vllm/issues/31661) | [Bug]: jina-reranker-m0 [image_index]   IndexError: list index out of range | @chen03191108-lab | 2026-01-04 | `bug`, `stale` |
| [#31647](https://github.com/vllm-project/vllm/issues/31647) | [Bug]: TypeError: BatchPrefillWithPagedKVCacheWrapper.plan() got an unexpected keyword argument 'fixed_split_size' | @sliedes | 2026-01-03 | `bug`, `stale` |
| [#31602](https://github.com/vllm-project/vllm/issues/31602) | [Performance]: If the next request is sent immediately after the previous one finishes, its TTFT will be relatively small; if the next request is sent 10 seconds after the previous one ends, its TTFT will be relatively large. | @xpzwzwz | 2026-01-01 | `performance`, `stale` |
| [#31484](https://github.com/vllm-project/vllm/issues/31484) | [Usage]: RuntimeError when running Qwen2.5-VL-7B-Instruct with vllm: Potential version incompatibility | @puyuan1996 | 2025-12-29 | `usage`, `stale` |
| [#31479](https://github.com/vllm-project/vllm/issues/31479) | [Feature]: Enable LoRA support for tower and connector in more MM models | @jeejeelee | 2025-12-29 | `help wanted`, `feature request`, `multi-modality` |
| [#31372](https://github.com/vllm-project/vllm/issues/31372) | [Feature]: Running paddocr‑vl consumes an excessively large amount of  memory. | @wszgrcy | 2025-12-26 | `feature request`, `stale` |
| [#31353](https://github.com/vllm-project/vllm/issues/31353) | [Bug]: KV Cache grows continuously with just one chat completion request using meta-llama/Llama-3.2-1B on L40 GPU with Flash Attention and finally completed after 10 minutes | @aravilli | 2025-12-25 | `bug`, `help wanted`, `stale` |
| [#31344](https://github.com/vllm-project/vllm/issues/31344) | [Usage]: how to pass param logits_processors in  AsyncEngineArgs? | @cqray1990 | 2025-12-25 | `usage`, `stale` |
| [#31205](https://github.com/vllm-project/vllm/issues/31205) | ValueError: Qwen3OmniMoeThinkerForConditionalGeneration does not support LoRA yet. | @VJJJJJJ1 | 2025-12-23 | `usage` |
| [#31063](https://github.com/vllm-project/vllm/issues/31063) | [Bug]: Qwen3VL-8B-instruct-FP8 has larger result diff rate than sglang compared with transformers | @DamonsJ | 2025-12-20 | `bug`, `stale` |
| [#30839](https://github.com/vllm-project/vllm/issues/30839) | [RFC]: Enabling Zero-Copy Video with PyNvVideoCodec and IPC | @brandonpelfrey | 2025-12-17 | `RFC`, `unstale` |
| [#30819](https://github.com/vllm-project/vllm/issues/30819) | [Bug]: vLLM inference stuck when requesting video description on VLM models | @sidezrw | 2025-12-16 | `bug`, `stale` |
| [#30779](https://github.com/vllm-project/vllm/issues/30779) | [Bug]: v0.11.2 can not support Qwen2.5-Omni- | @GoGo-UpUp | 2025-12-16 | `bug`, `stale` |
| [#30777](https://github.com/vllm-project/vllm/issues/30777) | [Bug]: whisper-large-v3-turbo have accuracy problem on nightly build | @cyysky | 2025-12-16 | `bug`, `stale` |
| [#30493](https://github.com/vllm-project/vllm/issues/30493) | [Bug]: 5090 RTX seems to be broken | @mobicham | 2025-12-11 | `bug` |
| [#30180](https://github.com/vllm-project/vllm/issues/30180) | [Bug]: Inference of Qwen3-VL-235B failed | @ving666 | 2025-12-06 | `bug`, `stale` |
| [#30167](https://github.com/vllm-project/vllm/issues/30167) | [Bug][ROCm]: `vision_embeddings` in transformers inaccurate without math SDP | @AndreasKaratzas | 2025-12-06 | `bug`, `rocm` |
| [#30129](https://github.com/vllm-project/vllm/issues/30129) | [Feature]: About video input for qwen3vl | @lingcco | 2025-12-05 | `feature request`, `stale` |
| [#29968](https://github.com/vllm-project/vllm/issues/29968) | [Bug]: Ministral 3 - streaming tool call not working | @alew3 | 2025-12-03 | `bug` |
| [#29863](https://github.com/vllm-project/vllm/issues/29863) | [Bug]: fails to inference prompt ends with '.' ':' for video inputs | @xgwang | 2025-12-02 | `bug` |
| [#29814](https://github.com/vllm-project/vllm/issues/29814) | [Bug]: Qwen3-VL-235B-A22B-Instruct-FP8 tools doesn't respond in the tools but in content for hermes parswer | @joestein-ssc | 2025-12-01 | `bug` |
| [#29373](https://github.com/vllm-project/vllm/issues/29373) | [Bug]: Multinode inference request with Ray and vLLM crashes - regression from vLLM v0.7.3 | @ysimeonovatnvidia | 2025-11-25 | `bug`, `ray` |
| [#29369](https://github.com/vllm-project/vllm/issues/29369) | [Bug]: nixl connector PD disagg stuck at INFO logging level | @herotai214 | 2025-11-25 | `bug` |
| [#29362](https://github.com/vllm-project/vllm/issues/29362) | [RFC]: Resettle examples. | @noooop | 2025-11-25 | `RFC`, `keep-open` |
| [#29280](https://github.com/vllm-project/vllm/issues/29280) | [Feature]: Selective Token Logprobs Tracking | @Zoher15 | 2025-11-23 | `feature request` |
| [#29174](https://github.com/vllm-project/vllm/issues/29174) | [Bug]: Qwen3 Omni thinking unstable output | @catsled | 2025-11-21 | `bug` |
| [#28919](https://github.com/vllm-project/vllm/issues/28919) | [Bug]: Qwen3-32B use eagle3 crash | @wanghuihhh | 2025-11-18 | `bug`, `unstale` |
| [#28707](https://github.com/vllm-project/vllm/issues/28707) | [Bug]: run fail in nvidia b300 | @Xerxes-cn | 2025-11-14 | `bug`, `unstale` |
| [#28691](https://github.com/vllm-project/vllm/issues/28691) | [Feature]: GGUF model with architecture qwen3vlmoe is not supported yet. | @ivanbaldo | 2025-11-13 | `feature request`, `unstale` |
| [#28640](https://github.com/vllm-project/vllm/issues/28640) | [Bug]: LoRA/Adapter Loading Error with Qwen3-VL-8B-Instruct Multimodal Model in vLLM Deployment (AssertionError in lora_shrink_op) | @pidanshourouzhou | 2025-11-13 | `bug`, `unstale` |
| [#28568](https://github.com/vllm-project/vllm/issues/28568) | [Bug]: FlashInfer attention backend on Hopper fails with llama4-scout and llama3 with fp8 kvcache | @ProExpertProg | 2025-11-12 | `bug`, `stale`, `nvidia` |
| [#28467](https://github.com/vllm-project/vllm/issues/28467) | [Bug]: Persistent CPU RAM Growth and Container Restarts with QWEN2-2B on SageMaker, Unaffected by Cache Settings | @bakas-george | 2025-11-11 | `bug`, `unstale` |
| [#28388](https://github.com/vllm-project/vllm/issues/28388) | [Bug]: 新版的vllm已经废弃了v0代码，而对qwen-omni系列的模型支持仅限于v0，似乎是因为这个原因，我们无法使用最新版的vllm推理qwen-omni模型 | @Lee-xeo | 2025-11-10 | `bug` |
| [#28375](https://github.com/vllm-project/vllm/issues/28375) | [Bug]: Engine stuck at CPU when running long video input on Qwen3-vl model | @MegaGriffin | 2025-11-10 | `bug`, `stale` |
| [#28307](https://github.com/vllm-project/vllm/issues/28307) | [Bug]: `repetition_penalty` leads to engine failure when using vllm serve... | @Blaze-DSP | 2025-11-07 | `bug` |
| [#27932](https://github.com/vllm-project/vllm/issues/27932) | [Feature]: Qwen3 Omini AttributeError: 'Qwen3OmniMoeProcessor' object has no attribute '_get_num_multimodal_tokens' | @rongcg5620 | 2025-11-02 | `feature request`, `stale` |
| [#27557](https://github.com/vllm-project/vllm/issues/27557) | [Bug]: Engine core proc EngineCore_DP0 died unexpectedly, shutting down client. | @wmj9346464543 | 2025-10-27 | `bug` |
| [#27430](https://github.com/vllm-project/vllm/issues/27430) | [Bug]: vLLM (TP=8) on 235B model triggers "CUDA error: unspecified launch failure" and persistent "ERR!" state in nvidia-smi | @whwangovo | 2025-10-23 | `bug`, `nvidia` |
| [#27364](https://github.com/vllm-project/vllm/issues/27364) | [Bug]: Qwen3-VL {4B,8B} FP8 on vLLM returns only exclamation marks ("!!!!!...") on Jetson Thor | @mantyni | 2025-10-22 | `bug`, `unstale` |
| [#27340](https://github.com/vllm-project/vllm/issues/27340) | [Bug]: Qwen3-VL-2B-Instruct vllm推理报错 | @mllmivy-ship-it | 2025-10-22 | `bug`, `stale` |
| [#27120](https://github.com/vllm-project/vllm/issues/27120) | [Bug]: some models won't stream from s3: Qwen/Qwen3-VL-30B-A3B-Instruct-FP8 | @liebman | 2025-10-17 | `bug`, `stale` |
| [#27116](https://github.com/vllm-project/vllm/issues/27116) | [Bug]: vLLM failure (ray.exceptions.RayChannelTimeoutError) observed since v0.10.2, with tp=8, pp=4 | @TheRValiquette | 2025-10-17 | `bug`, `ray`, `stale` |
| [#27094](https://github.com/vllm-project/vllm/issues/27094) | [RFC]: Remove redundant multi-modal input preprocessing during disaggregated inference | @MerHS | 2025-10-17 | `RFC`, `unstale` |
| [#26777](https://github.com/vllm-project/vllm/issues/26777) | [Bug]: Deploying Qwen2.5-VL-7B with Eagle3, the service crashes when concurrency increases. | @taoyun951753 | 2025-10-14 | `bug`, `stale` |
| [#26480](https://github.com/vllm-project/vllm/issues/26480) | [Bug][v0.11.0]: gpt-oss-120b generates with no output | @AlessandroSpallina | 2025-10-09 | `bug` |
| [#26420](https://github.com/vllm-project/vllm/issues/26420) | [Bug]: vLLM server not starting when running Qwen/Qwen3-VL-30B-A3B-Instruct on 2 A6000 GPUs.  | @dhruvil237 | 2025-10-08 | `bug`, `stale` |
| [#26239](https://github.com/vllm-project/vllm/issues/26239) | [Feature]: support reasoning-parser for Qwen3-VL | @SlavikCA | 2025-10-05 | `feature request`, `unstale` |
| [#25950](https://github.com/vllm-project/vllm/issues/25950) | [RFC]: Generalized KV cache reuse | @iddo10 | 2025-09-30 | `RFC`, `stale` |
| [#25943](https://github.com/vllm-project/vllm/issues/25943) | [Feature]: Disable ANSII colors in logs of OpenAI compatible server | @apohllo | 2025-09-30 | `feature request`, `stale` |
| [#25750](https://github.com/vllm-project/vllm/issues/25750) | [Feature]: Tracking Whisper feature requests | @NickLucche | 2025-09-26 | `help wanted`, `good first issue`, `feature request`, `keep-open`, `multi-modality` |
| [#25502](https://github.com/vllm-project/vllm/issues/25502) | [Bug]: This flash attention build does not support tanh softcapping: gemma-2-2b-it on H100 NVL | @yashaswipiplani | 2025-09-23 | `bug`, `stale` |
| [#25383](https://github.com/vllm-project/vllm/issues/25383) | [Usage]: Is there a simple way to pass embedding directly in V1 | @bnuzhanyu | 2025-09-22 | `usage`, `unstale` |
| [#25192](https://github.com/vllm-project/vllm/issues/25192) | [Bug]: Make Whisper model compatible with cuda graphs | @russellb | 2025-09-18 | `bug`, `help wanted`, `stale` |
| [#25172](https://github.com/vllm-project/vllm/issues/25172) | [Feature]: FlexAttention + encoder_decoder support | @JakubCerven | 2025-09-18 | `feature request`, `stale` |
| [#24728](https://github.com/vllm-project/vllm/issues/24728) | [Performance]: Multi-Modal Benchmark on NVIDIA A100 – Qwen2.5-VL / MiniCPM-V-4 / InternVL3_5-4B / InternVL3_5-2B | @player0718 | 2025-09-12 | `performance`, `keep-open` |
| [#24430](https://github.com/vllm-project/vllm/issues/24430) | [Bug]: torch.AcceleratorError: CUDA error: an illegal memory access was encountered | @sleepwalker2017 | 2025-09-08 | `bug` |
| [#24289](https://github.com/vllm-project/vllm/issues/24289) | [Bug]: module 'triton.language' has no attribute 'constexpr_function' | @ZanePoe | 2025-09-05 | `bug` |
| [#23943](https://github.com/vllm-project/vllm/issues/23943) | [Feature]: Any plans to add nvidia/parakeet-tdt-0.6b-v3 to vllm? | @lukaLLM | 2025-08-29 | `feature request` |
| [#23404](https://github.com/vllm-project/vllm/issues/23404) | [Bug]: Qwen3 vLLM Structured Output Ignores Field Descriptions | @seabnavin19 | 2025-08-22 | `bug`, `unstale` |
| [#22817](https://github.com/vllm-project/vllm/issues/22817) | [RFC]: Disaggregated Everything - Token In <> Token Out API Server | @robertgshaw2-redhat | 2025-08-13 | `RFC`, `keep-open`, `rl` |
| [#22424](https://github.com/vllm-project/vllm/issues/22424) | [Bug]: Voxtral-Small-24B-2507 Does Not Support Pipeline-Parallel | @NaiveYan | 2025-08-07 | `bug`, `stale` |
| [#21995](https://github.com/vllm-project/vllm/issues/21995) | [RFC] Run HF processing on GPU | @DarkLight1337 | 2025-07-31 | `RFC`, `keep-open`, `multi-modality` |
| [#20261](https://github.com/vllm-project/vllm/issues/20261) | [Bug]: Prefix caching ignores visual input, causing incorrect multimodal outputs under concurrency | @Richar-Du | 2025-06-30 | `bug`, `unstale` |
| [#20149](https://github.com/vllm-project/vllm/issues/20149) | [Feature]: Add Support for Updating Lora Weights | @alex-jw-brooks | 2025-06-26 | `feature request`, `stale` |
| [#19668](https://github.com/vllm-project/vllm/issues/19668) | [Bug]: vllm, EngineCore encountered a fatal error TimeoutError | @surajssd | 2025-06-15 | `bug` |
| [#19445](https://github.com/vllm-project/vllm/issues/19445) | [Bug]: Docker image CUDA error on RTX 2080 Ti | @arkadijs | 2025-06-10 | `bug`, `unstale` |
| [#17972](https://github.com/vllm-project/vllm/issues/17972) | [Bug]: vLLM server hangs and timeouts after initial requests | @samuellimabraz | 2025-05-12 | `bug`, `stale` |
| [#16966](https://github.com/vllm-project/vllm/issues/16966) | [Bug]: vllm 0.8.4 whisper possible memory leak? | @shenenqing | 2025-04-22 | `bug`, `stale` |
| [#16313](https://github.com/vllm-project/vllm/issues/16313) | [Feature]: Support structured output and tool call together | @Wendong-Fan | 2025-04-09 | `feature request`, `unstale` |
| [#14174](https://github.com/vllm-project/vllm/issues/14174) | [Feature]: will whisper add language detection? | @xiayq1 | 2025-03-04 | `feature request`, `unstale` |
| [#13941](https://github.com/vllm-project/vllm/issues/13941) | [Bug]: wake up OOM (72B model in 8*A800(40G)) | @LugerW-A | 2025-02-27 | `bug`, `unstale` |
| [#10849](https://github.com/vllm-project/vllm/issues/10849) | [Feature]: add DoRA support | @cmhungsteve | 2024-12-03 | `feature request`, `keep-open` |
| [#7863](https://github.com/vllm-project/vllm/issues/7863) | [New Model]: Request to integrate Chexagent Multimodel in vLLM | @MohnishJain | 2024-08-26 | `new-model`, `stale` |
| [#4194](https://github.com/vllm-project/vllm/issues/4194) | [RFC]: Multi-modality Support on vLLM | @ywang96 | 2024-04-19 | `RFC`, `keep-open`, `multi-modality` |

---

## 二、按问题类型分类

### Bug / 错误报告 (310)

- [#40856](https://github.com/vllm-project/vllm/issues/40856) **[Bug]: VLLM running qwen3.6 for image inference occasionally reports 500 Internal Server Error** _(created: 2026-04-25)_
- [#40831](https://github.com/vllm-project/vllm/issues/40831) **[Bug]: TurboQuant KV × any speculative decoding (MTP or ngram) produces degenerate token loops — confirmed across dense and hybrid attention** _(created: 2026-04-24)_
- [#40821](https://github.com/vllm-project/vllm/issues/40821) **[Bug]: Deepseek V4 failed to load on RTX PRO 6000** _(created: 2026-04-24)_
- [#40807](https://github.com/vllm-project/vllm/issues/40807) **[Bug]: TurboQuant KV + spec-decode + chunked-prefill crashes CUDA graph capture at query_start_loc.tolist() in continuation-prefill path (Qwen3-Next hybrid dense)** _(created: 2026-04-24)_
- [#40765](https://github.com/vllm-project/vllm/issues/40765) **[Bug]: runai_streamer loads both Ministral consolidated and HF sharded safetensors** _(created: 2026-04-24)_
- [#40758](https://github.com/vllm-project/vllm/issues/40758) **[CI Failure]: `Qwen3.6-35B-A3B-FP8` fails on `NVIDIA GB10` with `cutlass_scaled_mm` / `cutlass_gemm_caller Error Internal` under vLLM nightly + CUDA 13.0** _(created: 2026-04-24)_
- [#40742](https://github.com/vllm-project/vllm/issues/40742) **[Bug]: CUDA graph capture crashes during startup due to Inductor autotuning torch.cuda.synchronize() inside graph capture (FULL_DECODE_ONLY + MLA + FP8) when PDL is enabled** _(created: 2026-04-23)_
- [#40741](https://github.com/vllm-project/vllm/issues/40741) **[Feature]: Make opencv-python-headless an optional dependency for FIPS compliance** _(created: 2026-04-23)_
- [#40707](https://github.com/vllm-project/vllm/issues/40707) **[Bug]: Scheduling deadlock in _mamba_block_aligned_split with multiple large multimodal inputs on hybrid Mamba models** _(created: 2026-04-23)_
- [#40661](https://github.com/vllm-project/vllm/issues/40661) **[Bug]: CUBLAS_STATUS_EXECUTION_FAILED during CUDA graph compilation of BF16 vision encoder on NVIDIA Jetson AGX Thor (vLLM 0.19.0 regression)** _(created: 2026-04-23)_
- [#40652](https://github.com/vllm-project/vllm/issues/40652) **[Bug]: Kimi 2.6 on 8x A100 SMX4 leads to NVLink Crash Coredump** _(created: 2026-04-22)_
- [#40649](https://github.com/vllm-project/vllm/issues/40649) **[Bug]: KeyError on model.layers.N.self_attn.attn during initialize_attn_backend with pipeline_parallel_size=4 (V1 engine + Ray)** _(created: 2026-04-22)_
- [#40642](https://github.com/vllm-project/vllm/issues/40642) **KimiK25ForConditionalGeneration failed to be inspected — SIGSEGV in registry subprocess during process exit
  (GB200)** _(created: 2026-04-22)_
- [#40620](https://github.com/vllm-project/vllm/issues/40620) **[RFC]: Unified Device Capability Abstraction for Cross-Platform Feature Detection** _(created: 2026-04-22)_
- [#40616](https://github.com/vllm-project/vllm/issues/40616) **[Bug]:  `/v2/embed` with `input_type` returns misleading 400 on nemotron-embed-vl** _(created: 2026-04-22)_
- [#40591](https://github.com/vllm-project/vllm/issues/40591) **[Bug]: Regression in 0.19.1 - Gemma 4 26B MoE fails to load packed experts (KeyError: down_proj_packed). Worked in dev6.** _(created: 2026-04-22)_
- [#40585](https://github.com/vllm-project/vllm/issues/40585) **[Bug]: qwen3.5 can not use --decode-context-parallel-size with --enable-prefix-caching** _(created: 2026-04-22)_
- [#40444](https://github.com/vllm-project/vllm/issues/40444) **[Bug]: Per-attention-head quantization is currently available only with the Flash Attention backend and requires the calibration pathway provided by llm-compressor.** _(created: 2026-04-21)_
- [#40443](https://github.com/vllm-project/vllm/issues/40443) **[BUG] Port-allocation race between ApiServer processes in hybrid-LB mode (ZMQError: Address already in use)** _(created: 2026-04-21)_
- [#40420](https://github.com/vllm-project/vllm/issues/40420) **[Bug]: TurboQuant `_continuation_prefill` OOMs and kills engine at long-context prefill (~185K actual tokens)** _(created: 2026-04-21)_
- [#40417](https://github.com/vllm-project/vllm/issues/40417) **[Bug]: `token_capacity_kv_cache_groups` (#40384) should also exclude `SlidingWindowSpec` / `ChunkedLocalAttentionSpec`** _(created: 2026-04-21)_
- [#40381](https://github.com/vllm-project/vllm/issues/40381) **[Bug]: Buffer overflow when allocating memory error on Qwen3.5-122B-A10B-GPTQ-Int4 and NVFP4** _(created: 2026-04-20)_
- [#40343](https://github.com/vllm-project/vllm/issues/40343) **[Installation]: 有提供cuda 12.6+python3.12的vllm预编译的whl包吗？ 以开发者模型需要本地安装下，发布的都是cuda13版本的，不适配cuda12.6的本机的版本型号** _(created: 2026-04-20)_
- [#40318](https://github.com/vllm-project/vllm/issues/40318) **[Bug]: Mistral3 text-only startup fails when text_config.architectures is None** _(created: 2026-04-20)_
- [#40290](https://github.com/vllm-project/vllm/issues/40290) **[Bug]: Gemma 4 (31B/26B-A4B) vision outputs only <pad> under fp16 — vision_tower standardize overflows** _(created: 2026-04-19)_
- [#40165](https://github.com/vllm-project/vllm/issues/40165) **[Bug]: HunyuanOCR crashes with "query and key must have the same dtype" during inference (vLLM 0.19.0, RTX 3050)** _(created: 2026-04-17)_
- [#40121](https://github.com/vllm-project/vllm/issues/40121) **[Bug]: CUDA graph replay triggers Xid 13 illegal memory access on Qwen3-32B-AWQ with TP=2 on dual RTX 3090** _(created: 2026-04-17)_
- [#40113](https://github.com/vllm-project/vllm/issues/40113) **[CI Failure]: Multi-Modal Processor (CPU)** _(created: 2026-04-17)_
- [#40106](https://github.com/vllm-project/vllm/issues/40106) **[Bug]: Gemma4 multimodal: missing vision-aware bidirectional attention mask for use_bidirectional_attention="vision" models** _(created: 2026-04-17)_
- [#40095](https://github.com/vllm-project/vllm/issues/40095) **[Bug]: Gemma4MultimodalEmbedder normalization order different from Transformers, causing bad audio inference** _(created: 2026-04-17)_
- [#40080](https://github.com/vllm-project/vllm/issues/40080) **[Bug]: Gemma 4 (31B / 26B-A4B) generates infinite repetition loops, especially with structured output (JSON schema)** _(created: 2026-04-17)_
- [#40047](https://github.com/vllm-project/vllm/issues/40047) **[Bug][Tracking Issue]: NaNs in CUDA Graph padding regions corrupt activations in some per-token kernels** _(created: 2026-04-16)_
- [#40038](https://github.com/vllm-project/vllm/issues/40038) **[Bug]: cudaErrorIllegalAddress during PIECEWISE CUDA graph replay with MoE LoRA: stale buffer addresses in `moe_lora_align_block_size`** _(created: 2026-04-16)_
- [#40018](https://github.com/vllm-project/vllm/issues/40018) **[Bug]: `ROCM_AITER_MLA_SPARSE` prefill produces garbage for prompt_len > ~20K tokens on gfx950 (GLM-5.1-FP8)** _(created: 2026-04-16)_
- [#40002](https://github.com/vllm-project/vllm/issues/40002) **[Bug]: Inconsistent KV Cache reporting and system hang on long context requests (Gemma-4 26B AWQ Int4)** _(created: 2026-04-16)_
- [#39996](https://github.com/vllm-project/vllm/issues/39996) **[Bug] Fatal AssertionError: Encoder KV cache fails to evict tokens, exceeding max_model_len in long-lived WebSocket sessions** _(created: 2026-04-16)_
- [#39928](https://github.com/vllm-project/vllm/issues/39928) **[Bug]: Qwen3.5 DFlash gives strange responses on SM90** _(created: 2026-04-15)_
- [#39919](https://github.com/vllm-project/vllm/issues/39919) **[Bug]: DeepSeek OCR doesn't work on vllm 0.19** _(created: 2026-04-15)_
- [#39915](https://github.com/vllm-project/vllm/issues/39915) **[Bug]: Engine core initialization failed (Parent process exited) on 2xH100 with Llama-3.3-70B-FP8 (TP/PP=2)** _(created: 2026-04-15)_
- [#39903](https://github.com/vllm-project/vllm/issues/39903) **[Bug]: Significant Cross-Instance Inference Variance in vLLM v0.18.0 on H20 (~10-point gap) Qwen3.5-35B-A3B** _(created: 2026-04-15)_
- [#39839](https://github.com/vllm-project/vllm/issues/39839) **[Bug]:  MTP DeepSeek and Eagle Flash Attention Failures in Spec Decode Unit Tests** _(created: 2026-04-14)_
- [#39814](https://github.com/vllm-project/vllm/issues/39814) **[Bug]: FlashInferFP8ScaledMMLinearKernel segfaults on Blackwell (sm100)** _(created: 2026-04-14)_
- [#39788](https://github.com/vllm-project/vllm/issues/39788) **[Bug]: CUDA OOM with Kimi-K2.5 NVFP4 on both TP4 and TP8** _(created: 2026-04-14)_
- [#39766](https://github.com/vllm-project/vllm/issues/39766) **[RFC]: Support Mooncake Based ECConnector for EPD** _(created: 2026-04-14)_
- [#39761](https://github.com/vllm-project/vllm/issues/39761) **[Bug]:CUDA illegal instruction during decode (V1 Engine + NVFP4) on aarch64 (NVIDIA GB10)** _(created: 2026-04-14)_
- [#39749](https://github.com/vllm-project/vllm/issues/39749) **[Roadmap] [Draft] vLLM Roadmap Q2 2026** _(created: 2026-04-13)_
- [#39735](https://github.com/vllm-project/vllm/issues/39735) **[Feature]: Expose Word-Level Timestamps in `/v1/realtime` API for Voxtral Realtime** _(created: 2026-04-13)_
- [#39687](https://github.com/vllm-project/vllm/issues/39687) **[Bug]: vllm(g0e39202ca) vllm serve: error: argument --limit-mm-per-prompt: Value image=4,audio=1 cannot be converted to <function** _(created: 2026-04-13)_
- [#39682](https://github.com/vllm-project/vllm/issues/39682) **[BUGS] vLLM V1 Engine Hangs After Weight Loading on Blackwell (sm_121) Multi-Node Ray Setup (TP=2)** _(created: 2026-04-13)_
- [#39681](https://github.com/vllm-project/vllm/issues/39681) **[Bug]: Gemma4 multimodal crashes with "pixel_values contains inconsistent shapes" when concurrent image requests have different resolutions** _(created: 2026-04-13)_
- [#39631](https://github.com/vllm-project/vllm/issues/39631) **[Bug]: Abnormal Scores in Batch Processing of Image-Text Pairs with qwen3-VL-reranker Model** _(created: 2026-04-12)_
- [#39485](https://github.com/vllm-project/vllm/issues/39485) **[Bug]: Runtime error on ROCm platform serving Deepseek-R1 using VLLM_ROCM_USE_AITER=1** _(created: 2026-04-10)_
- [#39474](https://github.com/vllm-project/vllm/issues/39474) **[Bug] Regression: GPTQ models fail to load on Intel XPU in v0.19.0 (missing XPU branches in gptq.py)** _(created: 2026-04-10)_
- [#39408](https://github.com/vllm-project/vllm/issues/39408) **[Usage]: qwen3-asr-1.7b pre-allocated encoder cache size limit** _(created: 2026-04-09)_
- [#39407](https://github.com/vllm-project/vllm/issues/39407) **[Bug]: Gemma 4 31B FP8_BLOCK checkpoint produces garbage repetitive output — logit saturation at softcap wall due to absorbed activation scales being double-applied** _(created: 2026-04-09)_
- [#39371](https://github.com/vllm-project/vllm/issues/39371) **DSA module construction corrupts CUDA RNG state (Offset increment outside graph capture)** _(created: 2026-04-09)_
- [#39348](https://github.com/vllm-project/vllm/issues/39348) **[Bug]: Qwen3.5-9B-AWQ on ROCm/vLLM 0.19.0 can get stuck generating endless "!" inside JSON schema output** _(created: 2026-04-08)_
- [#39319](https://github.com/vllm-project/vllm/issues/39319) **[Bug]: vLLM docker container with Qwen3.5 - Connection error** _(created: 2026-04-08)_
- [#39288](https://github.com/vllm-project/vllm/issues/39288) **[Bug]: FlashInfer CUTLASS MoE backend causes CUDA illegal memory access on H100 during CUDA graph capture (Qwen3-Next-80B BF16)** _(created: 2026-04-08)_
- [#39231](https://github.com/vllm-project/vllm/issues/39231) **[Bug]: Qwen3.5 Text Only Model (Qwen3_5ForCausalLM)** _(created: 2026-04-07)_
- [#39210](https://github.com/vllm-project/vllm/issues/39210) **[Bug] Embedding/pooling models crash on B200 (SM 10.0) — encoder attention hardcodes FA2 which lacks SM100 support** _(created: 2026-04-07)_
- [#39202](https://github.com/vllm-project/vllm/issues/39202) **[Bug]: Crash on Transcription (size for tensor a must match the size of tensor b) with reproduce** _(created: 2026-04-07)_
- [#39149](https://github.com/vllm-project/vllm/issues/39149) **[Bug]: Segfault in Triton LLVM (MachineCSE / translateLLVMIRToASM) when serving Qwen3.5-4B on RTX 4090 (WSL2) with vLLM 0.19.0** _(created: 2026-04-07)_
- [#39137](https://github.com/vllm-project/vllm/issues/39137) **[Bug]: fp8_e5m2 kv-cache gate in _init_kv_cache_quant fires on any quantized checkpoint, not only fp8 checkpoints** _(created: 2026-04-07)_
- [#39133](https://github.com/vllm-project/vllm/issues/39133) **[Bug]: Gemma 4 31B INT4 on 2×24GB GPUs (TP=2): GPU KV cache size is 25,200 tokens at max_model_len=131072, gpu_memory_utilization=0.96, BF16 KV** _(created: 2026-04-07)_
- [#39096](https://github.com/vllm-project/vllm/issues/39096) **[Bug]: Batch invariance breaks with torch.compile and/or CUDA graphs on SM<90** _(created: 2026-04-06)_
- [#39061](https://github.com/vllm-project/vllm/issues/39061) **[Bug]: Gemma4 vision encoder crashes with ValueError: Expected hidden_size to be 5376, but found: 72** _(created: 2026-04-06)_
- [#39057](https://github.com/vllm-project/vllm/issues/39057) **[Bug]: Deepseek v3.2 RuntimeError: Worker failed with error "Assertion error"** _(created: 2026-04-06)_
- [#39049](https://github.com/vllm-project/vllm/issues/39049) **[Bug]: Gemma 4 FP8 dynamic quantization = gibberish output** _(created: 2026-04-05)_
- [#39015](https://github.com/vllm-project/vllm/issues/39015) **[Bug]: torch.distributed.DistNetworkError: The server socket has failed to listen on any local network address. port: 29500, useIpv6: false, code: -98, name: EADDRINUSE, message: address already in use** _(created: 2026-04-05)_
- [#39010](https://github.com/vllm-project/vllm/issues/39010) **[Bug]: Hang During CUDA Graph Capture on ROCM in 0.19** _(created: 2026-04-05)_
- [#38999](https://github.com/vllm-project/vllm/issues/38999) **[Bug]: Gemma 4 MoE (26B-A4B) crashes with `--data-parallel-size > 1` — AssertionError in cuda_communicator all_gather** _(created: 2026-04-04)_
- [#38994](https://github.com/vllm-project/vllm/issues/38994) **Qwen-3.5 9B often producing repetitive/garbled output with Intel Backend** _(created: 2026-04-04)_
- [#38976](https://github.com/vllm-project/vllm/issues/38976) **[Bug]:TimeoutError: RPC call to sample_tokens timed out. when pp is on under xpu env** _(created: 2026-04-04)_
- [#38967](https://github.com/vllm-project/vllm/issues/38967) **[Bug] vLLM >= 0.18.0 NCCL segfault (cuMemCreate) with TP>1 on RTX 4090 (SM 89)** _(created: 2026-04-04)_
- [#38912](https://github.com/vllm-project/vllm/issues/38912) **Gemma 4 MoE NVFP4: expert_params_mapping doesn't handle scale key suffixes** _(created: 2026-04-03)_
- [#38903](https://github.com/vllm-project/vllm/issues/38903) **[Bug]: Cross-request context contamination with async scheduling + pipeline parallelism on multi-node** _(created: 2026-04-03)_
- [#38886](https://github.com/vllm-project/vllm/issues/38886) **[Bug]: Gemma 4 E4B weight loading fails `Gemma4ClippableLinear` parameter `input_max` not recognized** _(created: 2026-04-03)_
- [#38884](https://github.com/vllm-project/vllm/issues/38884) **[Bug]: Gemma 4 torch._dynamo.exc.TorchRuntimeError: Dynamo failed to run FX node with fake tensors** _(created: 2026-04-03)_
- [#38754](https://github.com/vllm-project/vllm/issues/38754) **[Bug]: GPT OSS Router GEMM Causing NaNs** _(created: 2026-04-01)_
- [#38740](https://github.com/vllm-project/vllm/issues/38740) **[Transformers v5] NemotronParseForConditionalGeneration** _(created: 2026-04-01)_
- [#38713](https://github.com/vllm-project/vllm/issues/38713) **[Bug]: Error when trying to serve MiniMax 2.5 on 4 H100 nodes with 4 GPUS** _(created: 2026-04-01)_
- [#38710](https://github.com/vllm-project/vllm/issues/38710) **[Bug]: heterogeneous disaggregated serving XPU (Prefill) + CPU (Decode) accuracy issue** _(created: 2026-04-01)_
- [#38660](https://github.com/vllm-project/vllm/issues/38660) **[Bug]: CUDA assert in triton attention for MolmoWeb models (Molmo2 architecture with different max_position_embeddings)** _(created: 2026-03-31)_
- [#38658](https://github.com/vllm-project/vllm/issues/38658) **[Bug]: MLA attention casts activations to int32 when using Marlin FP8 on GPUs without native FP8 support (sm < 89)** _(created: 2026-03-31)_
- [#38656](https://github.com/vllm-project/vllm/issues/38656) **[Bug]: qwen 3.5 model launch get stuck for quite a long time** _(created: 2026-03-31)_
- [#38603](https://github.com/vllm-project/vllm/issues/38603) **[Bug]: Streaming last chunk contains non-empty tool_calls with empty fields "type" causing type validation error ** _(created: 2026-03-31)_
- [#38591](https://github.com/vllm-project/vllm/issues/38591) **Bug: ValueError: too many values to unpack in dispatch_cpu_unquantized_gemm when loading Qwen3.5-4B** _(created: 2026-03-30)_
- [#38586](https://github.com/vllm-project/vllm/issues/38586) **[Bug]: Whisper online benchmark with profiling error: TypeError: multi_modal_content must be a dict containing 'audio'** _(created: 2026-03-30)_
- [#38560](https://github.com/vllm-project/vllm/issues/38560) **[Bug]: reasoning_effort passed to MistralCommonTokenizer.apply_chat_template breaks Mistral Small 4 chat completions on vLLM 0.18.0** _(created: 2026-03-30)_
- [#38551](https://github.com/vllm-project/vllm/issues/38551) **[Bug]: AssertionError: Encoder cache miss crashes engine with MTP + multimodal under high concurrency** _(created: 2026-03-30)_
- [#38486](https://github.com/vllm-project/vllm/issues/38486) **[Bug]: cuda graph takes too much memory for qwen 3.5** _(created: 2026-03-29)_
- [#38472](https://github.com/vllm-project/vllm/issues/38472) **[Bug]: [xPyD]Potential OOM when using v1 P2pNcclConnector as KV cache transport: KV cache accumulation on decode instance.** _(created: 2026-03-29)_
- [#38470](https://github.com/vllm-project/vllm/issues/38470) **[Bug]: When using the Sonnet dataset for benchmark testing, if the input length is too small, the CPU usage becomes abnormally high with no error logs, making it impossible to run the benchmark properly.** _(created: 2026-03-29)_
- [#38459](https://github.com/vllm-project/vllm/issues/38459) **[Bug]: `limit_mm_per_prompt` is ineffective for Qwen3-VL** _(created: 2026-03-29)_
- [#38428](https://github.com/vllm-project/vllm/issues/38428) **[Bug]: V1 Engine: EngineDeadError (AssertionError) on max_model_len overflow during realtime audio streaming** _(created: 2026-03-28)_
- [#38425](https://github.com/vllm-project/vllm/issues/38425) **[Transformers v5] InternVL2** _(created: 2026-03-28)_
- [#38411](https://github.com/vllm-project/vllm/issues/38411) **[Bug]: Vision encoder crashes on SM100 (Jetson Thor) — `_vllm_fa2_C` compiled for SM80-only, no override available for vision encoder** _(created: 2026-03-28)_
- [#38389](https://github.com/vllm-project/vllm/issues/38389) **[Transformers v5] IsaacForConditionalGeneration** _(created: 2026-03-27)_
- [#38379](https://github.com/vllm-project/vllm/issues/38379) **Upgrade to Transformers v5** _(created: 2026-03-27)_
- [#38351](https://github.com/vllm-project/vllm/issues/38351) **[Bug]: When use_audio_in_video is enabled in qwen3-omni, the output may exhibit issues such as empty or repetitive output.** _(created: 2026-03-27)_
- [#38297](https://github.com/vllm-project/vllm/issues/38297) **[Bug]: Gemma3n concurrent audio requests crash EngineCore — missing dynamic_dims on audio sequence dimension** _(created: 2026-03-27)_
- [#38235](https://github.com/vllm-project/vllm/issues/38235) **[Feature]: Quantization support (AWQ / GPTQ / FP8) for mistralai/Voxtral-Mini-4B-Realtime-2602** _(created: 2026-03-26)_
- [#38233](https://github.com/vllm-project/vllm/issues/38233) **[Bug]: Voxtral-Mini-4B-Realtime hangs/crashes on multiple sessions due to encoder_cache_usage saturation on 16GB GPU** _(created: 2026-03-26)_
- [#38208](https://github.com/vllm-project/vllm/issues/38208) **[Bug]: CUDA Illegal Instruction during CUDA Graph capture with Nemotron-3-Nano NVFP4 on sm_121** _(created: 2026-03-26)_
- [#38203](https://github.com/vllm-project/vllm/issues/38203) **[Bug]: M2.5 tool call result is badcase, deploy 1p1d with nixl connector, P and D use DP8-EP-TP1** _(created: 2026-03-26)_
- [#38196](https://github.com/vllm-project/vllm/issues/38196) **GDN attention backend crashes with ngram speculative decoding on mixed decode batches** _(created: 2026-03-26)_
- [#38176](https://github.com/vllm-project/vllm/issues/38176) **[Bug]: qwen3 235B model with latest vllm is going to generate only 1 token.** _(created: 2026-03-26)_
- [#38106](https://github.com/vllm-project/vllm/issues/38106) **[Bug]: tool_choice="required" + speculative decoding with lukealonso/Qwen3.5-397B-A17B-NVFP4 leads to failed tool calls.** _(created: 2026-03-25)_
- [#38079](https://github.com/vllm-project/vllm/issues/38079) **[RFC] Redesign enable_return_routed_experts to avoid blocking EngineCore event loop** _(created: 2026-03-25)_
- [#38077](https://github.com/vllm-project/vllm/issues/38077) **[Bug]: Qwen3.5-9B answer !!!!!!!!!** _(created: 2026-03-25)_
- [#38056](https://github.com/vllm-project/vllm/issues/38056) **[Bug]: ImportError: flash_attn.ops.triton.rotary not found on older versions (< v2.1.2)** _(created: 2026-03-25)_
- [#38022](https://github.com/vllm-project/vllm/issues/38022) **[Bug]: Marlin MoE kernel fails with MXFP4-quantized GPT-OSS 20B - Invalid thread config for non-aligned dimensions (K=2880, N=2880)** _(created: 2026-03-24)_
- [#38004](https://github.com/vllm-project/vllm/issues/38004) **[Bug]: Speech-to-Text endpoint may return 501 but not documented in OpenAPI** _(created: 2026-03-24)_
- [#37996](https://github.com/vllm-project/vllm/issues/37996) **[Bug]: Qwen3.5 397B GPTQ model outputs all exclamation points on ROCM** _(created: 2026-03-24)_
- [#37992](https://github.com/vllm-project/vllm/issues/37992) **[Bug]: RuntimeError triton error during profile_run with Qwen3.5-MoE vision encoder on ROCm** _(created: 2026-03-24)_
- [#37981](https://github.com/vllm-project/vllm/issues/37981) **[Bug]: v0.18.0 fails to run MiniCPM-o-4.5** _(created: 2026-03-24)_
- [#37977](https://github.com/vllm-project/vllm/issues/37977) **[Bug][Model] Eagle2.5-VL applies ImageNet normalization instead of SigLIP2** _(created: 2026-03-24)_
- [#37890](https://github.com/vllm-project/vllm/issues/37890) **[Bug]: NaNs in vLLM using DeepSeek-R1-0528-NVFP4-v2** _(created: 2026-03-23)_
- [#37858](https://github.com/vllm-project/vllm/issues/37858) **[Bug]: does not have the attribute 'FakeTensorMode'** _(created: 2026-03-23)_
- [#37828](https://github.com/vllm-project/vllm/issues/37828) **[Bug]: Intel ARC 140v not supported as XE2 cutlass kernel** _(created: 2026-03-22)_
- [#37746](https://github.com/vllm-project/vllm/issues/37746) **[Bug] prompt_logprobs causes livelock with IsHybrid models (Qwen3.5) in DP mode** _(created: 2026-03-21)_
- [#37737](https://github.com/vllm-project/vllm/issues/37737) **[Bug]: Missing logprobs for `<tool_call>` in streaming chat completions** _(created: 2026-03-21)_
- [#37736](https://github.com/vllm-project/vllm/issues/37736) **[CI Failure]:  Gemma3 OOMs with transformers backend** _(created: 2026-03-21)_
- [#37729](https://github.com/vllm-project/vllm/issues/37729) **[Bug]: V1 engine core deadlocks under concurrent load (fp8 + prefix caching + Qwen3.5)** _(created: 2026-03-21)_
- [#37675](https://github.com/vllm-project/vllm/issues/37675) **[Bug]: deepgemm compile error** _(created: 2026-03-20)_
- [#37659](https://github.com/vllm-project/vllm/issues/37659) **[Bug]: RuntimeError: 基于megatron grpo Qwen3-Omni模型时，出现RuntimeError: Event device index  does not match recording stream's device index** _(created: 2026-03-20)_
- [#37602](https://github.com/vllm-project/vllm/issues/37602) **[Bug]: Qwen3.5-122B-A10B-FP8 EngineCore crash on concurrent image requests** _(created: 2026-03-19)_
- [#37581](https://github.com/vllm-project/vllm/issues/37581) **[Bug]: /v1/chat/completions/render` crashes for Qwen/Qwen3-ASR-0.6B multimodal audio, and chat audio returns empty/junk** _(created: 2026-03-19)_
- [#37563](https://github.com/vllm-project/vllm/issues/37563) **mm_fp4 trtllm backend leaks padding scales into real rows (use_8x4_sf_layout=True)** _(created: 2026-03-19)_
- [#37553](https://github.com/vllm-project/vllm/issues/37553) **[Bug]: Mistral-Small-4-119B-2603 fails on 8x RTX 3090 (SM 8.6) with vLLM v0.17.1: no valid MLA attention backend** _(created: 2026-03-19)_
- [#37551](https://github.com/vllm-project/vllm/issues/37551) **[Bug] vLLM 0.17.1: `zai-org/GLM-OCR` has `mtp_graph < no_mtp_graph` despite high acceptance** _(created: 2026-03-19)_
- [#37472](https://github.com/vllm-project/vllm/issues/37472) **[Bug] V1 engine hangs on encoder cache profiling on AMD gfx1151 (MIOpen missing solver DB)** _(created: 2026-03-18)_
- [#37459](https://github.com/vllm-project/vllm/issues/37459) **[CI Failure]: MultiModal Models Extended 2 - isaac test case OOMs** _(created: 2026-03-18)_
- [#37451](https://github.com/vllm-project/vllm/issues/37451) **[Bug]: 0.17.1 - vllm serve deepseek-ai/DeepSeek-OCR-2 on H100 crashes during Capturing CUDA graphs (decode, FULL)** _(created: 2026-03-18)_
- [#37431](https://github.com/vllm-project/vllm/issues/37431) **Mamba-2 Triton kernels crash with illegal instruction on SM121 (DGX Spark) without CUDA_LAUNCH_BLOCKING=1** _(created: 2026-03-18)_
- [#37392](https://github.com/vllm-project/vllm/issues/37392) **[Bug]:推理时报错，模型关闭了。部署的Qwen3.5-122B-A10B-FP8模型** _(created: 2026-03-18)_
- [#37367](https://github.com/vllm-project/vllm/issues/37367) **[Bug]: gcc: internal compiler error: Segmentation fault signal terminated program cc1** _(created: 2026-03-18)_
- [#37363](https://github.com/vllm-project/vllm/issues/37363) **fix(compilation): fix piecewise CUDA graph bugs with splitting_ops** _(created: 2026-03-18)_
- [#37325](https://github.com/vllm-project/vllm/issues/37325) **[Bug][ARM CPU] Build/Runtime error: no matching function for call to ‘at::vec::CPU_CAPABILITY::VecMask<long int, 4>::VecMask(int&)’ when serving Qwen3-VL-8B-Instruct** _(created: 2026-03-17)_
- [#37255](https://github.com/vllm-project/vllm/issues/37255) **[Bug]: Failed to run distributed inference due to error list index out of range in omp_cpuids_list** _(created: 2026-03-17)_
- [#37242](https://github.com/vllm-project/vllm/issues/37242) **[Community] RTX 5090 (Blackwell sm_120) + WSL2 2.7.0: CUDA graphs work — benchmarks + full config** _(created: 2026-03-17)_
- [#37185](https://github.com/vllm-project/vllm/issues/37185) **[Bug]: The 2-bit model repeatedly outputs !!!!, while Transformers works correctly.** _(created: 2026-03-16)_
- [#37159](https://github.com/vllm-project/vllm/issues/37159) **[Bug]: vLLM crashed with V100 by running zai-org/GLM-OCR** _(created: 2026-03-16)_
- [#37140](https://github.com/vllm-project/vllm/issues/37140) **[Usage]: CUDA error: the provided PTX was compiled with an unsupported toolchain.** _(created: 2026-03-16)_
- [#37096](https://github.com/vllm-project/vllm/issues/37096) **[Bug]: v0.17.0-aarch64 onwards will run out of CUDA memory for gpt-oss-120b on GH200 144GB** _(created: 2026-03-15)_
- [#37060](https://github.com/vllm-project/vllm/issues/37060) **[Bug]: sm110: torch.AcceleratorError: CUDA error: an illegal instruction was encountered** _(created: 2026-03-14)_
- [#37035](https://github.com/vllm-project/vllm/issues/37035) **[Bug]: cudaErrorIllegalAddress in gdn_attn.py:237 when using qwen3_next_mtp with num_speculative_tokens=5 under load** _(created: 2026-03-14)_
- [#36906](https://github.com/vllm-project/vllm/issues/36906) **[Bug]: EAGLE3 speculative decoding + multimodal crash under high concurrency** _(created: 2026-03-12)_
- [#36872](https://github.com/vllm-project/vllm/issues/36872) **[Bug]: Gibberish output and collapsing generation throughput with Qwen3.5-35B-A3B-FP8 and speculative decoding enabled** _(created: 2026-03-12)_
- [#36861](https://github.com/vllm-project/vllm/issues/36861) **[Bug]: Why does setting `--pipeline-parallel-size > 1` result in an OOM error, but `--tensor-parallel-size> 1` does not?** _(created: 2026-03-12)_
- [#36852](https://github.com/vllm-project/vllm/issues/36852) **[Bug]: GPU failure during repeated model loading when using --enable-prefix-caching with KV transfer (LMCacheConnectorV1)** _(created: 2026-03-12)_
- [#36843](https://github.com/vllm-project/vllm/issues/36843) **[Bug]: VLLM 0.17.1 initial mtp with FLASH_ATTN randomly crash** _(created: 2026-03-12)_
- [#36811](https://github.com/vllm-project/vllm/issues/36811) **[Bug]: CUDA illegal memory access on GPTQ Marlin** _(created: 2026-03-11)_
- [#36796](https://github.com/vllm-project/vllm/issues/36796) **[Bug]: CPU offload errors on nightly with NVIDIA GH200 Unified Memory (UMA)** _(created: 2026-03-11)_
- [#36793](https://github.com/vllm-project/vllm/issues/36793) **[Bug]: TP=2 DP=2 Broken for Qwen3-Next W4A16** _(created: 2026-03-11)_
- [#36781](https://github.com/vllm-project/vllm/issues/36781) **[Bug]: vLLM 0.17.0 failed to serve Qwen3-30B-A3B-Instruct-2507 after adding `--enable_lora`** _(created: 2026-03-11)_
- [#36748](https://github.com/vllm-project/vllm/issues/36748) **[Bug]: In DP mode, waiting request stack in a few DP ranks.** _(created: 2026-03-11)_
- [#36589](https://github.com/vllm-project/vllm/issues/36589) **[Bug]: SM 7.5 extreme slowness hangs indefinitely on T4 (vllm 0.17.0 with Qwen3.5-27B)** _(created: 2026-03-10)_
- [#36585](https://github.com/vllm-project/vllm/issues/36585) **[Bug]: qwen3.5-27b-gptq deploy fail** _(created: 2026-03-10)_
- [#36566](https://github.com/vllm-project/vllm/issues/36566) **[Bug]:Qwen3.5-35B-A3B vllm v0.17.0 ERROR 03-10 00:52:24 [multiproc_executor.py:261] Worker proc VllmWorker-0 died unexpectedly, shutting down executor.** _(created: 2026-03-10)_
- [#36476](https://github.com/vllm-project/vllm/issues/36476) **[Bug]: vllm 0.17.0 启动Qwen3.5-122B-A10B失败** _(created: 2026-03-09)_
- [#36450](https://github.com/vllm-project/vllm/issues/36450) **[Bug]: Qwen3.5 AWQ models crash during inference on RTX 5090 (Blackwell) with Triton OOM in solve_tril despite successful model load** _(created: 2026-03-09)_
- [#36407](https://github.com/vllm-project/vllm/issues/36407) **[Bug]: KeyError in get_layers_from_vllm_config with pipeline parallelism (vLLM 0.16.0)** _(created: 2026-03-08)_
- [#36350](https://github.com/vllm-project/vllm/issues/36350) **[Bug]: Qwen 3.5 4B fail on first request on Intel XPU (Arc Graphics B580)** _(created: 2026-03-07)_
- [#36275](https://github.com/vllm-project/vllm/issues/36275) **[Bug]: Qwen3.5 4b incompatibility** _(created: 2026-03-06)_
- [#36245](https://github.com/vllm-project/vllm/issues/36245) **[Bug]: Gemma3 mmproj-*.gguf is not downloaded in 'download_gguf'** _(created: 2026-03-06)_
- [#36193](https://github.com/vllm-project/vllm/issues/36193) **[Bug]: Unsupported Activation Function for Step-3.5-Flash** _(created: 2026-03-06)_
- [#36180](https://github.com/vllm-project/vllm/issues/36180) **[Bug]: meta-llama/Llama-3.2-1B-Instruct Fails With ROCM_ATTN Due To Seg Fault** _(created: 2026-03-05)_
- [#36148](https://github.com/vllm-project/vllm/issues/36148) **[Bug]: Qwen3-VL-Reranker-8B fails with shape mismatch error when loading with --quantization bitsandbytes** _(created: 2026-03-05)_
- [#36116](https://github.com/vllm-project/vllm/issues/36116) **[Bug]: Pseudo-Streaming Issue When Using Tool Parser with Qwen3-Coder and MiniMax-M2** _(created: 2026-03-05)_
- [#36103](https://github.com/vllm-project/vllm/issues/36103) **[Bug]: torch._inductor.exc.InductorError:** _(created: 2026-03-05)_
- [#36073](https://github.com/vllm-project/vllm/issues/36073) **[Bug]: Prolonged Latency in Some Streaming Responses in Function Call Mode（MiniMax model）** _(created: 2026-03-05)_
- [#36015](https://github.com/vllm-project/vllm/issues/36015) **[Bug]: Realtime audio transcription (Voxtral) silently hangs after ~10 minutes due to unhandled TimeoutError in background task** _(created: 2026-03-04)_
- [#35980](https://github.com/vllm-project/vllm/issues/35980) **[Bug]: Why does deploying Qwen3-32B-AWQ via vllm:v0.10.1.1 result in different outputs for the same input?** _(created: 2026-03-04)_
- [#35950](https://github.com/vllm-project/vllm/issues/35950) **[Bug]: ValueError: too many values to unpack (expected 2)** _(created: 2026-03-04)_
- [#35944](https://github.com/vllm-project/vllm/issues/35944) **[Bug]: jetson agx thor报错** _(created: 2026-03-04)_
- [#35863](https://github.com/vllm-project/vllm/issues/35863) **[Bug]: Voxtral-Realtime stops returning transcribed text starting from the 3rd concurrent session** _(created: 2026-03-03)_
- [#35820](https://github.com/vllm-project/vllm/issues/35820) **[Bug]: deploy Qwen3.5-27B error** _(created: 2026-03-03)_
- [#35778](https://github.com/vllm-project/vllm/issues/35778) **[Bug]: Regression: terrible mixed prefill-decode performance with CUDA graphs enabled.** _(created: 2026-03-02)_
- [#35755](https://github.com/vllm-project/vllm/issues/35755) **[Bug]: AsyncScheduler crashes with AssertionError during Realtime ASR streaming (num_output_placeholders underflow)** _(created: 2026-03-02)_
- [#35743](https://github.com/vllm-project/vllm/issues/35743) **[Bug]: Qwen 3.5 27B AWQ 4bit capturing CUDA graph fails** _(created: 2026-03-02)_
- [#35659](https://github.com/vllm-project/vllm/issues/35659) **[Bug]: cudaErrorIllegalAddress under sustained parallel load with CUDA Graphs on Blackwell SM120 (NVFP4 MoE)** _(created: 2026-03-01)_
- [#35642](https://github.com/vllm-project/vllm/issues/35642) **[Bug]: HIP build in Docker: offload-arch stderr contaminates compiler flags via cmake/utils.cmake and CMAKE_HIP_FLAGS** _(created: 2026-03-01)_
- [#35624](https://github.com/vllm-project/vllm/issues/35624) **[Bug]: Qwen3-Omni Model Fails when try to l** _(created: 2026-02-28)_
- [#35608](https://github.com/vllm-project/vllm/issues/35608) **[Bug]: vllm 0.16.0+image encountered CUDA error: CUBLAS_STATUS_INVALID_VALUE when calling `cublasGemmEx** _(created: 2026-02-28)_
- [#35569](https://github.com/vllm-project/vllm/issues/35569) **[Bug]: [ROCm] ROCM_ATTN backend shows ~8.5% systematic score deviation on Qwen3-VL-Reranker pooling** _(created: 2026-02-28)_
- [#35566](https://github.com/vllm-project/vllm/issues/35566) **CUDA illegal memory access in MoE layer with MiniMax-M2.5 NVFP4 on Blackwell (SM120)** _(created: 2026-02-28)_
- [#35521](https://github.com/vllm-project/vllm/issues/35521) **[Bug]: Suffix decoding crashes with assert total_num_scheduled_tokens > 0** _(created: 2026-02-27)_
- [#35519](https://github.com/vllm-project/vllm/issues/35519) **[Bug]: Qwen3.5 NVFP4 models crash on ARM64 GB10 DGX Spark (CUDA illegal instruction during generation)** _(created: 2026-02-27)_
- [#35407](https://github.com/vllm-project/vllm/issues/35407) **[CI] DBO with DP+EP accuracy regression on GSM8K evaluation** _(created: 2026-02-26)_
- [#35303](https://github.com/vllm-project/vllm/issues/35303) **[Bug] CompressedTensorsWNA16MarlinMoEMethod registers g_idx params unconditionally, crashes with actorder=null AWQ MoE models** _(created: 2026-02-25)_
- [#35285](https://github.com/vllm-project/vllm/issues/35285) **[Bug]: Large Video Request cause vLLM Progress Core Dump** _(created: 2026-02-25)_
- [#35276](https://github.com/vllm-project/vllm/issues/35276) **[Bug]: OpenAI transcribe prompt parameter with whisper return hallucinated transcription** _(created: 2026-02-25)_
- [#35169](https://github.com/vllm-project/vllm/issues/35169) **[Bug]: Memory Access Fault during Step-3.5-Flash inference (ROCM)** _(created: 2026-02-24)_
- [#35091](https://github.com/vllm-project/vllm/issues/35091) **[Bug]: Triton CompilationError in speculative decoding (draft_model)** _(created: 2026-02-23)_
- [#35089](https://github.com/vllm-project/vllm/issues/35089) **[RFC]: In-Tree AMD Zen CPU Backend via zentorch** _(created: 2026-02-23)_
- [#35087](https://github.com/vllm-project/vllm/issues/35087) **[Bug]: DeepSeek 3.2 P/D Disaggregation Support** _(created: 2026-02-23)_
- [#35028](https://github.com/vllm-project/vllm/issues/35028) **[Bug]: RuntimeError: CUDA error: CUBLAS_STATUS_INVALID_VALUE when calling `cublasGemmEx** _(created: 2026-02-21)_
- [#34759](https://github.com/vllm-project/vllm/issues/34759) **[Bug]: nvidia/Llama-3.3-70B-Instruct-NVFP4 Degraded / Gibberish Output with TRITON_ATTN** _(created: 2026-02-17)_
- [#34545](https://github.com/vllm-project/vllm/issues/34545) **[Installation]: unrecognized arguments: --omni** _(created: 2026-02-14)_
- [#34505](https://github.com/vllm-project/vllm/issues/34505) **[Bug]: Qwen 2.5 Omni cuda graph has error** _(created: 2026-02-13)_
- [#34406](https://github.com/vllm-project/vllm/issues/34406) **[Bug]: Instruction following capability is deteriorating：Output introduces parameter  defined in functioncall incorrectly** _(created: 2026-02-12)_
- [#34323](https://github.com/vllm-project/vllm/issues/34323) **[CI Failure]: Spawned tests can fail silently** _(created: 2026-02-11)_
- [#34295](https://github.com/vllm-project/vllm/issues/34295) **[Bug]:** _(created: 2026-02-11)_
- [#34250](https://github.com/vllm-project/vllm/issues/34250) **[Bug]: using vllm on Qwen3-Omni-30B-A3B-Instruct: Failed to apply prompt replacement for mm_items['audio'][0].** _(created: 2026-02-10)_
- [#34210](https://github.com/vllm-project/vllm/issues/34210) **[Bug]: Enable DBO on Qwen3-VL-235B-A22B raise TypeError: 'NoneType' object is not subscriptable** _(created: 2026-02-10)_
- [#34201](https://github.com/vllm-project/vllm/issues/34201) **[Bug]: AttributeError: 'Parameter' object has no attribute 'weight_loader'** _(created: 2026-02-10)_
- [#34154](https://github.com/vllm-project/vllm/issues/34154) **[Bug]: Failing to run DeepSeek-OCR on Radeon GPUs (memory fault)** _(created: 2026-02-09)_
- [#33920](https://github.com/vllm-project/vllm/issues/33920) **[Bug]: GLM-4.7-Flash OOM during sampler warmup with tensor parallelism on RTX 4090** _(created: 2026-02-05)_
- [#33916](https://github.com/vllm-project/vllm/issues/33916) **[Bug] IndexError: list index out of range in chat_completion_stream_generator with --tool-call-parser=mistral during streaming tool calls** _(created: 2026-02-05)_
- [#33913](https://github.com/vllm-project/vllm/issues/33913) **[Usage]: Compile with `TORCH_USE_CUDA_DSA` to enable device-side assertions.** _(created: 2026-02-05)_
- [#33871](https://github.com/vllm-project/vllm/issues/33871) **[Bug]: The local deployment achieves about 30% higher accuracy compared to the server deployment.** _(created: 2026-02-05)_
- [#33865](https://github.com/vllm-project/vllm/issues/33865) **[Bug]: OpenAI-compatible Embeddings API intermittently crashes with multimodal cache assertion (`Expected a cached item for mm_hash`) on Qwen3-VL-Embedding-8B** _(created: 2026-02-05)_
- [#33828](https://github.com/vllm-project/vllm/issues/33828) **[Bug]: mistral3 offline multimodal inference example failing with prompt placeholder error** _(created: 2026-02-04)_
- [#33672](https://github.com/vllm-project/vllm/issues/33672) **[Bug]: HTTP API multimodal embedding causes image_pad token duplication, producing incorrect results** _(created: 2026-02-03)_
- [#33654](https://github.com/vllm-project/vllm/issues/33654) **[Bug]: The content of response from Kimi-K2.5 is empty.** _(created: 2026-02-03)_
- [#33628](https://github.com/vllm-project/vllm/issues/33628) **[Bug]: Failed to run distributed inference with Gloo backend on aarch64** _(created: 2026-02-03)_
- [#33424](https://github.com/vllm-project/vllm/issues/33424) **[Bug]: LoRA MoE Not Matching HF Output** _(created: 2026-01-30)_
- [#33338](https://github.com/vllm-project/vllm/issues/33338) **[Bug]: Crash when using presence_penalty with Qwen3-VL in v0.11.0** _(created: 2026-01-29)_
- [#33336](https://github.com/vllm-project/vllm/issues/33336) **[Bug]: AttributeError: 'Step3VLProcessor' object has no attribute '_get_num_multimodal_tokens'** _(created: 2026-01-29)_
- [#33307](https://github.com/vllm-project/vllm/issues/33307) **[Bug]: Latency spikes at input_len=1024 with batch_size=16 (TP1 & TP2)** _(created: 2026-01-29)_
- [#33204](https://github.com/vllm-project/vllm/issues/33204) **[Bug]: Qwen3-VL-Embedding model produces different embeddings than official qwen_vl_utils implementation** _(created: 2026-01-27)_
- [#33107](https://github.com/vllm-project/vllm/issues/33107) **[Bug]: Whisper large-v3 accuracy degradation in vLLM 0.14.1 (134.56% WER) on L40S - works fine in 0.12.0** _(created: 2026-01-26)_
- [#32925](https://github.com/vllm-project/vllm/issues/32925) **[Bug]: tensorize_vllm_model double gpu** _(created: 2026-01-23)_
- [#32921](https://github.com/vllm-project/vllm/issues/32921) **[Bug]: gpt-oss-20b streaming last reasoning content part into content** _(created: 2026-01-23)_
- [#32897](https://github.com/vllm-project/vllm/issues/32897) **[Bug]: PaddleOCR-VL crash** _(created: 2026-01-23)_
- [#32864](https://github.com/vllm-project/vllm/issues/32864) **[Bug] [ROCm]: Loading Qwen3-MoE-MXFP4 Weights in v0.14.** _(created: 2026-01-22)_
- [#32858](https://github.com/vllm-project/vllm/issues/32858) **[Bug]: Use vllm to obtain Qwen3VL last token hidden status** _(created: 2026-01-22)_
- [#32826](https://github.com/vllm-project/vllm/issues/32826) **[Bug]: MiniMax-M2.1 NVFP4 fails on RTX PRO 6000 Blackwell (SM120) with expert parallel** _(created: 2026-01-22)_
- [#32732](https://github.com/vllm-project/vllm/issues/32732) **[Bug]: Regression in v0.14.0: "No valid attention backend found" for nvidia/DeepSeek-R1-0528-NVFP4 on RTX Pro 6000 (Blackwell)** _(created: 2026-01-20)_
- [#32685](https://github.com/vllm-project/vllm/issues/32685) **[Feature]: Support multi-modal inputs for OpenAI Response API** _(created: 2026-01-20)_
- [#32659](https://github.com/vllm-project/vllm/issues/32659) **[RFC]: Tracking follow-up progress on Encode–Prefill–Decode Disaggregation** _(created: 2026-01-20)_
- [#32636](https://github.com/vllm-project/vllm/issues/32636) **[Bug]: Invalid base64-encoded string for audio input** _(created: 2026-01-20)_
- [#32604](https://github.com/vllm-project/vllm/issues/32604) **[Bug]: LMCache CPU kv offload cause decode speed degrade** _(created: 2026-01-19)_
- [#32600](https://github.com/vllm-project/vllm/issues/32600) **[Bug]:** _(created: 2026-01-19)_
- [#32588](https://github.com/vllm-project/vllm/issues/32588) **[Bug]: Wrong timestamps if audio > 30s** _(created: 2026-01-19)_
- [#32468](https://github.com/vllm-project/vllm/issues/32468) **[Bug]: Engine core proc EngineCore_DP0 died unexpectedly, shutting down client.** _(created: 2026-01-16)_
- [#32410](https://github.com/vllm-project/vllm/issues/32410) **[Bug][XPU]: Failed to serve w4a16 Qwen3 VL MoE on Intel Arc Pro B60** _(created: 2026-01-15)_
- [#32338](https://github.com/vllm-project/vllm/issues/32338) **[Feature]: support for LlavaQwenForCausalLM** _(created: 2026-01-14)_
- [#32259](https://github.com/vllm-project/vllm/issues/32259) **[Bug]: offline infer of mm model cache** _(created: 2026-01-13)_
- [#32193](https://github.com/vllm-project/vllm/issues/32193) **[Bug]: vLLM engine crash under burst load despite expected request queuing (72 concurrent API calls)** _(created: 2026-01-12)_
- [#32176](https://github.com/vllm-project/vllm/issues/32176) **[Bug]: deepseekv3.2 core dumped with cpu_offload_gb** _(created: 2026-01-12)_
- [#32151](https://github.com/vllm-project/vllm/issues/32151) **[Bug]: jina-reranker-m0 infer error** _(created: 2026-01-12)_
- [#32113](https://github.com/vllm-project/vllm/issues/32113) **[Bug]: Qwen3-VL-8B inference with video** _(created: 2026-01-11)_
- [#32068](https://github.com/vllm-project/vllm/issues/32068) **[Bug]: Recompile in LLama model** _(created: 2026-01-10)_
- [#32012](https://github.com/vllm-project/vllm/issues/32012) **[Bug]: Qwen3-VL-MoE fails loading when using bitsandbytes quantization** _(created: 2026-01-09)_
- [#31961](https://github.com/vllm-project/vllm/issues/31961) **[Performance]: EPD Disaggregation Performance Testing Scripts** _(created: 2026-01-08)_
- [#31936](https://github.com/vllm-project/vllm/issues/31936) **[Bug]: run deepseek v3.2 failed，not support RTX PRO 6000 * 8？** _(created: 2026-01-08)_
- [#31884](https://github.com/vllm-project/vllm/issues/31884) **[Bug]: run Qwen3-30B-A3B on 8*A800 2  nodes with DP failed zmq.error.ZMQError** _(created: 2026-01-07)_
- [#31871](https://github.com/vllm-project/vllm/issues/31871) **[Bug]: Streaming mode with --tool-call-parser hermes returns raw text instead of parsed tool_calls** _(created: 2026-01-07)_
- [#31782](https://github.com/vllm-project/vllm/issues/31782) **[Feature]: Support compressed-tensors NVFP4 quantization for MoE models (Nemotron-H non-gated MoE)** _(created: 2026-01-06)_
- [#31709](https://github.com/vllm-project/vllm/issues/31709) **[Bug]: After upgrade to 0.11.2, vllm crashs with Qwen3.** _(created: 2026-01-05)_
- [#31661](https://github.com/vllm-project/vllm/issues/31661) **[Bug]: jina-reranker-m0 [image_index]   IndexError: list index out of range** _(created: 2026-01-04)_
- [#31647](https://github.com/vllm-project/vllm/issues/31647) **[Bug]: TypeError: BatchPrefillWithPagedKVCacheWrapper.plan() got an unexpected keyword argument 'fixed_split_size'** _(created: 2026-01-03)_
- [#31484](https://github.com/vllm-project/vllm/issues/31484) **[Usage]: RuntimeError when running Qwen2.5-VL-7B-Instruct with vllm: Potential version incompatibility** _(created: 2025-12-29)_
- [#31372](https://github.com/vllm-project/vllm/issues/31372) **[Feature]: Running paddocr‑vl consumes an excessively large amount of  memory.** _(created: 2025-12-26)_
- [#31353](https://github.com/vllm-project/vllm/issues/31353) **[Bug]: KV Cache grows continuously with just one chat completion request using meta-llama/Llama-3.2-1B on L40 GPU with Flash Attention and finally completed after 10 minutes** _(created: 2025-12-25)_
- [#31344](https://github.com/vllm-project/vllm/issues/31344) **[Usage]: how to pass param logits_processors in  AsyncEngineArgs?** _(created: 2025-12-25)_
- [#31205](https://github.com/vllm-project/vllm/issues/31205) **ValueError: Qwen3OmniMoeThinkerForConditionalGeneration does not support LoRA yet.** _(created: 2025-12-23)_
- [#31063](https://github.com/vllm-project/vllm/issues/31063) **[Bug]: Qwen3VL-8B-instruct-FP8 has larger result diff rate than sglang compared with transformers** _(created: 2025-12-20)_
- [#30839](https://github.com/vllm-project/vllm/issues/30839) **[RFC]: Enabling Zero-Copy Video with PyNvVideoCodec and IPC** _(created: 2025-12-17)_
- [#30819](https://github.com/vllm-project/vllm/issues/30819) **[Bug]: vLLM inference stuck when requesting video description on VLM models** _(created: 2025-12-16)_
- [#30779](https://github.com/vllm-project/vllm/issues/30779) **[Bug]: v0.11.2 can not support Qwen2.5-Omni-** _(created: 2025-12-16)_
- [#30777](https://github.com/vllm-project/vllm/issues/30777) **[Bug]: whisper-large-v3-turbo have accuracy problem on nightly build** _(created: 2025-12-16)_
- [#30493](https://github.com/vllm-project/vllm/issues/30493) **[Bug]: 5090 RTX seems to be broken** _(created: 2025-12-11)_
- [#30180](https://github.com/vllm-project/vllm/issues/30180) **[Bug]: Inference of Qwen3-VL-235B failed** _(created: 2025-12-06)_
- [#30167](https://github.com/vllm-project/vllm/issues/30167) **[Bug][ROCm]: `vision_embeddings` in transformers inaccurate without math SDP** _(created: 2025-12-06)_
- [#29968](https://github.com/vllm-project/vllm/issues/29968) **[Bug]: Ministral 3 - streaming tool call not working** _(created: 2025-12-03)_
- [#29863](https://github.com/vllm-project/vllm/issues/29863) **[Bug]: fails to inference prompt ends with '.' ':' for video inputs** _(created: 2025-12-02)_
- [#29814](https://github.com/vllm-project/vllm/issues/29814) **[Bug]: Qwen3-VL-235B-A22B-Instruct-FP8 tools doesn't respond in the tools but in content for hermes parswer** _(created: 2025-12-01)_
- [#29373](https://github.com/vllm-project/vllm/issues/29373) **[Bug]: Multinode inference request with Ray and vLLM crashes - regression from vLLM v0.7.3** _(created: 2025-11-25)_
- [#29369](https://github.com/vllm-project/vllm/issues/29369) **[Bug]: nixl connector PD disagg stuck at INFO logging level** _(created: 2025-11-25)_
- [#29362](https://github.com/vllm-project/vllm/issues/29362) **[RFC]: Resettle examples.** _(created: 2025-11-25)_
- [#29174](https://github.com/vllm-project/vllm/issues/29174) **[Bug]: Qwen3 Omni thinking unstable output** _(created: 2025-11-21)_
- [#28919](https://github.com/vllm-project/vllm/issues/28919) **[Bug]: Qwen3-32B use eagle3 crash** _(created: 2025-11-18)_
- [#28707](https://github.com/vllm-project/vllm/issues/28707) **[Bug]: run fail in nvidia b300** _(created: 2025-11-14)_
- [#28691](https://github.com/vllm-project/vllm/issues/28691) **[Feature]: GGUF model with architecture qwen3vlmoe is not supported yet.** _(created: 2025-11-13)_
- [#28640](https://github.com/vllm-project/vllm/issues/28640) **[Bug]: LoRA/Adapter Loading Error with Qwen3-VL-8B-Instruct Multimodal Model in vLLM Deployment (AssertionError in lora_shrink_op)** _(created: 2025-11-13)_
- [#28568](https://github.com/vllm-project/vllm/issues/28568) **[Bug]: FlashInfer attention backend on Hopper fails with llama4-scout and llama3 with fp8 kvcache** _(created: 2025-11-12)_
- [#28467](https://github.com/vllm-project/vllm/issues/28467) **[Bug]: Persistent CPU RAM Growth and Container Restarts with QWEN2-2B on SageMaker, Unaffected by Cache Settings** _(created: 2025-11-11)_
- [#28388](https://github.com/vllm-project/vllm/issues/28388) **[Bug]: 新版的vllm已经废弃了v0代码，而对qwen-omni系列的模型支持仅限于v0，似乎是因为这个原因，我们无法使用最新版的vllm推理qwen-omni模型** _(created: 2025-11-10)_
- [#28375](https://github.com/vllm-project/vllm/issues/28375) **[Bug]: Engine stuck at CPU when running long video input on Qwen3-vl model** _(created: 2025-11-10)_
- [#28307](https://github.com/vllm-project/vllm/issues/28307) **[Bug]: `repetition_penalty` leads to engine failure when using vllm serve...** _(created: 2025-11-07)_
- [#27932](https://github.com/vllm-project/vllm/issues/27932) **[Feature]: Qwen3 Omini AttributeError: 'Qwen3OmniMoeProcessor' object has no attribute '_get_num_multimodal_tokens'** _(created: 2025-11-02)_
- [#27557](https://github.com/vllm-project/vllm/issues/27557) **[Bug]: Engine core proc EngineCore_DP0 died unexpectedly, shutting down client.** _(created: 2025-10-27)_
- [#27430](https://github.com/vllm-project/vllm/issues/27430) **[Bug]: vLLM (TP=8) on 235B model triggers "CUDA error: unspecified launch failure" and persistent "ERR!" state in nvidia-smi** _(created: 2025-10-23)_
- [#27364](https://github.com/vllm-project/vllm/issues/27364) **[Bug]: Qwen3-VL {4B,8B} FP8 on vLLM returns only exclamation marks ("!!!!!...") on Jetson Thor** _(created: 2025-10-22)_
- [#27340](https://github.com/vllm-project/vllm/issues/27340) **[Bug]: Qwen3-VL-2B-Instruct vllm推理报错** _(created: 2025-10-22)_
- [#27120](https://github.com/vllm-project/vllm/issues/27120) **[Bug]: some models won't stream from s3: Qwen/Qwen3-VL-30B-A3B-Instruct-FP8** _(created: 2025-10-17)_
- [#27116](https://github.com/vllm-project/vllm/issues/27116) **[Bug]: vLLM failure (ray.exceptions.RayChannelTimeoutError) observed since v0.10.2, with tp=8, pp=4** _(created: 2025-10-17)_
- [#26777](https://github.com/vllm-project/vllm/issues/26777) **[Bug]: Deploying Qwen2.5-VL-7B with Eagle3, the service crashes when concurrency increases.** _(created: 2025-10-14)_
- [#26480](https://github.com/vllm-project/vllm/issues/26480) **[Bug][v0.11.0]: gpt-oss-120b generates with no output** _(created: 2025-10-09)_
- [#26420](https://github.com/vllm-project/vllm/issues/26420) **[Bug]: vLLM server not starting when running Qwen/Qwen3-VL-30B-A3B-Instruct on 2 A6000 GPUs. ** _(created: 2025-10-08)_
- [#25502](https://github.com/vllm-project/vllm/issues/25502) **[Bug]: This flash attention build does not support tanh softcapping: gemma-2-2b-it on H100 NVL** _(created: 2025-09-23)_
- [#25192](https://github.com/vllm-project/vllm/issues/25192) **[Bug]: Make Whisper model compatible with cuda graphs** _(created: 2025-09-18)_
- [#25172](https://github.com/vllm-project/vllm/issues/25172) **[Feature]: FlexAttention + encoder_decoder support** _(created: 2025-09-18)_
- [#24430](https://github.com/vllm-project/vllm/issues/24430) **[Bug]: torch.AcceleratorError: CUDA error: an illegal memory access was encountered** _(created: 2025-09-08)_
- [#24289](https://github.com/vllm-project/vllm/issues/24289) **[Bug]: module 'triton.language' has no attribute 'constexpr_function'** _(created: 2025-09-05)_
- [#23404](https://github.com/vllm-project/vllm/issues/23404) **[Bug]: Qwen3 vLLM Structured Output Ignores Field Descriptions** _(created: 2025-08-22)_
- [#22817](https://github.com/vllm-project/vllm/issues/22817) **[RFC]: Disaggregated Everything - Token In <> Token Out API Server** _(created: 2025-08-13)_
- [#22424](https://github.com/vllm-project/vllm/issues/22424) **[Bug]: Voxtral-Small-24B-2507 Does Not Support Pipeline-Parallel** _(created: 2025-08-07)_
- [#21995](https://github.com/vllm-project/vllm/issues/21995) **[RFC] Run HF processing on GPU** _(created: 2025-07-31)_
- [#20261](https://github.com/vllm-project/vllm/issues/20261) **[Bug]: Prefix caching ignores visual input, causing incorrect multimodal outputs under concurrency** _(created: 2025-06-30)_
- [#20149](https://github.com/vllm-project/vllm/issues/20149) **[Feature]: Add Support for Updating Lora Weights** _(created: 2025-06-26)_
- [#19668](https://github.com/vllm-project/vllm/issues/19668) **[Bug]: vllm, EngineCore encountered a fatal error TimeoutError** _(created: 2025-06-15)_
- [#19445](https://github.com/vllm-project/vllm/issues/19445) **[Bug]: Docker image CUDA error on RTX 2080 Ti** _(created: 2025-06-10)_
- [#17972](https://github.com/vllm-project/vllm/issues/17972) **[Bug]: vLLM server hangs and timeouts after initial requests** _(created: 2025-05-12)_
- [#16966](https://github.com/vllm-project/vllm/issues/16966) **[Bug]: vllm 0.8.4 whisper possible memory leak?** _(created: 2025-04-22)_
- [#13941](https://github.com/vllm-project/vllm/issues/13941) **[Bug]: wake up OOM (72B model in 8*A800(40G))** _(created: 2025-02-27)_

### Feature Request / 功能请求 (178)

- [#40856](https://github.com/vllm-project/vllm/issues/40856) **[Bug]: VLLM running qwen3.6 for image inference occasionally reports 500 Internal Server Error** _(created: 2026-04-25)_
- [#40802](https://github.com/vllm-project/vllm/issues/40802) **[Feature]: Deepseek V4 cannot run ,Please support SM120 GPU,example rtx5090  rtxpro6000** _(created: 2026-04-24)_
- [#40781](https://github.com/vllm-project/vllm/issues/40781) **[Feature]: vllm support audio otel tracing?** _(created: 2026-04-24)_
- [#40742](https://github.com/vllm-project/vllm/issues/40742) **[Bug]: CUDA graph capture crashes during startup due to Inductor autotuning torch.cuda.synchronize() inside graph capture (FULL_DECODE_ONLY + MLA + FP8) when PDL is enabled** _(created: 2026-04-23)_
- [#40741](https://github.com/vllm-project/vllm/issues/40741) **[Feature]: Make opencv-python-headless an optional dependency for FIPS compliance** _(created: 2026-04-23)_
- [#40707](https://github.com/vllm-project/vllm/issues/40707) **[Bug]: Scheduling deadlock in _mamba_block_aligned_split with multiple large multimodal inputs on hybrid Mamba models** _(created: 2026-04-23)_
- [#40652](https://github.com/vllm-project/vllm/issues/40652) **[Bug]: Kimi 2.6 on 8x A100 SMX4 leads to NVLink Crash Coredump** _(created: 2026-04-22)_
- [#40620](https://github.com/vllm-project/vllm/issues/40620) **[RFC]: Unified Device Capability Abstraction for Cross-Platform Feature Detection** _(created: 2026-04-22)_
- [#40585](https://github.com/vllm-project/vllm/issues/40585) **[Bug]: qwen3.5 can not use --decode-context-parallel-size with --enable-prefix-caching** _(created: 2026-04-22)_
- [#40543](https://github.com/vllm-project/vllm/issues/40543) **[Feature]: Dynamic PDL Enablement** _(created: 2026-04-21)_
- [#40536](https://github.com/vllm-project/vllm/issues/40536) **[Feature]: Support iterative in-place weight editing on TP workers (online RLHF / steering / abliteration)** _(created: 2026-04-21)_
- [#40443](https://github.com/vllm-project/vllm/issues/40443) **[BUG] Port-allocation race between ApiServer processes in hybrid-LB mode (ZMQError: Address already in use)** _(created: 2026-04-21)_
- [#40420](https://github.com/vllm-project/vllm/issues/40420) **[Bug]: TurboQuant `_continuation_prefill` OOMs and kills engine at long-context prefill (~185K actual tokens)** _(created: 2026-04-21)_
- [#40417](https://github.com/vllm-project/vllm/issues/40417) **[Bug]: `token_capacity_kv_cache_groups` (#40384) should also exclude `SlidingWindowSpec` / `ChunkedLocalAttentionSpec`** _(created: 2026-04-21)_
- [#40381](https://github.com/vllm-project/vllm/issues/40381) **[Bug]: Buffer overflow when allocating memory error on Qwen3.5-122B-A10B-GPTQ-Int4 and NVFP4** _(created: 2026-04-20)_
- [#40343](https://github.com/vllm-project/vllm/issues/40343) **[Installation]: 有提供cuda 12.6+python3.12的vllm预编译的whl包吗？ 以开发者模型需要本地安装下，发布的都是cuda13版本的，不适配cuda12.6的本机的版本型号** _(created: 2026-04-20)_
- [#40113](https://github.com/vllm-project/vllm/issues/40113) **[CI Failure]: Multi-Modal Processor (CPU)** _(created: 2026-04-17)_
- [#40106](https://github.com/vllm-project/vllm/issues/40106) **[Bug]: Gemma4 multimodal: missing vision-aware bidirectional attention mask for use_bidirectional_attention="vision" models** _(created: 2026-04-17)_
- [#40095](https://github.com/vllm-project/vllm/issues/40095) **[Bug]: Gemma4MultimodalEmbedder normalization order different from Transformers, causing bad audio inference** _(created: 2026-04-17)_
- [#40047](https://github.com/vllm-project/vllm/issues/40047) **[Bug][Tracking Issue]: NaNs in CUDA Graph padding regions corrupt activations in some per-token kernels** _(created: 2026-04-16)_
- [#40038](https://github.com/vllm-project/vllm/issues/40038) **[Bug]: cudaErrorIllegalAddress during PIECEWISE CUDA graph replay with MoE LoRA: stale buffer addresses in `moe_lora_align_block_size`** _(created: 2026-04-16)_
- [#40002](https://github.com/vllm-project/vllm/issues/40002) **[Bug]: Inconsistent KV Cache reporting and system hang on long context requests (Gemma-4 26B AWQ Int4)** _(created: 2026-04-16)_
- [#39993](https://github.com/vllm-project/vllm/issues/39993) **[Usage]: does vllm support Qwen3_5ForCausalLM architecture inference? not just Qwen3_5ForConditionalGeneration?** _(created: 2026-04-16)_
- [#39979](https://github.com/vllm-project/vllm/issues/39979) **[RFC]: Ultimate Better Observability.** _(created: 2026-04-16)_
- [#39919](https://github.com/vllm-project/vllm/issues/39919) **[Bug]: DeepSeek OCR doesn't work on vllm 0.19** _(created: 2026-04-15)_
- [#39839](https://github.com/vllm-project/vllm/issues/39839) **[Bug]:  MTP DeepSeek and Eagle Flash Attention Failures in Spec Decode Unit Tests** _(created: 2026-04-14)_
- [#39766](https://github.com/vllm-project/vllm/issues/39766) **[RFC]: Support Mooncake Based ECConnector for EPD** _(created: 2026-04-14)_
- [#39735](https://github.com/vllm-project/vllm/issues/39735) **[Feature]: Expose Word-Level Timestamps in `/v1/realtime` API for Voxtral Realtime** _(created: 2026-04-13)_
- [#39722](https://github.com/vllm-project/vllm/issues/39722) **Gibberish with flashinfer_nvlink_two_sided on GB200/arm64** _(created: 2026-04-13)_
- [#39708](https://github.com/vllm-project/vllm/issues/39708) **[Feature]: Pre-ViT visual token pruning for VLMs (PixelPrune)** _(created: 2026-04-13)_
- [#39701](https://github.com/vllm-project/vllm/issues/39701) **[RFC] Replace routing replay with CUDA-graph-compatible device cache approach** _(created: 2026-04-13)_
- [#39682](https://github.com/vllm-project/vllm/issues/39682) **[BUGS] vLLM V1 Engine Hangs After Weight Loading on Blackwell (sm_121) Multi-Node Ray Setup (TP=2)** _(created: 2026-04-13)_
- [#39681](https://github.com/vllm-project/vllm/issues/39681) **[Bug]: Gemma4 multimodal crashes with "pixel_values contains inconsistent shapes" when concurrent image requests have different resolutions** _(created: 2026-04-13)_
- [#39680](https://github.com/vllm-project/vllm/issues/39680) **[Performance]: Qwen3.5 with mtp is slower than without** _(created: 2026-04-13)_
- [#39583](https://github.com/vllm-project/vllm/issues/39583) **[RFC]: Deprecate bitsandbytes and GGUF quantization support** _(created: 2026-04-11)_
- [#39504](https://github.com/vllm-project/vllm/issues/39504) **[RFC]: Enable prompt_embeds content parts in Chat Completions API** _(created: 2026-04-10)_
- [#39408](https://github.com/vllm-project/vllm/issues/39408) **[Usage]: qwen3-asr-1.7b pre-allocated encoder cache size limit** _(created: 2026-04-09)_
- [#39210](https://github.com/vllm-project/vllm/issues/39210) **[Bug] Embedding/pooling models crash on B200 (SM 10.0) — encoder attention hardcodes FA2 which lacks SM100 support** _(created: 2026-04-07)_
- [#39158](https://github.com/vllm-project/vllm/issues/39158) **[RFC][Test]: Unified Platform-Aware Test Skip Mechanism** _(created: 2026-04-07)_
- [#39137](https://github.com/vllm-project/vllm/issues/39137) **[Bug]: fp8_e5m2 kv-cache gate in _init_kv_cache_quant fires on any quantized checkpoint, not only fp8 checkpoints** _(created: 2026-04-07)_
- [#39076](https://github.com/vllm-project/vllm/issues/39076) **[RFC]: Entropy-Gated Online KV Block Expiration During Active Decode** _(created: 2026-04-06)_
- [#39015](https://github.com/vllm-project/vllm/issues/39015) **[Bug]: torch.distributed.DistNetworkError: The server socket has failed to listen on any local network address. port: 29500, useIpv6: false, code: -98, name: EADDRINUSE, message: address already in use** _(created: 2026-04-05)_
- [#38999](https://github.com/vllm-project/vllm/issues/38999) **[Bug]: Gemma 4 MoE (26B-A4B) crashes with `--data-parallel-size > 1` — AssertionError in cuda_communicator all_gather** _(created: 2026-04-04)_
- [#38925](https://github.com/vllm-project/vllm/issues/38925) **[Feature]: Support lightweight import of vllm protocol types without torch dependency** _(created: 2026-04-03)_
- [#38918](https://github.com/vllm-project/vllm/issues/38918) **[Usage]: Gemma4 on Turing GPUs (SM 7.5): all attention backends hit shared memory limits** _(created: 2026-04-03)_
- [#38903](https://github.com/vllm-project/vllm/issues/38903) **[Bug]: Cross-request context contamination with async scheduling + pipeline parallelism on multi-node** _(created: 2026-04-03)_
- [#38886](https://github.com/vllm-project/vllm/issues/38886) **[Bug]: Gemma 4 E4B weight loading fails `Gemma4ClippableLinear` parameter `input_max` not recognized** _(created: 2026-04-03)_
- [#38809](https://github.com/vllm-project/vllm/issues/38809) **[Feature]: How to disable chat template when using vllm serve** _(created: 2026-04-02)_
- [#38760](https://github.com/vllm-project/vllm/issues/38760) **[RFC]: Per-iteration forward pass metrics with accurate engine-level timing** _(created: 2026-04-01)_
- [#38754](https://github.com/vllm-project/vllm/issues/38754) **[Bug]: GPT OSS Router GEMM Causing NaNs** _(created: 2026-04-01)_
- [#38740](https://github.com/vllm-project/vllm/issues/38740) **[Transformers v5] NemotronParseForConditionalGeneration** _(created: 2026-04-01)_
- [#38658](https://github.com/vllm-project/vllm/issues/38658) **[Bug]: MLA attention casts activations to int32 when using Marlin FP8 on GPUs without native FP8 support (sm < 89)** _(created: 2026-03-31)_
- [#38603](https://github.com/vllm-project/vllm/issues/38603) **[Bug]: Streaming last chunk contains non-empty tool_calls with empty fields "type" causing type validation error ** _(created: 2026-03-31)_
- [#38591](https://github.com/vllm-project/vllm/issues/38591) **Bug: ValueError: too many values to unpack in dispatch_cpu_unquantized_gemm when loading Qwen3.5-4B** _(created: 2026-03-30)_
- [#38560](https://github.com/vllm-project/vllm/issues/38560) **[Bug]: reasoning_effort passed to MistralCommonTokenizer.apply_chat_template breaks Mistral Small 4 chat completions on vLLM 0.18.0** _(created: 2026-03-30)_
- [#38531](https://github.com/vllm-project/vllm/issues/38531) **[Feature]: Support direct binary/multipart file upload for video and image in OpenAI-compatible API** _(created: 2026-03-30)_
- [#38425](https://github.com/vllm-project/vllm/issues/38425) **[Transformers v5] InternVL2** _(created: 2026-03-28)_
- [#38411](https://github.com/vllm-project/vllm/issues/38411) **[Bug]: Vision encoder crashes on SM100 (Jetson Thor) — `_vllm_fa2_C` compiled for SM80-only, no override available for vision encoder** _(created: 2026-03-28)_
- [#38389](https://github.com/vllm-project/vllm/issues/38389) **[Transformers v5] IsaacForConditionalGeneration** _(created: 2026-03-27)_
- [#38379](https://github.com/vllm-project/vllm/issues/38379) **Upgrade to Transformers v5** _(created: 2026-03-27)_
- [#38351](https://github.com/vllm-project/vllm/issues/38351) **[Bug]: When use_audio_in_video is enabled in qwen3-omni, the output may exhibit issues such as empty or repetitive output.** _(created: 2026-03-27)_
- [#38297](https://github.com/vllm-project/vllm/issues/38297) **[Bug]: Gemma3n concurrent audio requests crash EngineCore — missing dynamic_dims on audio sequence dimension** _(created: 2026-03-27)_
- [#38235](https://github.com/vllm-project/vllm/issues/38235) **[Feature]: Quantization support (AWQ / GPTQ / FP8) for mistralai/Voxtral-Mini-4B-Realtime-2602** _(created: 2026-03-26)_
- [#38175](https://github.com/vllm-project/vllm/issues/38175) **[RFC]: Support ViT Full CUDA Graph (Tracker)** _(created: 2026-03-26)_
- [#38079](https://github.com/vllm-project/vllm/issues/38079) **[RFC] Redesign enable_return_routed_experts to avoid blocking EngineCore event loop** _(created: 2026-03-25)_
- [#38004](https://github.com/vllm-project/vllm/issues/38004) **[Bug]: Speech-to-Text endpoint may return 501 but not documented in OpenAPI** _(created: 2026-03-24)_
- [#37828](https://github.com/vllm-project/vllm/issues/37828) **[Bug]: Intel ARC 140v not supported as XE2 cutlass kernel** _(created: 2026-03-22)_
- [#37737](https://github.com/vllm-project/vllm/issues/37737) **[Bug]: Missing logprobs for `<tool_call>` in streaming chat completions** _(created: 2026-03-21)_
- [#37602](https://github.com/vllm-project/vllm/issues/37602) **[Bug]: Qwen3.5-122B-A10B-FP8 EngineCore crash on concurrent image requests** _(created: 2026-03-19)_
- [#37563](https://github.com/vllm-project/vllm/issues/37563) **mm_fp4 trtllm backend leaks padding scales into real rows (use_8x4_sf_layout=True)** _(created: 2026-03-19)_
- [#37553](https://github.com/vllm-project/vllm/issues/37553) **[Bug]: Mistral-Small-4-119B-2603 fails on 8x RTX 3090 (SM 8.6) with vLLM v0.17.1: no valid MLA attention backend** _(created: 2026-03-19)_
- [#37423](https://github.com/vllm-project/vllm/issues/37423) **[Feature]: Allow passing `images` to CompletionRequest** _(created: 2026-03-18)_
- [#37367](https://github.com/vllm-project/vllm/issues/37367) **[Bug]: gcc: internal compiler error: Segmentation fault signal terminated program cc1** _(created: 2026-03-18)_
- [#37363](https://github.com/vllm-project/vllm/issues/37363) **fix(compilation): fix piecewise CUDA graph bugs with splitting_ops** _(created: 2026-03-18)_
- [#37175](https://github.com/vllm-project/vllm/issues/37175) **[Usage]:wen3.5-35B-A3B (FP8) with vLLM 0.17.1 , the first request takes significantly longer than subsequent requests** _(created: 2026-03-16)_
- [#37140](https://github.com/vllm-project/vllm/issues/37140) **[Usage]: CUDA error: the provided PTX was compiled with an unsupported toolchain.** _(created: 2026-03-16)_
- [#37075](https://github.com/vllm-project/vllm/issues/37075) **[RFC]: Opt-in Media URL Cache for `MediaConnector`** _(created: 2026-03-14)_
- [#37035](https://github.com/vllm-project/vllm/issues/37035) **[Bug]: cudaErrorIllegalAddress in gdn_attn.py:237 when using qwen3_next_mtp with num_speculative_tokens=5 under load** _(created: 2026-03-14)_
- [#36998](https://github.com/vllm-project/vllm/issues/36998) **[RFC]: Observation Plugin for Intercepting & Routing on Activations** _(created: 2026-03-13)_
- [#36872](https://github.com/vllm-project/vllm/issues/36872) **[Bug]: Gibberish output and collapsing generation throughput with Qwen3.5-35B-A3B-FP8 and speculative decoding enabled** _(created: 2026-03-12)_
- [#36852](https://github.com/vllm-project/vllm/issues/36852) **[Bug]: GPU failure during repeated model loading when using --enable-prefix-caching with KV transfer (LMCacheConnectorV1)** _(created: 2026-03-12)_
- [#36781](https://github.com/vllm-project/vllm/issues/36781) **[Bug]: vLLM 0.17.0 failed to serve Qwen3-30B-A3B-Instruct-2507 after adding `--enable_lora`** _(created: 2026-03-11)_
- [#36780](https://github.com/vllm-project/vllm/issues/36780) **[RFC][NixlConnector]: Add support for hybrid SSM-FA models** _(created: 2026-03-11)_
- [#36748](https://github.com/vllm-project/vllm/issues/36748) **[Bug]: In DP mode, waiting request stack in a few DP ranks.** _(created: 2026-03-11)_
- [#36627](https://github.com/vllm-project/vllm/issues/36627) **[Performance]: qwen3.5 vs qwen3** _(created: 2026-03-10)_
- [#36589](https://github.com/vllm-project/vllm/issues/36589) **[Bug]: SM 7.5 extreme slowness hangs indefinitely on T4 (vllm 0.17.0 with Qwen3.5-27B)** _(created: 2026-03-10)_
- [#36566](https://github.com/vllm-project/vllm/issues/36566) **[Bug]:Qwen3.5-35B-A3B vllm v0.17.0 ERROR 03-10 00:52:24 [multiproc_executor.py:261] Worker proc VllmWorker-0 died unexpectedly, shutting down executor.** _(created: 2026-03-10)_
- [#36450](https://github.com/vllm-project/vllm/issues/36450) **[Bug]: Qwen3.5 AWQ models crash during inference on RTX 5090 (Blackwell) with Triton OOM in solve_tril despite successful model load** _(created: 2026-03-09)_
- [#36350](https://github.com/vllm-project/vllm/issues/36350) **[Bug]: Qwen 3.5 4B fail on first request on Intel XPU (Arc Graphics B580)** _(created: 2026-03-07)_
- [#36328](https://github.com/vllm-project/vllm/issues/36328) **[Feature]: Include mm_hash and mm_transfer_params in Disaggregated Encoder response to prevent redundant data fetching** _(created: 2026-03-07)_
- [#36193](https://github.com/vllm-project/vllm/issues/36193) **[Bug]: Unsupported Activation Function for Step-3.5-Flash** _(created: 2026-03-06)_
- [#35980](https://github.com/vllm-project/vllm/issues/35980) **[Bug]: Why does deploying Qwen3-32B-AWQ via vllm:v0.10.1.1 result in different outputs for the same input?** _(created: 2026-03-04)_
- [#35944](https://github.com/vllm-project/vllm/issues/35944) **[Bug]: jetson agx thor报错** _(created: 2026-03-04)_
- [#35908](https://github.com/vllm-project/vllm/issues/35908) **[RFC]: Model-specific realtime streaming abstraction** _(created: 2026-03-03)_
- [#35778](https://github.com/vllm-project/vllm/issues/35778) **[Bug]: Regression: terrible mixed prefill-decode performance with CUDA graphs enabled.** _(created: 2026-03-02)_
- [#35771](https://github.com/vllm-project/vllm/issues/35771) **[RFC][torch.compile]: Disable Sequence Parallelism (SP) for piecewise compilation** _(created: 2026-03-02)_
- [#35767](https://github.com/vllm-project/vllm/issues/35767) **[Enhancement]: Qwen3-ASR realtime endpoint produces degraded output — stateless segments, no cross-segment context, raw format leaks** _(created: 2026-03-02)_
- [#35755](https://github.com/vllm-project/vllm/issues/35755) **[Bug]: AsyncScheduler crashes with AssertionError during Realtime ASR streaming (num_output_placeholders underflow)** _(created: 2026-03-02)_
- [#35659](https://github.com/vllm-project/vllm/issues/35659) **[Bug]: cudaErrorIllegalAddress under sustained parallel load with CUDA Graphs on Blackwell SM120 (NVFP4 MoE)** _(created: 2026-03-01)_
- [#35566](https://github.com/vllm-project/vllm/issues/35566) **CUDA illegal memory access in MoE layer with MiniMax-M2.5 NVFP4 on Blackwell (SM120)** _(created: 2026-02-28)_
- [#35467](https://github.com/vllm-project/vllm/issues/35467) **[Performance]: non-optimal performance of `linear` for medium batches** _(created: 2026-02-27)_
- [#35407](https://github.com/vllm-project/vllm/issues/35407) **[CI] DBO with DP+EP accuracy regression on GSM8K evaluation** _(created: 2026-02-26)_
- [#35285](https://github.com/vllm-project/vllm/issues/35285) **[Bug]: Large Video Request cause vLLM Progress Core Dump** _(created: 2026-02-25)_
- [#35272](https://github.com/vllm-project/vllm/issues/35272) **[Feature]: Make the Qwen3-ASR use `request_prompt` from user input** _(created: 2026-02-25)_
- [#35254](https://github.com/vllm-project/vllm/issues/35254) **[Performance]: curious new kernels from vllm 0.11.1** _(created: 2026-02-25)_
- [#35141](https://github.com/vllm-project/vllm/issues/35141) **[Feature]: Sequence Parallel Support for Model Runner V2** _(created: 2026-02-23)_
- [#35089](https://github.com/vllm-project/vllm/issues/35089) **[RFC]: In-Tree AMD Zen CPU Backend via zentorch** _(created: 2026-02-23)_
- [#35087](https://github.com/vllm-project/vllm/issues/35087) **[Bug]: DeepSeek 3.2 P/D Disaggregation Support** _(created: 2026-02-23)_
- [#34994](https://github.com/vllm-project/vllm/issues/34994) **[Feature]: Infrastructure Improvements for ROCm CI** _(created: 2026-02-20)_
- [#34536](https://github.com/vllm-project/vllm/issues/34536) **[Feature]: Reasoning output for offline inference** _(created: 2026-02-13)_
- [#34518](https://github.com/vllm-project/vllm/issues/34518) **[Feature]: [Whisper] Support for decoder prefix and custom task tokens in transcription API** _(created: 2026-02-13)_
- [#34406](https://github.com/vllm-project/vllm/issues/34406) **[Bug]: Instruction following capability is deteriorating：Output introduces parameter  defined in functioncall incorrectly** _(created: 2026-02-12)_
- [#34212](https://github.com/vllm-project/vllm/issues/34212) **[Performance]: W4Afp8 is slower than FP8-W8A8** _(created: 2026-02-10)_
- [#34210](https://github.com/vllm-project/vllm/issues/34210) **[Bug]: Enable DBO on Qwen3-VL-235B-A22B raise TypeError: 'NoneType' object is not subscriptable** _(created: 2026-02-10)_
- [#33980](https://github.com/vllm-project/vllm/issues/33980) **[RFC]: Sparse attention KV cache offloading to support longer sequence length** _(created: 2026-02-06)_
- [#33913](https://github.com/vllm-project/vllm/issues/33913) **[Usage]: Compile with `TORCH_USE_CUDA_DSA` to enable device-side assertions.** _(created: 2026-02-05)_
- [#33865](https://github.com/vllm-project/vllm/issues/33865) **[Bug]: OpenAI-compatible Embeddings API intermittently crashes with multimodal cache assertion (`Expected a cached item for mm_hash`) on Qwen3-VL-Embedding-8B** _(created: 2026-02-05)_
- [#33628](https://github.com/vllm-project/vllm/issues/33628) **[Bug]: Failed to run distributed inference with Gloo backend on aarch64** _(created: 2026-02-03)_
- [#33458](https://github.com/vllm-project/vllm/issues/33458) **[Feature][Speculative Decoding]: Multi Modal Draft Model Support** _(created: 2026-01-31)_
- [#33398](https://github.com/vllm-project/vllm/issues/33398) **[RFC]: Layerwise KV cache offloading to support longer sequence length** _(created: 2026-01-30)_
- [#33311](https://github.com/vllm-project/vllm/issues/33311) **[Feature]: support pixel_values_videos input for VLM** _(created: 2026-01-29)_
- [#33301](https://github.com/vllm-project/vllm/issues/33301) **[RFC]: Support FP8 LoRA inference** _(created: 2026-01-29)_
- [#33204](https://github.com/vllm-project/vllm/issues/33204) **[Bug]: Qwen3-VL-Embedding model produces different embeddings than official qwen_vl_utils implementation** _(created: 2026-01-27)_
- [#32897](https://github.com/vllm-project/vllm/issues/32897) **[Bug]: PaddleOCR-VL crash** _(created: 2026-01-23)_
- [#32858](https://github.com/vllm-project/vllm/issues/32858) **[Bug]: Use vllm to obtain Qwen3VL last token hidden status** _(created: 2026-01-22)_
- [#32733](https://github.com/vllm-project/vllm/issues/32733) **[RFC]: [P/D] Prefill compute optimizations with bi-directional KV cache transfers between P and D nodes** _(created: 2026-01-20)_
- [#32685](https://github.com/vllm-project/vllm/issues/32685) **[Feature]: Support multi-modal inputs for OpenAI Response API** _(created: 2026-01-20)_
- [#32659](https://github.com/vllm-project/vllm/issues/32659) **[RFC]: Tracking follow-up progress on Encode–Prefill–Decode Disaggregation** _(created: 2026-01-20)_
- [#32588](https://github.com/vllm-project/vllm/issues/32588) **[Bug]: Wrong timestamps if audio > 30s** _(created: 2026-01-19)_
- [#32541](https://github.com/vllm-project/vllm/issues/32541) **[Feature]: LoRa adapter support for Qwen3VLForConditionalGeneration** _(created: 2026-01-18)_
- [#32468](https://github.com/vllm-project/vllm/issues/32468) **[Bug]: Engine core proc EngineCore_DP0 died unexpectedly, shutting down client.** _(created: 2026-01-16)_
- [#32338](https://github.com/vllm-project/vllm/issues/32338) **[Feature]: support for LlavaQwenForCausalLM** _(created: 2026-01-14)_
- [#32320](https://github.com/vllm-project/vllm/issues/32320) **[Performance]: Issue in serving concurrent streams** _(created: 2026-01-14)_
- [#32193](https://github.com/vllm-project/vllm/issues/32193) **[Bug]: vLLM engine crash under burst load despite expected request queuing (72 concurrent API calls)** _(created: 2026-01-12)_
- [#32046](https://github.com/vllm-project/vllm/issues/32046) **Google MedASR** _(created: 2026-01-09)_
- [#31961](https://github.com/vllm-project/vllm/issues/31961) **[Performance]: EPD Disaggregation Performance Testing Scripts** _(created: 2026-01-08)_
- [#31936](https://github.com/vllm-project/vllm/issues/31936) **[Bug]: run deepseek v3.2 failed，not support RTX PRO 6000 * 8？** _(created: 2026-01-08)_
- [#31782](https://github.com/vllm-project/vllm/issues/31782) **[Feature]: Support compressed-tensors NVFP4 quantization for MoE models (Nemotron-H non-gated MoE)** _(created: 2026-01-06)_
- [#31709](https://github.com/vllm-project/vllm/issues/31709) **[Bug]: After upgrade to 0.11.2, vllm crashs with Qwen3.** _(created: 2026-01-05)_
- [#31602](https://github.com/vllm-project/vllm/issues/31602) **[Performance]: If the next request is sent immediately after the previous one finishes, its TTFT will be relatively small; if the next request is sent 10 seconds after the previous one ends, its TTFT will be relatively large.** _(created: 2026-01-01)_
- [#31479](https://github.com/vllm-project/vllm/issues/31479) **[Feature]: Enable LoRA support for tower and connector in more MM models** _(created: 2025-12-29)_
- [#31372](https://github.com/vllm-project/vllm/issues/31372) **[Feature]: Running paddocr‑vl consumes an excessively large amount of  memory.** _(created: 2025-12-26)_
- [#31353](https://github.com/vllm-project/vllm/issues/31353) **[Bug]: KV Cache grows continuously with just one chat completion request using meta-llama/Llama-3.2-1B on L40 GPU with Flash Attention and finally completed after 10 minutes** _(created: 2025-12-25)_
- [#31205](https://github.com/vllm-project/vllm/issues/31205) **ValueError: Qwen3OmniMoeThinkerForConditionalGeneration does not support LoRA yet.** _(created: 2025-12-23)_
- [#30839](https://github.com/vllm-project/vllm/issues/30839) **[RFC]: Enabling Zero-Copy Video with PyNvVideoCodec and IPC** _(created: 2025-12-17)_
- [#30819](https://github.com/vllm-project/vllm/issues/30819) **[Bug]: vLLM inference stuck when requesting video description on VLM models** _(created: 2025-12-16)_
- [#30779](https://github.com/vllm-project/vllm/issues/30779) **[Bug]: v0.11.2 can not support Qwen2.5-Omni-** _(created: 2025-12-16)_
- [#30129](https://github.com/vllm-project/vllm/issues/30129) **[Feature]: About video input for qwen3vl** _(created: 2025-12-05)_
- [#29863](https://github.com/vllm-project/vllm/issues/29863) **[Bug]: fails to inference prompt ends with '.' ':' for video inputs** _(created: 2025-12-02)_
- [#29373](https://github.com/vllm-project/vllm/issues/29373) **[Bug]: Multinode inference request with Ray and vLLM crashes - regression from vLLM v0.7.3** _(created: 2025-11-25)_
- [#29369](https://github.com/vllm-project/vllm/issues/29369) **[Bug]: nixl connector PD disagg stuck at INFO logging level** _(created: 2025-11-25)_
- [#29280](https://github.com/vllm-project/vllm/issues/29280) **[Feature]: Selective Token Logprobs Tracking** _(created: 2025-11-23)_
- [#29174](https://github.com/vllm-project/vllm/issues/29174) **[Bug]: Qwen3 Omni thinking unstable output** _(created: 2025-11-21)_
- [#28919](https://github.com/vllm-project/vllm/issues/28919) **[Bug]: Qwen3-32B use eagle3 crash** _(created: 2025-11-18)_
- [#28691](https://github.com/vllm-project/vllm/issues/28691) **[Feature]: GGUF model with architecture qwen3vlmoe is not supported yet.** _(created: 2025-11-13)_
- [#28307](https://github.com/vllm-project/vllm/issues/28307) **[Bug]: `repetition_penalty` leads to engine failure when using vllm serve...** _(created: 2025-11-07)_
- [#27932](https://github.com/vllm-project/vllm/issues/27932) **[Feature]: Qwen3 Omini AttributeError: 'Qwen3OmniMoeProcessor' object has no attribute '_get_num_multimodal_tokens'** _(created: 2025-11-02)_
- [#27094](https://github.com/vllm-project/vllm/issues/27094) **[RFC]: Remove redundant multi-modal input preprocessing during disaggregated inference** _(created: 2025-10-17)_
- [#26239](https://github.com/vllm-project/vllm/issues/26239) **[Feature]: support reasoning-parser for Qwen3-VL** _(created: 2025-10-05)_
- [#25950](https://github.com/vllm-project/vllm/issues/25950) **[RFC]: Generalized KV cache reuse** _(created: 2025-09-30)_
- [#25943](https://github.com/vllm-project/vllm/issues/25943) **[Feature]: Disable ANSII colors in logs of OpenAI compatible server** _(created: 2025-09-30)_
- [#25750](https://github.com/vllm-project/vllm/issues/25750) **[Feature]: Tracking Whisper feature requests** _(created: 2025-09-26)_
- [#25502](https://github.com/vllm-project/vllm/issues/25502) **[Bug]: This flash attention build does not support tanh softcapping: gemma-2-2b-it on H100 NVL** _(created: 2025-09-23)_
- [#25383](https://github.com/vllm-project/vllm/issues/25383) **[Usage]: Is there a simple way to pass embedding directly in V1** _(created: 2025-09-22)_
- [#25192](https://github.com/vllm-project/vllm/issues/25192) **[Bug]: Make Whisper model compatible with cuda graphs** _(created: 2025-09-18)_
- [#25172](https://github.com/vllm-project/vllm/issues/25172) **[Feature]: FlexAttention + encoder_decoder support** _(created: 2025-09-18)_
- [#24728](https://github.com/vllm-project/vllm/issues/24728) **[Performance]: Multi-Modal Benchmark on NVIDIA A100 – Qwen2.5-VL / MiniCPM-V-4 / InternVL3_5-4B / InternVL3_5-2B** _(created: 2025-09-12)_
- [#23943](https://github.com/vllm-project/vllm/issues/23943) **[Feature]: Any plans to add nvidia/parakeet-tdt-0.6b-v3 to vllm?** _(created: 2025-08-29)_
- [#22424](https://github.com/vllm-project/vllm/issues/22424) **[Bug]: Voxtral-Small-24B-2507 Does Not Support Pipeline-Parallel** _(created: 2025-08-07)_
- [#21995](https://github.com/vllm-project/vllm/issues/21995) **[RFC] Run HF processing on GPU** _(created: 2025-07-31)_
- [#20149](https://github.com/vllm-project/vllm/issues/20149) **[Feature]: Add Support for Updating Lora Weights** _(created: 2025-06-26)_
- [#17972](https://github.com/vllm-project/vllm/issues/17972) **[Bug]: vLLM server hangs and timeouts after initial requests** _(created: 2025-05-12)_
- [#16966](https://github.com/vllm-project/vllm/issues/16966) **[Bug]: vllm 0.8.4 whisper possible memory leak?** _(created: 2025-04-22)_
- [#16313](https://github.com/vllm-project/vllm/issues/16313) **[Feature]: Support structured output and tool call together** _(created: 2025-04-09)_
- [#14174](https://github.com/vllm-project/vllm/issues/14174) **[Feature]: will whisper add language detection?** _(created: 2025-03-04)_
- [#10849](https://github.com/vllm-project/vllm/issues/10849) **[Feature]: add DoRA support** _(created: 2024-12-03)_
- [#7863](https://github.com/vllm-project/vllm/issues/7863) **[New Model]: Request to integrate Chexagent Multimodel in vLLM** _(created: 2024-08-26)_
- [#4194](https://github.com/vllm-project/vllm/issues/4194) **[RFC]: Multi-modality Support on vLLM** _(created: 2024-04-19)_

### Performance / 性能问题 (51)

- [#40608](https://github.com/vllm-project/vllm/issues/40608) **[Kimi] Track Kimi K2.5/K2.6 MLA + EAGLE serving on Blackwell (DCP4/DCP8, FP8 KV, draft backend split)** _(created: 2026-04-22)_
- [#40543](https://github.com/vllm-project/vllm/issues/40543) **[Feature]: Dynamic PDL Enablement** _(created: 2026-04-21)_
- [#40536](https://github.com/vllm-project/vllm/issues/40536) **[Feature]: Support iterative in-place weight editing on TP workers (online RLHF / steering / abliteration)** _(created: 2026-04-21)_
- [#40420](https://github.com/vllm-project/vllm/issues/40420) **[Bug]: TurboQuant `_continuation_prefill` OOMs and kills engine at long-context prefill (~185K actual tokens)** _(created: 2026-04-21)_
- [#39979](https://github.com/vllm-project/vllm/issues/39979) **[RFC]: Ultimate Better Observability.** _(created: 2026-04-16)_
- [#39788](https://github.com/vllm-project/vllm/issues/39788) **[Bug]: CUDA OOM with Kimi-K2.5 NVFP4 on both TP4 and TP8** _(created: 2026-04-14)_
- [#39766](https://github.com/vllm-project/vllm/issues/39766) **[RFC]: Support Mooncake Based ECConnector for EPD** _(created: 2026-04-14)_
- [#39749](https://github.com/vllm-project/vllm/issues/39749) **[Roadmap] [Draft] vLLM Roadmap Q2 2026** _(created: 2026-04-13)_
- [#39701](https://github.com/vllm-project/vllm/issues/39701) **[RFC] Replace routing replay with CUDA-graph-compatible device cache approach** _(created: 2026-04-13)_
- [#39680](https://github.com/vllm-project/vllm/issues/39680) **[Performance]: Qwen3.5 with mtp is slower than without** _(created: 2026-04-13)_
- [#38586](https://github.com/vllm-project/vllm/issues/38586) **[Bug]: Whisper online benchmark with profiling error: TypeError: multi_modal_content must be a dict containing 'audio'** _(created: 2026-03-30)_
- [#38472](https://github.com/vllm-project/vllm/issues/38472) **[Bug]: [xPyD]Potential OOM when using v1 P2pNcclConnector as KV cache transport: KV cache accumulation on decode instance.** _(created: 2026-03-29)_
- [#38470](https://github.com/vllm-project/vllm/issues/38470) **[Bug]: When using the Sonnet dataset for benchmark testing, if the input length is too small, the CPU usage becomes abnormally high with no error logs, making it impossible to run the benchmark properly.** _(created: 2026-03-29)_
- [#38235](https://github.com/vllm-project/vllm/issues/38235) **[Feature]: Quantization support (AWQ / GPTQ / FP8) for mistralai/Voxtral-Mini-4B-Realtime-2602** _(created: 2026-03-26)_
- [#38175](https://github.com/vllm-project/vllm/issues/38175) **[RFC]: Support ViT Full CUDA Graph (Tracker)** _(created: 2026-03-26)_
- [#38056](https://github.com/vllm-project/vllm/issues/38056) **[Bug]: ImportError: flash_attn.ops.triton.rotary not found on older versions (< v2.1.2)** _(created: 2026-03-25)_
- [#37736](https://github.com/vllm-project/vllm/issues/37736) **[CI Failure]:  Gemma3 OOMs with transformers backend** _(created: 2026-03-21)_
- [#37459](https://github.com/vllm-project/vllm/issues/37459) **[CI Failure]: MultiModal Models Extended 2 - isaac test case OOMs** _(created: 2026-03-18)_
- [#37392](https://github.com/vllm-project/vllm/issues/37392) **[Bug]:推理时报错，模型关闭了。部署的Qwen3.5-122B-A10B-FP8模型** _(created: 2026-03-18)_
- [#37242](https://github.com/vllm-project/vllm/issues/37242) **[Community] RTX 5090 (Blackwell sm_120) + WSL2 2.7.0: CUDA graphs work — benchmarks + full config** _(created: 2026-03-17)_
- [#37168](https://github.com/vllm-project/vllm/issues/37168) **[RFC]:  Active Coordination and Two-Zone Scheduling Mechanism for KV Cache in Long-Running Agents** _(created: 2026-03-16)_
- [#36872](https://github.com/vllm-project/vllm/issues/36872) **[Bug]: Gibberish output and collapsing generation throughput with Qwen3.5-35B-A3B-FP8 and speculative decoding enabled** _(created: 2026-03-12)_
- [#36861](https://github.com/vllm-project/vllm/issues/36861) **[Bug]: Why does setting `--pipeline-parallel-size > 1` result in an OOM error, but `--tensor-parallel-size> 1` does not?** _(created: 2026-03-12)_
- [#36627](https://github.com/vllm-project/vllm/issues/36627) **[Performance]: qwen3.5 vs qwen3** _(created: 2026-03-10)_
- [#36589](https://github.com/vllm-project/vllm/issues/36589) **[Bug]: SM 7.5 extreme slowness hangs indefinitely on T4 (vllm 0.17.0 with Qwen3.5-27B)** _(created: 2026-03-10)_
- [#36450](https://github.com/vllm-project/vllm/issues/36450) **[Bug]: Qwen3.5 AWQ models crash during inference on RTX 5090 (Blackwell) with Triton OOM in solve_tril despite successful model load** _(created: 2026-03-09)_
- [#36073](https://github.com/vllm-project/vllm/issues/36073) **[Bug]: Prolonged Latency in Some Streaming Responses in Function Call Mode（MiniMax model）** _(created: 2026-03-05)_
- [#35848](https://github.com/vllm-project/vllm/issues/35848) **[RFC]: Revamp Ray Distributed Executor Backend (from Ray team)** _(created: 2026-03-03)_
- [#35778](https://github.com/vllm-project/vllm/issues/35778) **[Bug]: Regression: terrible mixed prefill-decode performance with CUDA graphs enabled.** _(created: 2026-03-02)_
- [#35467](https://github.com/vllm-project/vllm/issues/35467) **[Performance]: non-optimal performance of `linear` for medium batches** _(created: 2026-02-27)_
- [#35407](https://github.com/vllm-project/vllm/issues/35407) **[CI] DBO with DP+EP accuracy regression on GSM8K evaluation** _(created: 2026-02-26)_
- [#35254](https://github.com/vllm-project/vllm/issues/35254) **[Performance]: curious new kernels from vllm 0.11.1** _(created: 2026-02-25)_
- [#35089](https://github.com/vllm-project/vllm/issues/35089) **[RFC]: In-Tree AMD Zen CPU Backend via zentorch** _(created: 2026-02-23)_
- [#34212](https://github.com/vllm-project/vllm/issues/34212) **[Performance]: W4Afp8 is slower than FP8-W8A8** _(created: 2026-02-10)_
- [#33980](https://github.com/vllm-project/vllm/issues/33980) **[RFC]: Sparse attention KV cache offloading to support longer sequence length** _(created: 2026-02-06)_
- [#33920](https://github.com/vllm-project/vllm/issues/33920) **[Bug]: GLM-4.7-Flash OOM during sampler warmup with tensor parallelism on RTX 4090** _(created: 2026-02-05)_
- [#33398](https://github.com/vllm-project/vllm/issues/33398) **[RFC]: Layerwise KV cache offloading to support longer sequence length** _(created: 2026-01-30)_
- [#33307](https://github.com/vllm-project/vllm/issues/33307) **[Bug]: Latency spikes at input_len=1024 with batch_size=16 (TP1 & TP2)** _(created: 2026-01-29)_
- [#32659](https://github.com/vllm-project/vllm/issues/32659) **[RFC]: Tracking follow-up progress on Encode–Prefill–Decode Disaggregation** _(created: 2026-01-20)_
- [#32604](https://github.com/vllm-project/vllm/issues/32604) **[Bug]: LMCache CPU kv offload cause decode speed degrade** _(created: 2026-01-19)_
- [#32320](https://github.com/vllm-project/vllm/issues/32320) **[Performance]: Issue in serving concurrent streams** _(created: 2026-01-14)_
- [#32193](https://github.com/vllm-project/vllm/issues/32193) **[Bug]: vLLM engine crash under burst load despite expected request queuing (72 concurrent API calls)** _(created: 2026-01-12)_
- [#31961](https://github.com/vllm-project/vllm/issues/31961) **[Performance]: EPD Disaggregation Performance Testing Scripts** _(created: 2026-01-08)_
- [#31803](https://github.com/vllm-project/vllm/issues/31803) **[RFC]: Video Frame Sparsification via Pixel-Level Similarity for Efficient Multimodal Long-Video Inference** _(created: 2026-01-06)_
- [#31602](https://github.com/vllm-project/vllm/issues/31602) **[Performance]: If the next request is sent immediately after the previous one finishes, its TTFT will be relatively small; if the next request is sent 10 seconds after the previous one ends, its TTFT will be relatively large.** _(created: 2026-01-01)_
- [#31372](https://github.com/vllm-project/vllm/issues/31372) **[Feature]: Running paddocr‑vl consumes an excessively large amount of  memory.** _(created: 2025-12-26)_
- [#30839](https://github.com/vllm-project/vllm/issues/30839) **[RFC]: Enabling Zero-Copy Video with PyNvVideoCodec and IPC** _(created: 2025-12-17)_
- [#25192](https://github.com/vllm-project/vllm/issues/25192) **[Bug]: Make Whisper model compatible with cuda graphs** _(created: 2025-09-18)_
- [#24728](https://github.com/vllm-project/vllm/issues/24728) **[Performance]: Multi-Modal Benchmark on NVIDIA A100 – Qwen2.5-VL / MiniCPM-V-4 / InternVL3_5-4B / InternVL3_5-2B** _(created: 2025-09-12)_
- [#21995](https://github.com/vllm-project/vllm/issues/21995) **[RFC] Run HF processing on GPU** _(created: 2025-07-31)_
- [#13941](https://github.com/vllm-project/vllm/issues/13941) **[Bug]: wake up OOM (72B model in 8*A800(40G))** _(created: 2025-02-27)_

### CUDA Graph / 计算图 (26)

- [#40807](https://github.com/vllm-project/vllm/issues/40807) **[Bug]: TurboQuant KV + spec-decode + chunked-prefill crashes CUDA graph capture at query_start_loc.tolist() in continuation-prefill path (Qwen3-Next hybrid dense)** _(created: 2026-04-24)_
- [#40742](https://github.com/vllm-project/vllm/issues/40742) **[Bug]: CUDA graph capture crashes during startup due to Inductor autotuning torch.cuda.synchronize() inside graph capture (FULL_DECODE_ONLY + MLA + FP8) when PDL is enabled** _(created: 2026-04-23)_
- [#40661](https://github.com/vllm-project/vllm/issues/40661) **[Bug]: CUBLAS_STATUS_EXECUTION_FAILED during CUDA graph compilation of BF16 vision encoder on NVIDIA Jetson AGX Thor (vLLM 0.19.0 regression)** _(created: 2026-04-23)_
- [#40543](https://github.com/vllm-project/vllm/issues/40543) **[Feature]: Dynamic PDL Enablement** _(created: 2026-04-21)_
- [#40121](https://github.com/vllm-project/vllm/issues/40121) **[Bug]: CUDA graph replay triggers Xid 13 illegal memory access on Qwen3-32B-AWQ with TP=2 on dual RTX 3090** _(created: 2026-04-17)_
- [#40047](https://github.com/vllm-project/vllm/issues/40047) **[Bug][Tracking Issue]: NaNs in CUDA Graph padding regions corrupt activations in some per-token kernels** _(created: 2026-04-16)_
- [#40038](https://github.com/vllm-project/vllm/issues/40038) **[Bug]: cudaErrorIllegalAddress during PIECEWISE CUDA graph replay with MoE LoRA: stale buffer addresses in `moe_lora_align_block_size`** _(created: 2026-04-16)_
- [#39371](https://github.com/vllm-project/vllm/issues/39371) **DSA module construction corrupts CUDA RNG state (Offset increment outside graph capture)** _(created: 2026-04-09)_
- [#39288](https://github.com/vllm-project/vllm/issues/39288) **[Bug]: FlashInfer CUTLASS MoE backend causes CUDA illegal memory access on H100 during CUDA graph capture (Qwen3-Next-80B BF16)** _(created: 2026-04-08)_
- [#39096](https://github.com/vllm-project/vllm/issues/39096) **[Bug]: Batch invariance breaks with torch.compile and/or CUDA graphs on SM<90** _(created: 2026-04-06)_
- [#39010](https://github.com/vllm-project/vllm/issues/39010) **[Bug]: Hang During CUDA Graph Capture on ROCM in 0.19** _(created: 2026-04-05)_
- [#38754](https://github.com/vllm-project/vllm/issues/38754) **[Bug]: GPT OSS Router GEMM Causing NaNs** _(created: 2026-04-01)_
- [#38486](https://github.com/vllm-project/vllm/issues/38486) **[Bug]: cuda graph takes too much memory for qwen 3.5** _(created: 2026-03-29)_
- [#38208](https://github.com/vllm-project/vllm/issues/38208) **[Bug]: CUDA Illegal Instruction during CUDA Graph capture with Nemotron-3-Nano NVFP4 on sm_121** _(created: 2026-03-26)_
- [#38175](https://github.com/vllm-project/vllm/issues/38175) **[RFC]: Support ViT Full CUDA Graph (Tracker)** _(created: 2026-03-26)_
- [#37563](https://github.com/vllm-project/vllm/issues/37563) **mm_fp4 trtllm backend leaks padding scales into real rows (use_8x4_sf_layout=True)** _(created: 2026-03-19)_
- [#37451](https://github.com/vllm-project/vllm/issues/37451) **[Bug]: 0.17.1 - vllm serve deepseek-ai/DeepSeek-OCR-2 on H100 crashes during Capturing CUDA graphs (decode, FULL)** _(created: 2026-03-18)_
- [#37363](https://github.com/vllm-project/vllm/issues/37363) **fix(compilation): fix piecewise CUDA graph bugs with splitting_ops** _(created: 2026-03-18)_
- [#37242](https://github.com/vllm-project/vllm/issues/37242) **[Community] RTX 5090 (Blackwell sm_120) + WSL2 2.7.0: CUDA graphs work — benchmarks + full config** _(created: 2026-03-17)_
- [#35778](https://github.com/vllm-project/vllm/issues/35778) **[Bug]: Regression: terrible mixed prefill-decode performance with CUDA graphs enabled.** _(created: 2026-03-02)_
- [#35743](https://github.com/vllm-project/vllm/issues/35743) **[Bug]: Qwen 3.5 27B AWQ 4bit capturing CUDA graph fails** _(created: 2026-03-02)_
- [#35659](https://github.com/vllm-project/vllm/issues/35659) **[Bug]: cudaErrorIllegalAddress under sustained parallel load with CUDA Graphs on Blackwell SM120 (NVFP4 MoE)** _(created: 2026-03-01)_
- [#35467](https://github.com/vllm-project/vllm/issues/35467) **[Performance]: non-optimal performance of `linear` for medium batches** _(created: 2026-02-27)_
- [#35141](https://github.com/vllm-project/vllm/issues/35141) **[Feature]: Sequence Parallel Support for Model Runner V2** _(created: 2026-02-23)_
- [#34505](https://github.com/vllm-project/vllm/issues/34505) **[Bug]: Qwen 2.5 Omni cuda graph has error** _(created: 2026-02-13)_
- [#25192](https://github.com/vllm-project/vllm/issues/25192) **[Bug]: Make Whisper model compatible with cuda graphs** _(created: 2025-09-18)_

### EPD / 编解码分离 (12)

- [#39766](https://github.com/vllm-project/vllm/issues/39766) **[RFC]: Support Mooncake Based ECConnector for EPD** _(created: 2026-04-14)_
- [#38760](https://github.com/vllm-project/vllm/issues/38760) **[RFC]: Per-iteration forward pass metrics with accurate engine-level timing** _(created: 2026-04-01)_
- [#38710](https://github.com/vllm-project/vllm/issues/38710) **[Bug]: heterogeneous disaggregated serving XPU (Prefill) + CPU (Decode) accuracy issue** _(created: 2026-04-01)_
- [#36328](https://github.com/vllm-project/vllm/issues/36328) **[Feature]: Include mm_hash and mm_transfer_params in Disaggregated Encoder response to prevent redundant data fetching** _(created: 2026-03-07)_
- [#35087](https://github.com/vllm-project/vllm/issues/35087) **[Bug]: DeepSeek 3.2 P/D Disaggregation Support** _(created: 2026-02-23)_
- [#32733](https://github.com/vllm-project/vllm/issues/32733) **[RFC]: [P/D] Prefill compute optimizations with bi-directional KV cache transfers between P and D nodes** _(created: 2026-01-20)_
- [#32659](https://github.com/vllm-project/vllm/issues/32659) **[RFC]: Tracking follow-up progress on Encode–Prefill–Decode Disaggregation** _(created: 2026-01-20)_
- [#31961](https://github.com/vllm-project/vllm/issues/31961) **[Performance]: EPD Disaggregation Performance Testing Scripts** _(created: 2026-01-08)_
- [#29369](https://github.com/vllm-project/vllm/issues/29369) **[Bug]: nixl connector PD disagg stuck at INFO logging level** _(created: 2025-11-25)_
- [#29362](https://github.com/vllm-project/vllm/issues/29362) **[RFC]: Resettle examples.** _(created: 2025-11-25)_
- [#27094](https://github.com/vllm-project/vllm/issues/27094) **[RFC]: Remove redundant multi-modal input preprocessing during disaggregated inference** _(created: 2025-10-17)_
- [#22817](https://github.com/vllm-project/vllm/issues/22817) **[RFC]: Disaggregated Everything - Token In <> Token Out API Server** _(created: 2025-08-13)_

### Prefix Caching / 前缀缓存 (6)

- [#39680](https://github.com/vllm-project/vllm/issues/39680) **[Performance]: Qwen3.5 with mtp is slower than without** _(created: 2026-04-13)_
- [#38754](https://github.com/vllm-project/vllm/issues/38754) **[Bug]: GPT OSS Router GEMM Causing NaNs** _(created: 2026-04-01)_
- [#37729](https://github.com/vllm-project/vllm/issues/37729) **[Bug]: V1 engine core deadlocks under concurrent load (fp8 + prefix caching + Qwen3.5)** _(created: 2026-03-21)_
- [#37168](https://github.com/vllm-project/vllm/issues/37168) **[RFC]:  Active Coordination and Two-Zone Scheduling Mechanism for KV Cache in Long-Running Agents** _(created: 2026-03-16)_
- [#33398](https://github.com/vllm-project/vllm/issues/33398) **[RFC]: Layerwise KV cache offloading to support longer sequence length** _(created: 2026-01-30)_
- [#20261](https://github.com/vllm-project/vllm/issues/20261) **[Bug]: Prefix caching ignores visual input, causing incorrect multimodal outputs under concurrency** _(created: 2025-06-30)_

### ViT / 视觉编码器 (6)

- [#40095](https://github.com/vllm-project/vllm/issues/40095) **[Bug]: Gemma4MultimodalEmbedder normalization order different from Transformers, causing bad audio inference** _(created: 2026-04-17)_
- [#39708](https://github.com/vllm-project/vllm/issues/39708) **[Feature]: Pre-ViT visual token pruning for VLMs (PixelPrune)** _(created: 2026-04-13)_
- [#38886](https://github.com/vllm-project/vllm/issues/38886) **[Bug]: Gemma 4 E4B weight loading fails `Gemma4ClippableLinear` parameter `input_max` not recognized** _(created: 2026-04-03)_
- [#38175](https://github.com/vllm-project/vllm/issues/38175) **[RFC]: Support ViT Full CUDA Graph (Tracker)** _(created: 2026-03-26)_
- [#37977](https://github.com/vllm-project/vllm/issues/37977) **[Bug][Model] Eagle2.5-VL applies ImageNet normalization instead of SigLIP2** _(created: 2026-03-24)_
- [#35980](https://github.com/vllm-project/vllm/issues/35980) **[Bug]: Why does deploying Qwen3-32B-AWQ via vllm:v0.10.1.1 result in different outputs for the same input?** _(created: 2026-03-04)_

### Video Multimodal / 视频多模态 (23)

- [#38811](https://github.com/vllm-project/vllm/issues/38811) **[Usage]: Qwen3-VL inference on video complains of lack of metadata** _(created: 2026-04-02)_
- [#38531](https://github.com/vllm-project/vllm/issues/38531) **[Feature]: Support direct binary/multipart file upload for video and image in OpenAI-compatible API** _(created: 2026-03-30)_
- [#38351](https://github.com/vllm-project/vllm/issues/38351) **[Bug]: When use_audio_in_video is enabled in qwen3-omni, the output may exhibit issues such as empty or repetitive output.** _(created: 2026-03-27)_
- [#37168](https://github.com/vllm-project/vllm/issues/37168) **[RFC]:  Active Coordination and Two-Zone Scheduling Mechanism for KV Cache in Long-Running Agents** _(created: 2026-03-16)_
- [#37075](https://github.com/vllm-project/vllm/issues/37075) **[RFC]: Opt-in Media URL Cache for `MediaConnector`** _(created: 2026-03-14)_
- [#35908](https://github.com/vllm-project/vllm/issues/35908) **[RFC]: Model-specific realtime streaming abstraction** _(created: 2026-03-03)_
- [#35285](https://github.com/vllm-project/vllm/issues/35285) **[Bug]: Large Video Request cause vLLM Progress Core Dump** _(created: 2026-02-25)_
- [#33311](https://github.com/vllm-project/vllm/issues/33311) **[Feature]: support pixel_values_videos input for VLM** _(created: 2026-01-29)_
- [#33204](https://github.com/vllm-project/vllm/issues/33204) **[Bug]: Qwen3-VL-Embedding model produces different embeddings than official qwen_vl_utils implementation** _(created: 2026-01-27)_
- [#32685](https://github.com/vllm-project/vllm/issues/32685) **[Feature]: Support multi-modal inputs for OpenAI Response API** _(created: 2026-01-20)_
- [#32338](https://github.com/vllm-project/vllm/issues/32338) **[Feature]: support for LlavaQwenForCausalLM** _(created: 2026-01-14)_
- [#32113](https://github.com/vllm-project/vllm/issues/32113) **[Bug]: Qwen3-VL-8B inference with video** _(created: 2026-01-11)_
- [#31803](https://github.com/vllm-project/vllm/issues/31803) **[RFC]: Video Frame Sparsification via Pixel-Level Similarity for Efficient Multimodal Long-Video Inference** _(created: 2026-01-06)_
- [#30839](https://github.com/vllm-project/vllm/issues/30839) **[RFC]: Enabling Zero-Copy Video with PyNvVideoCodec and IPC** _(created: 2025-12-17)_
- [#30819](https://github.com/vllm-project/vllm/issues/30819) **[Bug]: vLLM inference stuck when requesting video description on VLM models** _(created: 2025-12-16)_
- [#30129](https://github.com/vllm-project/vllm/issues/30129) **[Feature]: About video input for qwen3vl** _(created: 2025-12-05)_
- [#29863](https://github.com/vllm-project/vllm/issues/29863) **[Bug]: fails to inference prompt ends with '.' ':' for video inputs** _(created: 2025-12-02)_
- [#29174](https://github.com/vllm-project/vllm/issues/29174) **[Bug]: Qwen3 Omni thinking unstable output** _(created: 2025-11-21)_
- [#28388](https://github.com/vllm-project/vllm/issues/28388) **[Bug]: 新版的vllm已经废弃了v0代码，而对qwen-omni系列的模型支持仅限于v0，似乎是因为这个原因，我们无法使用最新版的vllm推理qwen-omni模型** _(created: 2025-11-10)_
- [#28375](https://github.com/vllm-project/vllm/issues/28375) **[Bug]: Engine stuck at CPU when running long video input on Qwen3-vl model** _(created: 2025-11-10)_
- [#27557](https://github.com/vllm-project/vllm/issues/27557) **[Bug]: Engine core proc EngineCore_DP0 died unexpectedly, shutting down client.** _(created: 2025-10-27)_
- [#24728](https://github.com/vllm-project/vllm/issues/24728) **[Performance]: Multi-Modal Benchmark on NVIDIA A100 – Qwen2.5-VL / MiniCPM-V-4 / InternVL3_5-4B / InternVL3_5-2B** _(created: 2025-09-12)_
- [#10849](https://github.com/vllm-project/vllm/issues/10849) **[Feature]: add DoRA support** _(created: 2024-12-03)_

### Audio / Speech / 音频语音 (46)

- [#40781](https://github.com/vllm-project/vllm/issues/40781) **[Feature]: vllm support audio otel tracing?** _(created: 2026-04-24)_
- [#40113](https://github.com/vllm-project/vllm/issues/40113) **[CI Failure]: Multi-Modal Processor (CPU)** _(created: 2026-04-17)_
- [#40095](https://github.com/vllm-project/vllm/issues/40095) **[Bug]: Gemma4MultimodalEmbedder normalization order different from Transformers, causing bad audio inference** _(created: 2026-04-17)_
- [#39996](https://github.com/vllm-project/vllm/issues/39996) **[Bug] Fatal AssertionError: Encoder KV cache fails to evict tokens, exceeding max_model_len in long-lived WebSocket sessions** _(created: 2026-04-16)_
- [#39687](https://github.com/vllm-project/vllm/issues/39687) **[Bug]: vllm(g0e39202ca) vllm serve: error: argument --limit-mm-per-prompt: Value image=4,audio=1 cannot be converted to <function** _(created: 2026-04-13)_
- [#39408](https://github.com/vllm-project/vllm/issues/39408) **[Usage]: qwen3-asr-1.7b pre-allocated encoder cache size limit** _(created: 2026-04-09)_
- [#38886](https://github.com/vllm-project/vllm/issues/38886) **[Bug]: Gemma 4 E4B weight loading fails `Gemma4ClippableLinear` parameter `input_max` not recognized** _(created: 2026-04-03)_
- [#38586](https://github.com/vllm-project/vllm/issues/38586) **[Bug]: Whisper online benchmark with profiling error: TypeError: multi_modal_content must be a dict containing 'audio'** _(created: 2026-03-30)_
- [#38560](https://github.com/vllm-project/vllm/issues/38560) **[Bug]: reasoning_effort passed to MistralCommonTokenizer.apply_chat_template breaks Mistral Small 4 chat completions on vLLM 0.18.0** _(created: 2026-03-30)_
- [#38428](https://github.com/vllm-project/vllm/issues/38428) **[Bug]: V1 Engine: EngineDeadError (AssertionError) on max_model_len overflow during realtime audio streaming** _(created: 2026-03-28)_
- [#38351](https://github.com/vllm-project/vllm/issues/38351) **[Bug]: When use_audio_in_video is enabled in qwen3-omni, the output may exhibit issues such as empty or repetitive output.** _(created: 2026-03-27)_
- [#38297](https://github.com/vllm-project/vllm/issues/38297) **[Bug]: Gemma3n concurrent audio requests crash EngineCore — missing dynamic_dims on audio sequence dimension** _(created: 2026-03-27)_
- [#38235](https://github.com/vllm-project/vllm/issues/38235) **[Feature]: Quantization support (AWQ / GPTQ / FP8) for mistralai/Voxtral-Mini-4B-Realtime-2602** _(created: 2026-03-26)_
- [#38233](https://github.com/vllm-project/vllm/issues/38233) **[Bug]: Voxtral-Mini-4B-Realtime hangs/crashes on multiple sessions due to encoder_cache_usage saturation on 16GB GPU** _(created: 2026-03-26)_
- [#38176](https://github.com/vllm-project/vllm/issues/38176) **[Bug]: qwen3 235B model with latest vllm is going to generate only 1 token.** _(created: 2026-03-26)_
- [#38004](https://github.com/vllm-project/vllm/issues/38004) **[Bug]: Speech-to-Text endpoint may return 501 but not documented in OpenAPI** _(created: 2026-03-24)_
- [#37581](https://github.com/vllm-project/vllm/issues/37581) **[Bug]: /v1/chat/completions/render` crashes for Qwen/Qwen3-ASR-0.6B multimodal audio, and chat audio returns empty/junk** _(created: 2026-03-19)_
- [#37075](https://github.com/vllm-project/vllm/issues/37075) **[RFC]: Opt-in Media URL Cache for `MediaConnector`** _(created: 2026-03-14)_
- [#36585](https://github.com/vllm-project/vllm/issues/36585) **[Bug]: qwen3.5-27b-gptq deploy fail** _(created: 2026-03-10)_
- [#36015](https://github.com/vllm-project/vllm/issues/36015) **[Bug]: Realtime audio transcription (Voxtral) silently hangs after ~10 minutes due to unhandled TimeoutError in background task** _(created: 2026-03-04)_
- [#35908](https://github.com/vllm-project/vllm/issues/35908) **[RFC]: Model-specific realtime streaming abstraction** _(created: 2026-03-03)_
- [#35863](https://github.com/vllm-project/vllm/issues/35863) **[Bug]: Voxtral-Realtime stops returning transcribed text starting from the 3rd concurrent session** _(created: 2026-03-03)_
- [#35767](https://github.com/vllm-project/vllm/issues/35767) **[Enhancement]: Qwen3-ASR realtime endpoint produces degraded output — stateless segments, no cross-segment context, raw format leaks** _(created: 2026-03-02)_
- [#35755](https://github.com/vllm-project/vllm/issues/35755) **[Bug]: AsyncScheduler crashes with AssertionError during Realtime ASR streaming (num_output_placeholders underflow)** _(created: 2026-03-02)_
- [#35642](https://github.com/vllm-project/vllm/issues/35642) **[Bug]: HIP build in Docker: offload-arch stderr contaminates compiler flags via cmake/utils.cmake and CMAKE_HIP_FLAGS** _(created: 2026-03-01)_
- [#35276](https://github.com/vllm-project/vllm/issues/35276) **[Bug]: OpenAI transcribe prompt parameter with whisper return hallucinated transcription** _(created: 2026-02-25)_
- [#35272](https://github.com/vllm-project/vllm/issues/35272) **[Feature]: Make the Qwen3-ASR use `request_prompt` from user input** _(created: 2026-02-25)_
- [#34518](https://github.com/vllm-project/vllm/issues/34518) **[Feature]: [Whisper] Support for decoder prefix and custom task tokens in transcription API** _(created: 2026-02-13)_
- [#34323](https://github.com/vllm-project/vllm/issues/34323) **[CI Failure]: Spawned tests can fail silently** _(created: 2026-02-11)_
- [#34250](https://github.com/vllm-project/vllm/issues/34250) **[Bug]: using vllm on Qwen3-Omni-30B-A3B-Instruct: Failed to apply prompt replacement for mm_items['audio'][0].** _(created: 2026-02-10)_
- [#33107](https://github.com/vllm-project/vllm/issues/33107) **[Bug]: Whisper large-v3 accuracy degradation in vLLM 0.14.1 (134.56% WER) on L40S - works fine in 0.12.0** _(created: 2026-01-26)_
- [#32636](https://github.com/vllm-project/vllm/issues/32636) **[Bug]: Invalid base64-encoded string for audio input** _(created: 2026-01-20)_
- [#32588](https://github.com/vllm-project/vllm/issues/32588) **[Bug]: Wrong timestamps if audio > 30s** _(created: 2026-01-19)_
- [#32320](https://github.com/vllm-project/vllm/issues/32320) **[Performance]: Issue in serving concurrent streams** _(created: 2026-01-14)_
- [#32046](https://github.com/vllm-project/vllm/issues/32046) **Google MedASR** _(created: 2026-01-09)_
- [#30777](https://github.com/vllm-project/vllm/issues/30777) **[Bug]: whisper-large-v3-turbo have accuracy problem on nightly build** _(created: 2025-12-16)_
- [#29174](https://github.com/vllm-project/vllm/issues/29174) **[Bug]: Qwen3 Omni thinking unstable output** _(created: 2025-11-21)_
- [#28388](https://github.com/vllm-project/vllm/issues/28388) **[Bug]: 新版的vllm已经废弃了v0代码，而对qwen-omni系列的模型支持仅限于v0，似乎是因为这个原因，我们无法使用最新版的vllm推理qwen-omni模型** _(created: 2025-11-10)_
- [#25943](https://github.com/vllm-project/vllm/issues/25943) **[Feature]: Disable ANSII colors in logs of OpenAI compatible server** _(created: 2025-09-30)_
- [#25750](https://github.com/vllm-project/vllm/issues/25750) **[Feature]: Tracking Whisper feature requests** _(created: 2025-09-26)_
- [#25192](https://github.com/vllm-project/vllm/issues/25192) **[Bug]: Make Whisper model compatible with cuda graphs** _(created: 2025-09-18)_
- [#25172](https://github.com/vllm-project/vllm/issues/25172) **[Feature]: FlexAttention + encoder_decoder support** _(created: 2025-09-18)_
- [#23943](https://github.com/vllm-project/vllm/issues/23943) **[Feature]: Any plans to add nvidia/parakeet-tdt-0.6b-v3 to vllm?** _(created: 2025-08-29)_
- [#20149](https://github.com/vllm-project/vllm/issues/20149) **[Feature]: Add Support for Updating Lora Weights** _(created: 2025-06-26)_
- [#16966](https://github.com/vllm-project/vllm/issues/16966) **[Bug]: vllm 0.8.4 whisper possible memory leak?** _(created: 2025-04-22)_
- [#14174](https://github.com/vllm-project/vllm/issues/14174) **[Feature]: will whisper add language detection?** _(created: 2025-03-04)_

### VL Model: Qwen-VL系列 (8)

- [#38470](https://github.com/vllm-project/vllm/issues/38470) **[Bug]: When using the Sonnet dataset for benchmark testing, if the input length is too small, the CPU usage becomes abnormally high with no error logs, making it impossible to run the benchmark properly.** _(created: 2026-03-29)_
- [#35285](https://github.com/vllm-project/vllm/issues/35285) **[Bug]: Large Video Request cause vLLM Progress Core Dump** _(created: 2026-02-25)_
- [#34212](https://github.com/vllm-project/vllm/issues/34212) **[Performance]: W4Afp8 is slower than FP8-W8A8** _(created: 2026-02-10)_
- [#31484](https://github.com/vllm-project/vllm/issues/31484) **[Usage]: RuntimeError when running Qwen2.5-VL-7B-Instruct with vllm: Potential version incompatibility** _(created: 2025-12-29)_
- [#31063](https://github.com/vllm-project/vllm/issues/31063) **[Bug]: Qwen3VL-8B-instruct-FP8 has larger result diff rate than sglang compared with transformers** _(created: 2025-12-20)_
- [#27094](https://github.com/vllm-project/vllm/issues/27094) **[RFC]: Remove redundant multi-modal input preprocessing during disaggregated inference** _(created: 2025-10-17)_
- [#26777](https://github.com/vllm-project/vllm/issues/26777) **[Bug]: Deploying Qwen2.5-VL-7B with Eagle3, the service crashes when concurrency increases.** _(created: 2025-10-14)_
- [#24728](https://github.com/vllm-project/vllm/issues/24728) **[Performance]: Multi-Modal Benchmark on NVIDIA A100 – Qwen2.5-VL / MiniCPM-V-4 / InternVL3_5-4B / InternVL3_5-2B** _(created: 2025-09-12)_

### VL Model: Qwen-Omni系列 (2)

- [#30779](https://github.com/vllm-project/vllm/issues/30779) **[Bug]: v0.11.2 can not support Qwen2.5-Omni-** _(created: 2025-12-16)_
- [#28388](https://github.com/vllm-project/vllm/issues/28388) **[Bug]: 新版的vllm已经废弃了v0代码，而对qwen-omni系列的模型支持仅限于v0，似乎是因为这个原因，我们无法使用最新版的vllm推理qwen-omni模型** _(created: 2025-11-10)_

### VL Model: LLaVA系列 (3)

- [#32338](https://github.com/vllm-project/vllm/issues/32338) **[Feature]: support for LlavaQwenForCausalLM** _(created: 2026-01-14)_
- [#10849](https://github.com/vllm-project/vllm/issues/10849) **[Feature]: add DoRA support** _(created: 2024-12-03)_
- [#7863](https://github.com/vllm-project/vllm/issues/7863) **[New Model]: Request to integrate Chexagent Multimodel in vLLM** _(created: 2024-08-26)_

### VL Model: InternVL系列 (3)

- [#38425](https://github.com/vllm-project/vllm/issues/38425) **[Transformers v5] InternVL2** _(created: 2026-03-28)_
- [#37977](https://github.com/vllm-project/vllm/issues/37977) **[Bug][Model] Eagle2.5-VL applies ImageNet normalization instead of SigLIP2** _(created: 2026-03-24)_
- [#24728](https://github.com/vllm-project/vllm/issues/24728) **[Performance]: Multi-Modal Benchmark on NVIDIA A100 – Qwen2.5-VL / MiniCPM-V-4 / InternVL3_5-4B / InternVL3_5-2B** _(created: 2025-09-12)_

### VL Model: 其他模型 (3)

- [#37981](https://github.com/vllm-project/vllm/issues/37981) **[Bug]: v0.18.0 fails to run MiniCPM-o-4.5** _(created: 2026-03-24)_
- [#31479](https://github.com/vllm-project/vllm/issues/31479) **[Feature]: Enable LoRA support for tower and connector in more MM models** _(created: 2025-12-29)_
- [#24728](https://github.com/vllm-project/vllm/issues/24728) **[Performance]: Multi-Modal Benchmark on NVIDIA A100 – Qwen2.5-VL / MiniCPM-V-4 / InternVL3_5-4B / InternVL3_5-2B** _(created: 2025-09-12)_

### Image Token / 输入处理 (6)

- [#40106](https://github.com/vllm-project/vllm/issues/40106) **[Bug]: Gemma4 multimodal: missing vision-aware bidirectional attention mask for use_bidirectional_attention="vision" models** _(created: 2026-04-17)_
- [#39708](https://github.com/vllm-project/vllm/issues/39708) **[Feature]: Pre-ViT visual token pruning for VLMs (PixelPrune)** _(created: 2026-04-13)_
- [#39681](https://github.com/vllm-project/vllm/issues/39681) **[Bug]: Gemma4 multimodal crashes with "pixel_values contains inconsistent shapes" when concurrent image requests have different resolutions** _(created: 2026-04-13)_
- [#33311](https://github.com/vllm-project/vllm/issues/33311) **[Feature]: support pixel_values_videos input for VLM** _(created: 2026-01-29)_
- [#31803](https://github.com/vllm-project/vllm/issues/31803) **[RFC]: Video Frame Sparsification via Pixel-Level Similarity for Efficient Multimodal Long-Video Inference** _(created: 2026-01-06)_
- [#31205](https://github.com/vllm-project/vllm/issues/31205) **ValueError: Qwen3OmniMoeThinkerForConditionalGeneration does not support LoRA yet.** _(created: 2025-12-23)_

### Compatibility / 兼容性 (205)

- [#40856](https://github.com/vllm-project/vllm/issues/40856) **[Bug]: VLLM running qwen3.6 for image inference occasionally reports 500 Internal Server Error** _(created: 2026-04-25)_
- [#40821](https://github.com/vllm-project/vllm/issues/40821) **[Bug]: Deepseek V4 failed to load on RTX PRO 6000** _(created: 2026-04-24)_
- [#40807](https://github.com/vllm-project/vllm/issues/40807) **[Bug]: TurboQuant KV + spec-decode + chunked-prefill crashes CUDA graph capture at query_start_loc.tolist() in continuation-prefill path (Qwen3-Next hybrid dense)** _(created: 2026-04-24)_
- [#40765](https://github.com/vllm-project/vllm/issues/40765) **[Bug]: runai_streamer loads both Ministral consolidated and HF sharded safetensors** _(created: 2026-04-24)_
- [#40758](https://github.com/vllm-project/vllm/issues/40758) **[CI Failure]: `Qwen3.6-35B-A3B-FP8` fails on `NVIDIA GB10` with `cutlass_scaled_mm` / `cutlass_gemm_caller Error Internal` under vLLM nightly + CUDA 13.0** _(created: 2026-04-24)_
- [#40741](https://github.com/vllm-project/vllm/issues/40741) **[Feature]: Make opencv-python-headless an optional dependency for FIPS compliance** _(created: 2026-04-23)_
- [#40661](https://github.com/vllm-project/vllm/issues/40661) **[Bug]: CUBLAS_STATUS_EXECUTION_FAILED during CUDA graph compilation of BF16 vision encoder on NVIDIA Jetson AGX Thor (vLLM 0.19.0 regression)** _(created: 2026-04-23)_
- [#40652](https://github.com/vllm-project/vllm/issues/40652) **[Bug]: Kimi 2.6 on 8x A100 SMX4 leads to NVLink Crash Coredump** _(created: 2026-04-22)_
- [#40649](https://github.com/vllm-project/vllm/issues/40649) **[Bug]: KeyError on model.layers.N.self_attn.attn during initialize_attn_backend with pipeline_parallel_size=4 (V1 engine + Ray)** _(created: 2026-04-22)_
- [#40591](https://github.com/vllm-project/vllm/issues/40591) **[Bug]: Regression in 0.19.1 - Gemma 4 26B MoE fails to load packed experts (KeyError: down_proj_packed). Worked in dev6.** _(created: 2026-04-22)_
- [#40585](https://github.com/vllm-project/vllm/issues/40585) **[Bug]: qwen3.5 can not use --decode-context-parallel-size with --enable-prefix-caching** _(created: 2026-04-22)_
- [#40381](https://github.com/vllm-project/vllm/issues/40381) **[Bug]: Buffer overflow when allocating memory error on Qwen3.5-122B-A10B-GPTQ-Int4 and NVFP4** _(created: 2026-04-20)_
- [#40318](https://github.com/vllm-project/vllm/issues/40318) **[Bug]: Mistral3 text-only startup fails when text_config.architectures is None** _(created: 2026-04-20)_
- [#40290](https://github.com/vllm-project/vllm/issues/40290) **[Bug]: Gemma 4 (31B/26B-A4B) vision outputs only <pad> under fp16 — vision_tower standardize overflows** _(created: 2026-04-19)_
- [#40165](https://github.com/vllm-project/vllm/issues/40165) **[Bug]: HunyuanOCR crashes with "query and key must have the same dtype" during inference (vLLM 0.19.0, RTX 3050)** _(created: 2026-04-17)_
- [#40121](https://github.com/vllm-project/vllm/issues/40121) **[Bug]: CUDA graph replay triggers Xid 13 illegal memory access on Qwen3-32B-AWQ with TP=2 on dual RTX 3090** _(created: 2026-04-17)_
- [#40080](https://github.com/vllm-project/vllm/issues/40080) **[Bug]: Gemma 4 (31B / 26B-A4B) generates infinite repetition loops, especially with structured output (JSON schema)** _(created: 2026-04-17)_
- [#40038](https://github.com/vllm-project/vllm/issues/40038) **[Bug]: cudaErrorIllegalAddress during PIECEWISE CUDA graph replay with MoE LoRA: stale buffer addresses in `moe_lora_align_block_size`** _(created: 2026-04-16)_
- [#40002](https://github.com/vllm-project/vllm/issues/40002) **[Bug]: Inconsistent KV Cache reporting and system hang on long context requests (Gemma-4 26B AWQ Int4)** _(created: 2026-04-16)_
- [#39919](https://github.com/vllm-project/vllm/issues/39919) **[Bug]: DeepSeek OCR doesn't work on vllm 0.19** _(created: 2026-04-15)_
- [#39903](https://github.com/vllm-project/vllm/issues/39903) **[Bug]: Significant Cross-Instance Inference Variance in vLLM v0.18.0 on H20 (~10-point gap) Qwen3.5-35B-A3B** _(created: 2026-04-15)_
- [#39788](https://github.com/vllm-project/vllm/issues/39788) **[Bug]: CUDA OOM with Kimi-K2.5 NVFP4 on both TP4 and TP8** _(created: 2026-04-14)_
- [#39761](https://github.com/vllm-project/vllm/issues/39761) **[Bug]:CUDA illegal instruction during decode (V1 Engine + NVFP4) on aarch64 (NVIDIA GB10)** _(created: 2026-04-14)_
- [#39722](https://github.com/vllm-project/vllm/issues/39722) **Gibberish with flashinfer_nvlink_two_sided on GB200/arm64** _(created: 2026-04-13)_
- [#39701](https://github.com/vllm-project/vllm/issues/39701) **[RFC] Replace routing replay with CUDA-graph-compatible device cache approach** _(created: 2026-04-13)_
- [#39687](https://github.com/vllm-project/vllm/issues/39687) **[Bug]: vllm(g0e39202ca) vllm serve: error: argument --limit-mm-per-prompt: Value image=4,audio=1 cannot be converted to <function** _(created: 2026-04-13)_
- [#39682](https://github.com/vllm-project/vllm/issues/39682) **[BUGS] vLLM V1 Engine Hangs After Weight Loading on Blackwell (sm_121) Multi-Node Ray Setup (TP=2)** _(created: 2026-04-13)_
- [#39681](https://github.com/vllm-project/vllm/issues/39681) **[Bug]: Gemma4 multimodal crashes with "pixel_values contains inconsistent shapes" when concurrent image requests have different resolutions** _(created: 2026-04-13)_
- [#39631](https://github.com/vllm-project/vllm/issues/39631) **[Bug]: Abnormal Scores in Batch Processing of Image-Text Pairs with qwen3-VL-reranker Model** _(created: 2026-04-12)_
- [#39583](https://github.com/vllm-project/vllm/issues/39583) **[RFC]: Deprecate bitsandbytes and GGUF quantization support** _(created: 2026-04-11)_
- [#39485](https://github.com/vllm-project/vllm/issues/39485) **[Bug]: Runtime error on ROCm platform serving Deepseek-R1 using VLLM_ROCM_USE_AITER=1** _(created: 2026-04-10)_
- [#39348](https://github.com/vllm-project/vllm/issues/39348) **[Bug]: Qwen3.5-9B-AWQ on ROCm/vLLM 0.19.0 can get stuck generating endless "!" inside JSON schema output** _(created: 2026-04-08)_
- [#39319](https://github.com/vllm-project/vllm/issues/39319) **[Bug]: vLLM docker container with Qwen3.5 - Connection error** _(created: 2026-04-08)_
- [#39288](https://github.com/vllm-project/vllm/issues/39288) **[Bug]: FlashInfer CUTLASS MoE backend causes CUDA illegal memory access on H100 during CUDA graph capture (Qwen3-Next-80B BF16)** _(created: 2026-04-08)_
- [#39210](https://github.com/vllm-project/vllm/issues/39210) **[Bug] Embedding/pooling models crash on B200 (SM 10.0) — encoder attention hardcodes FA2 which lacks SM100 support** _(created: 2026-04-07)_
- [#39202](https://github.com/vllm-project/vllm/issues/39202) **[Bug]: Crash on Transcription (size for tensor a must match the size of tensor b) with reproduce** _(created: 2026-04-07)_
- [#39149](https://github.com/vllm-project/vllm/issues/39149) **[Bug]: Segfault in Triton LLVM (MachineCSE / translateLLVMIRToASM) when serving Qwen3.5-4B on RTX 4090 (WSL2) with vLLM 0.19.0** _(created: 2026-04-07)_
- [#39104](https://github.com/vllm-project/vllm/issues/39104) **[Usage]: The qwen3.5 model generates a random stream of words in thought mode.** _(created: 2026-04-06)_
- [#39061](https://github.com/vllm-project/vllm/issues/39061) **[Bug]: Gemma4 vision encoder crashes with ValueError: Expected hidden_size to be 5376, but found: 72** _(created: 2026-04-06)_
- [#39057](https://github.com/vllm-project/vllm/issues/39057) **[Bug]: Deepseek v3.2 RuntimeError: Worker failed with error "Assertion error"** _(created: 2026-04-06)_
- [#39049](https://github.com/vllm-project/vllm/issues/39049) **[Bug]: Gemma 4 FP8 dynamic quantization = gibberish output** _(created: 2026-04-05)_
- [#39010](https://github.com/vllm-project/vllm/issues/39010) **[Bug]: Hang During CUDA Graph Capture on ROCM in 0.19** _(created: 2026-04-05)_
- [#38999](https://github.com/vllm-project/vllm/issues/38999) **[Bug]: Gemma 4 MoE (26B-A4B) crashes with `--data-parallel-size > 1` — AssertionError in cuda_communicator all_gather** _(created: 2026-04-04)_
- [#38994](https://github.com/vllm-project/vllm/issues/38994) **Qwen-3.5 9B often producing repetitive/garbled output with Intel Backend** _(created: 2026-04-04)_
- [#38976](https://github.com/vllm-project/vllm/issues/38976) **[Bug]:TimeoutError: RPC call to sample_tokens timed out. when pp is on under xpu env** _(created: 2026-04-04)_
- [#38967](https://github.com/vllm-project/vllm/issues/38967) **[Bug] vLLM >= 0.18.0 NCCL segfault (cuMemCreate) with TP>1 on RTX 4090 (SM 89)** _(created: 2026-04-04)_
- [#38918](https://github.com/vllm-project/vllm/issues/38918) **[Usage]: Gemma4 on Turing GPUs (SM 7.5): all attention backends hit shared memory limits** _(created: 2026-04-03)_
- [#38903](https://github.com/vllm-project/vllm/issues/38903) **[Bug]: Cross-request context contamination with async scheduling + pipeline parallelism on multi-node** _(created: 2026-04-03)_
- [#38886](https://github.com/vllm-project/vllm/issues/38886) **[Bug]: Gemma 4 E4B weight loading fails `Gemma4ClippableLinear` parameter `input_max` not recognized** _(created: 2026-04-03)_
- [#38884](https://github.com/vllm-project/vllm/issues/38884) **[Bug]: Gemma 4 torch._dynamo.exc.TorchRuntimeError: Dynamo failed to run FX node with fake tensors** _(created: 2026-04-03)_
- [#38811](https://github.com/vllm-project/vllm/issues/38811) **[Usage]: Qwen3-VL inference on video complains of lack of metadata** _(created: 2026-04-02)_
- [#38710](https://github.com/vllm-project/vllm/issues/38710) **[Bug]: heterogeneous disaggregated serving XPU (Prefill) + CPU (Decode) accuracy issue** _(created: 2026-04-01)_
- [#38660](https://github.com/vllm-project/vllm/issues/38660) **[Bug]: CUDA assert in triton attention for MolmoWeb models (Molmo2 architecture with different max_position_embeddings)** _(created: 2026-03-31)_
- [#38656](https://github.com/vllm-project/vllm/issues/38656) **[Bug]: qwen 3.5 model launch get stuck for quite a long time** _(created: 2026-03-31)_
- [#38603](https://github.com/vllm-project/vllm/issues/38603) **[Bug]: Streaming last chunk contains non-empty tool_calls with empty fields "type" causing type validation error ** _(created: 2026-03-31)_
- [#38586](https://github.com/vllm-project/vllm/issues/38586) **[Bug]: Whisper online benchmark with profiling error: TypeError: multi_modal_content must be a dict containing 'audio'** _(created: 2026-03-30)_
- [#38551](https://github.com/vllm-project/vllm/issues/38551) **[Bug]: AssertionError: Encoder cache miss crashes engine with MTP + multimodal under high concurrency** _(created: 2026-03-30)_
- [#38531](https://github.com/vllm-project/vllm/issues/38531) **[Feature]: Support direct binary/multipart file upload for video and image in OpenAI-compatible API** _(created: 2026-03-30)_
- [#38459](https://github.com/vllm-project/vllm/issues/38459) **[Bug]: `limit_mm_per_prompt` is ineffective for Qwen3-VL** _(created: 2026-03-29)_
- [#38379](https://github.com/vllm-project/vllm/issues/38379) **Upgrade to Transformers v5** _(created: 2026-03-27)_
- [#38351](https://github.com/vllm-project/vllm/issues/38351) **[Bug]: When use_audio_in_video is enabled in qwen3-omni, the output may exhibit issues such as empty or repetitive output.** _(created: 2026-03-27)_
- [#38208](https://github.com/vllm-project/vllm/issues/38208) **[Bug]: CUDA Illegal Instruction during CUDA Graph capture with Nemotron-3-Nano NVFP4 on sm_121** _(created: 2026-03-26)_
- [#38203](https://github.com/vllm-project/vllm/issues/38203) **[Bug]: M2.5 tool call result is badcase, deploy 1p1d with nixl connector, P and D use DP8-EP-TP1** _(created: 2026-03-26)_
- [#38196](https://github.com/vllm-project/vllm/issues/38196) **GDN attention backend crashes with ngram speculative decoding on mixed decode batches** _(created: 2026-03-26)_
- [#38106](https://github.com/vllm-project/vllm/issues/38106) **[Bug]: tool_choice="required" + speculative decoding with lukealonso/Qwen3.5-397B-A17B-NVFP4 leads to failed tool calls.** _(created: 2026-03-25)_
- [#38077](https://github.com/vllm-project/vllm/issues/38077) **[Bug]: Qwen3.5-9B answer !!!!!!!!!** _(created: 2026-03-25)_
- [#38056](https://github.com/vllm-project/vllm/issues/38056) **[Bug]: ImportError: flash_attn.ops.triton.rotary not found on older versions (< v2.1.2)** _(created: 2026-03-25)_
- [#38022](https://github.com/vllm-project/vllm/issues/38022) **[Bug]: Marlin MoE kernel fails with MXFP4-quantized GPT-OSS 20B - Invalid thread config for non-aligned dimensions (K=2880, N=2880)** _(created: 2026-03-24)_
- [#37996](https://github.com/vllm-project/vllm/issues/37996) **[Bug]: Qwen3.5 397B GPTQ model outputs all exclamation points on ROCM** _(created: 2026-03-24)_
- [#37992](https://github.com/vllm-project/vllm/issues/37992) **[Bug]: RuntimeError triton error during profile_run with Qwen3.5-MoE vision encoder on ROCm** _(created: 2026-03-24)_
- [#37981](https://github.com/vllm-project/vllm/issues/37981) **[Bug]: v0.18.0 fails to run MiniCPM-o-4.5** _(created: 2026-03-24)_
- [#37928](https://github.com/vllm-project/vllm/issues/37928) **[Usage]: v0.18.0 nvidia/nemotron-colembed-vl-4b-v2 /embeddings 404** _(created: 2026-03-23)_
- [#37858](https://github.com/vllm-project/vllm/issues/37858) **[Bug]: does not have the attribute 'FakeTensorMode'** _(created: 2026-03-23)_
- [#37828](https://github.com/vllm-project/vllm/issues/37828) **[Bug]: Intel ARC 140v not supported as XE2 cutlass kernel** _(created: 2026-03-22)_
- [#37746](https://github.com/vllm-project/vllm/issues/37746) **[Bug] prompt_logprobs causes livelock with IsHybrid models (Qwen3.5) in DP mode** _(created: 2026-03-21)_
- [#37737](https://github.com/vllm-project/vllm/issues/37737) **[Bug]: Missing logprobs for `<tool_call>` in streaming chat completions** _(created: 2026-03-21)_
- [#37675](https://github.com/vllm-project/vllm/issues/37675) **[Bug]: deepgemm compile error** _(created: 2026-03-20)_
- [#37602](https://github.com/vllm-project/vllm/issues/37602) **[Bug]: Qwen3.5-122B-A10B-FP8 EngineCore crash on concurrent image requests** _(created: 2026-03-19)_
- [#37581](https://github.com/vllm-project/vllm/issues/37581) **[Bug]: /v1/chat/completions/render` crashes for Qwen/Qwen3-ASR-0.6B multimodal audio, and chat audio returns empty/junk** _(created: 2026-03-19)_
- [#37551](https://github.com/vllm-project/vllm/issues/37551) **[Bug] vLLM 0.17.1: `zai-org/GLM-OCR` has `mtp_graph < no_mtp_graph` despite high acceptance** _(created: 2026-03-19)_
- [#37451](https://github.com/vllm-project/vllm/issues/37451) **[Bug]: 0.17.1 - vllm serve deepseek-ai/DeepSeek-OCR-2 on H100 crashes during Capturing CUDA graphs (decode, FULL)** _(created: 2026-03-18)_
- [#37431](https://github.com/vllm-project/vllm/issues/37431) **Mamba-2 Triton kernels crash with illegal instruction on SM121 (DGX Spark) without CUDA_LAUNCH_BLOCKING=1** _(created: 2026-03-18)_
- [#37367](https://github.com/vllm-project/vllm/issues/37367) **[Bug]: gcc: internal compiler error: Segmentation fault signal terminated program cc1** _(created: 2026-03-18)_
- [#37325](https://github.com/vllm-project/vllm/issues/37325) **[Bug][ARM CPU] Build/Runtime error: no matching function for call to ‘at::vec::CPU_CAPABILITY::VecMask<long int, 4>::VecMask(int&)’ when serving Qwen3-VL-8B-Instruct** _(created: 2026-03-17)_
- [#37159](https://github.com/vllm-project/vllm/issues/37159) **[Bug]: vLLM crashed with V100 by running zai-org/GLM-OCR** _(created: 2026-03-16)_
- [#37096](https://github.com/vllm-project/vllm/issues/37096) **[Bug]: v0.17.0-aarch64 onwards will run out of CUDA memory for gpt-oss-120b on GH200 144GB** _(created: 2026-03-15)_
- [#37060](https://github.com/vllm-project/vllm/issues/37060) **[Bug]: sm110: torch.AcceleratorError: CUDA error: an illegal instruction was encountered** _(created: 2026-03-14)_
- [#37035](https://github.com/vllm-project/vllm/issues/37035) **[Bug]: cudaErrorIllegalAddress in gdn_attn.py:237 when using qwen3_next_mtp with num_speculative_tokens=5 under load** _(created: 2026-03-14)_
- [#36906](https://github.com/vllm-project/vllm/issues/36906) **[Bug]: EAGLE3 speculative decoding + multimodal crash under high concurrency** _(created: 2026-03-12)_
- [#36872](https://github.com/vllm-project/vllm/issues/36872) **[Bug]: Gibberish output and collapsing generation throughput with Qwen3.5-35B-A3B-FP8 and speculative decoding enabled** _(created: 2026-03-12)_
- [#36852](https://github.com/vllm-project/vllm/issues/36852) **[Bug]: GPU failure during repeated model loading when using --enable-prefix-caching with KV transfer (LMCacheConnectorV1)** _(created: 2026-03-12)_
- [#36843](https://github.com/vllm-project/vllm/issues/36843) **[Bug]: VLLM 0.17.1 initial mtp with FLASH_ATTN randomly crash** _(created: 2026-03-12)_
- [#36811](https://github.com/vllm-project/vllm/issues/36811) **[Bug]: CUDA illegal memory access on GPTQ Marlin** _(created: 2026-03-11)_
- [#36796](https://github.com/vllm-project/vllm/issues/36796) **[Bug]: CPU offload errors on nightly with NVIDIA GH200 Unified Memory (UMA)** _(created: 2026-03-11)_
- [#36793](https://github.com/vllm-project/vllm/issues/36793) **[Bug]: TP=2 DP=2 Broken for Qwen3-Next W4A16** _(created: 2026-03-11)_
- [#36781](https://github.com/vllm-project/vllm/issues/36781) **[Bug]: vLLM 0.17.0 failed to serve Qwen3-30B-A3B-Instruct-2507 after adding `--enable_lora`** _(created: 2026-03-11)_
- [#36627](https://github.com/vllm-project/vllm/issues/36627) **[Performance]: qwen3.5 vs qwen3** _(created: 2026-03-10)_
- [#36585](https://github.com/vllm-project/vllm/issues/36585) **[Bug]: qwen3.5-27b-gptq deploy fail** _(created: 2026-03-10)_
- [#36566](https://github.com/vllm-project/vllm/issues/36566) **[Bug]:Qwen3.5-35B-A3B vllm v0.17.0 ERROR 03-10 00:52:24 [multiproc_executor.py:261] Worker proc VllmWorker-0 died unexpectedly, shutting down executor.** _(created: 2026-03-10)_
- [#36476](https://github.com/vllm-project/vllm/issues/36476) **[Bug]: vllm 0.17.0 启动Qwen3.5-122B-A10B失败** _(created: 2026-03-09)_
- [#36450](https://github.com/vllm-project/vllm/issues/36450) **[Bug]: Qwen3.5 AWQ models crash during inference on RTX 5090 (Blackwell) with Triton OOM in solve_tril despite successful model load** _(created: 2026-03-09)_
- [#36407](https://github.com/vllm-project/vllm/issues/36407) **[Bug]: KeyError in get_layers_from_vllm_config with pipeline parallelism (vLLM 0.16.0)** _(created: 2026-03-08)_
- [#36350](https://github.com/vllm-project/vllm/issues/36350) **[Bug]: Qwen 3.5 4B fail on first request on Intel XPU (Arc Graphics B580)** _(created: 2026-03-07)_
- [#36275](https://github.com/vllm-project/vllm/issues/36275) **[Bug]: Qwen3.5 4b incompatibility** _(created: 2026-03-06)_
- [#36245](https://github.com/vllm-project/vllm/issues/36245) **[Bug]: Gemma3 mmproj-*.gguf is not downloaded in 'download_gguf'** _(created: 2026-03-06)_
- [#36193](https://github.com/vllm-project/vllm/issues/36193) **[Bug]: Unsupported Activation Function for Step-3.5-Flash** _(created: 2026-03-06)_
- [#36180](https://github.com/vllm-project/vllm/issues/36180) **[Bug]: meta-llama/Llama-3.2-1B-Instruct Fails With ROCM_ATTN Due To Seg Fault** _(created: 2026-03-05)_
- [#36103](https://github.com/vllm-project/vllm/issues/36103) **[Bug]: torch._inductor.exc.InductorError:** _(created: 2026-03-05)_
- [#35950](https://github.com/vllm-project/vllm/issues/35950) **[Bug]: ValueError: too many values to unpack (expected 2)** _(created: 2026-03-04)_
- [#35820](https://github.com/vllm-project/vllm/issues/35820) **[Bug]: deploy Qwen3.5-27B error** _(created: 2026-03-03)_
- [#35778](https://github.com/vllm-project/vllm/issues/35778) **[Bug]: Regression: terrible mixed prefill-decode performance with CUDA graphs enabled.** _(created: 2026-03-02)_
- [#35743](https://github.com/vllm-project/vllm/issues/35743) **[Bug]: Qwen 3.5 27B AWQ 4bit capturing CUDA graph fails** _(created: 2026-03-02)_
- [#35659](https://github.com/vllm-project/vllm/issues/35659) **[Bug]: cudaErrorIllegalAddress under sustained parallel load with CUDA Graphs on Blackwell SM120 (NVFP4 MoE)** _(created: 2026-03-01)_
- [#35624](https://github.com/vllm-project/vllm/issues/35624) **[Bug]: Qwen3-Omni Model Fails when try to l** _(created: 2026-02-28)_
- [#35569](https://github.com/vllm-project/vllm/issues/35569) **[Bug]: [ROCm] ROCM_ATTN backend shows ~8.5% systematic score deviation on Qwen3-VL-Reranker pooling** _(created: 2026-02-28)_
- [#35521](https://github.com/vllm-project/vllm/issues/35521) **[Bug]: Suffix decoding crashes with assert total_num_scheduled_tokens > 0** _(created: 2026-02-27)_
- [#35519](https://github.com/vllm-project/vllm/issues/35519) **[Bug]: Qwen3.5 NVFP4 models crash on ARM64 GB10 DGX Spark (CUDA illegal instruction during generation)** _(created: 2026-02-27)_
- [#35303](https://github.com/vllm-project/vllm/issues/35303) **[Bug] CompressedTensorsWNA16MarlinMoEMethod registers g_idx params unconditionally, crashes with actorder=null AWQ MoE models** _(created: 2026-02-25)_
- [#35169](https://github.com/vllm-project/vllm/issues/35169) **[Bug]: Memory Access Fault during Step-3.5-Flash inference (ROCM)** _(created: 2026-02-24)_
- [#35091](https://github.com/vllm-project/vllm/issues/35091) **[Bug]: Triton CompilationError in speculative decoding (draft_model)** _(created: 2026-02-23)_
- [#35087](https://github.com/vllm-project/vllm/issues/35087) **[Bug]: DeepSeek 3.2 P/D Disaggregation Support** _(created: 2026-02-23)_
- [#35028](https://github.com/vllm-project/vllm/issues/35028) **[Bug]: RuntimeError: CUDA error: CUBLAS_STATUS_INVALID_VALUE when calling `cublasGemmEx** _(created: 2026-02-21)_
- [#34759](https://github.com/vllm-project/vllm/issues/34759) **[Bug]: nvidia/Llama-3.3-70B-Instruct-NVFP4 Degraded / Gibberish Output with TRITON_ATTN** _(created: 2026-02-17)_
- [#34518](https://github.com/vllm-project/vllm/issues/34518) **[Feature]: [Whisper] Support for decoder prefix and custom task tokens in transcription API** _(created: 2026-02-13)_
- [#34505](https://github.com/vllm-project/vllm/issues/34505) **[Bug]: Qwen 2.5 Omni cuda graph has error** _(created: 2026-02-13)_
- [#34406](https://github.com/vllm-project/vllm/issues/34406) **[Bug]: Instruction following capability is deteriorating：Output introduces parameter  defined in functioncall incorrectly** _(created: 2026-02-12)_
- [#34295](https://github.com/vllm-project/vllm/issues/34295) **[Bug]:** _(created: 2026-02-11)_
- [#34210](https://github.com/vllm-project/vllm/issues/34210) **[Bug]: Enable DBO on Qwen3-VL-235B-A22B raise TypeError: 'NoneType' object is not subscriptable** _(created: 2026-02-10)_
- [#34209](https://github.com/vllm-project/vllm/issues/34209) **[Usage]: How to decrease Encoder cache budget from 16384 tokens to something lower?** _(created: 2026-02-10)_
- [#34201](https://github.com/vllm-project/vllm/issues/34201) **[Bug]: AttributeError: 'Parameter' object has no attribute 'weight_loader'** _(created: 2026-02-10)_
- [#34154](https://github.com/vllm-project/vllm/issues/34154) **[Bug]: Failing to run DeepSeek-OCR on Radeon GPUs (memory fault)** _(created: 2026-02-09)_
- [#33920](https://github.com/vllm-project/vllm/issues/33920) **[Bug]: GLM-4.7-Flash OOM during sampler warmup with tensor parallelism on RTX 4090** _(created: 2026-02-05)_
- [#33916](https://github.com/vllm-project/vllm/issues/33916) **[Bug] IndexError: list index out of range in chat_completion_stream_generator with --tool-call-parser=mistral during streaming tool calls** _(created: 2026-02-05)_
- [#33871](https://github.com/vllm-project/vllm/issues/33871) **[Bug]: The local deployment achieves about 30% higher accuracy compared to the server deployment.** _(created: 2026-02-05)_
- [#33865](https://github.com/vllm-project/vllm/issues/33865) **[Bug]: OpenAI-compatible Embeddings API intermittently crashes with multimodal cache assertion (`Expected a cached item for mm_hash`) on Qwen3-VL-Embedding-8B** _(created: 2026-02-05)_
- [#33828](https://github.com/vllm-project/vllm/issues/33828) **[Bug]: mistral3 offline multimodal inference example failing with prompt placeholder error** _(created: 2026-02-04)_
- [#33672](https://github.com/vllm-project/vllm/issues/33672) **[Bug]: HTTP API multimodal embedding causes image_pad token duplication, producing incorrect results** _(created: 2026-02-03)_
- [#33654](https://github.com/vllm-project/vllm/issues/33654) **[Bug]: The content of response from Kimi-K2.5 is empty.** _(created: 2026-02-03)_
- [#33628](https://github.com/vllm-project/vllm/issues/33628) **[Bug]: Failed to run distributed inference with Gloo backend on aarch64** _(created: 2026-02-03)_
- [#33534](https://github.com/vllm-project/vllm/issues/33534) **[Installation]: vllm with ray in AWS cluster** _(created: 2026-02-02)_
- [#33424](https://github.com/vllm-project/vllm/issues/33424) **[Bug]: LoRA MoE Not Matching HF Output** _(created: 2026-01-30)_
- [#33338](https://github.com/vllm-project/vllm/issues/33338) **[Bug]: Crash when using presence_penalty with Qwen3-VL in v0.11.0** _(created: 2026-01-29)_
- [#33307](https://github.com/vllm-project/vllm/issues/33307) **[Bug]: Latency spikes at input_len=1024 with batch_size=16 (TP1 & TP2)** _(created: 2026-01-29)_
- [#33204](https://github.com/vllm-project/vllm/issues/33204) **[Bug]: Qwen3-VL-Embedding model produces different embeddings than official qwen_vl_utils implementation** _(created: 2026-01-27)_
- [#33107](https://github.com/vllm-project/vllm/issues/33107) **[Bug]: Whisper large-v3 accuracy degradation in vLLM 0.14.1 (134.56% WER) on L40S - works fine in 0.12.0** _(created: 2026-01-26)_
- [#32921](https://github.com/vllm-project/vllm/issues/32921) **[Bug]: gpt-oss-20b streaming last reasoning content part into content** _(created: 2026-01-23)_
- [#32897](https://github.com/vllm-project/vllm/issues/32897) **[Bug]: PaddleOCR-VL crash** _(created: 2026-01-23)_
- [#32864](https://github.com/vllm-project/vllm/issues/32864) **[Bug] [ROCm]: Loading Qwen3-MoE-MXFP4 Weights in v0.14.** _(created: 2026-01-22)_
- [#32826](https://github.com/vllm-project/vllm/issues/32826) **[Bug]: MiniMax-M2.1 NVFP4 fails on RTX PRO 6000 Blackwell (SM120) with expert parallel** _(created: 2026-01-22)_
- [#32732](https://github.com/vllm-project/vllm/issues/32732) **[Bug]: Regression in v0.14.0: "No valid attention backend found" for nvidia/DeepSeek-R1-0528-NVFP4 on RTX Pro 6000 (Blackwell)** _(created: 2026-01-20)_
- [#32636](https://github.com/vllm-project/vllm/issues/32636) **[Bug]: Invalid base64-encoded string for audio input** _(created: 2026-01-20)_
- [#32604](https://github.com/vllm-project/vllm/issues/32604) **[Bug]: LMCache CPU kv offload cause decode speed degrade** _(created: 2026-01-19)_
- [#32600](https://github.com/vllm-project/vllm/issues/32600) **[Bug]:** _(created: 2026-01-19)_
- [#32410](https://github.com/vllm-project/vllm/issues/32410) **[Bug][XPU]: Failed to serve w4a16 Qwen3 VL MoE on Intel Arc Pro B60** _(created: 2026-01-15)_
- [#32338](https://github.com/vllm-project/vllm/issues/32338) **[Feature]: support for LlavaQwenForCausalLM** _(created: 2026-01-14)_
- [#32259](https://github.com/vllm-project/vllm/issues/32259) **[Bug]: offline infer of mm model cache** _(created: 2026-01-13)_
- [#32176](https://github.com/vllm-project/vllm/issues/32176) **[Bug]: deepseekv3.2 core dumped with cpu_offload_gb** _(created: 2026-01-12)_
- [#32151](https://github.com/vllm-project/vllm/issues/32151) **[Bug]: jina-reranker-m0 infer error** _(created: 2026-01-12)_
- [#32068](https://github.com/vllm-project/vllm/issues/32068) **[Bug]: Recompile in LLama model** _(created: 2026-01-10)_
- [#32012](https://github.com/vllm-project/vllm/issues/32012) **[Bug]: Qwen3-VL-MoE fails loading when using bitsandbytes quantization** _(created: 2026-01-09)_
- [#31936](https://github.com/vllm-project/vllm/issues/31936) **[Bug]: run deepseek v3.2 failed，not support RTX PRO 6000 * 8？** _(created: 2026-01-08)_
- [#31884](https://github.com/vllm-project/vllm/issues/31884) **[Bug]: run Qwen3-30B-A3B on 8*A800 2  nodes with DP failed zmq.error.ZMQError** _(created: 2026-01-07)_
- [#31871](https://github.com/vllm-project/vllm/issues/31871) **[Bug]: Streaming mode with --tool-call-parser hermes returns raw text instead of parsed tool_calls** _(created: 2026-01-07)_
- [#31709](https://github.com/vllm-project/vllm/issues/31709) **[Bug]: After upgrade to 0.11.2, vllm crashs with Qwen3.** _(created: 2026-01-05)_
- [#31661](https://github.com/vllm-project/vllm/issues/31661) **[Bug]: jina-reranker-m0 [image_index]   IndexError: list index out of range** _(created: 2026-01-04)_
- [#31647](https://github.com/vllm-project/vllm/issues/31647) **[Bug]: TypeError: BatchPrefillWithPagedKVCacheWrapper.plan() got an unexpected keyword argument 'fixed_split_size'** _(created: 2026-01-03)_
- [#31484](https://github.com/vllm-project/vllm/issues/31484) **[Usage]: RuntimeError when running Qwen2.5-VL-7B-Instruct with vllm: Potential version incompatibility** _(created: 2025-12-29)_
- [#31353](https://github.com/vllm-project/vllm/issues/31353) **[Bug]: KV Cache grows continuously with just one chat completion request using meta-llama/Llama-3.2-1B on L40 GPU with Flash Attention and finally completed after 10 minutes** _(created: 2025-12-25)_
- [#31063](https://github.com/vllm-project/vllm/issues/31063) **[Bug]: Qwen3VL-8B-instruct-FP8 has larger result diff rate than sglang compared with transformers** _(created: 2025-12-20)_
- [#30819](https://github.com/vllm-project/vllm/issues/30819) **[Bug]: vLLM inference stuck when requesting video description on VLM models** _(created: 2025-12-16)_
- [#30779](https://github.com/vllm-project/vllm/issues/30779) **[Bug]: v0.11.2 can not support Qwen2.5-Omni-** _(created: 2025-12-16)_
- [#30777](https://github.com/vllm-project/vllm/issues/30777) **[Bug]: whisper-large-v3-turbo have accuracy problem on nightly build** _(created: 2025-12-16)_
- [#30493](https://github.com/vllm-project/vllm/issues/30493) **[Bug]: 5090 RTX seems to be broken** _(created: 2025-12-11)_
- [#30180](https://github.com/vllm-project/vllm/issues/30180) **[Bug]: Inference of Qwen3-VL-235B failed** _(created: 2025-12-06)_
- [#30167](https://github.com/vllm-project/vllm/issues/30167) **[Bug][ROCm]: `vision_embeddings` in transformers inaccurate without math SDP** _(created: 2025-12-06)_
- [#29863](https://github.com/vllm-project/vllm/issues/29863) **[Bug]: fails to inference prompt ends with '.' ':' for video inputs** _(created: 2025-12-02)_
- [#29373](https://github.com/vllm-project/vllm/issues/29373) **[Bug]: Multinode inference request with Ray and vLLM crashes - regression from vLLM v0.7.3** _(created: 2025-11-25)_
- [#28919](https://github.com/vllm-project/vllm/issues/28919) **[Bug]: Qwen3-32B use eagle3 crash** _(created: 2025-11-18)_
- [#28707](https://github.com/vllm-project/vllm/issues/28707) **[Bug]: run fail in nvidia b300** _(created: 2025-11-14)_
- [#28640](https://github.com/vllm-project/vllm/issues/28640) **[Bug]: LoRA/Adapter Loading Error with Qwen3-VL-8B-Instruct Multimodal Model in vLLM Deployment (AssertionError in lora_shrink_op)** _(created: 2025-11-13)_
- [#28568](https://github.com/vllm-project/vllm/issues/28568) **[Bug]: FlashInfer attention backend on Hopper fails with llama4-scout and llama3 with fp8 kvcache** _(created: 2025-11-12)_
- [#28467](https://github.com/vllm-project/vllm/issues/28467) **[Bug]: Persistent CPU RAM Growth and Container Restarts with QWEN2-2B on SageMaker, Unaffected by Cache Settings** _(created: 2025-11-11)_
- [#28388](https://github.com/vllm-project/vllm/issues/28388) **[Bug]: 新版的vllm已经废弃了v0代码，而对qwen-omni系列的模型支持仅限于v0，似乎是因为这个原因，我们无法使用最新版的vllm推理qwen-omni模型** _(created: 2025-11-10)_
- [#27932](https://github.com/vllm-project/vllm/issues/27932) **[Feature]: Qwen3 Omini AttributeError: 'Qwen3OmniMoeProcessor' object has no attribute '_get_num_multimodal_tokens'** _(created: 2025-11-02)_
- [#27430](https://github.com/vllm-project/vllm/issues/27430) **[Bug]: vLLM (TP=8) on 235B model triggers "CUDA error: unspecified launch failure" and persistent "ERR!" state in nvidia-smi** _(created: 2025-10-23)_
- [#27364](https://github.com/vllm-project/vllm/issues/27364) **[Bug]: Qwen3-VL {4B,8B} FP8 on vLLM returns only exclamation marks ("!!!!!...") on Jetson Thor** _(created: 2025-10-22)_
- [#27340](https://github.com/vllm-project/vllm/issues/27340) **[Bug]: Qwen3-VL-2B-Instruct vllm推理报错** _(created: 2025-10-22)_
- [#27120](https://github.com/vllm-project/vllm/issues/27120) **[Bug]: some models won't stream from s3: Qwen/Qwen3-VL-30B-A3B-Instruct-FP8** _(created: 2025-10-17)_
- [#26777](https://github.com/vllm-project/vllm/issues/26777) **[Bug]: Deploying Qwen2.5-VL-7B with Eagle3, the service crashes when concurrency increases.** _(created: 2025-10-14)_
- [#26480](https://github.com/vllm-project/vllm/issues/26480) **[Bug][v0.11.0]: gpt-oss-120b generates with no output** _(created: 2025-10-09)_
- [#26420](https://github.com/vllm-project/vllm/issues/26420) **[Bug]: vLLM server not starting when running Qwen/Qwen3-VL-30B-A3B-Instruct on 2 A6000 GPUs. ** _(created: 2025-10-08)_
- [#25943](https://github.com/vllm-project/vllm/issues/25943) **[Feature]: Disable ANSII colors in logs of OpenAI compatible server** _(created: 2025-09-30)_
- [#25502](https://github.com/vllm-project/vllm/issues/25502) **[Bug]: This flash attention build does not support tanh softcapping: gemma-2-2b-it on H100 NVL** _(created: 2025-09-23)_
- [#25192](https://github.com/vllm-project/vllm/issues/25192) **[Bug]: Make Whisper model compatible with cuda graphs** _(created: 2025-09-18)_
- [#24289](https://github.com/vllm-project/vllm/issues/24289) **[Bug]: module 'triton.language' has no attribute 'constexpr_function'** _(created: 2025-09-05)_
- [#23404](https://github.com/vllm-project/vllm/issues/23404) **[Bug]: Qwen3 vLLM Structured Output Ignores Field Descriptions** _(created: 2025-08-22)_
- [#22424](https://github.com/vllm-project/vllm/issues/22424) **[Bug]: Voxtral-Small-24B-2507 Does Not Support Pipeline-Parallel** _(created: 2025-08-07)_
- [#20261](https://github.com/vllm-project/vllm/issues/20261) **[Bug]: Prefix caching ignores visual input, causing incorrect multimodal outputs under concurrency** _(created: 2025-06-30)_
- [#20149](https://github.com/vllm-project/vllm/issues/20149) **[Feature]: Add Support for Updating Lora Weights** _(created: 2025-06-26)_
- [#19668](https://github.com/vllm-project/vllm/issues/19668) **[Bug]: vllm, EngineCore encountered a fatal error TimeoutError** _(created: 2025-06-15)_
- [#19445](https://github.com/vllm-project/vllm/issues/19445) **[Bug]: Docker image CUDA error on RTX 2080 Ti** _(created: 2025-06-10)_
- [#17972](https://github.com/vllm-project/vllm/issues/17972) **[Bug]: vLLM server hangs and timeouts after initial requests** _(created: 2025-05-12)_
- [#16966](https://github.com/vllm-project/vllm/issues/16966) **[Bug]: vllm 0.8.4 whisper possible memory leak?** _(created: 2025-04-22)_
- [#13941](https://github.com/vllm-project/vllm/issues/13941) **[Bug]: wake up OOM (72B model in 8*A800(40G))** _(created: 2025-02-27)_
- [#4194](https://github.com/vllm-project/vllm/issues/4194) **[RFC]: Multi-modality Support on vLLM** _(created: 2024-04-19)_

### Documentation / 文档 (60)

- [#40802](https://github.com/vllm-project/vllm/issues/40802) **[Feature]: Deepseek V4 cannot run ,Please support SM120 GPU,example rtx5090  rtxpro6000** _(created: 2026-04-24)_
- [#40781](https://github.com/vllm-project/vllm/issues/40781) **[Feature]: vllm support audio otel tracing?** _(created: 2026-04-24)_
- [#40591](https://github.com/vllm-project/vllm/issues/40591) **[Bug]: Regression in 0.19.1 - Gemma 4 26B MoE fails to load packed experts (KeyError: down_proj_packed). Worked in dev6.** _(created: 2026-04-22)_
- [#40543](https://github.com/vllm-project/vllm/issues/40543) **[Feature]: Dynamic PDL Enablement** _(created: 2026-04-21)_
- [#39814](https://github.com/vllm-project/vllm/issues/39814) **[Bug]: FlashInferFP8ScaledMMLinearKernel segfaults on Blackwell (sm100)** _(created: 2026-04-14)_
- [#39766](https://github.com/vllm-project/vllm/issues/39766) **[RFC]: Support Mooncake Based ECConnector for EPD** _(created: 2026-04-14)_
- [#39761](https://github.com/vllm-project/vllm/issues/39761) **[Bug]:CUDA illegal instruction during decode (V1 Engine + NVFP4) on aarch64 (NVIDIA GB10)** _(created: 2026-04-14)_
- [#39708](https://github.com/vllm-project/vllm/issues/39708) **[Feature]: Pre-ViT visual token pruning for VLMs (PixelPrune)** _(created: 2026-04-13)_
- [#39504](https://github.com/vllm-project/vllm/issues/39504) **[RFC]: Enable prompt_embeds content parts in Chat Completions API** _(created: 2026-04-10)_
- [#39474](https://github.com/vllm-project/vllm/issues/39474) **[Bug] Regression: GPTQ models fail to load on Intel XPU in v0.19.0 (missing XPU branches in gptq.py)** _(created: 2026-04-10)_
- [#39319](https://github.com/vllm-project/vllm/issues/39319) **[Bug]: vLLM docker container with Qwen3.5 - Connection error** _(created: 2026-04-08)_
- [#38999](https://github.com/vllm-project/vllm/issues/38999) **[Bug]: Gemma 4 MoE (26B-A4B) crashes with `--data-parallel-size > 1` — AssertionError in cuda_communicator all_gather** _(created: 2026-04-04)_
- [#38994](https://github.com/vllm-project/vllm/issues/38994) **Qwen-3.5 9B often producing repetitive/garbled output with Intel Backend** _(created: 2026-04-04)_
- [#38967](https://github.com/vllm-project/vllm/issues/38967) **[Bug] vLLM >= 0.18.0 NCCL segfault (cuMemCreate) with TP>1 on RTX 4090 (SM 89)** _(created: 2026-04-04)_
- [#38925](https://github.com/vllm-project/vllm/issues/38925) **[Feature]: Support lightweight import of vllm protocol types without torch dependency** _(created: 2026-04-03)_
- [#38918](https://github.com/vllm-project/vllm/issues/38918) **[Usage]: Gemma4 on Turing GPUs (SM 7.5): all attention backends hit shared memory limits** _(created: 2026-04-03)_
- [#38713](https://github.com/vllm-project/vllm/issues/38713) **[Bug]: Error when trying to serve MiniMax 2.5 on 4 H100 nodes with 4 GPUS** _(created: 2026-04-01)_
- [#38660](https://github.com/vllm-project/vllm/issues/38660) **[Bug]: CUDA assert in triton attention for MolmoWeb models (Molmo2 architecture with different max_position_embeddings)** _(created: 2026-03-31)_
- [#38560](https://github.com/vllm-project/vllm/issues/38560) **[Bug]: reasoning_effort passed to MistralCommonTokenizer.apply_chat_template breaks Mistral Small 4 chat completions on vLLM 0.18.0** _(created: 2026-03-30)_
- [#38486](https://github.com/vllm-project/vllm/issues/38486) **[Bug]: cuda graph takes too much memory for qwen 3.5** _(created: 2026-03-29)_
- [#38351](https://github.com/vllm-project/vllm/issues/38351) **[Bug]: When use_audio_in_video is enabled in qwen3-omni, the output may exhibit issues such as empty or repetitive output.** _(created: 2026-03-27)_
- [#38233](https://github.com/vllm-project/vllm/issues/38233) **[Bug]: Voxtral-Mini-4B-Realtime hangs/crashes on multiple sessions due to encoder_cache_usage saturation on 16GB GPU** _(created: 2026-03-26)_
- [#38004](https://github.com/vllm-project/vllm/issues/38004) **[Bug]: Speech-to-Text endpoint may return 501 but not documented in OpenAPI** _(created: 2026-03-24)_
- [#37431](https://github.com/vllm-project/vllm/issues/37431) **Mamba-2 Triton kernels crash with illegal instruction on SM121 (DGX Spark) without CUDA_LAUNCH_BLOCKING=1** _(created: 2026-03-18)_
- [#37423](https://github.com/vllm-project/vllm/issues/37423) **[Feature]: Allow passing `images` to CompletionRequest** _(created: 2026-03-18)_
- [#37367](https://github.com/vllm-project/vllm/issues/37367) **[Bug]: gcc: internal compiler error: Segmentation fault signal terminated program cc1** _(created: 2026-03-18)_
- [#37096](https://github.com/vllm-project/vllm/issues/37096) **[Bug]: v0.17.0-aarch64 onwards will run out of CUDA memory for gpt-oss-120b on GH200 144GB** _(created: 2026-03-15)_
- [#36627](https://github.com/vllm-project/vllm/issues/36627) **[Performance]: qwen3.5 vs qwen3** _(created: 2026-03-10)_
- [#36566](https://github.com/vllm-project/vllm/issues/36566) **[Bug]:Qwen3.5-35B-A3B vllm v0.17.0 ERROR 03-10 00:52:24 [multiproc_executor.py:261] Worker proc VllmWorker-0 died unexpectedly, shutting down executor.** _(created: 2026-03-10)_
- [#36450](https://github.com/vllm-project/vllm/issues/36450) **[Bug]: Qwen3.5 AWQ models crash during inference on RTX 5090 (Blackwell) with Triton OOM in solve_tril despite successful model load** _(created: 2026-03-09)_
- [#36015](https://github.com/vllm-project/vllm/issues/36015) **[Bug]: Realtime audio transcription (Voxtral) silently hangs after ~10 minutes due to unhandled TimeoutError in background task** _(created: 2026-03-04)_
- [#35950](https://github.com/vllm-project/vllm/issues/35950) **[Bug]: ValueError: too many values to unpack (expected 2)** _(created: 2026-03-04)_
- [#35863](https://github.com/vllm-project/vllm/issues/35863) **[Bug]: Voxtral-Realtime stops returning transcribed text starting from the 3rd concurrent session** _(created: 2026-03-03)_
- [#35642](https://github.com/vllm-project/vllm/issues/35642) **[Bug]: HIP build in Docker: offload-arch stderr contaminates compiler flags via cmake/utils.cmake and CMAKE_HIP_FLAGS** _(created: 2026-03-01)_
- [#35608](https://github.com/vllm-project/vllm/issues/35608) **[Bug]: vllm 0.16.0+image encountered CUDA error: CUBLAS_STATUS_INVALID_VALUE when calling `cublasGemmEx** _(created: 2026-02-28)_
- [#35303](https://github.com/vllm-project/vllm/issues/35303) **[Bug] CompressedTensorsWNA16MarlinMoEMethod registers g_idx params unconditionally, crashes with actorder=null AWQ MoE models** _(created: 2026-02-25)_
- [#35276](https://github.com/vllm-project/vllm/issues/35276) **[Bug]: OpenAI transcribe prompt parameter with whisper return hallucinated transcription** _(created: 2026-02-25)_
- [#35141](https://github.com/vllm-project/vllm/issues/35141) **[Feature]: Sequence Parallel Support for Model Runner V2** _(created: 2026-02-23)_
- [#34536](https://github.com/vllm-project/vllm/issues/34536) **[Feature]: Reasoning output for offline inference** _(created: 2026-02-13)_
- [#34323](https://github.com/vllm-project/vllm/issues/34323) **[CI Failure]: Spawned tests can fail silently** _(created: 2026-02-11)_
- [#34250](https://github.com/vllm-project/vllm/issues/34250) **[Bug]: using vllm on Qwen3-Omni-30B-A3B-Instruct: Failed to apply prompt replacement for mm_items['audio'][0].** _(created: 2026-02-10)_
- [#33986](https://github.com/vllm-project/vllm/issues/33986) **[Tracking issue]: Issue Tracker for Qwen/Qwen3-VL-Embedding & Qwen/Qwen3-VL-Reranker** _(created: 2026-02-06)_
- [#33920](https://github.com/vllm-project/vllm/issues/33920) **[Bug]: GLM-4.7-Flash OOM during sampler warmup with tensor parallelism on RTX 4090** _(created: 2026-02-05)_
- [#33828](https://github.com/vllm-project/vllm/issues/33828) **[Bug]: mistral3 offline multimodal inference example failing with prompt placeholder error** _(created: 2026-02-04)_
- [#33336](https://github.com/vllm-project/vllm/issues/33336) **[Bug]: AttributeError: 'Step3VLProcessor' object has no attribute '_get_num_multimodal_tokens'** _(created: 2026-01-29)_
- [#32925](https://github.com/vllm-project/vllm/issues/32925) **[Bug]: tensorize_vllm_model double gpu** _(created: 2026-01-23)_
- [#32732](https://github.com/vllm-project/vllm/issues/32732) **[Bug]: Regression in v0.14.0: "No valid attention backend found" for nvidia/DeepSeek-R1-0528-NVFP4 on RTX Pro 6000 (Blackwell)** _(created: 2026-01-20)_
- [#32685](https://github.com/vllm-project/vllm/issues/32685) **[Feature]: Support multi-modal inputs for OpenAI Response API** _(created: 2026-01-20)_
- [#32046](https://github.com/vllm-project/vllm/issues/32046) **Google MedASR** _(created: 2026-01-09)_
- [#31372](https://github.com/vllm-project/vllm/issues/31372) **[Feature]: Running paddocr‑vl consumes an excessively large amount of  memory.** _(created: 2025-12-26)_
- [#31063](https://github.com/vllm-project/vllm/issues/31063) **[Bug]: Qwen3VL-8B-instruct-FP8 has larger result diff rate than sglang compared with transformers** _(created: 2025-12-20)_
- [#29968](https://github.com/vllm-project/vllm/issues/29968) **[Bug]: Ministral 3 - streaming tool call not working** _(created: 2025-12-03)_
- [#29362](https://github.com/vllm-project/vllm/issues/29362) **[RFC]: Resettle examples.** _(created: 2025-11-25)_
- [#28388](https://github.com/vllm-project/vllm/issues/28388) **[Bug]: 新版的vllm已经废弃了v0代码，而对qwen-omni系列的模型支持仅限于v0，似乎是因为这个原因，我们无法使用最新版的vllm推理qwen-omni模型** _(created: 2025-11-10)_
- [#27120](https://github.com/vllm-project/vllm/issues/27120) **[Bug]: some models won't stream from s3: Qwen/Qwen3-VL-30B-A3B-Instruct-FP8** _(created: 2025-10-17)_
- [#27094](https://github.com/vllm-project/vllm/issues/27094) **[RFC]: Remove redundant multi-modal input preprocessing during disaggregated inference** _(created: 2025-10-17)_
- [#25750](https://github.com/vllm-project/vllm/issues/25750) **[Feature]: Tracking Whisper feature requests** _(created: 2025-09-26)_
- [#19445](https://github.com/vllm-project/vllm/issues/19445) **[Bug]: Docker image CUDA error on RTX 2080 Ti** _(created: 2025-06-10)_
- [#10849](https://github.com/vllm-project/vllm/issues/10849) **[Feature]: add DoRA support** _(created: 2024-12-03)_
- [#4194](https://github.com/vllm-project/vllm/issues/4194) **[RFC]: Multi-modality Support on vLLM** _(created: 2024-04-19)_

### 其他 / Other (5)

- [#38256](https://github.com/vllm-project/vllm/issues/38256) **[RFC]: Incremental MoE Expert Offloading — GPU Cache + Async Pipeline** _(created: 2026-03-26)_
- [#37674](https://github.com/vllm-project/vllm/issues/37674) **[Usage]: serve部署模型后，调用chat.completions输入给模型的text中image_pad token被提到了prompt开头** _(created: 2026-03-20)_
- [#37113](https://github.com/vllm-project/vllm/issues/37113) **[SM120][GLM-5.1] NVFP4 DCP/MTP stack tracker** _(created: 2026-03-15)_
- [#35550](https://github.com/vllm-project/vllm/issues/35550) **[RFC]: Batch-Aware Expert Pruning for MoE Decode (XShare)** _(created: 2026-02-27)_
- [#32218](https://github.com/vllm-project/vllm/issues/32218) **[RFC]: Consolidate Multimodal Related Info** _(created: 2026-01-12)_

---

## 三、分类统计

| 问题类型 | Issue 数量 |
|----------|-----------|
| Bug / 错误报告 | 310 |
| Feature Request / 功能请求 | 178 |
| Performance / 性能问题 | 51 |
| CUDA Graph / 计算图 | 26 |
| EPD / 编解码分离 | 12 |
| Prefix Caching / 前缀缓存 | 6 |
| ViT / 视觉编码器 | 6 |
| Video Multimodal / 视频多模态 | 23 |
| Audio / Speech / 音频语音 | 46 |
| VL Model: Qwen-VL系列 | 8 |
| VL Model: Qwen-Omni系列 | 2 |
| VL Model: LLaVA系列 | 3 |
| VL Model: InternVL系列 | 3 |
| VL Model: 其他模型 | 3 |
| Image Token / 输入处理 | 6 |
| Compatibility / 兼容性 | 205 |
| Documentation / 文档 | 60 |
| 其他 / Other | 5 |
