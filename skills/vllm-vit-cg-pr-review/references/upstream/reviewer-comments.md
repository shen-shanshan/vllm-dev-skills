# ViT CUDA graph reviewer comments (raw corpus)

> Harvested 2026-11-26 by `scripts/update_skill.py` from the PRs linked in [#38175](https://github.com/vllm-project/vllm/issues/38175). Reviewers: @shen-shanshan, @Isotr0py; author replies are kept for context. Distilled rules live in `../vit-cg-review-rules.md` and `../reviewer-playbook.md` — this file is the evidence they were built from. Do not edit by hand.

## PR #40660: [MM][Perf][CG] Support ViT full cudagraphs for mllama4 (@ghost, merged)

### `vllm/model_executor/models/mllama4.py:730`
```diff

```

- **@shen-shanshan**: Please also update this model to:

Doc: https://docs.vllm.ai/en/latest/design/cuda_graphs_multimodal/#model-integration-via-supportsencodercudagraph

Example: https://github.com/vllm-project/vllm/blob/main/examples/offline_inference/vision_language.py#L2466-L2469 (if needed)

CI test: https://github.com/vllm-project/vllm/blob/main/tests/models/multimodal/generation/test_vit_cudagraph.py#L44-L58

- **@ghost**: @shen-shanshan done

### `tests/models/multimodal/generation/test_vit_cudagraph.py:65`
```diff
+        model="meta-llama/Llama-4-Scout-17B-16E-Instruct",
+        modalities=["image"],
+        image_prompt=(
+            "<|begin_of_text|><|header_start|>user<|header_end|>\n\n"
+            "<|image|><|eot|><|header_start|>assistant<|header_end|>\n\n"
+        ),
```

- **@shen-shanshan**: Maybe we should add a question (like "What is in this image?") in the prompt template, referring to Qwen3-VL below.

### `vllm/model_executor/models/mllama4.py:742`
```diff
@@ -737,6 +739,7 @@ class Llama4ForConditionalGeneration(
     }
 
     supports_encoder_tp_data = True
+    supports_encoder_cudagraph = True
```

- **@shen-shanshan**: I suppose this is not used?

### `vllm/model_executor/models/mllama4.py:892`
```diff
+    def get_encoder_cudagraph_per_item_output_tokens(
+        self,
+        mm_kwargs: dict[str, object],
+    ) -> list[int]:
+        patches_per_chunk = self.get_image_patches_per_chunk()
+        patches_per_image = mm_kwargs["patches_per_image"]
```

- **@shen-shanshan**: I'm not so familiar with this model, the `patches_per_image` here actually means `chunks_per_image`, am I understanding correctly?

### `vllm/model_executor/models/mllama4.py:970`
```diff
+    ):
+        from vllm.v1.worker.encoder_cudagraph_defs import (
+            EncoderCudaGraphReplayBuffers,
+        )
+
+        return EncoderCudaGraphReplayBuffers(buffers={})
```

- **@shen-shanshan**: Maybe we can put this as a default implementation into the ViT CG model interface, after which we can reduce the interface needed overwrite for models whose ViT doesn't require additional buffers.

- **@ghost**: I'd prefer to do this as a follow-up later bc i feel its out of scope here. just confirm if you would rather I do it right now.

- **@shen-shanshan**: > I'd prefer to do this as a follow-up later bc i feel its out of scope here. just confirm if you would rather I do it right now.

Yeah, we can put this in a following PR.

### `vllm/model_executor/models/mllama4.py:982`
```diff
+        mm_kwargs: dict[str, object],
+    ) -> torch.Tensor:
+        return self.encode_image_chunks(
+            mm_kwargs["pixel_values"],
+            use_data_parallel=False,
+        ).flatten(0, 1)
```

- **@shen-shanshan**: Why we fix the value of `use_data_parallel` as `False`? I'm not sure if this model supports ViT DP mode?

### `tests/models/multimodal/generation/test_vit_cudagraph.py:59`
```diff
@@ -42,6 +42,21 @@ def qwen_vl_chat_template(content: str) -> str:
 
 
 MODEL_CONFIGS: dict[str, VitCudagraphTestConfig] = {
+    "llama4": VitCudagraphTestConfig(
+        model="meta-llama/Llama-4-Scout-17B-16E-Instruct",
```

- **@shen-shanshan**: I'm not sure if this 17B model would cause OOM issues in the CI.

FYI, https://github.com/vllm-project/vllm/pull/43082.

- **@ghost**: yea... this is tricky. llama4 cg test is probably too heavy. I can shrink the config more but it looks like the only options are moving the llama4 cg testing to a different hardware config or not testing llama4. 

what r ur thoughts?

- **@Isotr0py**: You can use dummy weights to test this model, because we only test basic functionality here.  You can refer to Step3-VL below to reduce num layers in config.

### `vllm/model_executor/models/mllama4.py:867`
```diff
+        mm_kwargs: dict[str, Any],
+    ) -> str:
+        return "image"
+
+    def get_max_frames_per_video(self) -> int:
+        return 0
```

- **@shen-shanshan**: This method is not needed now for image-only models, since https://github.com/vllm-project/vllm/pull/41234/changes#diff-8214882028a01f717c0aac8d8261b29bd24e965d393b13aaa37acd96c26c6482R54-R57 has set it default to 1.

### `vllm/model_executor/models/mllama4.py:898`
```diff
+
+    def get_encoder_cudagraph_per_item_input_sizes(
+        self,
+        mm_kwargs: dict[str, Any],
+    ) -> list[int]:
+        return mm_kwargs["patches_per_image"].tolist()
```

- **@shen-shanshan**: These 3 interfaces has been merged into 1 method in https://github.com/vllm-project/vllm/pull/41234. You should update it.

### `vllm/model_executor/models/mllama4.py:898`
```diff
+
+    def get_encoder_cudagraph_per_item_input_sizes(
+        self,
+        mm_kwargs: dict[str, Any],
+    ) -> list[int]:
+        return mm_kwargs["patches_per_image"].tolist()
```

- **@shen-shanshan**: I'm not sure the `input_sizes` per item here should be `chunks_per_image` or `chunks_per_image * patches_per_chunk`? which one (`chunk` or `patch`) represents the actual input image token in this model? Could you please double check it? Thanks.

- **@ghost**: input_size should be chunks_per_image. The actual encoder input tensor is pixel_values[total_num_chunks, C, H, W]. output_tokens is chunks_per_image * patches_per_chunk.

- **@shen-shanshan**: > input_size should be chunks_per_image. The actual encoder input tensor is pixel_values[total_num_chunks, C, H, W]. output_tokens is chunks_per_image * patches_per_chunk.

OK.

### `tests/models/multimodal/generation/test_common.py:700`
```diff
@@ -697,7 +697,7 @@ def _granite4_vision_vllm_to_hf_output(vllm_output, model):
         dtype="bfloat16",
         auto_cls=AutoModelForImageTextToText,
         tensor_parallel_size=4,
-        marks=multi_gpu_marks(num_gpus=4),
+        marks=[pytest.mark.core_model, *multi_gpu_marks(num_gpus=4)],
```

- **@Isotr0py**: ```suggestion
        marks=[multi_gpu_marks(num_gpus=4)],
```
Let's revert this one, this test is too big to fit our CI.

### `tests/models/multimodal/processing/test_mllama4.py:1`
```diff

```

- **@Isotr0py**: I think the e2e test in `tests/models/multimodal/generation/test_vit_cudagraph.py` is enough. We can revert change in this file.

### `vllm/model_executor/models/mllama4.py:947`
```diff
+            device=device,
+            dtype=dtype,
+        )
+
+        return EncoderCudaGraphCaptureInputs(
+            mm_kwargs={"pixel_values": dummy_pixel_values},
```

- **@Isotr0py**: Just a note: We might need to update the interface if https://github.com/vllm-project/vllm/pull/42288 merge before this one.

- **@ghost**: will keep that in mind. 👍

- **@Isotr0py**: Can you update this one?

### `vllm/model_executor/models/mllama4.py:852`
```diff
         )
 
     def _process_image_input(
         self, image_input: Llama4ImagePatchInputs
     ) -> MultiModalEmbeddings:
-        assert self.vision_model and self.multi_modal_projector
```

- **@Isotr0py**: QQ: Why do we have to remove this assertion?

- **@ghost**: yea this was a mistake in hindsight. the normal __init__ always initializes vision_model and multi_modal_projector so i figured this assert mightve been redundant. its not relevant though.

### `vllm/model_executor/models/mllama4.py:634`
```diff
@@ -630,12 +631,12 @@ def _call_hf_processor(
                 (image_size[0] // tile_size, image_size[1] // tile_size)
                 for image_size in best_fit_sizes
             ]
-            patches_per_image = [
+            chunks_per_image = [
```

- **@Isotr0py**: Seems this rename is redundant, can you revert it?

### `vllm/model_executor/models/mllama4.py:973`
```diff
+        buffers: dict[str, torch.Tensor],
+    ) -> torch.Tensor:
+        return self.encode_image_chunks(
+            mm_kwargs["pixel_values"],
+            use_data_parallel=False,
+        ).flatten(0, 1)
```

- **@Isotr0py**: I think this will cause incorrect graph capture with ViT DP? Can you validate it with `--mm-encoder-tp-mode data`?

- **@ghost**: validated and tested. everything works fine with `--mm-encoder-tp-mode data`

<details><summary>config</summary>
<p>

```
vllm serve meta-llama/Llama-4-Scout-17B-16E-Instruct \
  --tensor-parallel-size 8 \
  --max-model-len 1024 \
  --max-num-seqs 1 \
  --gpu-memory-utilization 0.85 \
  --limit-mm-per-prompt '{"image": 1}' \
  --mm-encoder-tp-mode data \
  --port 8000 \
  --compilation-config '{"cudagraph_mm_encoder": true, "encoder_cudagraph_max_vision_items_per_batch": 1}'
  
  ```
</p>
</details> 

relevant logs file:
[serve_tp8_data_cg_tiny_1024.log](https://github.com/user-attachments/files/28659277/serve_tp8_data_cg_tiny_1024.log)

### review @shen-shanshan [COMMENTED]
Sorry for late, I'm busy with working these days. I left some comments and questions.

### review @shen-shanshan [COMMENTED]
I left some comments and questions. Others LGTM.

### review @shen-shanshan [APPROVED]
LGTM. Also CC @Isotr0py

### review @Isotr0py [COMMENTED]
I think this PR has unrelated changes out of vit cuda graph scope. We should clean it uo before merging it.

### review @Isotr0py [COMMENTED]
Sorry for missing this PR! Leave some final comments, PTAL!

## PR #38061: [MM][Perf][CG] Support ViT full CUDA graph for Qwen3-VL video inference (@shen-shanshan, merged)

### `vllm/model_executor/models/qwen3_vl.py:1829`
```diff
+        # that subsequent replays with the same grid_thw pattern can reuse them.
+        self._replay_buffer_cache[cache_key] = buffers
+
+        # TODO(shen-shanshan): Design more appropriate cache eviction strategy.
+        if len(self._replay_buffer_cache) > 10:
+            self._replay_buffer_cache.popitem(last=False)
```

- **@b-mu**: I think we have to either use `dict.popitem()`, which taks no arg, or  use `collections.OrderedDict.popitem()`, which accepts `last=`.

- **@shen-shanshan**: In fact, during my benchmark, this cache has little perf benefits, so I suppose maybe we can directly remove this caching mechanism currently. And we can add this back in a separate PR with more detailed design.

- **@b-mu**: I think the bug is pretty straightforward to fix. That's said, if you do have concrete results to show that this caching mechanism has no benefit, then it's better to remove and follow-up if a better design shew up.

- **@shen-shanshan**: Yeah, I prefer to add this in another PR, since I haven't consider this caching mechanism in detail (such as eviction strategies, cache size configuration, ...). I suppose it's better to merge this PR first to enable full CG for video inference first, then optimize performance. I will remove these codes soon.

### `vllm/v1/worker/encoder_cudagraph.py:178`
```diff
@@ -157,10 +172,11 @@ def _capture_budget_graph(self, token_budget: int):
             output = self.model.encoder_cudagraph_forward(mm_kwargs, buffers)
             output_buffer.copy_(output)
 
-        input_key = self.config.input_key
+        input_key = self.config.input_key_by_modality["image"]
```

- **@b-mu**: It would be nice to have a comment here to explain that, since the image and video modalities share the same per-patch shape in this model (through the shared PatchEmbed), so we can use the image dummy inputs to capture CUDA graph for both image and video.

- **@shen-shanshan**: I agree with you. I have noted this in `prepare_encoder_cudagraph_capture_inputs()` of `qwen3_vl.py`, but I suppose we should also add comments in the `EncoderCudaGraphManager` class.

### `vllm/model_executor/models/qwen3_vl.py:1636`
```diff
         self,
         vllm_config,
     ) -> tuple[int, int]:
         # Min: estimated smallest possible encoder input.
-        # 224x224 image → 16x16 patches, spatial_merge_size=2 → 8x8 = 64 tokens
+        # 224x224 image → 14x14 patches, spatial_merge_size=2 → 8x8 = 64 tokens
```

- **@lgeiger**: I think this should still be 16x16 for Qwen3VL or am I missing something?

- **@shen-shanshan**: Sorry, I just refer to https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/qwen3_vl.py#L345-L348, maybe the default value is wrong?

- **@b-mu**: I believe the patch size is 14 and there're `224/14 = 16` patches per side, and thus caused the confusion.

- **@shen-shanshan**: > I believe the patch size is 14 and there're `224/14 = 16` patches per side, and thus caused the confusion.

Oh, I understand, the 16x16 here is the total patch num, not patch size.

### `docs/design/cuda_graphs_multimodal.md:96`
```diff
@@ -80,10 +93,13 @@ Three fields in `CompilationConfig` control encoder CUDA Graphs:
 
 * `cudagraph_mm_encoder` (`bool`, default `False`) — enable CUDA Graph capture for multimodal encoder. When enabled, captures the full encoder forward as a CUDA Graph for each token budget level.
 * `encoder_cudagraph_token_budgets` (`list[int]`, default `[]`) — token budget levels for capture. If empty (default), auto-inferred from model architecture as power-of-2 levels. User-provided values override auto-inference.
-* `encoder_cudagraph_max_images_per_batch` (`int`, default `0`) — maximum number of images per batch during capture. If 0 (default), auto-inferred as `max_budget // min_budget`.
+* `encoder_cudagraph_max_mm_items_per_batch` (`int`, default `0`) — maximum number of images per batch during capture. If 0 (default), auto-inferred as `max_budget // min_budget`.
```

- **@tjtanaa**: If I understand correctly, this refers to any type of mm items, can we update the description to be clearer?

```
maximum number of `mm_items` ( images/videos per batch during capture.
```

- **@shen-shanshan**: OK, I will check all misleading comments like this in this PR again.

### `vllm/model_executor/models/qwen3_vl.py:1784`
```diff
+        self,
+        mm_kwargs: dict[str, Any],
+    ) -> str:
+        if "image_grid_thw" in mm_kwargs:
+            return "image"
+        return "video"
```

- **@Isotr0py**: What if for image + video mixed inputs? I think it will use image CG in this case?

- **@shen-shanshan**: In fact, I didn't consider mixed (image + video) inputs in this PR, since it'
s just experimental now. I prefer to enable this feature (mixed inputs) in a following PR.

- **@shen-shanshan**: Maybe we can add a note in the doc to illustrate this.

- **@ywang96**: Does this mean in order to use this feature with video input, users will have to turn off the image modality at launch time?

- **@shen-shanshan**: > Does this mean in order to use this feature with video input, users will have to turn off the image modality at launch time?

I suppose users don't need to explicitly set `--limit-mm-per-prompt '{"image": 0}'` (do you mean this?) as long as there are no image inputs mixed in video inputs in one request. But as you said, setting this is clearer. Maybe we should add this to the usage doc?

Currently, I'm thinking of how to support mixed inputs in a simple way. It may need to separate the processing logic for images and videos respectively for optimal performance.

### `vllm/config/compilation.py:523`
```diff
     Example: [2048, 4096, 8192, 13824]"""
 
-    encoder_cudagraph_max_images_per_batch: int = 0
-    """Maximum number of images per batch for encoder CUDA graph capture.
+    encoder_cudagraph_max_mm_items_per_batch: int = 0
+    """Maximum number of images/videos per batch for encoder CUDA graph capture.
```

- **@ywang96**: I'm slightly concerned about this since the naming suggests that we basically include audio here as well.

How about `encoder_cudagraph_max_vision_items_per_batch`? Please also update this config name everywhere in the example/doc.

- **@shen-shanshan**: I agree with this. `vision_items` is clearer since we don't plan to support CG for audio encode currently.

- **@ywang96**: Sounds good - let's update with `encoder_cudagraph_max_vision_items_per_batch` then!

- **@shen-shanshan**: Updated. Let's wait for CI finish. 😃

### `vllm/model_executor/models/qwen3_vl.py:709`
```diff
             max_batch_size: If set, pad cu_seqlens to this size
                 (needed for CUDA graph capture/replay).
+            max_frames_per_batch: If set, overrides max_batch_size for
+                cu_seqlens padding. For video inputs each item contributes
+                T attention sequences (frames); this sizes the buffer to
+                the total frame budget so video replays never overflow.
```

- **@ywang96**: Does this mean if a video has more frames than this value, it falls back to eager?

- **@shen-shanshan**: Yes. If the smallest num of frames of one video doesn't fit in this budget, CG will not be enabled.

### comment @shen-shanshan
CC @wangshangsam @Isotr0py

### comment @shen-shanshan
@ywang96 Could you help review this PR? Since it has been pending for a long time. 😃

## PR #35963: [Feature] ViT Full CUDA Graph (@b-mu, merged)

### `vllm/v1/worker/gpu_model_runner.py:2506`
```diff
+                    cudagraph_output = None
+                    if (
+                        self.encoder_cudagraph_manager is not None
+                        and modality == "image"
+                        and "pixel_values" in mm_kwargs_group
+                        and "image_grid_thw" in mm_kwargs_group
```

- **@Isotr0py**: Hmmm, I feel this is too model-specific, and it 's difficult to use for other models with different mm_kwargs naming.

Can we execute `encoder_cudagraph_manage` with mm_kwargs_group directly?

- **@b-mu**: To address the general concern about encoder cudagraph manager being too model specific, I'm thinking about having a class `SupportsEncoderCudaGraph(Protocol)` in `vllm/model_executor/models/interfaces.py`. To use encoder cudagraph support for a model, the model needs to implement a list of methods to tell the encoder cudagraph manager how to extract inputs, sequence metadata, etc. The encoder cudagraph manager would be model agnostic. The specific concern here in gpu_model_runner.py can use `mm_kwargs_group` directly. What do you think?

- **@Isotr0py**: Sounds great.

- **@b-mu**: We introduced `SupportsEncoderCudaGraph` protocol in `interfaces.py` with 9 protocol methods. The manager is now model-agnostic — all Qwen3-VL-specific logic lives in `qwen3_vl.py` implementing the protocol.

The specific concern here in `gpu_model_runner.py` now uses `mm_kwargs_batch` directly and uses `self.encoder_cudagraph_manager.supports_modality(modality)` instead of checking for specific keys.

### `vllm/v1/worker/gpu/mm/encoder_cudagraph.py:188`
```diff
+            token_budget, self.max_batch_size
+        )
+
+        dummy_pixel_values, dummy_grid_thw = self._prepare_dummy_inputs(
+            dummy_grid_config
+        )
```

- **@Isotr0py**: I feel this is something belongs to `DummyInputsBuilder`, otherwise the CG manager's dummy data creation could be quite complicated if we want to support other models.

- **@b-mu**: Dummy input generation has been moved out of the manager and into the protocol method `prepare_encoder_cudagraph_capture_inputs()`. Each model implements its own dummy input logic (e.g., in qwen3_vl.py).

### `vllm/model_executor/models/qwen3_vl.py:621`
```diff
@@ -531,44 +532,55 @@ def forward(
         self,
         x: torch.Tensor,
         grid_thw: torch.Tensor | list[list[int]],
+        *,
+        encoder_metadata: dict[str, torch.Tensor] | None = None,
```

- **@Isotr0py**: I think we can add a `prepare_encoder_metadata` method to avoid duplicated implementation:
```python3
def prepare_encoder_metadata(...) -> dict[str, Any]:
    ...
    max_seqlen = torch.tensor(
        MMEncoderAttention.compute_max_seqlen(self.attn_backend, cu_seqlens),
        dtype=torch.int32,
        device=self.device,
    )
    cu_seqlens = MMEncoderAttention.maybe_recompute_cu_seqlens(
        self.attn_backend,
        cu_seqlens,
        self.hidden_size,
        self.tp_size,
    )
    cu_seqlens = torch.from_numpy(cu_seqlens).to(self.device, non_blocking=True)
    return {"max_seq_len": ...}

def forward(
    self,
    x: torch.Tensor,
    grid_thw: torch.Tensor | list[list[int]],
    *,
    encoder_metadata: dict[str, torch.Tensor] | None = None,
):
    ...
    if encoder_metadata is None:
        encoder_metadata = self.prepare_encoder_metadata(...)
    ...
    for layer_num, blk in enumerate(self.blocks):
        hidden_states = blk(
            hidden_states,
            **encoder_metadata,
        )
```
Then we can reuse it in `prepare_encoder_cudagraph_capture_inputs` and `prepare_encoder_cudagraph_replay_buffers`:
```python3
def prepare_encoder_cudagraph_replay_buffers(
        self,
        mm_kwargs: dict[str, Any],
        max_batch_size: int,
    ):
        from vllm.v1.worker.gpu.mm.encoder_cudagraph_defs import (
            EncoderCudaGraphReplayBuffers,
        )
        buffers = self.visual.prepare_encoder_metadata(...)
        ...
        return EncoderCudaGraphReplayBuffers(buffers=buffers)
```

WDYT?

- **@b-mu**: good catch, I have added the helper `prepare_encoder_metadata()` and used it for eager forward pass, cudagraph capture/replay for qwen3 vl.

### `vllm/v1/worker/gpu/mm/encoder_cudagraph.py:370`
```diff
+            get_tensor_model_parallel_rank,
+            get_tensor_model_parallel_world_size,
+        )
+        from vllm.model_executor.models.vision import (
+            get_load_balance_assignment,
+        )
```

- **@Isotr0py**: I think we can move it to top imports.

- **@b-mu**: done.

### `vllm/v1/worker/gpu/mm/encoder_cudagraph.py:434`
```diff
+
+        Assumes 2D output tensors [tokens, hidden]. Follows the same
+        pad -> all_gather -> unpad -> reorder algorithm as
+        run_dp_sharded_mrope_vision_model() in the eager path.
+        """
+        from vllm.distributed import tensor_model_parallel_all_gather
```

- **@Isotr0py**: Ditto.

- **@b-mu**: done.

### `vllm/v1/worker/gpu/mm/encoder_cudagraph.py:1`
```diff

```

- **@Isotr0py**: @ywang96 I remember `v1/worker/gpu` is reserved for v2 model runner, is it fine to place it here and only execute in v1 as well?

### `vllm/config/compilation.py:505`
```diff
+
+    encoder_cudagraph_token_budgets: list[int] = field(default_factory=list)
+    """Token budget levels for encoder CUDA graph capture.
+    Each budget defines a fixed token capacity. At runtime, images are greedy-packed
+    into the smallest fitting budget and the corresponding CUDA graph is replayed.
+    Example: [2048, 4096, 8192, 13824]"""
```

- **@Isotr0py**: I think asking user to pass `encoder_cudagraph_token_budgets` manually will cause poor user experience. Is it possible to automatically infer best `encoder_cudagraph_token_budgets` (at least cover most of possible image size) from `get_mm_max_tokens_per_item`?
https://github.com/vllm-project/vllm/blob/fad09e8a1f51b31eba1f42ff5d651256c77a734d/vllm/multimodal/encoder_budget.py#L15-L41

- **@b-mu**: It seems that `get_mm_max_tokens_per_item` returns the theoretical maximum that the model can consume, which might not get hit by a dataset. Capturing a CUDA graph that's too large could waste GPU memory. On the other hand, if an image is large enough, the kernel launch overhead might not be that visible, so the price of falling back to eager is low. 

I'm thinking about adding a new method in the protocol to provide the `(min budget, max budget)` of that model. For qwen3-vl, I'm thinking about using `max_num_batched_tokens` as the maximum budget, and using `224x224 ~ 64` tokens as the minimum budget. What do you think?

- **@Isotr0py**: > I'm thinking about using max_num_batched_tokens as the maximum budget, and using 224x224 ~ 64 tokens as the minimum budget.

Sounds great.

### review @Isotr0py [COMMENTED]
With a first glance, my concern is the generality of current encoder CG manager. I feel current implementation is too qwen3vl-specific (MRoPE + ViT RoPE), and it's difficult to boardcast this CG support to other ViTs.

### review @Isotr0py [APPROVED]
As an experimental functionality, the model interface and cg manager overall LGTM now. But still have a minor suggestion about UX.

### comment @shen-shanshan
Is there a plan to launch this PR? I have done a draft locally to support full CG for video inference based on this PR.

### comment @Isotr0py
Can you fix pre-commit?

## PR #40576: [MM][Perf][CG] Support ViT full CUDA graph for glm4_1v image and video inference  (@grYe99, merged)

### `vllm/model_executor/models/glm4_1v.py:1743`
```diff
@@ -1385,7 +1704,12 @@ def get_video_replacement_glm4v(item_idx: int):
     dummy_inputs=Glm4vDummyInputsBuilder,
 )
 class Glm4vForConditionalGeneration(
```

- **@shen-shanshan**: Please also update this model to:

Doc: https://docs.vllm.ai/en/latest/design/cuda_graphs_multimodal/#model-integration-via-supportsencodercudagraph

Example: https://github.com/vllm-project/vllm/blob/main/examples/offline_inference/vision_language.py#L2466-L2469 (if needed)

CI test: https://github.com/vllm-project/vllm/blob/main/tests/models/multimodal/generation/test_vit_cudagraph.py#L44-L58

- **@grYe99**: @shen-shanshan done

### `vllm/model_executor/models/glm4_1v.py:973`
```diff
+                    self.num_grid_per_side,
+                    self.spatial_merge_size,
+                    self.dtype,
+                )
+            )
+        return torch.cat(outputs, dim=0)
```

- **@Isotr0py**: Can you double check if Qwen3-VL and GLM-4.1V's VisiomEmbed implementation are fully equivalent with converged numeric results?

I think GLM-4.1V uses cubic interpolation instead of Qwen3-VL's bilinear interpolation.

- **@grYe99**: @Isotr0py Thanks for spotting this. Indeed the original GLM-4.1V uses bicubic interpolation for vision position embeddings. I have updated code and passed functional tests.

### `examples/generate/multimodal/vision_language_offline.py:591`
```diff
@@ -588,7 +588,6 @@ def run_glm4_1v(questions: list[str], modality: str) -> ModelRequestData:
             "fps": 1,
         },
         limit_mm_per_prompt=mm_limit,
-        enforce_eager=True,
```

- **@shen-shanshan**: I suppose this `enforce_eager=True` is irrelevant to ViT CG? We don't need to delete this.

- **@grYe99**: done in be3040e3766efa98c12e9c5774640e9929ce0a52

### `vllm/model_executor/models/glm4_1v.py:777`
```diff
         return max_seqlen
 
+    def pos_embeds_interpolate(self, grid_thw: list[list[int]]) -> torch.Tensor:
+        device = self.embeddings.position_embedding.weight.device
+        dtype = self.dtype
+        all_embeds = []
```

- **@shen-shanshan**: Would you mind adding a method description here? I don't quite understand this, since it's really different from the original implementation.

- **@grYe99**: done, description added

### `vllm/model_executor/models/glm4_1v.py:1719`
```diff
+        # of tokens becomes data-dependent (i.e., the retained tokens are
+        # dynamically selected based on inter-frame differences) and therefore
+        # cannot be captured by CUDA Graphs. As a result, video CUDA Graphs are
+        # only enabled when EVS is disabled.
+        if not self.is_multimodal_pruning_enabled:
+            modalities.append("video")
```

- **@shen-shanshan**: I suppose this should update to https://github.com/vllm-project/vllm/pull/38040/changes#diff-89b6b3d99f8961ecdc0cf8cf16b0f9e091a6227ee2a9bfb4f04b88043278e0e0R1771-R1776.

- **@grYe99**: done in be3040e3766efa98c12e9c5774640e9929ce0a52

### `vllm/model_executor/models/glm4_1v.py:1815`
```diff
+    def get_encoder_cudagraph_per_item_input_sizes(
+        self,
+        mm_kwargs: dict[str, Any],
+    ) -> list[int]:
+        grid_thw = self._get_grid_thw_by_modality(mm_kwargs)
+        return [t * h * w for t, h, w in grid_thw]
```

- **@shen-shanshan**: These 3 interfaces has been merged into 1 method in https://github.com/vllm-project/vllm/pull/41234. You should update it.

- **@grYe99**: done in be3040e3766efa98c12e9c5774640e9929ce0a52

### review @shen-shanshan [COMMENTED]
Sorry for late, I'm busy with working these days. I left some comments and questions.

### review @shen-shanshan [APPROVED]
Done with my pass. Also CC @Isotr0py

### review @Isotr0py [APPROVED]
LGTM, thanks!

### comment @Isotr0py
Sry for missing this! Can you provide some multimodal accuracy benchmark results like MMMU etc?

## PR #42224: [MM][CG] Enable encoder Cudagraph for Step3VL (@Soyaazz, merged)

### `examples/generate/multimodal/vision_language_offline.py:2593`
```diff
     enable_vit_cuda_graph = args.enable_vit_cuda_graph
 
-    if enable_vit_cuda_graph and model in MODELS_SUPPORT_VIT_CUDA_GRAPH:
+    assert model in MODELS_SUPPORT_VIT_CUDA_GRAPH
+
+    if enable_vit_cuda_graph:
```

- **@shen-shanshan**: Why put `assert model in MODELS_SUPPORT_VIT_CUDA_GRAPH` before `if enable_vit_cuda_graph: ...`? I suppose this will get an error for models that don't support ViT CG?

- **@Soyaazz**: You're right, that was an oversight in the implementation. I'll have it fixed soon.

### `vllm/model_executor/models/step3_vl.py:897`
```diff
+        self,
+        output: torch.Tensor,
+        mm_kwargs: dict[str, Any],
+        indices: list[int],
+        per_item_out_tokens: list[int],
+    ) -> list[torch.Tensor]:
```

- **@shen-shanshan**: Maybe we can extract this method as a ViT CG interface like `postprocess_encoder_output()`, which will use default `_scatter_output_slices()` for other models?

- **@Soyaazz**: Sounds good! I'll take care of it in the next commit.

### `vllm/model_executor/models/interfaces.py:1519`
```diff
+    offset = 0
+    for idx in indices:
+        n_tok = per_item_out_tokens[idx]
+        sliced = output[offset : offset + n_tok]
+        dest[idx] = sliced.clone() if clone else sliced
+        offset += n_tok
```

- **@Isotr0py**: Can move this to `vllm/model_executor/models/utils.py`.

- **@Soyaazz**: Done. 

Note: moving this function to `vllm/model_executor/models/utils.py` directly would introduce a circular import, so I've kept the import inside the function body instead of at the module level. See the change [here](https://github.com/vllm-project/vllm/pull/42224/changes#diff-2ae789d702f81147ec584f3c0077aa3dedb6ebd424320dc607a44b86e3780874R1614).

### `vllm/model_executor/models/step3_vl.py:726`
```diff
+    def get_encoder_cudagraph_budget_range(
+        self,
+        vllm_config: "VllmConfig",
+    ) -> tuple[int, int]:
+        # An image without patches
+        min_budget = Step3VLForConditionalGeneration._compute_spatial_tokens(
```

- **@Isotr0py**: ```suggestion
        min_budget = self._compute_spatial_tokens(
```
 I think `self._compute_spatial_tokens` should work as well

### `vllm/model_executor/models/step3_vl.py:832`
```diff
+        from vllm.v1.worker.encoder_cudagraph_defs import (
+            EncoderCudaGraphCaptureInputs,
+        )
+
+        # For pixel_value, the max input size is max_batch_size
+        img_output_tokens = Step3VLForConditionalGeneration._compute_spatial_tokens(
```

- **@Isotr0py**: ```suggestion
        img_output_tokens = self._compute_spatial_tokens(
```
ditto.

### `docs/design/cuda_graphs_multimodal.md:90`
```diff
@@ -86,6 +87,7 @@ Models opt-in to encoder CUDA Graphs by implementing the [SupportsEncoderCudaGra
 | Architecture | Models | CG for Image | CG for Video |
 | ------------ | ------ | ------------ | ------------ |
 | `Qwen2_5_VLForConditionalGeneration` | `Qwen2.5-VL` | ✅︎ | ✅︎ |
+| `Step3VLForConditionalGeneration` | `Step3-VL` | ✅︎ | ❌︎ |
```

- **@Isotr0py**: Keep in alphabet order.

- **@Soyaazz**: ditto

### `examples/generate/multimodal/vision_language_offline.py:2561`
```diff
@@ -2557,6 +2557,8 @@ def run_tarsier2(questions: list[str], modality: str) -> ModelRequestData:
     "qwen2_5_vl",
     "qwen3_vl",
     "qwen3_vl_moe",
+    "qwen2_5_vl",
+    "stepvl",
```

- **@Isotr0py**: ```suggestion
    "stepvl",
```
Also keep in alphabet order.

- **@Soyaazz**: ditto

### review @Isotr0py [COMMENTED]
I think we can merge this PR first to initialize step-vl's encoder CG support. Then land #41714's graph_pool and profiling support, becuase it will have another potential conflicts with https://github.com/vllm-project/vllm/pull/42288 and needs more discussions about the design carefully.

### comment @shen-shanshan
Sorry for late, I'm busy with working these days. I'll take a look at this PR today.

### comment @shen-shanshan
IMO, the idea in https://github.com/vllm-project/vllm/pull/41714 of supporting graph pool and considering it in the profiling is critical. Maybe you two can discuss about the cooperation, merge into one PR and add another one as a co-author.

## PR #41992: [MM][Perf][CG] Support ViT full CUDA graph for Kimi-VL (@oguzhankir, merged)

### `docs/design/cuda_graphs_multimodal.md:88`
```diff
@@ -85,6 +85,7 @@ Models opt-in to encoder CUDA Graphs by implementing the [SupportsEncoderCudaGra
 | `InternVLChatModel` | `InternVL3.5`, `InternVL3`, `InternVL2.5`, `InternVL2` | ✅︎ | ✅︎ |
 | `Qwen2VLForConditionalGeneration` | `Qwen2-VL` | ✅︎ | ✅︎ |
 | `Qwen2_5_VLForConditionalGeneration` | `Qwen2.5-VL` | ✅︎ | ✅︎ |
+| `KimiVLForConditionalGeneration` | `Kimi-VL` | ✅︎ | ✗ |
```

- **@shen-shanshan**: Use the same ❌︎ looks more beautiful.

### `tests/models/multimodal/generation/test_vit_cudagraph.py:111`
```diff
@@ -78,6 +85,17 @@ def step3_vl_chat_template(content: str) -> str:
         needs_video_metadata=False,
         marks=[pytest.mark.core_model],
     ),
+    "kimi_vl": VitCudagraphTestConfig(
+        model="moonshotai/Kimi-VL-A3B-Instruct",
```

- **@shen-shanshan**: I'm not sure if this MoE model can be too big to fit in the CI.
FYI, https://github.com/vllm-project/vllm/pull/40660#discussion_r3316493915 and https://github.com/vllm-project/vllm/pull/43082.

- **@oguzhankir**: Switched to dummy weights (dummy_hf_overrides) + a single [1024] budget, following the step3_vl pattern in #43082. Keeps it within CI GPU memory; trust_remote_code kept since Kimi-VL needs it.

### `vllm/model_executor/models/kimi_vl.py:376`
```diff
+
+    def get_input_modality(
+        self,
+        mm_kwargs: dict[str, Any],
+    ) -> str:
+        return "image"
```

- **@shen-shanshan**: This method is not needed anymore. FYI, https://github.com/vllm-project/vllm/pull/44484.

### `vllm/model_executor/models/kimi_vl.py:461`
```diff
+        from vllm.v1.worker.encoder_cudagraph_defs import (
+            EncoderCudaGraphCaptureInputs,
+        )
+
+        kh, kw = self.config.vision_config.merge_kernel_size
+        per_mm_item_output = token_budget // max_batch_size
```

- **@shen-shanshan**: This need updating: https://github.com/vllm-project/vllm/pull/38040/changes#diff-89b6b3d99f8961ecdc0cf8cf16b0f9e091a6227ee2a9bfb4f04b88043278e0e0R1924-R1927.

### `vllm/model_executor/models/moonvit.py:730`
```diff
+        if device is None:
+            device = next(self.parameters()).device
+
+        # Normalize to a list of plain Python int pairs so the helpers
+        # below never need ``.tolist()`` on a tensor.
+        normalized: list[tuple[int, int]] = [(int(h), int(w)) for h, w in grid_hws_list]
```

- **@shen-shanshan**: I suppose the name `normalized` here is confusing.

- **@oguzhankir**: renamed to grid_pairs

### `vllm/model_executor/models/moonvit.py:751`
```diff
+                    seq_lens.cumsum(dtype=np.int32),
+                ]
+            )
+        else:
+            seq_lens = np.zeros(0, dtype=np.int32)
+            cu_seqlens_np = np.zeros(1, dtype=np.int32)
```

- **@shen-shanshan**: I suppose `normalized` here will not be `None`?

- **@oguzhankir**: right, it's never empty here. dropped the branch (manager guarantees max_batch_size >= 1), same as qwen3_vl.

### `vllm/model_executor/models/moonvit.py:704`
```diff
+        device: torch.device | None = None,
+    ) -> dict[str, Any]:
+        """Precompute every grid-dependent input the encoder needs.
+
+        Shared by the eager forward path and the CUDA graph
+        capture/replay path. Everything here is computed outside the
```

- **@shen-shanshan**: In this PR, I see `prepare_encoder_metadata()` is only used by CUDA graph path, not used by eager path?

- **@oguzhankir**: Right, fixed the docstring to match. Routing eager through it too would change the encoder output to packed format and need projector callers updated, so I left that as a follow-up to keep this PR focused (same as InternVL #41759). Happy to do it here if you'd prefer.

### review @shen-shanshan [COMMENTED]
I left some comments and questions.

### review @shen-shanshan [APPROVED]
Overall LGTM.

Can you run the following commands to compare the eager and vit cuda graph outputs?

```bash
python examples/generate/multimodal/vision_language_offline.py -m kimi_vl --modality "image"
python examples/generate/multimodal/vision_language_offline.py -m kimi_vl --modality "image" --enable-vit-cuda-graph
```

### comment @shen-shanshan
> Rebased on latest main and synced the encoder CUDA graph code to the current interface: migrated to get_encoder_cudagraph_item_specs (#41234) and the single values dict API (#42288) — dropped input_key_by_modality, moved pixel_values into buffer_keys, and updated capture/replay/forward (same pattern as the merged Qwen-VL / InternVL #41759 work). Resolved the test merge conflict. pre-commit and mkdocs build --strict pass locally; the failing pre-run-check / readthedocs are just the "needs ready label / 4 merged PRs" gate. @DarkLight1337 @ywang96 could you add the ready label when you get a chance? cc @Isotr0py @shen-shanshan who reviewed InternVL — this follows the same approach.

Sorry for late, I will take a look at this PR this week.

## PR #43591: [MM][CG] Gemma3 Encoder CUDA Graph (@Soyaazz, merged)

### `vllm/model_executor/models/gemma3_mm.py:705`
```diff
+
+    def get_input_modality(
+        self,
+        mm_kwargs: dict[str, Any],
+    ) -> str:
+        return "image"
```

- **@shen-shanshan**: This method is not needed now, FYI, https://github.com/vllm-project/vllm/pull/44484.

- **@Soyaazz**: Done that~

### `vllm/model_executor/models/gemma3_mm.py:708`
```diff
+        mm_kwargs: dict[str, Any],
+    ) -> str:
+        return "image"
+
+    def get_max_frames_per_video(self) -> int:
+        return 0
```

- **@shen-shanshan**: This method is not needed now, FYI, https://github.com/vllm-project/vllm/pull/41234/changes#diff-8214882028a01f717c0aac8d8261b29bd24e965d393b13aaa37acd96c26c6482R54-R57.

- **@Soyaazz**: ditto

### `vllm/model_executor/models/gemma3_mm.py:729`
```diff
+    ):
+        from vllm.v1.worker.encoder_cudagraph_defs import EncoderItemSpec
+
+        num_patches = mm_kwargs["num_patches"]
+        vision_cfg = self.config.vision_config
+        vit_positions_per_patch = (vision_cfg.image_size // vision_cfg.patch_size) ** 2
```

- **@shen-shanshan**: This value is fixed, we can make it as an `@property`, after which we can avoid calculating this value in each inference step.

- **@Soyaazz**: ditto

### `vllm/model_executor/models/gemma3_mm.py:742`
```diff
+
+        if len(indices) == 0:
+            return {
+                "pixel_values": pixel_values[:0],
+                "num_patches": num_patches[:0],
+            }
```

- **@shen-shanshan**: We can move this before calculating `cum_patches`.

- **@Soyaazz**: ditto

### `vllm/model_executor/models/gemma3_mm.py:773`
```diff
+
+        mm_tokens_per_image = self.config.mm_tokens_per_image
+        num_images = min(
+            (token_budget + mm_tokens_per_image - 1) // mm_tokens_per_image,
+            max_batch_size,
+        )
```

- **@shen-shanshan**: I suppose we should use `floor` instead of `ceil` here?

- **@Soyaazz**: It does make sense! already changed to `floor`.

### review @shen-shanshan [COMMENTED]
I left some comments and questions.

In addition, I see that for Gemma3, if the original image `ratio` > `min_ratio`, the mm inputs may contains N crops, does this PR support this?

### review @shen-shanshan [APPROVED]
LGTM.

### comment @shen-shanshan
> cc @shen-shanshan

You are so fast! I may review this PR later, since there are many ViT CG integration PRs are queuing. 🥲

## PR #45254: [MM][CG] Support ViT full CUDA graph for Ernie-4.5-VL image inference (@qyYue1389, merged)

### `docs/design/cuda_graphs_multimodal.md:121`
```diff
@@ -127,6 +127,7 @@ Models opt-in to encoder CUDA Graphs by implementing the [SupportsEncoderCudaGra
 | Architecture | Models | CG for Image | CG for Video | Dual-Path Graph |
 | ------------ | ------ | ------------ | ------------ | --------------- |
 | `DeepseekOCRForCausalLM` | `DeepSeek-OCR` | ✅︎ | ❌︎ | ✅︎ |
+| `Ernie4_5_VLMoeForConditionalGeneration` | `ERNIE-4.5-VL` | ✅︎ | ❌︎ | ❌︎ |
```

- **@shen-shanshan**: As far as I know, this model supports video inference, does this PR support?

- **@qyYue1389**: Yes it supports video inference, but this PR supports image only for now

ERNIE's video path applies a `use_temporal_conv` in the resampler that changes the output token count ([code](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/ernie45_vl.py#L1525)), which needs separate handling. Thus I plan to work on it as a follow-up PR 

This PR only adds the *image* encoder CUDA graph, and Eager video inference is unaffected. Thus ❌ in the Video column for now.

Does the follow-up PR plan sound good to you or any suggestions? Thanks!

### `vllm/model_executor/models/ernie45_vl.py:1609`
```diff
+        )
+
+    def get_input_modality(self, mm_kwargs: dict[str, Any]) -> str:
+        if "image_grid_thw" in mm_kwargs:
+            return "image"
+        raise AssertionError("This line should be unreachable.")
```

- **@shen-shanshan**: If it only supports image modality, this method is not needed, FYI, https://github.com/vllm-project/vllm/pull/44484.

- **@qyYue1389**: Thank you for pointing out! Removed

### `vllm/model_executor/models/ernie45_vl.py:1628`
```diff
+
+    def _get_grid_thw_by_modality(self, mm_kwargs: dict[str, Any]) -> list[list[int]]:
+        grid_thw = mm_kwargs["image_grid_thw"]
+        if not isinstance(grid_thw, list):
+            grid_thw = grid_thw.tolist()
+        return grid_thw
```

- **@shen-shanshan**: Ditto.

- **@qyYue1389**: Thank you for pointing out!

### `vllm/model_executor/models/ernie45_vl.py:1736`
```diff
+        # token budget. Run the resampler eagerly on the valid portion using
+        # the actual batch grid_thw, then scatter the post-merge embeddings.
+        grid_thw_list = self._get_grid_thw_by_modality(batch_mm_kwargs)
+        num_valid = sum(t * h * w for t, h, w in grid_thw_list)
+        grid_thw = torch.tensor(grid_thw_list, device=output.device)
+        image_embeds = self.resampler_model(output[:num_valid], grid_thw)
```

- **@shen-shanshan**: In `VariableResolutionResamplerModel`, if `use_temporal_conv=False`, I suppose it's CUDA graph compatible? Since it only contains linear layer and activation layer.

- **@qyYue1389**: Yes. If `use_temporal_conv=False` the resampler is just linear and activation and would be CUDA graph compatible. 

However `ERNIE-4.5-VL-28B-A3B` ships with `use_temporal_conv=True` from [`Ernie4_5_VLMoEConfig.__init__`](https://huggingface.co/baidu/ERNIE-4.5-VL-28B-A3B-PT/blob/00f623a8627294e3e1457ee73e308ddf0582e6f3/configuration_ernie4_5_vl.py#L565) in the model's bundled `configuration_ernie4_5_vl.py`
With that flag, `VariableResolutionResamplerModel.forward` always takes the `fwd_placeholder` path. Hence only ViT is put into the CUDA graph in this PR and the resampler stays on the eager path

Including the resampler in the CUDA graph for `use_temporal_conv=False` could be done in the follow-up PR

Does it sound good or any suggestions?

### review @shen-shanshan [COMMENTED]
I left some comments and questions.

### review @shen-shanshan [APPROVED]
LGTM. Also CC @Isotr0py

### review @Isotr0py [APPROVED]
LGTM, thanks!

### comment @Isotr0py
Can you verify the multimodal accuracy with ViT CG as well?

### comment @shen-shanshan
> Hi @shen-shanshan, thank you for your feedback. I finished adding changes and reply. Could you review again when getting a chance? Thank you!

Sorry for late, I'm busy with working these days. I'll take a look at this PR today.

## PR #41759: [MM][Perf][CG] Support ViT full CUDA graph for InternVL (@oguzhankir, merged)

### `vllm/model_executor/models/internvl.py:1009`
```diff
+
+    def get_encoder_cudagraph_per_item_input_sizes(
+        self,
+        mm_kwargs: dict[str, Any],
+    ) -> list[int]:
+        return self._get_internvl_patches_list(mm_kwargs)
```

- **@Isotr0py**: You need to update the interface after https://github.com/vllm-project/vllm/pull/41234

- **@oguzhankir**: Thanks for the review. Missed that #41234 landed. I'll update the interface and push shortly.

### `docs/design/cuda_graphs_multimodal.md:89`
```diff
@@ -86,6 +86,7 @@ Models opt-in to encoder CUDA Graphs by implementing the [SupportsEncoderCudaGra
 
 | Architecture | Models | CG for Image | CG for Video |
 | ------------ | ------ | ------------ | ------------ |
+| `InternVLChatModel` | `InternVL3`, `InternVL2.5`, `InternVL2` | ✅︎ | ✅︎ |
```

- **@shen-shanshan**: Does this PR support InternVL3_5?

- **@oguzhankir**: Yes. InternVL3.5 resolves to the same InternVLChatModel, so it's covered. Added it to the docs table.

### `tests/models/multimodal/generation/test_vit_cudagraph.py:59`
```diff
@@ -51,6 +55,16 @@ def step3_vl_chat_template(content: str) -> str:
 
 
 MODEL_CONFIGS: dict[str, VitCudagraphTestConfig] = {
+    "internvl": VitCudagraphTestConfig(
+        model="OpenGVLab/InternVL3-2B",
```

- **@shen-shanshan**: For CI test, we should use a model as small as possible, I suppose `OpenGVLab/InternVL3-1B` is better.

### `vllm/model_executor/models/internvl.py:555`
```diff
+    SupportsPP,
+    SupportsLoRA,
+    SupportsEncoderCudaGraph,
+):
     supports_encoder_tp_data = True
+    supports_encoder_cudagraph: ClassVar[Literal[True]] = True
```

- **@shen-shanshan**: This `supports_encoder_cudagraph` is not needed.

### `vllm/model_executor/models/internvl.py:964`
```diff
+        return "video"
+
+    def get_max_frames_per_video(self) -> int:
+        # InternVL has no attention-metadata buffers that depend on frame
+        # count (buffer_keys=[]), so any value is safe. Return 1.
+        return 1
```

- **@shen-shanshan**: This method is not needed, since https://github.com/vllm-project/vllm/pull/41234/changes#diff-8214882028a01f717c0aac8d8261b29bd24e965d393b13aaa37acd96c26c6482R54-R57 has set `max_frames_per_video` default to 1.

### review @shen-shanshan [COMMENTED]
I left some comments and questions. Others LGTM.

### review @shen-shanshan [APPROVED]
Done with my pass. Also CC @Isotr0py

### comment @shen-shanshan
> cc @Isotr0py @shen-shanshan

Sorry for late, I will take a look at this PR today.

### comment @shen-shanshan
@oguzhankir `input_key_by_modality` has been deprecated, you should sync the changes in https://github.com/vllm-project/vllm/pull/42288.

## PR #42785: [MM][CG] Enable encoder CUDA Graph for MiniCPM-V (@YunzhuLu, merged)

### `vllm/v1/worker/encoder_cudagraph.py:167`
```diff
-        self.budget_graphs: dict[int, BudgetGraphMetadata] = {}
-        self.graph_pool: Any | None = None
+        self._ordered_secondary_capture_axis_keys: tuple[Hashable, ...] | None = None
+        get_axis_keys_fn = getattr(
+            model, "get_encoder_cudagraph_secondary_capture_axis_keys", None
+        )
```

- **@shen-shanshan**: We can make this method as an ViT CG interface (default: return `None`).

### `vllm/v1/worker/encoder_cudagraph.py:224`
```diff
+            return select_fn(
+                mm_kwargs,
+                indices,
+                secondary_capture_axis_key=secondary_capture_axis_key,
+            )
+        return select_fn(mm_kwargs, indices)
```

- **@shen-shanshan**: Maybe it's better to pass `enable_secondary_capture_axis=True` (default as `False`) into `EncoderCudaGraphConfig` in the modeling files and add `secondary_capture_axis_key` (default as `None`) into params of `prepare_encoder_cudagraph_capture_inputs()` interface.

I suppose there is no need to add additional wrappers in `EncoderCudaGraphManager`.

Ditto: `_select_encoder_cudagraph_items()`.

### `vllm/v1/worker/encoder_cudagraph.py:253`
```diff
+        else:
+            for token_budget in self.token_budgets:
+                for (
+                    secondary_capture_axis_key
+                ) in self._ordered_secondary_capture_axis_keys:
+                    self._capture_budget_graph(token_budget, secondary_capture_axis_key)
```

- **@shen-shanshan**: Then we can use `if self.config.enable_secondary_capture_axis:` here, which is clearer and more readable.

### `vllm/v1/worker/encoder_cudagraph.py:496`
```diff
+            if self._ordered_secondary_capture_axis_keys is not None:
+                resolve_fn = getattr(
+                    self.model,
+                    "resolve_encoder_cudagraph_secondary_capture_axis_key",
+                    None,
+                )
```

- **@shen-shanshan**: Ditto.

### `vllm/v1/worker/encoder_cudagraph.py:472`
```diff
                 # graph_misses counted here for this eager fallback.
+                batch_mm_kwargs = self._select_encoder_cudagraph_items(
+                    mm_kwargs,
+                    batch_orig_indices,
+                    None,
+                )
```

- **@shen-shanshan**: We can make `secondary_capture_axis_key` default to `None`, then no need to explicitly pass `None` here.

### `docs/design/cuda_graphs_multimodal.md:96`
```diff
 | `Qwen3VLForConditionalGeneration` | `Qwen3-VL` | ✅︎ | ✅︎ |
 | `Qwen3_5ForConditionalGeneration` | `Qwen3.5` | ✅︎ | ✅︎ |
 | `Step3VLForConditionalGeneration` | `Step3-VL` | ✅︎ | ❌︎ |
+| `MiniCPMV` | `MiniCPMV2.5` | ✅︎ | ❌︎ |
+| `MiniCPMV` | `MiniCPMV2.6` | ✅︎ | ✅︎ |
+| `MiniCPMV` | `MiniCPMV4.0` | ✅︎ | ✅︎ |
```

- **@shen-shanshan**: Maybe It's better to merge 2.6 and 4.0 into one row, since they belong to the same model arch.

### `vllm/v1/worker/encoder_cudagraph_defs.py:93`
```diff
     )
     """Independently captured encoder paths keyed by their forward name."""
 
+    enable_secondary_capture_axis: bool = False
+    """Whether to enable secondary capture axis."""
+
```

- **@Isotr0py**: Is it possible to make arbitrary capture axis support?

### review @shen-shanshan [COMMENTED]
I left some initial comments, without diving into the detailed model implementations (I will take a look at them when I'm free).

### comment @shen-shanshan
> Hi @shen-shanshan, I've rebased this PR onto the latest interface changes. Would appreciate your review when you have a moment. Thanks！

Sorry for late, I will take a look at this PR this week.

## PR #43586: [MM][Perf][CG] Support dual-path ViT full CUDA graph for DeepSeek-OCR (@shen-shanshan, merged)

### `docs/design/cuda_graphs_multimodal.md:91`
```diff
@@ -88,6 +88,7 @@ Models opt-in to encoder CUDA Graphs by implementing the [SupportsEncoderCudaGra
 | `Qwen3VLForConditionalGeneration` | `Qwen3-VL` | ✅︎ | ✅︎ |
 | `Qwen3_5ForConditionalGeneration` | `Qwen3.5` | ✅︎ | ✅︎ |
 | `Step3VLForConditionalGeneration` | `Step3-VL` | ✅︎ | ❌︎ |
+| `DeepseekOCRForCausalLM` | `DeepSeek-OCR` | ✅︎ | ❌︎ |
```

- **@Isotr0py**: Let's keep this in alphabetical order:

### `vllm/model_executor/models/deepseek_ocr.py:678`
```diff
+        num_output_tokens = global_h * (global_w + 1)
+        self.global_image_output_token = num_output_tokens  # 272
+
+        local_h = local_w = math.ceil(local_input_side / downsample_ratio)
+        self.patch_side = local_h
+        self.single_patch_output_token = local_h * local_w  # 100
```

- **@Isotr0py**: I think we can make `image_side` and `global_image_output_token` etc as property, so we don't need to call `self._get_num_input_output_tokens` at `get_encoder_cudagraph_config`.
```python3
@property
def global_image_output_token(self):
...
```

### `vllm/model_executor/models/deepseek_ocr.py:831`
```diff
+
+        Only the global (SAM + CLIP + projector + newline) path is
+        captured by the CUDA graph. Local patches are handled eagerly in
+        ``postprocess_encoder_output()`` so that the graph avoids wasted
+        compute on zero-padded patch buffers when images are untiled.
+        """
```

- **@Isotr0py**: IIRC, encoder's computation bottlelack should be local patches forwarding instead of global view forwarding? So perhaps capturing graph for local patches should make more sense?

- **@shen-shanshan**: @Isotr0py I have implemented a new version - dual-path CUDA graph select for global images and local patches respectively, leading to certain degree of performance improvement for large images.

### `vllm/v1/worker/encoder_cudagraph.py:175`
```diff
+            )
+            # When `image_width <= 640 and image_height <= 640`, the mm inputs
+            # will only contain global image, without generating local patches.
+            self.local_token_budgets.insert(0, 0)
+            self.global_budget_graphs: dict[int, BudgetGraphMetadata] = {}
+            self.local_budget_graphs: dict[int, BudgetGraphMetadata] = {}
```

- **@Isotr0py**: Perhaps we can represent `budget_graphs` as `dict[str, dict[int, BudgetGraphMetadata]]` if there're multiple paths?

- **@shen-shanshan**: Good idea! Done.

### review @Isotr0py [APPROVED]
Overall LGTM, leave a nit.

### comment @shen-shanshan
CC @Isotr0py 

At first, I have tried to contain the `local_patch` encoding into the ViT cuda graph, but I find this solution has some drawbacks:

1. The grid and number of `local_patch` is dynamic according to the input images, and it also requires to add newline tokens to the end of each raw in the grid, which makes the tensor shape dynamic and unpredictable. Thus, we have to only put the encoding of the raw `local_patch` into the cuda graph, then add newline tokens in the `postprocess_encoder_output()`.
2. To make sure images with max number of `local_patch` could be correctly repalyed, we have to capture `max_crops` buffers for each input, even for images with size < `(640, 640)`, which will not be tiled and will not have local patches. In this case, the performance is worse than the eager execution, since it can lead to additional and redundant dummy `local_patch` replay.

Thus, I decide to only contain `global_image` encoding into the ViT cuda graph, with eager executing `local_patch` encoding, then assemble them in `postprocess_encoder_output()`. This is a tradeoff of performance for both images < `(640, 640)` and larger images.

Future Plan (in following PRs): I want to explore dual-path ViT cuda graph budget selecting mechanism for DeepSeek-OCR and Step3-VL to decouple the `global_image` replay and `local_patch` replay.

### comment @Isotr0py
Seems multimodal CI failures are related: https://buildkite.com/vllm/ci/builds/71879#019ebeee-1634-4a5e-b0a4-3280eadb8c8b

### comment @shen-shanshan
> Seems multimodal CI failures are related: https://buildkite.com/vllm/ci/builds/71879#019ebeee-1634-4a5e-b0a4-3280eadb8c8b

Sorry, there are still some issues, please don't merge. I will fix it recently.

### comment @shen-shanshan
@Isotr0py Could you please help re-trigger the CI? Thanks~

## PR #46005: [MM][CG] Support ViT full CUDA graph for DeepSeek-VL2 (@littlecircle0730, merged)

### `tests/v1/cudagraph/test_encoder_cudagraph.py:1050`
```diff
@@ -752,6 +759,286 @@ def test_video_model_returns_video_for_video_kwargs(self):
         assert model.get_input_modality(mm_kwargs) == "video"
 
 
+# ---------------------------------------------------------------------------
+# GPU tests — DeepSeek-VL2 batched-tile encoder CUDA graph
```

- **@shen-shanshan**: Please also add this model to `tests/models/multimodal/generation/test_vit_cudagraph.py` and update `examples/generate/multimodal/vision_language_offline.py`.

### `vllm/model_executor/models/deepseek_vl2.py:627`
```diff
+    def get_encoder_cudagraph_budget_range(
+        self, vllm_config: "VllmConfig"
+    ) -> tuple[int, int]:
+        h, w = self._proj_hw()
+        # Smallest valid image: 1 global + 1×1 local tile → 2 tiles total.
+        min_budget = h * (w + 1) * 2 + 1
```

- **@shen-shanshan**: QQ: Does DSVL2 use the same image size for both global image and local patches and always contains at least one local patch?

- **@littlecircle0730**: Thanks for the suggestion. 

For DeepSeek-VL2, global and local tiles use the same `vision_config.image_size` and are processed by the same vision encoder. Upstream DSVL2 concatenates all global/local tiles along the batch dimension and runs them together through a single `self.vision(total_tiles)` call. For a valid image, there is also always at least one local tile.

Because the per-tile input shape and encoder computation are identical, the current implementation represents the dynamic number of global/local tiles as a dynamic tile batch (`N_tiles`) rather than separate encoder paths.

Ref:
https://github.com/deepseek-ai/DeepSeek-VL2/blob/main/deepseek_vl2/models/modeling_deepseek_vl_v2.py

Would you still prefer using the multi-path CG abstraction here for consistency with the framework, even though both paths would have the same input shape/computation?

- **@shen-shanshan**: > Thanks for the suggestion.
> 
> For DeepSeek-VL2, global and local tiles use the same `vision_config.image_size` and are processed by the same vision encoder. Upstream DSVL2 concatenates all global/local tiles along the batch dimension and runs them together through a single `self.vision(total_tiles)` call. For a valid image, there is also always at least one local tile.
> 
> Because the per-tile input shape and encoder computation are identical, the current implementation represents the dynamic number of global/local tiles as a dynamic tile batch (`N_tiles`) rather than separate encoder paths.
> 
> Ref: https://github.com/deepseek-ai/DeepSeek-VL2/blob/main/deepseek_vl2/models/modeling_deepseek_vl_v2.py
> 
> Would you still prefer using the multi-path CG abstraction here for consistency with the framework, even though both paths would have the same input shape/computation?

Thanks for explanation, I understanded. I think the current implementation is acceptable, let's merge this initial support PR first, then optimize it in a following PR if others request for better performance.

### `vllm/model_executor/models/deepseek_vl2.py:611`
```diff
+            / self.projector_config.downsample_ratio
+        )
+        return h, w
+
+    def get_max_frames_per_video(self) -> int:
+        return 1
```

- **@shen-shanshan**: This is not needed now, FYI: https://github.com/vllm-project/vllm/blob/8bd7737a03bfcd4a1bca008f9f0e94729f8d67dc/vllm/v1/worker/encoder_cudagraph_defs.py#L87-L90

- **@littlecircle0730**: Deleted. Thank you.

### `vllm/model_executor/models/deepseek_vl2.py:604`
```diff
+        """Return (h, w) of the projector output per tile."""
+        h = w = math.ceil(
+            (self.vision_config.image_size // self.vision_config.patch_size)
+            / self.projector_config.downsample_ratio
+        )
+        return h, w
```

- **@shen-shanshan**: The returned value of `h, w` is fixed, but this method will be called multiple times, which will lead to additional cost on inference hot path. We can make it a property method or just cache the results.

- **@littlecircle0730**: Thanks for the suggestion. It makes sense!

### review @shen-shanshan [COMMENTED]
Sorry for late reply, I'm busy with working these days. I left some comments and questions.

Since this model uses similar pattern of DeepSeek-OCR, which processes the input image then generate one global image + dynamic num of local patches, I suppose it's better to use multi-path ViT CG for this pattern, FYI https://github.com/vllm-project/vllm/pull/43586 and https://github.com/vllm-project/vllm/pull/49934.

### review @shen-shanshan [COMMENTED]
Overall LGTM. Please resolve merge conflicts and the final comments, then we can get this in.

### review @shen-shanshan [APPROVED]
LGTM, please resolve merge conflicts.

## PR #40830: [MM][CG] Support ViT CG for Qwen2.5-VL (@johncalesp, merged)

### `tests/models/multimodal/generation/test_qwen2_5_vl.py:168`
```diff
+    not current_platform.is_cuda(), reason="ViT encoder CUDA graphs require CUDA"
+)
+def test_qwen2_5_vl_vision_transformer_cudagraph_matches_eager(
+    vllm_runner,
+    video_assets,
+) -> None:
```

- **@shen-shanshan**: Could we just reuse this test https://github.com/vllm-project/vllm/blob/main/tests/models/multimodal/generation/test_vit_cudagraph.py#L44-L58? We only need to add new configs for more models that have supported ViT CG.

- **@johncalesp**: No problem, I reverted the changes, we can add more later on if we need more coverage

### `vllm/model_executor/models/qwen2_5_vl.py:1591`
```diff
+        # Min: estimated smallest possible encoder input.
+        # 224x224 image → 16x16 patches (patch_size=14)
+        #                 spatial_merge_size=2 → 8x8 = 64 tokens
+        min_budget = 64
+        # Max: capped by max_num_batched_tokens
+        max_budget = vllm_config.scheduler_config.max_num_batched_tokens
```

- **@shen-shanshan**: Currently, there may be some issues with this `max_budget` auto-infer, we should sync the change in https://github.com/vllm-project/vllm/pull/40580/changes#diff-89b6b3d99f8961ecdc0cf8cf16b0f9e091a6227ee2a9bfb4f04b88043278e0e0R1805-R1809.

See https://github.com/vllm-project/vllm/pull/40580#discussion_r3122533201.

- **@johncalesp**: Thx for the feedback, just made the changes

### `vllm/model_executor/models/qwen2_5_vl.py:1666`
```diff
+    def get_encoder_cudagraph_per_item_input_sizes(
+        self,
+        mm_kwargs: dict[str, Any],
+    ) -> list[int]:
+        grid_thw = self._get_grid_thw_by_modality(mm_kwargs)
+        return [t * h * w for t, h, w in grid_thw]
```

- **@shen-shanshan**: Since these codes are totally the same as that of Qwen3-VL, maybe we should find out a way to eliminate redundancy (maybe in the future).

- **@johncalesp**: We can re-evaluate this in a future PR if this section of the code remain similar. I think for now it should be ok since it affects at most 3 models (qwen3, 2.5 and 2)

### `vllm/model_executor/models/qwen2_5_vl.py:1113`
```diff
@@ -1003,6 +1081,7 @@ def get_replacement_qwen2vl(item_idx: int, modality: str):
 class Qwen2_5_VLForConditionalGeneration(
```

- **@shen-shanshan**: Please also update this model to:

Doc: https://docs.vllm.ai/en/latest/design/cuda_graphs_multimodal/#model-integration-via-supportsencodercudagraph

Example: https://github.com/vllm-project/vllm/blob/main/examples/offline_inference/vision_language.py#L2466-L2469

- **@johncalesp**: done, btw the file vision_language.py has been moved. to vision_language_offline.py

### review @Isotr0py [APPROVED]
LGTM

### comment @shen-shanshan
> @shen-shanshan can you help review? Thanks

No problem.

## PR #40580: [MM][CG] Support `--enable-vit-cuda-graph` option for VLM examples (@shen-shanshan, merged)

### `examples/offline_inference/vision_language.py:2609`
```diff
+                1536,
+                2048,
+                2560,
+                3072,
+                3584,
+                4096,
```

- **@DarkLight1337**: Can we just determine this automatically? Different models may set different max_model_len in the examples

- **@shen-shanshan**: @DarkLight1337 I have removed the `encoder_cudagraph_token_budgets` hardcode, and let it auto-infer through `get_encoder_cudagraph_budget_range()` and `_generate_budgets()`.

When `max_model_len=4096`, the generated budgets are `[64, 128, 256, 512, 1024, 2048, 4096]`.

### `vllm/model_executor/models/qwen3_vl.py:1808`
```diff
         min_budget = 64
         # Max: capped by max_num_batched_tokens
-        max_budget = vllm_config.scheduler_config.max_num_batched_tokens
+        max_budget = min(
+            vllm_config.scheduler_config.max_num_batched_tokens,
+            self.model_config.max_model_len,
```

- **@DarkLight1337**: I thought `max_num_batched_tokens` isn't allowed to exceed `max_model_len` in the first place

- **@DarkLight1337**: Actually nvm, the cap is `max_num_seqs * max_model_len`

- **@DarkLight1337**: I'm still not sure whether we really need to cap it based on `max_model_len` though, isn't this budget in terms of the batch instead of a single prompt?

- **@shen-shanshan**: > I thought `max_num_batched_tokens` isn't allowed to exceed `max_model_len` in the first place

During my test, the log is:

```
Using max model len 4096
Chunked prefill is enabled with max_num_batched_tokens=8192.
```

- **@shen-shanshan**: > I'm still not sure whether we really need to cap it based on `max_model_len` though, isn't this budget in terms of the batch inside of a single prompt?

I suppose the budget is for batch multimodal inputs from scheduled encoder inputs.

- **@shen-shanshan**: But according to my test, if the `max_budget` > `max_model_len`, there will be an error like:

```
/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:111: operator(): block: [1181,0,0], thread: [11,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
```

Maybe there is still some thing wrong needs to be fixed for ViT CG.

- **@DarkLight1337**: Can you add a TODO to the code to fix it later?

- **@shen-shanshan**: > Can you add a TODO to the code to fix it later?

OK. I will solve this later.

### comment @shen-shanshan
@DarkLight1337 The CI seems get stuck, could you please help retrigger it? 😂

## PR #52256: [ROCm][CI] Enable ViT CUDA graph tests on AMD gfx950 GPUs (@shen-shanshan, merged)

### `.buildkite/test-amd.yaml:3689`
```diff
+  working_dir: "/vllm-workspace/tests"
+  source_file_dependencies:
+  - vllm/
+  - tests/models/multimodal
+  commands:
+  - pytest -v -s models/multimodal/generation/test_vit_cudagraph.py -m core_model
```

- **@AndreasKaratzas**: Do we need this? Can this be covered in the test group that upstream CI covers it too?

- **@shen-shanshan**: @AndreasKaratzas The `models/multimodal -m core_model` sweeps in the MI300 and MI355 "Multi-Modal Models (Standard) 4" steps now `--ignore .../test_vit_cudagraph.py` to avoid running it twice, so the new section could be `optional: true`.

- **@AndreasKaratzas**: I'm saying that I would prefer that the two test groups are integrated to the existing ones instead of adding new ones.

- **@shen-shanshan**: Oh sorry for misunderstanding, I will update it soon.

- **@shen-shanshan**: > I'm saying that I would prefer that the two test groups are integrated to the existing ones instead of adding new ones.

@AndreasKaratzas Updated.

### `.buildkite/test-amd.yaml:3851`
```diff
+  - vllm/config/compilation.py
+  - vllm/compilation
+  - vllm/platforms/rocm.py
+  commands:
+  - pytest -v -s v1/cudagraph/test_encoder_cudagraph.py
+
```

- **@AndreasKaratzas**: Do we need this? Can this be covered in the test group that upstream CI covers it too?

- **@shen-shanshan**: This is a new added test for ROCm path.

- **@AndreasKaratzas**: I'm saying that I would prefer that the two test groups are integrated to the existing ones instead of adding new ones.

### comment @shen-shanshan
CC @Isotr0py @AndreasKaratzas

## PR #42288: Adjust design around encoder_cudagraph_forward (@wdhongtw, merged)

### `vllm/model_executor/models/qwen3_vl.py:2014`
```diff
         buffers: dict[str, torch.Tensor],
     ) -> torch.Tensor:
-        pixel_values = self._get_pixel_values_by_modality(mm_kwargs)
-        grid_thw = self._get_grid_thw_by_modality(mm_kwargs)
-        return self.visual(pixel_values, grid_thw, encoder_metadata=buffers)
+        return self.visual(values, None, encoder_metadata=buffers)
```

- **@Isotr0py**: I think this could hit edge case for non Qwen-style ViT, because passing single pixel values maybe not enough to finish CUDA graph capture for other ViTs (especially clip/siglip variants that can have two coupled vision towers).

For example, deepseek-ocr's ViT can require `pixel_values` and `patch_embeds` at the same time, while we can hardly prepare `patch_embeds` independently in the buffer since they're coupled.
https://github.com/vllm-project/vllm/blob/c8a6e272e0d3994f299ae522975514fcd98b7c76/vllm/model_executor/models/deepencoder.py#L655-L661

- **@wdhongtw**: Interesting... maybe only one input tensor (and only one input key to extract input tensor from `mm_kwargs`, the current design of `EncoderCudaGraphConfig`) is not general enough. Or maybe the separation of *input* and *metadata buffer* is not appropriate from the beginning.

Need to have a deeper look and do some experiment on `DeepseekOCRForCausalLM` architecture before making a conclusion. Will be back ASAP.
Thanks @Isotr0py.

### review @shen-shanshan [APPROVED]
LGTM. Nice refactor! https://github.com/vllm-project/vllm/pull/43586 also needs this PR.

### review @Isotr0py [APPROVED]
Thanks for your patience!

### comment @Isotr0py
I'm not familiar with the difference between JAX graph and CUDA graph actually tbh. 😅.

But from the design concept, I would prefer to the second solution. For solution1, `metadata_buffers['pixel_values']` looks a bit confusing, especially with `inputs=False`. While solution2 is much more intuitive about inputs and metadata.

Also cc @shen-shanshan: does Ascend NPU also have similar cases about ACL graph capturing?

### comment @shen-shanshan
> I'm not familiar with the difference between JAX graph and CUDA graph actually tbh. 😅.
> 
> But from the design concept, I would prefer to the second solution. For solution1, `metadata_buffers['pixel_values']` looks a bit confusing, especially with `inputs=False`. While solution2 is much more intuitive about inputs and metadata.
> 
> Also cc @shen-shanshan: does Ascend NPU also have similar cases about ACL graph capturing?

With regard to ACL graph, it doesn't need to consider the python level things mentioned above, but it need to do some full graph update operations since ascend attention op requires host-side `seq_lens`.

### comment @Isotr0py
> Does this looks good?

I think this looks good. Thanks for the simplification!

### comment @shen-shanshan
> @Isotr0py cc @shen-shanshan Sorry for pinging again... Is there anything I can help for this PR?

I will take a look today.

### comment @Isotr0py
> I'm wondering whether a manual merge is an option here?

I don't have access to force merge, requested force merge from those who have access. BTW, we can also wait https://github.com/vllm-project/vllm/pull/43846 to merge as well.

## PR #41234: [Multimodal] Simplify ViT CUDA graph interfaces (@Isotr0py, merged)

### `vllm/v1/worker/encoder_cudagraph.py:262`
```diff
@@ -257,7 +259,7 @@ def _run_budget_graph(
         # Copy the input tensor. Buffers are sized for the full budget;
         # actual inputs may be smaller. Zero then slice-copy so padded
         # positions are invisible to attention (cu_seqlens masks them out).
-        input_key = self.config.input_key_by_modality[
+        input_key = input_key = self.config.input_key_by_modality[
```

- **@shen-shanshan**: This `input_key = input_key = ...` is changed by mistake?

- **@Isotr0py**: Ooops, good catch!

### `vllm/model_executor/models/interfaces.py:1556`
```diff
-    def get_encoder_cudagraph_per_item_input_sizes(
-        self,
-        mm_kwargs: dict[str, Any],
-    ) -> list[int]:
-        """Return input size (e.g. patch count) for each item.
+    ) -> list["EncoderItemSpec"]:
```

- **@shen-shanshan**: Since https://github.com/vllm-project/vllm/pull/40830 has been merged, maybe we should also make Qwen2.5-VL adapt to these new interfaces.

### review @shen-shanshan [APPROVED]
LGTM.

### comment @Isotr0py
cc @shen-shanshan @b-mu about ViT CUDA graph cleanup.

### comment @shen-shanshan
> https://github.com/vllm-project/vllm/blob/4454a18695287a8b0ef5701e5390636d692e3aa5/docs/design/cuda_graphs_multimodal.md?plain=1#L71
> 
> should be updated as well

Done in https://github.com/vllm-project/vllm/pull/44388.

## PR #44720: [Doc] Document Qwen3.6 (dense + MoE) ViT CUDA graph support (@harsha20032020, merged)

### `tests/models/multimodal/generation/test_vit_cudagraph.py:142`
```diff
+                dummy_hf_overrides,
+                model_arch="Qwen3_5MoeForConditionalGeneration",
+            ),
+        },
+        marks=[pytest.mark.core_model],
+    ),
```

- **@shen-shanshan**: For CI test, we should use models as small as possible. 27B and 35B are too big, you need to use dummy weights. FYI, https://github.com/vllm-project/vllm/pull/43082/changes#diff-1cf2f4bc8aaa68c74da471fe5168da8c506731f1285344e4edef03a395d6520aR103-R125.

### `docs/design/cuda_graphs_multimodal.md:90`
```diff
 | `Qwen2VLForConditionalGeneration` | `Qwen2-VL` | ✅︎ | ✅︎ |
 | `Qwen2_5_VLForConditionalGeneration` | `Qwen2.5-VL` | ✅︎ | ✅︎ |
 | `Qwen3VLForConditionalGeneration` | `Qwen3-VL` | ✅︎ | ✅︎ |
-| `Qwen3_5ForConditionalGeneration` | `Qwen3.5` | ✅︎ | ✅︎ |
+| `Qwen3_5ForConditionalGeneration` | `Qwen3.5`, `Qwen3.6` | ✅︎ | ✅︎ |
+| `Qwen3_5MoeForConditionalGeneration` | `Qwen3.5-MoE`, `Qwen3.6-MoE` | ✅︎ | ✅︎ |
```

- **@shen-shanshan**: If `Qwen3.6` belongs to the same model arch with `Qwen3.5`, maybe we don't need to add an additional CI test for this model? I suppose doc and example updating is enough? CC @Isotr0py

- **@Isotr0py**: Agree, dense and MoE both uses same ViT encoder, so adding these additional CI doesn't really make sense, just wasting the valuable CI resource.

### `examples/generate/multimodal/vision_language_offline.py:2620`
```diff
@@ -2530,6 +2616,8 @@ def run_tarsier2(questions: list[str], modality: str) -> ModelRequestData:
     "qwen3_vl_moe": run_qwen3_vl_moe,
     "qwen3_5": run_qwen3_5,
     "qwen3_5_moe": run_qwen3_5_moe,
+    "qwen3_6": run_qwen3_6,
+    "qwen3_6_moe": run_qwen3_6_moe,
```

- **@Isotr0py**: There is no need to add these new examples becasue Qwen3.6 still uses `qwen3_5` as model type: https://huggingface.co/Qwen/Qwen3.6-27B/blob/main/config.json#L7

### review @shen-shanshan [APPROVED]
LGTM.

## PR #49934: [1/N] Unify multiple-path encoder cuda graph support (@Isotr0py, merged)

### `vllm/model_executor/models/deepseek_ocr.py:945`
```diff

```

- **@shen-shanshan**: The method desc should be updated too.

### `vllm/model_executor/models/step3_vl.py:898`
```diff
-        local_output: torch.Tensor | None = None,
     ):
         """CPU-side per-item merge after dual-path graph replay.
 
         ``output`` contains global-image features and ``local_output``
         contains local-patch features (or ``None`` when there are no patches).
```

- **@shen-shanshan**: Ditto.

### `vllm/model_executor/models/deepseek_ocr.py:974`
```diff
 
         # Split local output into per-patch groups.
         local_flat = None
-        if total_patches > 0 and local_output is not None:
+        if total_patches > 0:
+            assert local_output is not None
```

- **@shen-shanshan**: If a batch only contains global images (without even one local patch), I suppose `local` key will not be added into the graph outputs (be skipped) and this `local_output` could be `None`? Step3-VL ditto.

### review @shen-shanshan [COMMENTED]
Great work! That's simpler. Only some small suggestions.

### comment @Isotr0py
cc @shen-shanshan

## PR #65084: [Multimodal] Refactor encoder cuda graph to support share graph buffer reuse (@Isotr0py, merged)

### `vllm/v1/worker/encoder_cudagraph.py:261`
```diff
+            for k in range(steps):
+                point = base + k * step
+                if min_budget <= point <= max_budget:
+                    points.add(point)
+            base *= 2
+        return sorted(points)
```

- **@shen-shanshan**: The new grid is anchored at powers of two `(2^n + k * 2^n / steps)`, not at `min_budget`. The old ladder doubled from `min_budget`, so it landed exactly on `per_item * 2^k` for fixed-granularity paths.

For example, the DeepSeek-OCR global path (272 tokens per image, `rows = budget // 272`) now gets `[272, 320, 384, 448, 512, 640, ..., 1024, 1536, ...]`.

- 4 global images (1088 tokens) used to fit 1088 exactly and now go to 1536, which captures 5 rows.
- 8 images go to 3072, which captures 11 rows.
- The 320/384/448/512 budgets all give `rows == 1` like 272, so they are redundant graphs that never get picked.

Could you anchor the grid at `min_budget` (e.g. `min_budget * 2^(k/4)`, rounded), or let a path declare a token granularity and dedupe budgets that map to the same row count?

- **@Pb314314**: Thanks @shen-shanshan, good catch! This is fixed in #65227, which builds on this PR: paths with fixed-size items (DeepSeek-OCR, Step3-VL) get budgets that are multiples of the item size, so 4 global images use 1088 and 8 use 2176 exactly, with no redundant graphs.

- **@Isotr0py**: Thanks @Pb314314, I picked your fix https://github.com/vllm-project/vllm/pull/65084/commits/e48c5fad354ddbf9a337bccb19e0acf0d4ba8c4c from https://github.com/vllm-project/vllm/pull/65227 (also added co-author). Let's keep these two PR splitted so that #65227 will be much smaller after this PR merged :)

### `docs/design/cuda_graphs_multimodal.md:87`
```diff
 **Budget generation.** Each path gets a separate budget list. For DeepSeek-OCR:
 
-* the `global` path — power-of-2 budgets starting at the global path minimum (e.g., `[272, 544, 1088, 2176, 4352, 8704, 13824]` for DeepSeek-OCR).
-* the `local` path — power-of-2 budgets starting at the local path minimum (e.g., `[0, 100, 200, 400, 800, 1600, 3200, 6400, 12800, 13824]` for DeepSeek-OCR). A budget of `0` is included when `allow_zero_tokens=True` to handle images with no local patches (images ≤ 640×640 that produce only global features).
+* the `global` path — subdivided-grid budgets starting at the global path minimum (e.g., `[272, 320, 384, 448, 512, 640, 768, 896, 1024, 1536, 2048, 3072, 4096, 6144, 8192, 12288, 13824]` for DeepSeek-OCR).
+* the `local` path — subdivided-grid budgets starting at the local path minimum (e.g., `[0, 100, 128, 160, 192, 224, 256, 320, 384, 448, 512, 640, 768, 896, 1024, 1536, 2048, 3072, 4096, 6144, 8192, 12288, 13824]` for DeepSeek-OCR). A budget of `0` is included when `allow_zero_tokens=True` to handle images with no local patches (images ≤ 640×640 that produce only global features).
```

- **@shen-shanshan**: The doc should also be updated here, others LGTM.

- **@Isotr0py**: Fixed in https://github.com/vllm-project/vllm/pull/65084/commits/12f1bb994edb4428f9da5cfbf60f8635f88a9f5f

### comment @shen-shanshan
What a coincidence! I just implemented a similar PR https://github.com/vllm-project/vllm/pull/65351 yesterday. 😂

## PR #37914: [Docs] Add Encoder (ViT) CUDA Graphs section to CUDA Graphs design doc (@b-mu, merged)

### `docs/design/cuda_graphs.md:1`
```diff

```

- **@Isotr0py**: I think we can separate ViT Cuda graph doc into single ones like `docs/design/torch_compile_multimodal.md`

- **@b-mu**: I have moved it to a seperate file `docs/design/cuda_graphs_multimodal.md`

### comment @shen-shanshan
@b-mu @wangshangsam 

Hello, I also want to help improve this feature (ViT CG) in vLLM and maybe we can propose a RFC to track all the related PRs. How do you think?

I suppose there are some other things we can do in the future:

1. Support video inference for Qwen3-VL ViT CG (I'm currently working on this).
2. Support more models, e.g., Qwen3.5, GLM-V, Kimi K2.5.
3. Make benchmark comparisons for more scenarios and offer tuning guide for users to better use feature, since it's not always beneficial to performance. https://github.com/vllm-project/vllm/pull/35963#issuecomment-4115457421

### comment @shen-shanshan
> With that being said, @shen-shanshan if you would like to commit time and maintain a tracking issue for ViT-cudagraph-related features, please feel very welcome to go ahead :)

Thanks for your patient explanation! Currently, I'm working on vLLM multi-modal related things full-time and I'm sure I can commit time to maintain a tracker issue (only focus on CG for ViT).

## PR #40445: [MM][CG] Optimize default `max_frames_per_batch` auto-infer for ViT CUDA graph video inference (@shen-shanshan, merged)

### `vllm/multimodal/registry.py:103`
```diff
@@ -100,6 +100,9 @@ class MultiModalRegistry:
     A registry that dispatches data processing according to the model.
     """
 
+    def __init__(self):
```

- **@DarkLight1337**: We are trying to move away from using registry, can you use a cache in global scope instead?

- **@shen-shanshan**: I have removed this and just get `processing_info` directly through `_create_processing_info()` according to gemini's suggestions.

### comment @shen-shanshan
All the CI failures seem due to `ImportError: cannot import name 'GenerationConfig' from 'transformers'`.

### comment @shen-shanshan
@DarkLight1337 Could you please help retry the CI? It seems an error unrelated to this PR. Thanks~

## PR #42796: [MM][CG] Avoid over-padding Qwen2.5-VL encoder cudagraph window metadata (@huanghua1994, merged)

### `vllm/v1/worker/encoder_cudagraph.py:269`
```diff
         """Get per-item output token counts as plain ints."""
         return [spec.output_tokens for spec in self._get_item_specs(mm_kwargs)]
 
+    @staticmethod
+    def _is_cu_seqlens_buffer(key: str) -> bool:
+        return key.startswith("cu_") and "seqlens" in key
```

- **@Isotr0py**: Can we set up the padding logic in `EncoderCudaGraphConfig` for better flexibility?
```python3
        return EncoderCudaGraphConfig(
            modalities=modalities,
            input_key_by_modality={
                "image": "pixel_values",
                "video": "pixel_values_videos",
            },
            buffer_keys=[
                "rotary_pos_emb_cos",
                "rotary_pos_emb_sin",
                "window_index",
                "reverse_indices",
                "cu_seqlens",
                "cu_window_seqlens",
                "max_seqlen_full",
                "max_seqlen_window",
                "sequence_lengths_full",
                "sequence_lengths_window",
            ],
            padding_logics={
                "cu_seqlens": cu_seqlens_padding_func,
                "sequence_lengths_full": sequence_lengths_padding_func,
                ....
            },
            out_hidden_size=self.visual.out_hidden_size,
            max_frames_per_video=max_frames,
        )
```

### review @Isotr0py [APPROVED]
Thanks!

### comment @Isotr0py
Let's merge https://github.com/vllm-project/vllm/pull/42787 first becuase its only CI failure is unrelated.

## PR #41736: [MM][CG] Support ViT CG for Qwen2-VL (@johncalesp, merged)

### review @Isotr0py [APPROVED]
Sorry for missing this!

### comment @shen-shanshan
LGTM.

### comment @shen-shanshan
> @shen-shanshan can we set the ready tag to run the CI?

I don't have the authority to add label... 

CC @DarkLight1337 @Isotr0py

## PR #46837: [MM][CG] Support ViT CUDA Graph for Gemma-4 (@anthonsu, merged)

### `tests/models/multimodal/generation/test_vit_cudagraph.py:253`
```diff
@@ -249,6 +249,26 @@ def gemma3_chat_template(content: str) -> str:
         },
         skip=True,  # TODO: Re-enable this once OOM issues are resolved on CI.
     ),
+    "gemma4": VitCudagraphTestConfig(
+        model="google/gemma-4-31B-it",
```

- **@Isotr0py**: ```suggestion
        model="google/gemma-4-E2B-it",
```
We can use 2B model here, then we don't need to use the dummy ones.

### review @Isotr0py [APPROVED]
Overall look reasonable, thanks!

## PR #43082: [CI] Fix "test_vit_cudagraph_[image|video][step3_vl]" failure (@haosdent, merged)

### `tests/models/multimodal/generation/test_vit_cudagraph.py:104`
```diff
     "step3_vl": VitCudagraphTestConfig(
         model="stepfun-ai/Step3-VL-10B",
+        modalities=["image"],
         image_prompt=step3_vl_chat_template("What is in this image?"),
-        video_prompt=None,
+        marks=[large_gpu_mark(min_gb=32)],
```

- **@Isotr0py**: Actually, skinpping the test on L4 CI could be risky, because there is no more test coverage for model's encoder CG impl.

I think a better solution is using a smaller dummy model through `hf_overrides` + reducing budget_size as @JisoLya suggested, especially we only check whether ViT CG can function normally for this model here.

## PR #44394: [MM][Perf] Add PaddleOCR-VL encoder CUDA graph support (@harsha20032020, open)

### review @shen-shanshan [COMMENTED]
Sorry for late reply, I'm busy with working these days. Btw, could you please rebase on the latest main and resolve the conflicts?

### review @shen-shanshan [COMMENTED]
I don't think the TTFT numbers in the description measure the encoder at all. `vllm bench latency` only builds random-token `TokensPrompt`s (vllm/benchmarks/latency.py), so no image is ever sent; `--limit-mm-per-prompt` only affects memory profiling. The ~6% delta is therefore noise or a side effect of graph capture, not the ViT graph.

Could you re-measure with a workload that actually carries images, e.g. `vllm bench serve --dataset-name random-mm` with a fixed 896x896 bucket, and report TTFT plus encoder time with graph on/off?

I used this command to bench the encoder in my history PR:

```bash
vllm bench mm-processor \
--model /shared/models/modelscope/models/deepseek-ai/DeepSeek-OCR \
--max-model-len 8192 \
--dataset-name random-mm \
--random-mm-base-items-per-request 1 \
--random-mm-num-mm-items-range-ratio 0.0 \
--random-mm-bucket-config '{(896, 896, 1): 1.0}' \
--random-mm-limit-mm-per-prompt '{"image": 1, "video": 0}' \
--num-prompts 100 \
--compilation-config '{"cudagraph_mm_encoder": true, "encoder_cudagraph_max_vision_items_per_batch": 8}'
```

## PR #46034: [Bugfix][MM][CG] Enable dual-path ViT CUDA graph for Step3-VL (@shen-shanshan, merged)

### comment @shen-shanshan
CC @Isotr0py

### comment @shen-shanshan
CC @DarkLight1337 CI failures seem unrelated.

## PR #50929: [MM][CG] Support ViT full CUDA graph for Kimi-K2.5 (@lk-chen, merged)

### review @Isotr0py [APPROVED]
Thanks!

### comment @Isotr0py
@codex review

## PR #58925: [Multimodal] Optimize encoder cudagraph padding logic (@Isotr0py, merged)

### review @shen-shanshan [APPROVED]
LGTM.

### comment @Isotr0py
@codex review

## PR #38040: [Fix] Misc Fixes in ViT CUDA Graph (@b-mu, merged)

### review @Isotr0py [APPROVED]
LGTM, requested force merge.

## PR #42151: [MM][Perf][CG] Support ViT full CUDA graph for Qwen3.5 (@shen-shanshan, merged)

### comment @shen-shanshan
CC @DarkLight1337 @Isotr0py

## PR #43321: Correcting the mock classes for MM GC tests (@wdhongtw, merged)

### review @shen-shanshan [APPROVED]
LGTM.

## PR #44388: [Doc] Update ViT CUDA graph interfaces (@shen-shanshan, merged)

### comment @shen-shanshan
CC @Isotr0py

## PR #44484: [MM][CG] Simplify ViT CUDA graph interfaces (@shen-shanshan, merged)

### comment @shen-shanshan
CC @Isotr0py

## PR #56277: [MM] Mistral vision Encoder CG support  (@NickLucche, merged)

### review @Isotr0py [APPROVED]
LGTM

## PR #60519: [Doc] Update ViT CUDA graph design doc (@shen-shanshan, merged)

### comment @shen-shanshan
CC @Isotr0py

## PR #65351: [Multimodal][Bugfix] Fix Step3-VL encoder output aliasing and guard shared CUDA graph capture buffers (@shen-shanshan, merged)

### comment @shen-shanshan
I have updated my PR, only left some minor fixes as follow-up to https://github.com/vllm-project/vllm/pull/65084. CC @Isotr0py
