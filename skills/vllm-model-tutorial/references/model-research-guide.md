# Model Research Guide for vLLM Model Tutorials

This reference provides detailed methodology for researching a model's architecture, paper, and vLLM implementation.

## Research Strategy Overview

Research should follow a **three-axis approach** in parallel where possible:

1. **Paper/Report Axis**: Model architecture from official publications
2. **Source Code Axis**: Implementation details from vLLM codebase
3. **Ecosystem Axis**: Community knowledge, comparisons, benchmarks

## 1. Finding Model Files in vLLM

### Locating the Model Implementation

Use `gh` CLI to search the vllm-project/vllm repository:

```bash
# Search for model file in model_executor/models
gh search code "repo:vllm-project/vllm {model_keyword}" --path vllm/model_executor/models --json path,repository

# For multimodal models, also check multimodal processing
gh search code "repo:vllm-project/vllm {model_keyword}" --path vllm/multimodal --json path,repository

# Search for config or registration
gh search code "repo:vllm-project/vllm {model_upper} path:vllm/model_executor/models path:registry" --json path
```

Common naming patterns in vLLM:
- Model file: `vllm/model_executor/models/{model_name_lower}.py` (e.g., `qwen3_vl.py`, `deepseek_v2.py`)
- Multimodal processor: `vllm/multimodal/processing/{model_name_lower}.py`
- Vision encoder: `vllm/model_executor/models/{model_name_lower}_vision.py` or integrated in model file
- Config: often in the same model file or `vllm/transformers_utils/configs/{model_name}.py`

### Key Code Patterns to Identify

When reading vLLM model implementations, locate these patterns:

**1. Model Registration**
```python
# Look for:
@register_model("ModelName")
class ModelForCausalLM(VllmModelForCausalLM):
    ...
```

**2. Attention Class**
```python
# Look for class ending in "Attention" extending vLLM base
class ModelAttention(VllmAttention):
    def forward(self, ...):
        ...
```

**3. MLP/MoE Class**
```python
# Look for MLP or MoE class
class ModelMLP(nn.Module):
    ...

class ModelMoE(nn.Module):
    ...
```

**4. Vision Encoder (VLM only)**
```python
# Look for vision-related classes
class ModelVisionTransformer(nn.Module):
    ...

class ModelVisionEncoder(nn.Module):
    ...
```

**5. Multimodal Processor**
```python
# In vllm/multimodal/processing/
class ModelMultiModalProcessor(BaseMultiModalProcessor):
    ...
```

### Tracing the Forward Pass

For each key component, trace the forward pass from input to output:

1. **Text path**: `input_ids` → embedding → transformer layers → lm_head → logits
2. **Vision path (VLM)**: `pixel_values` → ViT encoder → projection → merge with text embeddings
3. **Attention**: hidden_states → QKV projection → attention computation → output projection
4. **MoE**: hidden_states → gate/router → top-k selection → expert dispatch → weighted combine

## 2. Locating Technical Reports and Papers

### Search Strategy

```bash
# Use WebSearch with these queries:
"{ModelName} technical report arxiv"
"{ModelName} paper site:arxiv.org"
"{ModelName} technical report site:github.com"
"{ModelName} huggingface model card"
```

### Key Information to Extract from Papers

| Category | What to Extract |
|----------|----------------|
| Architecture | Hidden size, layers, heads, KV heads, intermediate size, activation, norm type |
| Attention | Attention type (MHA/GQA/MQA/MLA), head dimension, RoPE configuration |
| FFN/MoE | FFN type, expert count, top-k, routing mechanism, shared experts |
| VLM details | ViT architecture, patch size, image resolution, projection method, fusion strategy |
| Training | Training data, stages, context length, optimizer, learning rate schedule |
| Performance | Benchmarks, comparison tables, scaling behavior |

### For Model Family Evolution

When building the model series comparison, search for each predecessor:
- `"{ModelFamily}-V1" or original model`
- `"{ModelFamily}-V2" or second generation`
- `"{ModelFamily}-V2.5" or intermediate releases`
- Current model

For each variant, collect: release date, paper link, parameter count, key innovations, HF/ModelScope links.

## 3. Analyzing the LLM Architecture Gallery

The [LLM Architecture Gallery](https://sebastianraschka.com/llm-architecture-gallery/) provides standardized architecture cards. For each model, extract:

- Total and active parameters (MoE)
- Context window length
- License and release date
- Decoder type (dense, sparse MoE, hybrid, recurrent)
- Attention mechanism
- KV cache size per token
- Key architectural details
- AA Intelligence Index scores

Use this as a cross-reference to validate paper findings and ensure accuracy.

## 4. Available GitHub CLI Operations

When the vLLM repo is not cloned locally, use these `gh` commands:

```bash
# Search code in vLLM repo
gh search code "repo:vllm-project/vllm {query}" --path {path_filter} --json path

# Read a file from vLLM repo
gh api "repos/vllm-project/vllm/contents/vllm/model_executor/models/{model_file}.py" --jq '.content' | base64 -d

# List files in a directory
gh api "repos/vllm-project/vllm/contents/vllm/model_executor/models/" --jq '.[].name'

# Search issues mentioning the model
gh search issues "repo:vllm-project/vllm {model_name}" --limit 20 --json title,state,number

# Search PRs mentioning the model
gh search prs "repo:vllm-project/vllm {model_name} is:merged" --limit 10 --json title,number,mergedAt
```

## 5. Technical Principle Deep-Dive Templates

### MoE (Mixture of Experts)

When the model uses MoE, include:

1. **Router/Gate mechanism**: How tokens are routed to experts
   ```
   gate_logits = Linear(hidden → num_experts)
   weights = softmax(top_k_filter(gate_logits))
   ```
2. **Expert architecture**: FFN structure of each expert
3. **Load balancing**: Auxiliary loss, expert capacity, token dropping
4. **Expert parallelism**: How experts are distributed across GPUs in vLLM
5. **Key formula**: `MoE(x) = Σ w_i · E_i(x)` for i in top-k

### MLA (Multi-head Latent Attention)

When the model uses MLA (e.g., DeepSeek-V3):

1. **Latent compression**: How KV are compressed to low-dim latent space
2. **Decompression**: How latents are expanded back for attention computation
3. **Memory savings**: KV cache size comparison vs. standard MHA/GQA
4. **Key formula**: `c_t^{KV} = W^{DKV} h_t`, `k_t^C = W^{UK} c_t^{KV}`
5. **RoPE integration**: How positional encoding is applied in MLA

### GQA (Grouped-Query Attention)

When the model uses GQA:

1. **Head grouping**: How Q heads share KV heads
2. **Memory vs. quality tradeoff**: Compare MHA (1:1), GQA (N:1), MQA (all:1)
3. **KV cache calculation**: `kv_cache_size = 2 × num_kv_heads × head_dim × layers × bytes`

### ViT (Vision Transformer)

When the model is a VLM:

1. **Patch embedding**: Image → patches → linear projection → patch embeddings
2. **Position encoding**: Learnable or sinusoidal position encodings for 2D patches
3. **ViT forward**: Multi-head self-attention + FFN blocks (similar to text transformer)
4. **Feature extraction**: CLS token vs. full patch features
5. **Projection to LLM space**: How visual features are projected to match text embedding dimension

## 6. Common Model Categories and Their Patterns

### Dense Decoder-only LLMs
Models: Llama 4, Qwen3 (dense), Gemma 3, OLMo
- Likely use: GQA, SwiGLU FFN, RoPE, RMSNorm
- No ViT, no MoE
- vLLM code typically straightforward, extending standard base classes

### MoE Decoder-only LLMs
Models: DeepSeek-V3, Mixtral, Qwen3-MoE, Kimi K2
- Key additions: Router/gate, expert FFNs, load balancing, shared experts
- vLLM code: custom MoE layer, often with fused expert kernels
- Focus research on: routing mechanism, expert count, shared expert design

### VLM (Vision-Language Models)
Models: Qwen3-VL, InternVL3, LLaVA, MiniCPM-V
- Key additions: ViT encoder, multimodal projection, visual token integration
- vLLM code: separate vision model class + multimodal processor + fusion logic
- Focus research on: ViT architecture, image resolution handling, token merging strategy

### VLM + MoE
Models: Qwen3-VL-MoE (if exists)
- Combines all of the above complexities
- Most complex category — allocate more research time

## 7. Research Quality Checklist

Before starting to write the tutorial, verify:

- [ ] Located the model's official technical report or paper
- [ ] Extracted and verified all key hyperparameters
- [ ] Found the vLLM model file and read the key classes
- [ ] For VLM: found the multimodal processor and vision encoder files
- [ ] For MoE: understood the routing mechanism and expert configuration
- [ ] Built the model family evolution timeline with all predecessor models
- [ ] Collected HF and ModelScope links for all variants
- [ ] Cross-referenced architecture details against the LLM Architecture Gallery
- [ ] Identified all unique/special architectural components that need deep-dive explanation
