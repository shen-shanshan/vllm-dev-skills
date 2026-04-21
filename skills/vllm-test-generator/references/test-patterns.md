# vLLM Test Patterns Reference

## Table of Contents

1. [Conftest Fixtures](#1-conftest-fixtures)
2. [pytest Markers](#2-pytest-markers)
3. [Mock Patterns](#3-mock-patterns)
4. [Distributed / Multi-GPU Tests](#4-distributed--multi-gpu-tests)
5. [Quantization Tests](#5-quantization-tests)
6. [Kernel / CUDA Tests](#6-kernel--cuda-tests)
7. [v1 Engine Tests](#7-v1-engine-tests)
8. [API Server Tests (OpenAI)](#8-api-server-tests-openai)
9. [Buildkite CI Structure](#9-buildkite-ci-structure)

---

## 1. Conftest Fixtures

Available from `tests/conftest.py` (use relative imports inside `tests/`):

```python
from ..conftest import HfRunner, VllmRunner
```

### `hf_runner` fixture
Runs a HuggingFace model for ground-truth comparison.
```python
@pytest.mark.parametrize("model", ["hmellor/tiny-random-LlamaForCausalLM"])
def test_correctness(hf_runner, model):
    with hf_runner(model, dtype="float16") as hf:
        hf_outputs = hf.generate_greedy(["Hello"], max_new_tokens=5)
```

### `vllm_runner` fixture
Runs a vLLM model via the `LLM` class.
```python
@pytest.mark.parametrize("model", ["hmellor/tiny-random-LlamaForCausalLM"])
def test_correctness(vllm_runner, model):
    with vllm_runner(model, max_model_len=512) as vllm:
        outputs = vllm.generate_greedy(["Hello"], max_tokens=5)
```

### `example_prompts` fixture
Provides a list of standard test prompts — use this instead of hardcoding.

### Module-scoped LLM (avoid re-loading)
```python
import weakref
from vllm import LLM
from vllm.distributed import cleanup_dist_env_and_memory

@pytest.fixture(scope="module")
def llm():
    llm = LLM(model="distilbert/distilgpt2", enforce_eager=True,
               gpu_memory_utilization=0.10)
    yield weakref.proxy(llm)
    del llm
    cleanup_dist_env_and_memory()

@pytest.mark.skip_global_cleanup  # required when using module-scoped fixture
def test_something(llm):
    ...
```

---

## 2. pytest Markers

Common markers used in the vllm test suite:

```python
@pytest.mark.skip_global_cleanup      # used with module-scoped LLM fixtures
@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
@pytest.mark.skipif(condition, reason="...")
@pytest.mark.slow                     # marks slow tests (often excluded in fast CI runs)
@pytest.mark.core_model               # marks tests for core/standard model variants
```

Custom markers are defined in `pyproject.toml`. When adding a new custom marker, register it there.

---

## 3. Mock Patterns

### Mocking a function in a module
```python
from unittest.mock import patch, MagicMock

@patch("vllm.config.ModelConfig._get_num_layers")
def test_config_layers(mock_get_layers):
    mock_get_layers.return_value = 12
    config = ModelConfig(...)
    assert config.num_layers == 12
```

### Mocking a class attribute
```python
with patch.object(SomeClass, "attribute", new=MagicMock(return_value=42)):
    result = obj.method()
```

### Mocking environment variables
```python
@patch.dict(os.environ, {"VLLM_WORKER_MULTIPROC_METHOD": "spawn"})
def test_something():
    ...
```

### Testing import-time behavior
```python
import importlib

def test_import_side_effect(monkeypatch):
    monkeypatch.setenv("SOME_ENV", "value")
    import vllm.some_module
    importlib.reload(vllm.some_module)
    assert vllm.some_module.SOME_VALUE == "value"
```

---

## 4. Distributed / Multi-GPU Tests

```python
from tests.utils import multi_gpu_test

@multi_gpu_test(num_gpus=2)
@pytest.mark.parametrize("model", ["meta-llama/Llama-3.2-1B-Instruct"])
def test_tensor_parallel(model):
    llm = LLM(model=model, tensor_parallel_size=2)
    outputs = llm.generate(["Hello"], SamplingParams(max_tokens=5))
    assert outputs
```

The `@multi_gpu_test` decorator skips the test if fewer than `num_gpus` GPUs are available.

For Buildkite, distributed tests live in `tests/distributed/` and are configured in `.buildkite/test_areas/distributed.yaml`.

---

## 5. Quantization Tests

```python
from vllm import LLM, SamplingParams

def test_fp8_quantization():
    llm = LLM(
        model="distilbert/distilgpt2",
        quantization="fp8",
        gpu_memory_utilization=0.3,
        enforce_eager=True,
    )
    outputs = llm.generate(["Hello"], SamplingParams(max_tokens=5))
    assert len(outputs) > 0
```

Quantization tests go in `tests/quantization/` and CI config is `.buildkite/test_areas/quantization.yaml`.

---

## 6. Kernel / CUDA Tests

```python
import torch
import pytest
from vllm._custom_ops import some_kernel

@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_custom_kernel():
    x = torch.randn(16, 64, device="cuda", dtype=torch.float16)
    result = some_kernel(x)
    expected = ...
    torch.testing.assert_close(result, expected, atol=1e-3, rtol=1e-3)
```

Kernel tests go in `tests/kernels/` and CI config is `.buildkite/test_areas/kernels.yaml`.

---

## 7. v1 Engine Tests

The `tests/v1/` directory mirrors the structure of `vllm/v1/`:

- `tests/v1/core/` — scheduler, block manager
- `tests/v1/worker/` — GPU worker
- `tests/v1/engine/` — engine core
- `tests/v1/e2e/general/` — full pipeline e2e

```python
# Unit test for v1 scheduler
from vllm.v1.core.scheduler import Scheduler

def test_scheduler_add_request():
    scheduler = Scheduler(scheduler_config=..., cache_config=..., lora_config=None)
    req = make_request(...)
    scheduler.add_request(req)
    assert scheduler.num_running_reqs() == 1
```

```python
# E2E test in tests/v1/e2e/general/
from vllm import LLM, SamplingParams

def test_v1_basic_generate():
    llm = LLM(model="distilbert/distilgpt2", enforce_eager=True)
    outputs = llm.generate(["Hello"], SamplingParams(max_tokens=5))
    assert len(outputs) == 1
```

---

## 8. API Server Tests (OpenAI)

For testing the OpenAI-compatible API server, use the `openai_server` fixture:

```python
import openai
import pytest

@pytest.fixture(scope="module")
def server(vllm_runner):
    # The fixture starts an actual API server subprocess
    ...

def test_completion_endpoint(server):
    client = openai.OpenAI(base_url=server.url, api_key="token")
    resp = client.completions.create(
        model="distilbert/distilgpt2",
        prompt="Hello",
        max_tokens=5,
    )
    assert len(resp.choices) > 0
```

See `tests/entrypoints/openai/conftest.py` for the `openai_server` fixture definition.

---

## 9. Buildkite CI Structure

### Finding the right yaml

| Test location | Buildkite yaml |
|---|---|
| `tests/basic_correctness/` | `.buildkite/test_areas/basic_correctness.yaml` |
| `tests/entrypoints/` | `.buildkite/test_areas/entrypoints.yaml` |
| `tests/models/language/` | `.buildkite/test_areas/models_language.yaml` |
| `tests/models/multimodal/` | `.buildkite/test_areas/models_multimodal.yaml` |
| `tests/quantization/` | `.buildkite/test_areas/quantization.yaml` |
| `tests/kernels/` | `.buildkite/test_areas/kernels.yaml` |
| `tests/distributed/` | `.buildkite/test_areas/distributed.yaml` |
| `tests/lora/` | `.buildkite/test_areas/lora.yaml` |
| `tests/v1/e2e/` | `.buildkite/test_areas/e2e_integration.yaml` |
| `tests/` root (unit tests) | No change needed — run by misc.yaml catch-all |

### yaml file structure

```yaml
group: <Group Name>
depends_on:
  - image-build
steps:
- label: <Step Label>
  timeout_in_minutes: 30
  device: h200_18gb          # optional; omit for default runner
  working_dir: "/vllm-workspace/tests"
  source_file_dependencies:
  - vllm/                    # rebuild trigger: any vllm source change
  - tests/path/to/test_file.py
  commands:
  - export VLLM_WORKER_MULTIPROC_METHOD=spawn   # for multiproc tests
  - pytest -v -s path/to/test_file.py
```

### Fetching a yaml before editing

```bash
gh api repos/vllm-project/vllm/contents/.buildkite/test_areas/<area>.yaml \
  --jq '.content' | base64 -d
```

### When to add a new step vs. append to existing

- **Append** when the new test logically belongs to an existing label (same topic, similar timeout)
- **New step** when the test needs a different device, much longer timeout, or is logically separate
- **New yaml** only when the test area has no existing yaml at all
