---
name: vllm-test-generator
description: Generate test cases for the vllm-project/vllm repository (https://github.com/vllm-project/vllm). Use this skill when the user wants to write unit tests or integration/e2e tests for vllm code, functions, classes, or features. Triggered by requests like "帮我写XXX的测试用例", "生成XXX的单元测试", "为XXX功能写测试", "generate tests for XXX in vllm", "write a test for XXX vllm function". Do NOT use for vllm-ascend — use vllm-ascend-ut-generator instead.
---

# vLLM Test Generator

Generate well-structured tests for [vllm-project/vllm](https://github.com/vllm-project/vllm), following the conventions of the existing test suite.

## Step 1 — Fetch Context

Before writing, fetch the relevant source and existing tests:

```bash
# Browse existing tests for the target area
gh api repos/vllm-project/vllm/contents/tests/<subdir> --jq '[.[] | {name}]'

# Read a representative existing test file
gh api repos/vllm-project/vllm/contents/tests/<path>.py --jq '.content' | base64 -d

# Read the source under test if needed
gh api repos/vllm-project/vllm/contents/vllm/<path>.py --jq '.content' | base64 -d
```

## Step 2 — Classify the Test

| What is being tested | Type | Directory |
|---|---|---|
| Single function / class / method | Unit | `tests/` or matching subdir |
| Config parsing, data structures, utils | Unit | `tests/` root |
| CUDA kernels | Unit | `tests/kernels/` |
| Attention backends | Unit/Integration | `tests/v1/attention/` or `tests/kernels/` |
| Full model inference with LLM class | Integration | `tests/entrypoints/llm/` |
| OpenAI-compatible API | Integration | `tests/entrypoints/openai/` |
| Model correctness (HF vs vLLM) | Integration | `tests/basic_correctness/` or `tests/models/language/` |
| Quantization end-to-end | Integration | `tests/quantization/` |
| LoRA end-to-end | Integration | `tests/lora/` |
| Distributed / multi-GPU | Integration | `tests/distributed/` |
| v1 engine internals | Unit/Integration | `tests/v1/` matching subdir |
| v1 e2e scenarios | Integration | `tests/v1/e2e/general/` |

If the user specifies a directory, use it.

## Step 3 — Write the Test

### License header (required on every file)

```python
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
```

### Unit test pattern

```python
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import MagicMock, patch
import pytest

from vllm.<module> import <ClassName>


def test_<function_behavior>():
    # Arrange
    obj = <ClassName>(...)
    # Act
    result = obj.<method>(...)
    # Assert
    assert result == expected


@patch("vllm.<module>.<dependency>")
def test_<function_with_mock>(mock_dep):
    mock_dep.return_value = ...
    result = <function>(...)
    mock_dep.assert_called_once_with(...)
    assert result == expected


class Test<ClassName>:

    def test_<method>(self):
        ...
```

Key rules for unit tests:
- Never load a real model
- Mock external deps (`torch.cuda.*`, network calls, file I/O)
- Use `pytest.raises(ExceptionType)` to test error paths
- Prefer plain `pytest` functions; use classes only when grouping related tests

### Integration test pattern (with LLM class)

```python
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import weakref
import pytest
from vllm import LLM, SamplingParams
from vllm.distributed import cleanup_dist_env_and_memory

MODEL = "distilbert/distilgpt2"  # use a tiny model

PROMPTS = ["Hello, my name is", "The capital of France is"]


@pytest.fixture(scope="module")
def llm():
    llm = LLM(model=MODEL, max_num_batched_tokens=4096,
               gpu_memory_utilization=0.10, enforce_eager=True)
    yield weakref.proxy(llm)
    del llm
    cleanup_dist_env_and_memory()


@pytest.mark.skip_global_cleanup
def test_<feature>(llm: LLM):
    outputs = llm.generate(PROMPTS, sampling_params=SamplingParams(max_tokens=10))
    assert len(outputs) == len(PROMPTS)
```

### Integration test pattern (HF vs vLLM correctness)

```python
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
from ..conftest import HfRunner, VllmRunner
from ..models.utils import check_outputs_equal

MODELS = ["hmellor/tiny-random-LlamaForCausalLM"]


@pytest.mark.parametrize("model", MODELS)
def test_<feature>(hf_runner, vllm_runner, model, example_prompts):
    with hf_runner(model) as hf_model:
        hf_outputs = hf_model.generate_greedy(example_prompts, max_new_tokens=5)
    with vllm_runner(model) as vllm_model:
        vllm_outputs = vllm_model.generate_greedy(example_prompts, max_tokens=5)
    check_outputs_equal(outputs_0_lst=hf_outputs, outputs_1_lst=vllm_outputs,
                        name_0="hf", name_1="vllm")
```

## Step 4 — Choose a Small Model

Always use the smallest suitable model. Prefer these:

| Use case | Model | Size |
|---|---|---|
| General text / any architecture | `distilbert/distilgpt2` | ~117M |
| Tiny random weights (no GPU quality) | `hmellor/tiny-random-LlamaForCausalLM` | <10M |
| Tiny random weights (Gemma2) | `hmellor/tiny-random-Gemma2ForCausalLM` | <10M |
| Real small LLM | `meta-llama/Llama-3.2-1B-Instruct` | 1B |
| Multimodal VLM | `llava-hf/llava-1.5-7b-hf` or smaller VLM | 7B |
| Quantization | `distilbert/distilgpt2` with int8 | ~60M |

Use `gpu_memory_utilization=0.10` to `0.3` and `max_num_batched_tokens≤4096` for integration tests.

## Step 5 — Place the File

- File name: `test_<subject>.py`
- If user didn't specify a directory, use the mapping in Step 2
- Ensure `__init__.py` exists in the target directory (check with `gh api`)

## Step 6 — Update CI (Integration tests only)

vLLM uses **Buildkite** (not GitHub Actions) for its main CI. Configuration lives in `.buildkite/test_areas/*.yaml`.

Find the matching yaml for the test area:

```bash
gh api repos/vllm-project/vllm/contents/.buildkite/test_areas --jq '[.[] | {name}]'
gh api repos/vllm-project/vllm/contents/.buildkite/test_areas/<area>.yaml --jq '.content' | base64 -d
```

Add the new test file to `source_file_dependencies` and `commands` of the relevant step:

```yaml
steps:
- label: Basic Correctness
  source_file_dependencies:
  - vllm/
  - tests/basic_correctness/test_basic_correctness
  - tests/basic_correctness/test_my_new_test.py   # ← add here
  commands:
  - pytest -v -s basic_correctness/test_basic_correctness.py
  - pytest -v -s basic_correctness/test_my_new_test.py          # ← add here
```

If no existing yaml fits, create a new one following the same structure (group, depends_on, steps).

**Unit tests** (no model loading) do NOT need CI changes — they run as part of existing catch-all steps.

## Reference

For detailed annotated examples from the real test suite, see:
- [references/test-patterns.md](references/test-patterns.md) — read when you need a specific pattern (conftest fixtures, parametrize, mock usage, distributed markers)
