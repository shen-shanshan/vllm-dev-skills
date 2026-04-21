---
name: vllm-ascend-test-generator
description: Generate test cases for the vllm-ascend project. Use this skill when the user wants to write unit tests or end-to-end tests for vllm-ascend code, functions, scenarios, or features. Triggered by requests like "帮我写XXX的测试用例", "生成XXX的单元测试", "为XXX功能写测试", "generate tests for XXX", "write unit test for XXX function".
---

# vllm-ascend Unit Test Generator

Generate well-structured tests for the vllm-ascend project, following the patterns established in the existing test suite.

## Test Type Decision

Choose between **unit test (UT)** and **end-to-end test (E2E)** based on what is being tested:

| Scenario | Type | Location |
|---|---|---|
| Single function / class method | UT | `tests/ut/` |
| Internal logic, config parsing | UT | `tests/ut/` |
| Hardware-specific ops (no model load) | UT | `tests/ut/` |
| Model inference, full pipeline | E2E | `tests/e2e/singlecard/` |
| Multi-card / distributed scenario | E2E | `tests/e2e/multicard/` |
| Nightly / weekly regression | E2E | `tests/e2e/nightly/` or `tests/e2e/weekly/` |
| Quantization end-to-end | E2E | `tests/e2e/singlecard/` |
| LoRA, VLM, speculative decode | E2E | `tests/e2e/singlecard/` |

If the user specifies a target directory, use that instead.

## Directory Placement Within tests/ut/

Match the subdirectory to the module under test:

- `tests/ut/attention/` → attention backends
- `tests/ut/ops/` → custom ops
- `tests/ut/quantization/` → quantization logic
- `tests/ut/worker/` → worker classes
- `tests/ut/distributed/` → distributed utilities
- `tests/ut/core/` → core scheduler / engine logic
- `tests/ut/` (root) → platform, config, envs, utils

## Unit Test Pattern

```python
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
# Licensed under the Apache License, Version 2.0

import importlib
from unittest.mock import MagicMock, patch

import pytest

from tests.ut.base import TestBase
from vllm_ascend.<module> import <ClassName>


class Test<ClassName>(TestBase):

    @staticmethod
    def mock_vllm_config():
        mock = MagicMock()
        mock.compilation_config = MagicMock()
        mock.model_config = MagicMock()
        mock.parallel_config = MagicMock()
        mock.cache_config = MagicMock()
        mock.scheduler_config = MagicMock()
        mock.speculative_config = None
        return mock

    def setUp(self):
        self.obj = <ClassName>()

    def test_<method_name>(self):
        # arrange
        ...
        # act
        result = self.obj.<method>(...)
        # assert
        self.assertEqual(result, expected)

    @patch("vllm_ascend.<module>.<dependency>")
    def test_<method_with_mock>(self, mock_dep):
        mock_dep.return_value = ...
        result = self.obj.<method>(...)
        mock_dep.assert_called_once_with(...)
        self.assertEqual(result, expected)
```

Key rules for UT:
- Inherit from `TestBase` (not `unittest.TestCase` directly)
- Mock `torch.npu.*` and `torch_npu.*` — they are unavailable in CI
- Use `importlib.reload(module)` when testing code that runs at import time
- Use `self.assertLogs(logger="vllm", level="WARNING")` to assert log output
- Never load a real model in UT

## E2E Test Pattern

```python
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# Licensed under the Apache License, Version 2.0

import os
import pytest
from vllm import SamplingParams

from tests.e2e.conftest import VllmRunner

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

MODELS = [
    "Qwen/Qwen2.5-0.5B-Instruct",  # ~1GB, on ModelScope
]


@pytest.mark.parametrize("model", MODELS)
def test_<feature>(model) -> None:
    prompts = ["Hello, my name is"]
    with VllmRunner(model, max_model_len=512, gpu_memory_utilization=0.7) as runner:
        outputs = runner.generate_greedy(prompts, max_tokens=10)
    assert outputs is not None
    assert len(outputs) > 0
```

Key rules for E2E:
- Always use `VllmRunner` as a context manager
- Set `VLLM_WORKER_MULTIPROC_METHOD=spawn` at the top
- Use `@pytest.mark.parametrize` for multiple models
- Keep `max_model_len` small (≤1024) to limit NPU memory usage
- Prefer greedy decoding (`generate_greedy`) for deterministic tests

## Recommended Small Models (ModelScope)

Always prefer these over large models. All are downloadable from ModelScope:

| Use Case | Model ID | Size |
|---|---|---|
| General text generation | `Qwen/Qwen2.5-0.5B-Instruct` | ~1 GB |
| General text generation | `Qwen/Qwen2.5-1.5B-Instruct` | ~3 GB |
| MHA architecture | `openbmb/MiniCPM-2B-sft-bf16` | ~4 GB |
| GQA architecture | `OpenBMB/MiniCPM4-0.5B` | ~1 GB |
| Vision-language | `Qwen/Qwen2.5-VL-3B-Instruct` | ~6 GB |
| Quantization test | `Qwen/Qwen2.5-0.5B-Instruct` with W8A8 | ~0.5 GB |
| Audio / speech | `openai-mirror/whisper-large-v3-turbo` | ~1.6 GB |

For distributed / multi-card tests, use models no larger than 7B.

## Generation Workflow

1. **Identify** what is being tested (function, class, scenario, feature)
2. **Decide** UT vs E2E (see table above)
3. **Determine** output directory (user-specified → auto-detect from module path)
4. **Write** the test file following the patterns above
5. **Name** the file `test_<subject>.py`
6. **Add** license header (Apache 2.0, Huawei copyright 2025)

For detailed patterns and more examples, see:
- [references/test-patterns.md](references/test-patterns.md) — annotated examples from the real test suite

## Example Prompts

针对 PR `https://github.com/vllm-project/vllm-ascend/pull/7427` 中提到的问题和代码修改，帮我写一个测试用例。
我想模拟在单卡上连续启动两个 vllm-ascend 服务时（模型使用 `Qwen3-0.6B`），第二个服务会不会出现 Available KV cache memory 不够的情况。
你可以先思考能不能直接针对 `determine_available_memory()` 这个方法写 ut 来覆盖这种场景。
如果 ut 无法测试这种场景的话，你就写一个 e2e 测试来看护这种场景。
测试代码写完之后，记得要加到 `.github/workflows/` 目录下对应的 `.yaml` 文件中，这样才能在 GitHub 上触发 CI。
/vllm-ascend-ut-generator
