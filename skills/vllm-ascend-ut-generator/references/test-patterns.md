# vllm-ascend Test Patterns Reference

Annotated examples from the real test suite.

## Table of Contents
1. [Unit Test: Class with Mocks](#1-unit-test-class-with-mocks)
2. [Unit Test: Import-time Code (importlib.reload)](#2-unit-test-import-time-code)
3. [Unit Test: Log Assertion](#3-unit-test-log-assertion)
4. [E2E Test: Text Generation](#4-e2e-test-text-generation)
5. [E2E Test: Multimodal (VLM)](#5-e2e-test-multimodal-vlm)
6. [conftest.py Patterns](#6-conftest-patterns)

---

## 1. Unit Test: Class with Mocks

From `tests/ut/test_platform.py`:

```python
import importlib
from unittest.mock import MagicMock, patch

import torch
from tests.ut.base import TestBase
from vllm_ascend.platform import NPUPlatform


class TestNPUPlatform(TestBase):

    @staticmethod
    def mock_vllm_config():
        """Factory for a fully-mocked VllmConfig."""
        mock = MagicMock()
        mock.compilation_config = MagicMock()
        mock.model_config = MagicMock()
        mock.parallel_config = MagicMock()
        mock.cache_config = MagicMock()
        mock.scheduler_config = MagicMock()
        mock.speculative_config = None
        mock.compilation_config.pass_config.enable_sp = False
        mock.compilation_config.cudagraph_mode = None
        return mock

    def setUp(self):
        self.platform = NPUPlatform()

    @patch("torch.npu.get_device_name")
    def test_get_device_name(self, mock_get_device_name):
        mock_get_device_name.return_value = "Ascend910B2"
        result = self.platform.get_device_name(0)
        self.assertEqual(result, "Ascend910B2")
        mock_get_device_name.assert_called_once_with(0)

    def test_get_attn_backend_cls_mla(self):
        # No mocks needed for pure-logic methods
        from vllm.v1.attention.selector import AttentionSelectorConfig
        config = AttentionSelectorConfig(
            dtype=torch.float16, head_size=0, kv_cache_dtype=None,
            block_size=128, use_mla=True, use_sparse=False,
        )
        result = self.platform.get_attn_backend_cls("ascend", config)
        self.assertEqual(result, "vllm_ascend.attention.mla_v1.AscendMLABackend")
```

**Key points:**
- Static factory `mock_vllm_config()` shared across test methods
- `@patch` decorates individual methods, not the whole class
- Patch targets the *import location*, not the definition location: `torch.npu.get_device_name` not `vllm_ascend.platform.torch.npu.get_device_name`

---

## 2. Unit Test: Import-time Code

When the code under test executes at import time (module-level side effects), use `importlib.reload()`:

```python
@patch("vllm_ascend.ascend_config.init_ascend_config")
@patch("vllm_ascend.utils.get_ascend_device_type", return_value=AscendDeviceType.A3)
def test_check_and_update_config(self, mock_soc, mock_init_ascend):
    mock_init_ascend.return_value = self.mock_vllm_ascend_config()
    vllm_config = self.mock_vllm_config()

    # Reload forces the module to re-run with mocks in place
    from vllm_ascend import platform
    importlib.reload(platform)

    self.platform.check_and_update_config(vllm_config)
    mock_init_ascend.assert_called_once_with(vllm_config)
```

**When to use:** Any method that calls `init_ascend_config`, `adapt_patch`, or other module-level initializers.

---

## 3. Unit Test: Log Assertion

Assert that a specific warning or info log is emitted:

```python
def test_warns_when_model_config_missing(self):
    vllm_config = self.mock_vllm_config()
    vllm_config.model_config = None

    with self.assertLogs(logger="vllm", level="WARNING") as cm:
        from vllm_ascend import platform
        importlib.reload(platform)
        self.platform.check_and_update_config(vllm_config)

    self.assertTrue(any("Model config is missing" in line for line in cm.output))
```

**Note:** `assertLogs` requires at least one log line to be emitted or it raises `AssertionError`. Wrap in try/except if the log is optional.

---

## 4. E2E Test: Text Generation

From `tests/e2e/singlecard/test_models.py`:

```python
import os
import pytest
from tests.e2e.conftest import VllmRunner

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

MODELS = ["Qwen/Qwen2.5-0.5B-Instruct"]


@pytest.mark.parametrize("model", MODELS)
def test_basic_generation(model) -> None:
    prompts = ["Hello, my name is", "The capital of France is"]

    with VllmRunner(model, max_model_len=512, gpu_memory_utilization=0.7) as runner:
        outputs = runner.generate_greedy(prompts, max_tokens=20)

    assert outputs is not None
    assert len(outputs) == len(prompts)
```

**VllmRunner useful methods:**
- `runner.generate_greedy(prompts, max_tokens)` → list of str
- `runner.generate(prompts, sampling_params=...)` → list of outputs
- `runner.generate_greedy_logprobs(prompts, max_tokens)` → with logprobs
- `runner.encode(prompts)` → embeddings

---

## 5. E2E Test: Multimodal (VLM)

```python
import os
import pytest
from tests.e2e.conftest import VllmRunner

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

VLM_MODELS = ["Qwen/Qwen2.5-VL-3B-Instruct"]


@pytest.mark.parametrize("model", VLM_MODELS)
def test_vlm_image_inference(model) -> None:
    from PIL import Image
    import requests
    from io import BytesIO

    image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/320px-Cat03.jpg"
    image = Image.open(BytesIO(requests.get(image_url).content))

    prompts = ["Describe this image in one sentence."]
    images = [image]

    with VllmRunner(model, max_model_len=1024, gpu_memory_utilization=0.8,
                    dtype="bfloat16") as runner:
        outputs = runner.generate_greedy(prompts, max_tokens=50, images=images)

    assert outputs is not None
```

---

## 6. conftest Patterns

### tests/ut/conftest.py — mock hardware at session scope

```python
# Mocks placed at module level so they apply to the whole session
import sys
from unittest.mock import MagicMock

# Mock Triton (not available on NPU)
triton_mock = MagicMock()
sys.modules["triton"] = triton_mock
sys.modules["triton.runtime"] = triton_mock.runtime

# Mock torch_npu
torch_npu_mock = MagicMock()
torch_npu_mock.npu_device_properties.return_value.num_aic = 8
torch_npu_mock.npu_device_properties.return_value.num_vectorcore = 8
sys.modules["torch_npu"] = torch_npu_mock
```

### tests/e2e/conftest.py — memory guard fixture

```python
# Use the wait_until_npu_memory_free decorator on memory-sensitive tests:
from tests.e2e.conftest import wait_until_npu_memory_free

@wait_until_npu_memory_free
def test_large_model_inference():
    ...
```

---

## File Header Template

Every test file must start with:

```python
#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
```
