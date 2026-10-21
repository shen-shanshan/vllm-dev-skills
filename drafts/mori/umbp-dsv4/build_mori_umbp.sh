#!/usr/bin/env bash
# Build mori with UMBP from /home/shashen/vllm/mori and install it into vLLM's
# .venv only. The system mori (amd_mori 1.2.3.post1, no UMBP) stays untouched;
# the .venv copy shadows it (include-system-site-packages = true).
# Roll back with:
#   uv pip uninstall --python /home/shashen/vllm/vllm/.venv/bin/python amd_mori
set -euo pipefail
VENV_PY=/home/shashen/vllm/vllm/.venv/bin/python
MORI_SRC=/home/shashen/vllm/mori

# 1. UMBP's control plane links gRPC/Protobuf (system packages, not pip).
apt-get update
apt-get install -y libgrpc-dev libgrpc++-dev libprotobuf-dev protobuf-compiler-grpc

# 2. Build + install. BUILD_UMBP is explicit so a missing dependency is reported
#    loudly instead of silently building mori without UMBP.
cd "${MORI_SRC}"
git submodule update --init --recursive
BUILD_UMBP=ON uv pip install --python "${VENV_PY}" --no-build-isolation -v . \
  2>&1 | tee /tmp/mori_umbp_build.log

# 3. Verify: UMBP bindings load and the standalone server binary exists.
"${VENV_PY}" - <<'PY'
import os, mori
from mori.cpp import UMBPClient, UMBPConfig, UMBPStandaloneProcessConfig  # noqa: F401
pkg = os.path.dirname(mori.__file__)
print("mori:", mori.__file__)
print("umbp_standalone_server:", os.path.exists(os.path.join(pkg, "umbp_standalone_server")))
PY
