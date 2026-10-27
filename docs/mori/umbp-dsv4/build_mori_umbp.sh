#!/usr/bin/env bash
# Build mori with UMBP at the version vLLM pins (docker/Dockerfile.rocm_base
# MORI_BRANCH) and install it into vLLM's .venv only. The system mori
# (amd_mori 1.2.3.post1, no UMBP) stays untouched; the .venv copy shadows it
# (include-system-site-packages = true). See ../umbp-install.md for pitfalls.
# Roll back with:
#   uv pip uninstall --python /home/shashen/vllm/vllm/.venv/bin/python amd_mori
set -euo pipefail
VENV_PY=/home/shashen/vllm/vllm/.venv/bin/python
MORI_REPO_DIR=/home/shashen/vllm/mori          # personal fork checkout (main), left as-is
MORI_TAG=${MORI_TAG:-v1.2.3.post1}             # keep in sync with vLLM's MORI_BRANCH
MORI_SRC=/home/shashen/vllm/mori-${MORI_TAG}   # worktree at the pinned tag
GPU_ARCHS=${MORI_GPU_ARCHS:-gfx950}            # MI355X
ROCM_SDK=${ROCM_PATH:-/usr/local/lib/python3.12/dist-packages/_rocm_sdk_devel}

# 1. UMBP's control plane links gRPC/Protobuf (system packages, not pip).
apt-get update
apt-get install -y libgrpc-dev libgrpc++-dev libprotobuf-dev protobuf-compiler-grpc

# 2. Check out the pinned tag in a separate worktree. The fork has no upstream
#    tags, so fetch it from ROCm/mori explicitly.
if [ ! -d "${MORI_SRC}" ]; then
  git -C "${MORI_REPO_DIR}" fetch https://github.com/ROCm/mori.git \
    "refs/tags/${MORI_TAG}:refs/tags/${MORI_TAG}"
  git -C "${MORI_REPO_DIR}" worktree add "${MORI_SRC}" "${MORI_TAG}"
fi
cd "${MORI_SRC}"
git submodule update --init --recursive

# 3. Build + install.
#    - CMAKE_PREFIX_PATH: the pip ROCm SDK keeps NUMA's cmake config under
#      rocm_sysdeps (needed by hsakmt-config.cmake); same as vLLM's Dockerfile.
#    - BUILD_UMBP=ON: report a missing dependency loudly instead of silently
#      building mori without UMBP.
#    - rm -rf build: drop a stale CMake cache from a previous failed configure.
rm -rf build
CMAKE_PREFIX_PATH="${ROCM_SDK}:${ROCM_SDK}/lib/rocm_sysdeps" \
BUILD_UMBP=ON MORI_GPU_ARCHS="${GPU_ARCHS}" \
  uv pip install --python "${VENV_PY}" --no-build-isolation --no-deps -v . \
  2>&1 | tee /tmp/mori_umbp_build.log
grep -q "distributed control plane ENABLED" /tmp/mori_umbp_build.log \
  || { echo "UMBP control plane not enabled, see /tmp/mori_umbp_build.log" >&2; exit 1; }

# 4. Verify (from /tmp so the source tree's python/mori is not imported):
#    UMBP bindings load and the standalone server binary exists.
cd /tmp
"${VENV_PY}" - <<'PY'
import os, mori
from mori.cpp import (UMBPClient, UMBPConfig, UMBPDeploymentMode,  # noqa: F401
                      UMBPStandaloneProcessConfig, MemoryLocationType)
pkg = os.path.dirname(mori.__file__)
print("mori:", mori.__file__, mori.__version__)
server = os.path.join(pkg, "umbp_standalone_server")
assert os.path.exists(server), "umbp_standalone_server missing"
print("umbp_standalone_server:", server)
PY
