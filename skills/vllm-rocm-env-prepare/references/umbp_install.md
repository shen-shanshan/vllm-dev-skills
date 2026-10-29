# 在 vLLM 环境中安装带 UMBP 的 MORI

本文记录如何在 ROCm 容器（8× MI355X，gfx950）里，把带 UMBP 的 MORI 安装进 vLLM 的 `.venv`。
2026-10-27 按此流程安装成功并通过了验证。

## 背景

- 容器预装的是 `amd_mori 1.2.3.post1`，位置在 `/usr/local/lib/python3.12/dist-packages/mori`。这个 wheel **编译时没有开 UMBP**，所以：
  - `from mori.cpp import UMBPClient` 会报 `ImportError`；
  - 包里没有 `umbp_standalone_server` 这个二进制。
- vLLM 的 UMBP connector（`vllm/distributed/kv_transfer/kv_connector/v1/umbp/`）依赖 `mori.cpp` 中的以下符号：
  `UMBPClient`、`UMBPConfig`、`UMBPDeploymentMode`、`UMBPStandaloneProcessConfig`、`MemoryLocationType`。
- 要用 UMBP（embedded 或 standalone 模式都一样），就得**从源码编译 MORI，并打开 `BUILD_UMBP=ON`**。

## 版本选择：与 vLLM 固定的 MORI 版本保持一致

vLLM 在 `docker/Dockerfile.rocm_base` 里固定了 MORI 的版本：

```dockerfile
ARG MORI_BRANCH="v1.2.3.post1"
ARG MORI_REPO="https://github.com/ROCm/mori.git"
```

编译时应该用这个 tag，不要用 MORI main。`v1.2.3.post1` 对应 commit `67632e8`，这个版本已经包含 `src/umbp`、`python/mori/umbp`，vLLM 需要的 5 个符号也都在里面。

换版本前，可以先确认一下 vLLM 用到的符号在目标版本里是否存在：

```bash
cd /home/shashen/vllm/mori
for s in UMBPClient UMBPConfig UMBPDeploymentMode UMBPStandaloneProcessConfig MemoryLocationType; do
  echo "$s: $(git grep -l "\"$s\"" <TAG> -- src python | wc -l)"
done
```

## 安装步骤

### 1. 安装 gRPC 和 Protobuf 系统包

UMBP 的控制面（master、standalone server、peer）会链接 gRPC 和 Protobuf。这两样只能通过 apt 安装，pip 装不了：

```bash
apt-get update
apt-get install -y libgrpc-dev libgrpc++-dev libprotobuf-dev protobuf-compiler-grpc
```

Ubuntu 22.04 上装到的版本是 gRPC 1.30.2 和 protoc 3.12.4。

### 2. 用 worktree 拉出固定 tag

这一步不会影响 `mori/` 里现有的 main checkout：

```bash
cd /home/shashen/vllm/mori
git fetch https://github.com/ROCm/mori.git refs/tags/v1.2.3.post1:refs/tags/v1.2.3.post1
git worktree add /home/shashen/vllm/mori-v1.2.3.post1 v1.2.3.post1
cd /home/shashen/vllm/mori-v1.2.3.post1
git submodule update --init --recursive
```

### 3. 编译并安装到 vLLM 的 `.venv`

```bash
cd /home/shashen/vllm/mori-v1.2.3.post1
R=${ROCM_PATH:-/usr/local/lib/python3.12/dist-packages/_rocm_sdk_devel}
rm -rf build
CMAKE_PREFIX_PATH="$R:$R/lib/rocm_sysdeps" \
BUILD_UMBP=ON \
MORI_GPU_ARCHS=gfx950 \
uv pip install --python /home/shashen/vllm/vllm/.venv/bin/python \
  --no-build-isolation --no-deps -v . 2>&1 | tee /tmp/mori_umbp_build.log
```

在 MI355X 上编译大约需要 2–3 分钟，共 157 个 ninja target。

各参数的作用：

| 参数 | 作用 |
|---|---|
| `CMAKE_PREFIX_PATH=...:rocm_sysdeps` | 让 CMake 能找到 ROCm SDK 自带的 `NUMA` 等 sysdeps。写法与 Dockerfile 一致（见踩坑 1） |
| `BUILD_UMBP=ON` | 显式要求编译 UMBP。缺依赖时会直接报出来，而不是悄悄跳过（见踩坑 2） |
| `MORI_GPU_ARCHS=gfx950` | 只编译 MI355X 的架构，速度更快。这个变量优先级最高，会覆盖本地探测和 `PYTORCH_ROCM_ARCH` |
| `--python .venv/bin/python` | 只装进 vLLM 的 venv，系统里的 `amd_mori` 保持不动 |
| `--no-build-isolation` | 编译时要用到环境里已有的 torch 和 ROCm SDK |
| `--no-deps` | 不让 uv 改动 venv 里其他包的版本 |

### 4. 验证

```bash
cd /tmp   # 别在 mori 源码目录里跑，否则 import 到的是源码树里的 mori
/home/shashen/vllm/vllm/.venv/bin/python - <<'PY'
import os, mori
from mori.cpp import (UMBPClient, UMBPConfig, UMBPDeploymentMode,
                      UMBPStandaloneProcessConfig, MemoryLocationType)
import mori.umbp
pkg = os.path.dirname(mori.__file__)
print("mori:", mori.__file__, mori.__version__)
print([f for f in os.listdir(pkg) if "umbp" in f.lower()])
PY
```

预期输出如下：

- `mori` 的路径在 `.venv/lib/python3.12/site-packages/mori/` 下，版本是 `1.2.3.post1`；
- 包目录里有 `umbp`、`umbp_master`、`umbp_standalone_server`。

再检查 server 二进制和 vLLM 单测：

```bash
P=/home/shashen/vllm/vllm/.venv/lib/python3.12/site-packages/mori
ldd $P/umbp_standalone_server | grep "not found"   # 期望无输出

cd /home/shashen/vllm/vllm
.venv/bin/python -m pytest -q tests/v1/kv_connector/unit/test_umbp_*.py \
  tests/v1/core/test_prefix_replay.py
```

在 `rocm-dsv4-umbp` 分支上，结果是 302 passed。

编译日志里应该能看到下面这几行：

```
-- [UMBP] gRPC + Protobuf found — distributed control plane ENABLED
-- [UMBP] hipfile headers found — GDS transfer engine ENABLED (libhipfile resolved at runtime via dlopen)
```

### 回滚

```bash
uv pip uninstall --python /home/shashen/vllm/vllm/.venv/bin/python amd_mori
```

卸载之后，venv 会回落到系统里的 `amd_mori 1.2.3.post1`（不带 UMBP）。

## 容易踩坑的地方

### 1. CMake 报找不到 `NUMA`

```
CMake Error at .../CMakeFindDependencyMacro.cmake:76 (find_package):
  Could not find a package configuration file provided by "NUMA" ...
Call Stack: .../_rocm_sdk_devel/lib/cmake/hsakmt/hsakmt-config.cmake:34 (find_dependency)
            src/application/CMakeLists.txt:37 (find_package)
```

**原因**：这个容器里的 ROCm 是 pip 装的 TheRock SDK（`_rocm_sdk_devel`），没有 `/opt/rocm`。
`hsakmt-config.cmake` 会 `find_dependency(NUMA)`，而对应的 `numa-config.cmake` 在 `_rocm_sdk_devel/lib/rocm_sysdeps/lib/cmake/NUMA/` 下，CMake 默认不会去这个目录找。

**解决**：按 vLLM Dockerfile 的写法设置 `CMAKE_PREFIX_PATH="$ROCM_PATH:$ROCM_PATH/lib/rocm_sysdeps"`。
改完重新编译前要先 `rm -rf build`，否则 CMake 缓存里还是失败时的配置。

### 2. 不设 `BUILD_UMBP=ON` 时，缺依赖会被悄悄跳过

MORI 的 `setup.py` 检查到缺少 gRPC/Protobuf 时，会**打印一段警告，然后关掉 UMBP 继续编译**，最后装上的仍是一个不带 UMBP 的 mori。
如果 `pip install` 的输出没有看全，很容易误以为已经装好了。

所以一定要：

- 显式设置 `BUILD_UMBP=ON`；
- 装完后用第 4 步的 import 检查和 `umbp_standalone_server` 是否存在来确认，不要只看 install 是否成功。

### 3. 不要用 MORI main 编译

MORI main 和 vLLM CI 镜像使用的 MORI 版本不一致，API 或行为可能有差异，出问题时很难判断是 vLLM 的问题还是 MORI 的问题。
所以要 checkout 与 `docker/Dockerfile.rocm_base` 中 `MORI_BRANCH` 一致的 tag。

补充一个容易误判的现象：在不带 UMBP 的系统 mori 上执行 `import mori.umbp`，会报
`cannot import name 'CacheRemoteAdmission' from 'mori.cpp'`。
这是因为 Python 层引用了 UMBP 的 C++ 绑定，而这份 mori 编译时没有开 UMBP。这说明没开 UMBP，不是版本不匹配。

### 4. 本地 fork 里没有上游 tag

`/home/shashen/vllm/mori` 的 `origin` 指向个人 fork（`shen-shanshan/mori`），里面没有 tag。
需要显式从 `https://github.com/ROCm/mori.git` 拉 tag，见第 2 步。

### 5. 装到 `.venv`，不要改系统里的 mori

- vLLM 的 `.venv` 开了 `include-system-site-packages = true`，装进 venv 的 mori 会覆盖系统那份，回滚只要 uninstall 一下。
- 不要用系统 `pip install` 覆盖 `/usr/local/.../dist-packages/mori`，同一容器里的其他用户和进程可能还在用它。
- 安装后用 `.venv/bin/python -c "import mori; print(mori.__file__)"` 确认 import 到的是 venv 里的版本。

### 6. 验证时注意当前目录

如果在 mori 源码目录（或其上级目录）下运行 `python -c "import mori"`，可能 import 到源码树里的 `python/mori`，而不是安装好的包。这个副本没有编译出来的 `.so`，就会出现莫名其妙的 ImportError。
验证时请先 `cd /tmp`。

### 7. 旧的 protoc 版本

apt 装的 protoc 是 3.12，低于 3.15，默认不支持 proto3 `optional`。
MORI 的 CMake 会检测到这一点，并自动加上 `--experimental_allow_proto3_optional`，日志里会打印：

```
-- [UMBP] protoc 3.12 < 3.15 — enabling --experimental_allow_proto3_optional
```

这是正常现象，不需要另外装新版 protoc。

### 8. `umbp_standalone_server` 不支持 `--help`

这个二进制把第一个参数直接当作 gRPC 监听地址。所以执行 `umbp_standalone_server --help` 会报
`failed to start gRPC server on --help`，这不代表二进制坏了。
正确用法是 `umbp_standalone_server <host:port>`，完整的启动参数可以参考 `umbp-dsv4/run_umbp_server.sh`。

### 9. 一键脚本

本 skill 的 `scripts/build_mori_umbp.sh` 已按本文的流程实现，会依次完成 apt 依赖安装、拉取固定 tag 的 worktree、编译安装和验证。
脚本会检查编译日志里有没有 `distributed control plane ENABLED`，没有就报错退出，避免装上一个没开 UMBP 的版本。

可以通过环境变量覆盖默认值，例如 `MORI_TAG=v1.2.4 MORI_GPU_ARCHS=gfx942 bash build_mori_umbp.sh`。
如果 vLLM 升级了 `MORI_BRANCH`，记得同步修改脚本里 `MORI_TAG` 的默认值。

### 10. 报错 `Standalone UMBP requires MORI with lookup_timeout_ms support`

上游 `[UMBP][7/10]` 的 standalone runtime 要求 `UMBPStandaloneProcessConfig.lookup_timeout_ms`，但 MORI `v1.2.3.post1` 和 ROCm/mori main 都没有这个字段，vLLM 启动时会直接报这个错。

`rocm-dsv4-umbp` 分支已经做了兼容：MORI 没有这个字段、而且没有显式配置 `lookup_timeout_ms` 时，只打印一条
`MORI has no standalone lookup_timeout_ms; UMBP lookups are unbounded` 警告，然后照常启动。
所以不要在 `kv_connector_extra_config` 里配置 `lookup_timeout_ms`，配了仍然会报错。

### 11. `UMBP_DRAM_CAPACITY` 不要设得过大

`umbp_standalone_server` 会先把整个 DRAM 池 fault in，然后才创建 gRPC socket。
设成 1 TiB（不开 hugepages）时，等了 10 分钟以上 `standalone.grpc.sock` 还没出现；设成 128 GiB 时大约 1 分钟就好了。

- 判断是否卡住：`grep VmRSS /proc/<pid>/status` 会一直涨，主线程 `wchan` 是 `do_user_addr_fault`，这说明它还在 fault in，不是挂死。
- 按 workload 设置池大小即可。GSM8K 这类精度测试用 128 GiB 就足够；跑大池时考虑打开 `UMBP_DRAM_USE_HUGEPAGES=1`。
- 等 `ls /tmp/umbp/standalone.grpc.sock` 出现后再启动 vLLM。

DSV4-Pro + DSpark MTP 的完整精度测试脚本和结果在 `umbp-dsv4-mtp/` 下：
- `run_vllm_dsv4_mtp.sh`：按 InferenceX recipe 的 `override_dep8_c128` 启动服务，`UMBP=0/1` 切换。
- `run_gsm8k.sh`：lm_eval 装在独立的 `/home/shashen/vllm/lm_eval_venv` 中，GSM8K 数据集缓存在 `/shared/huggingface/datasets`。
- `results/`：GSM8K 结果和 metrics。
