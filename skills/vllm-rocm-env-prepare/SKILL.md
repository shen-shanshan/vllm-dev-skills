---
name: vllm-rocm-env-prepare
description: Prepare a vLLM ROCm development environment inside an already-running `vllm/vllm-openai-rocm` nightly container (MI355X / gfx950). Clones the dev repos into /home/shashen/vllm (vllm, vllm-dev-skills, InferenceX, ATOM, mori), symlinks vllm-dev-skills skills into ~/.claude/skills, writes git/vim/node/claude config, syncs vllm main and builds a no-compile editable `.venv` that reuses the image's precompiled binaries, installs pre-commit, locates model weights/datasets, and optionally builds MORI with UMBP into the venv. Every step is skipped when already done. Out of scope — pulling the image or starting the container. Use when the user has just entered a fresh container and asks to set up / initialize the vllm rocm dev environment. Triggered by requests like "帮我初始化容器内的 vllm rocm 开发环境", "准备 vllm rocm 开发环境", "搭建 vllm 开发环境", "安装带 UMBP 的 mori", "prepare vllm rocm dev env", "/vllm-rocm-env-prepare".
---

# vLLM ROCm Env Prepare

Set up the in-container vLLM ROCm dev environment. The container is started from `vllm/vllm-openai-rocm:nightly-rocm100` with `/home/shashen` mounted (persistent), `/mnt/hf_cache` mounted read-only at `/shared`, and `~/.claude` mounted from the host. Pulling the image and `docker run` are NOT this skill's job.

`<skill_dir>` below is this skill's directory. Run the steps in order; skip any step whose result already exists, and tell the user which steps were skipped. Steps 1–4 are the default; step 5 (MORI + UMBP) runs only when the user asks for it.

## 1. Clone repos

All repos live under `/home/shashen/vllm/`. Skip each clone whose directory already exists.

```bash
cd /home/shashen/vllm/
git clone https://github.com/shen-shanshan/vllm.git
git -C vllm remote add upstream https://github.com/vllm-project/vllm.git
git clone https://github.com/shen-shanshan/vllm-dev-skills.git
git clone https://github.com/shen-shanshan/InferenceX.git
git clone https://github.com/shen-shanshan/ATOM.git
git clone https://github.com/shen-shanshan/mori.git
```

For an existing `vllm` checkout, make sure the `upstream` remote exists (`git -C vllm remote -v`); add it if missing.

Then symlink every skill in `vllm-dev-skills/skills/` into `~/.claude/skills/`, skipping links that already point to the right place:

```bash
mkdir -p ~/.claude/skills
for d in /home/shashen/vllm/vllm-dev-skills/skills/*/; do ln -sfn "${d%/}" ~/.claude/skills/; done
```

Newly linked skills show up after `/reload-plugins` or a Claude Code restart.

## 2. Init config

```bash
bash <skill_dir>/scripts/init_config.sh
```

The script is idempotent: it skips the git config if `~/.gitconfig` already has a `[user]` section, skips the vim line if present, and skips the node.js install if `node` is on PATH. It then links the newest `/root/.local/share/claude/versions/*` to `/root/.local/bin/claude`. The git aliases `cb` / `sync` / `nb` / `db` / `pr` it writes are used in later steps.

## 3. Init vLLM

Sync `main` with upstream. Requires a clean working tree; if `vllm` has uncommitted changes or is on another branch with work in progress, stop and ask the user first.

```bash
cd /home/shashen/vllm/vllm
git cb main
git sync
mkdir -p dev
```

Copy `<skill_dir>/scripts/init_vllm.sh` to `/home/shashen/vllm/vllm/dev/init_vllm.sh` (skip if it already exists there) and run it from a shell **without** `.venv` active. It must live in `vllm/dev/`, because it treats its parent directory as the vLLM root.

```bash
cp -n <skill_dir>/scripts/init_vllm.sh /home/shashen/vllm/vllm/dev/
bash /home/shashen/vllm/vllm/dev/init_vllm.sh -y
```

The script prompts `[y/N]` before rewriting branch history. The agent cannot answer that prompt, so pass `-y`, which is equivalent to answering `y`. What the script does:

- Deletes `./.venv`.
- Resets or rebases the current branch onto the commit the Docker vLLM was built from, after saving the old tip as branch `pre-align-backup`.
- Copies the precompiled `*.so` and `vllm-rs` into the tree.
- Creates `.venv` with `--system-site-packages` and installs vLLM editable with `VLLM_TARGET_DEVICE=empty`.
- Verifies the install.

Check the tail of its output for `dev env ready` and `native extensions : OK`.

Then install pre-commit:

```bash
cd /home/shashen/vllm/vllm
source .venv/bin/activate
uv pip install pre-commit
pre-commit install
```

## 4. Model weights & datasets

When a task needs HF model weights or datasets:

1. Look for them under `/shared` first (e.g. `/shared/huggingface/...`).
2. `/shared` is read-only. If they are not there, download into a suitable directory under `/home/shashen/vllm/`.

Report the path you used.

## 5. (Optional) Install MORI with UMBP into the vLLM venv

Run only when the user asks for it, e.g. for the UMBP KV connector. This step needs `/home/shashen/vllm/mori` and the `.venv` from step 3.

```bash
bash <skill_dir>/scripts/build_mori_umbp.sh   # env overrides: MORI_TAG, MORI_GPU_ARCHS, ROCM_PATH
```

The script:

1. apt-installs gRPC and Protobuf.
2. Adds a worktree `/home/shashen/vllm/mori-<tag>` at the MORI tag vLLM pins (default `v1.2.3.post1`, which matches `MORI_BRANCH` in `docker/Dockerfile.rocm_base`).
3. Builds with `BUILD_UMBP=ON` into `.venv` only.
4. Fails unless the build log shows `distributed control plane ENABLED`.
5. Verifies the `mori.cpp` UMBP symbols and the `umbp_standalone_server` binary.

Before running, check `MORI_BRANCH` in `vllm/docker/Dockerfile.rocm_base` and pass `MORI_TAG=...` if it differs from the default. The build takes about 2–3 minutes. Roll back with `uv pip uninstall --python /home/shashen/vllm/vllm/.venv/bin/python amd_mori`.

On build or import errors (NUMA not found, silent UMBP skip, `CacheRemoteAdmission` ImportError, `lookup_timeout_ms`, slow standalone server start), read [references/umbp_install.md](references/umbp_install.md).

## Final report

Summarize for the user:

- Which steps ran and which were skipped.
- The vLLM branch and HEAD after alignment, and whether a force-push is needed (if a rebase happened).
- The venv path and vLLM version.
- Any failures, with the relevant log lines.
