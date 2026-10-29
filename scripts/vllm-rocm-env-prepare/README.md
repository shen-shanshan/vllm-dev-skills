# vllm-rocm-env-prepare skill 创建说明

该 skill 主要用于拉取镜像 `docker pull vllm/vllm-openai-rocm:nightly-rocm100` 并使用以下命令进入容器之后，容器内 vllm rocm 开发环境搭建的常用步骤，供 claude code 等 agent 按需调用（进入容器之前的步骤，比如拉取镜像等，不在此 skill 负责的范围内）。

```bash
docker run --rm -it \
    --name vllm-dev-sss \
    --group-add=video \
    --cap-add=SYS_PTRACE \
    --security-opt seccomp=unconfined \
    --device /dev/kfd \
    --device /dev/dri \
    --ipc=host \
    --network=host \
    --env HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    --env VLLM_ROCM_USE_AITER=1 \
    --env AITER_LOG_LEVEL=WARNING \
    --volume /home/shashen:/home/shashen \
    --volume /home/shashen/.local/share/claude:/root/.local/share/claude \
    --volume /home/shashen/.claude:/root/.claude \
    --volume /home/shashen/.claude.json:/root/.claude.json \
    --volume /mnt/hf_cache:/shared \
    --entrypoint /bin/bash \
    --shm-size 16G \
    --label spur_job_id=251945 \
    vllm/vllm-openai-rocm:nightly-rocm100
```

## 容器内常用目录说明

代码仓库：`git clone` 开发所需代码仓到 `/home/shashen/vllm/` 目录下（`/home/shashen` 为宿主机挂载进容器的目录，便于持久化）。若仓库已存在，则跳过该步骤。

```bash
cd /home/shashen/vllm/

git clone https://github.com/shen-shanshan/vllm.git
cd vllm
git remote add upstream https://github.com/vllm-project/vllm.git
cd ..

git clone https://github.com/shen-shanshan/vllm-dev-skills.git
# 告诉 agent 把 vllm-dev-skills/skills 中的 skill 软链接到 claude code 官方可识别的 skill 目录中。若 skill 已链接，则跳过该步骤。

git clone https://github.com/shen-shanshan/InferenceX.git
git clone https://github.com/shen-shanshan/ATOM.git
git clone https://github.com/shen-shanshan/mori.git
```

模型权重 & 数据集（Huggingface）：先找找 `/shared` 目录中有没有现成的。由于该目录是只读的，没有的话就下载到 `/home/shashen/vllm/` 下恰当的目录中。

## 初始化 config

执行 [init_config.sh](./init_config.sh) 脚本。执行之前先检查 `~/.gitconfig`，若 git 配置已存在，则跳过 git 配置的写入。

## 初始化 vllm

```bash
cd /home/shashen/vllm/vllm
# 切换并同步最新 main 分支：
git cb main
git sync

mkdir dev  # 目录已存在则跳过
cd dev
```

把 [init_vllm.sh](./init_vllm.sh) 拷贝到 `/home/shashen/vllm/vllm/dev` 目录下并执行，中途需要输入 `y` 确认。若脚本已存在，则跳过该步骤。

然后，启用虚拟环境并安装 pre-commit：

```bash
cd /home/shashen/vllm/vllm
source /home/shashen/vllm/vllm/.venv/bin/activate
uv pip install pre-commit
pre-commit install
```

## 安装带 UMBP 的 MORI 到当前虚拟环境中（Optional：按需调用）

参考 [umbp_install.md](./mori_umbp/umbp_install.md)。
