# Archify

## Installation

```bash
npx -y skills add tt-a1i/archify --skill archify --agent claude-code --global --copy --yes
```

## Usage

Open a repository and ask:

```text
Analyze this repository, then use archify to create a high-level runtime architecture diagram.
```

架构图：

```text
用 Archify 把下面这段自然语言系统描述画成高层架构图：[在这里描述用户、核心组件、主要路径、外部依赖和边界]。不需要代码库。只追问会实质影响图的缺失信息，其余不确定内容要标明而不是编造；保留 8–12 个核心组件和一条一眼可见的主路径。
```

工作流：

```text
用 Archify 工作流模式把下面的描述画成图：[粘贴参与者、主要步骤、决策、审批和异常路径]。不同负责方使用独立泳道，保留一条明确的成功主路径，缺失的负责人或未定分支要标明而不是编造。
```

时序图：

```text
用 Archify 时序模式绘制下面的交互：[粘贴参与者、调用、返回、回退和异步副作用]。确保消息顺序无歧义、标签简短，并明确标注未知行为。不需要代码库。
```
