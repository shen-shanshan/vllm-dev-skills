#!/usr/bin/env python3
"""Generate the AI-native workflow diagram (workflow_en.svg / workflow_zh.svg).

The diagram follows the hand-drawn draw.io style of
skills/vllm-technical-blog-writer/references/diagram-guide.md.

To add a skill: append it to the right stage in STAGES / MAINTAINER / EVOLVE
below (at most 3 nodes per row), then run `python3 docs/workflow/gen_workflow.py`.
Both language versions are regenerated together, and the Chinese one is
rendered to docs/blogs/images/ai-native-workflow.png when rsvg-convert exists.
"""

import shutil
import subprocess
from pathlib import Path
from xml.sax.saxutils import escape

OUT_DIR = Path(__file__).resolve().parent
# PNG copy for the project blog posts (Zhihu does not accept SVG)
BLOG_PNG = OUT_DIR.parent / "blogs" / "images" / "ai-native-workflow.png"

# (skill name, English role, Chinese role, AMD/ROCm-specific)
Skill = tuple[str, str, str, bool]

# (English stage label, Chinese stage label, skills)
STAGES: list[tuple[str, str, list[Skill]]] = [
    ("1 Discover", "1 发现问题", [
        ("vllm-dev-task-discovery", "open tasks & leftover issues", "社区公开任务与遗留问题", False),
        ("vllm-multimodal-open-issue-analyzer", "triage multimodal issues", "整理多模态 open issue", False),
        ("vllm-profile-trace-comparison", "kernel-level trace diff", "逐 kernel 对比 trace", False),
    ]),
    ("2 Learn", "2 学习原理", [
        ("vllm-feature-tutorial", "how a feature works", "学习特性原理", False),
        ("vllm-model-tutorial", "model architecture", "学习模型结构", False),
    ]),
    ("3 Design", "3 设计方案", [
        ("vllm-feature-design", "design doc + core code", "设计文档 + 核心代码", False),
        ("vllm-rfc-generator", "RFC for large features", "大特性提 RFC", False),
    ]),
    ("4 Develop", "4 开发代码", [
        ("vllm-rocm-env-prepare", "one-shot dev env setup", "一键准备开发环境", True),
        ("vllm-rocm-code-dev", "ROCm-only code changes", "只改 ROCm 路径的代码", True),
    ]),
    ("5 Test", "5 测试验证", [
        ("vllm-test-generator", "unit tests for the change", "为改动生成 UT", False),
        ("infx-bench-script-generator", "InferenceX bench scripts", "生成 InferenceX 测试脚本", True),
        ("vllm-benchmark-result-analysis", "before / after comparison", "对比改动前后的结果", False),
    ]),
    ("6 Submit", "6 提交代码", [
        ("vllm-pr-desc-generator", "concise PR description", "精简的 PR 描述", False),
    ]),
    ("7 Share", "7 撰写博客", [
        ("vllm-technical-blog-writer", "blog in your own style", "按个人风格写技术博客", False),
    ]),
]

MAINTAINER: tuple[str, str, list[Skill]] = ("Review", "审核 PR", [
    ("vllm-rocm-pr-review", "review ROCm / AMD PRs", "审核 ROCm / AMD PR", True),
    ("vllm-vit-cg-pr-review", "review ViT CUDA graph PRs", "审核 ViT CUDA Graph PR", False),
    ("vllm-pr-summary", "understand a PR quickly", "快速了解 PR 改动", False),
])

# Evolve row: note -> skill-creator -> note
EVOLVE = ("Evolve", "持续进化",
          ("repeated manual step", "重复的手动操作", "(3+ times)", "（3 次及以上）"),
          ("skill-creator", "turn it into a skill", "把它做成 skill", False),
          ("new skill", "新 skill", "joins this workflow", "加入工作流"))

TEXT = {
    "en": {
        "title": "AI-native vLLM Development Workflow",
        "dev": "As a Developer",
        "maint": "As a vLLM Maintainer",
        "evolve": "Grow the Workflow",
        "next": "next task",
        "auto": "auto-invoked",
        "lg_skill": "general skill",
        "lg_rocm": "AMD / ROCm skill",
        "lg_order": "workflow order",
        "lg_loop": "next iteration",
        "lg_auto": "agent calls it for you",
    },
    "zh": {
        "title": "AI-native vLLM 开发工作流",
        "dev": "作为开发者",
        "maint": "作为 vLLM Maintainer",
        "evolve": "让工作流持续进化",
        "next": "下一个任务",
        "auto": "自动调用",
        "lg_skill": "通用 skill",
        "lg_rocm": "AMD / ROCm skill",
        "lg_order": "工作流顺序",
        "lg_loop": "进入下一轮",
        "lg_auto": "由 agent 自动调用",
    },
}

# Layout (10px grid)
W = 1100
SEC_X, SEC_W = 20, 1060  # dashed section panels
LOOP_X = 40  # "next task" return line
ROW_X, ROW_W, ROW_H, ROW_STEP = 80, 980, 70, 80
LABEL_X, LABEL_W = 100, 100
NODE_X0, NODE_W, NODE_GAP, NODE_H = 220, 260, 20, 50
SEC_PAD_TOP = 40  # from section top to first row

STYLE = """
    text { font-family: "Comic Sans MS", "Comic Sans", "Microsoft YaHei", "PingFang SC", "Noto Sans CJK SC", cursive; font-size: 12px; fill: #000000; dominant-baseline: central; }
    .c { text-anchor: middle; }
    .w { fill: #ffffff; }
    .b { font-weight: bold; }
    .hl { fill: #cc0000; }
    .sys-title { font-size: 20px; font-weight: bold; font-style: italic; text-anchor: middle; }
    rect, path { stroke-width: 1; }
    .n-key { fill: #a20025; stroke: #6f0000; }
    .n-sys { fill: #76608a; stroke: #432d57; }
    .n-skill { fill: #dae8fc; stroke: #6c8ebf; }
    .n-rocm { fill: #d5e8d4; stroke: #82b366; }
    .n-note { fill: #f9f7ed; stroke: #36393d; }
    .ct-row { fill: #f5f5f5; stroke: #666666; }
    .ct-sec { fill: none; stroke: #000000; stroke-dasharray: 8 4; }
    .e-red { fill: none; stroke: #cc0000; marker-end: url(#arrow-red); }
    .e-blue { fill: none; stroke: #0066cc; marker-end: url(#arrow-blue); }
    .e-black { fill: none; stroke: #000000; marker-end: url(#arrow-black); }
    .e-auto { fill: none; stroke: #000000; stroke-dasharray: 4 3; marker-end: url(#arrow-black); }
"""

DEFS = """
    <marker id="arrow-black" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 L3,5z" fill="#000000"/></marker>
    <marker id="arrow-red" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 L3,5z" fill="#cc0000"/></marker>
    <marker id="arrow-blue" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 L3,5z" fill="#0066cc"/></marker>
"""


class Svg:
    def __init__(self) -> None:
        self.parts: list[str] = []

    def rect(self, x, y, w, h, cls, rx=0):
        r = f' rx="{rx}"' if rx else ""
        self.parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" class="{cls}"{r}/>')

    def text(self, x, y, s, cls="c"):
        self.parts.append(f'<text x="{x}" y="{y}" class="{cls}">{escape(s)}</text>')

    def path(self, d, cls):
        self.parts.append(f'<path d="{d}" class="{cls}"/>')


def node(svg, x, y, line1, line2, cls, white=False):
    svg.rect(x, y, NODE_W, NODE_H, cls, rx=6)
    tcls = "c w" if white else "c"
    cy = y + NODE_H // 2
    svg.text(x + NODE_W // 2, cy - 9, line1, tcls)
    svg.text(x + NODE_W // 2, cy + 9, line2, tcls)


def node_x(i):
    return NODE_X0 + i * (NODE_W + NODE_GAP)


def row(svg, y, label, label_cls):
    svg.rect(ROW_X, y, ROW_W, ROW_H, "ct-row")
    svg.rect(LABEL_X, y + 10, LABEL_W, NODE_H, label_cls)
    svg.text(LABEL_X + LABEL_W // 2, y + 10 + NODE_H // 2, label, "c w")


def skill_nodes(svg, y, skills, lang):
    centers = {}
    for i, (name, en, zh, rocm) in enumerate(skills):
        x = node_x(i)
        node(svg, x, y + 10, name, en if lang == "en" else zh,
             "n-rocm" if rocm else "n-skill")
        centers[name] = (x, y + 10)
    return centers


def build(lang: str) -> str:
    t = TEXT[lang]
    svg = Svg()
    li = 0 if lang == "en" else 1

    # Title
    svg.text(W // 2, 40, t["title"], "sys-title")

    # Developer section
    dev_top = 70
    first_row = dev_top + SEC_PAD_TOP
    n = len(STAGES)
    dev_h = SEC_PAD_TOP + n * ROW_STEP - (ROW_STEP - ROW_H) + 20
    svg.rect(SEC_X, dev_top, SEC_W, dev_h, "ct-sec", rx=20)
    svg.text(SEC_X + 16, dev_top + 16, t["dev"], "b")

    pos = {}
    for k, stage in enumerate(STAGES):
        y = first_row + k * ROW_STEP
        row(svg, y, stage[li], "n-key")
        pos.update(skill_nodes(svg, y, stage[2], lang))
        if k:  # red order arrow between consecutive stage labels
            svg.path(f"M{LABEL_X + LABEL_W // 2},{y - ROW_STEP + 10 + NODE_H} V{y + 10}", "e-red")

    # Blue return path: last stage -> first stage
    ly = first_row + (n - 1) * ROW_STEP + 10 + NODE_H // 2
    fy = first_row + 10 + NODE_H // 2
    svg.path(f"M{LABEL_X},{ly} H{LOOP_X} V{fy} H{LABEL_X}", "e-blue")
    mid = (ly + fy) // 2
    svg.parts.append(
        f'<text x="{LOOP_X + 20}" y="{mid}" class="c" transform="rotate(-90 {LOOP_X + 20} {mid})" '
        f'style="fill:#0066cc">{escape(t["next"])}</text>')

    # Dashed "auto-invoked" edge: vllm-rocm-code-dev -> vllm-pr-desc-generator
    sx, sy = pos["vllm-rocm-code-dev"]
    dx, dy = pos["vllm-pr-desc-generator"]
    gutter = sx + NODE_W + NODE_GAP // 2  # gap between node columns 2 and 3
    s_y, e_y = sy + NODE_H // 2, dy + NODE_H // 2
    svg.path(f"M{sx + NODE_W},{s_y} H{gutter} V{e_y} H{dx + NODE_W}", "e-auto")
    svg.text((dx + NODE_W + gutter) // 2, e_y - 12, t["auto"])

    # Maintainer section
    m_top = dev_top + dev_h + 20
    svg.rect(SEC_X, m_top, SEC_W, SEC_PAD_TOP + ROW_H + 20, "ct-sec", rx=20)
    svg.text(SEC_X + 16, m_top + 16, t["maint"], "b")
    row(svg, m_top + SEC_PAD_TOP, MAINTAINER[li], "n-sys")
    skill_nodes(svg, m_top + SEC_PAD_TOP, MAINTAINER[2], lang)

    # Evolve section
    e_top = m_top + SEC_PAD_TOP + ROW_H + 40
    svg.rect(SEC_X, e_top, SEC_W, SEC_PAD_TOP + ROW_H + 20, "ct-sec", rx=20)
    svg.text(SEC_X + 16, e_top + 16, t["evolve"], "b")
    ey = e_top + SEC_PAD_TOP
    row(svg, ey, EVOLVE[li], "n-sys")
    pre, sc, post = EVOLVE[2], EVOLVE[3], EVOLVE[4]
    node(svg, node_x(0), ey + 10, pre[li], pre[2 + li], "n-note")
    node(svg, node_x(1), ey + 10, sc[0], sc[1 + li], "n-skill")
    node(svg, node_x(2), ey + 10, post[li], post[2 + li], "n-note")
    cy = ey + 10 + NODE_H // 2
    for i in (0, 1):
        svg.path(f"M{node_x(i) + NODE_W},{cy} H{node_x(i + 1)}", "e-black")

    # Legend
    lg_y = e_top + SEC_PAD_TOP + ROW_H + 40
    x = SEC_X + 20
    for cls, key in (("n-skill", "lg_skill"), ("n-rocm", "lg_rocm")):
        svg.rect(x, lg_y, 30, 20, cls, rx=6)
        svg.text(x + 40, lg_y + 10, t[key], "")
        x += 200
    for cls, key in (("e-red", "lg_order"), ("e-blue", "lg_loop"), ("e-auto", "lg_auto")):
        svg.path(f"M{x},{lg_y + 10} H{x + 40}", cls)
        svg.text(x + 50, lg_y + 10, t[key], "")
        x += 200 if key != "lg_auto" else 0

    h = lg_y + 40
    body = "\n  ".join(svg.parts)
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        "<!-- Generated by docs/workflow/gen_workflow.py; edit the script, not this file. -->\n"
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{h}" viewBox="0 0 {W} {h}">\n'
        f"  <defs>{DEFS}  </defs>\n  <style>{STYLE}  </style>\n"
        f'  <rect x="0" y="0" width="{W}" height="{h}" fill="#ffffff"/>\n  {body}\n</svg>\n'
    )


if __name__ == "__main__":
    for lang in ("en", "zh"):
        out = OUT_DIR / f"workflow_{lang}.svg"
        out.write_text(build(lang), encoding="utf-8")
        print(f"wrote {out}")
    if shutil.which("rsvg-convert"):
        subprocess.run(["rsvg-convert", "-z", "2", "-o", str(BLOG_PNG),
                        str(OUT_DIR / "workflow_zh.svg")], check=True)
        print(f"wrote {BLOG_PNG}")
    else:
        print("rsvg-convert not found; blog PNG not updated")
