"""MkDocs hooks that generate the site pages from the repository.

Generated (git-ignored) files under website/docs/:
  index.md                 home page, built from README_zh.md
  feature/*.md, model/*.md the tutorials from skills/vllm-{feature,model}-tutorial/outputs/
  feature/index.md, model/index.md  card list pages
  assets/images/workflow_zh.svg     copied from docs/workflow/
The nav is built from website/catalog.yml.
"""

import re
import shutil
from pathlib import Path

import yaml

SITE_DIR = Path(__file__).resolve().parent
REPO = SITE_DIR.parent
DOCS = SITE_DIR / "docs"
GITHUB = "https://github.com/shen-shanshan/vllm-dev-skills"

SECTIONS = {
    "feature": ("vLLM Feature Tutorial", REPO / "skills/vllm-feature-tutorial/outputs",
                "vLLM 特性与模块的代码走读文档，由 `vllm-feature-tutorial` Skill 生成。"),
    "model": ("vLLM Model Tutorial", REPO / "skills/vllm-model-tutorial/outputs",
              "vLLM 中具体模型的结构与实现解析，由 `vllm-model-tutorial` Skill 生成。"),
}
OTHER_GROUP = "其他"

# Raw HTML tags allowed through; any other <word...> outside code is escaped so
# that tokens such as <image> or <start> in prose are not swallowed as tags.
HTML_TAGS = ("a|abbr|b|blockquote|br|center|code|del|details|div|em|font|h[1-6]|hr|i|img|ins|kbd|"
             "li|mark|ol|p|pre|s|small|span|strong|sub|summary|sup|table|tbody|td|th|thead|tr|u|ul")
STRAY_TAG = re.compile(rf"<(?!/?(?:{HTML_TAGS})\b)(?!https?:)(?=/?[A-Za-z_])")
INLINE_CODE = re.compile(r"(`+)(.+?)\1")
UPDATED = re.compile(r"最后更新\**\s*[:：]\s*\**\s*(\d{4}-\d{2}-\d{2})")
COMMIT = re.compile(r"commit\s*`?([0-9a-f]{7,40})`?")


def _load_catalog():
    return yaml.safe_load((SITE_DIR / "catalog.yml").read_text(encoding="utf-8"))


def _write(path: Path, text: str):
    # Only touch files whose content changed, so `mkdocs serve` does not loop.
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists() or path.read_text(encoding="utf-8") != text:
        path.write_text(text, encoding="utf-8")


def _escape_stray_tags(md: str) -> str:
    out, fence = [], None
    for line in md.split("\n"):
        stripped = line.lstrip()
        m = re.match(r"(`{3,}|~{3,})", stripped)
        if m:
            if fence is None:
                fence = m.group(1)
            elif stripped.startswith(fence):
                fence = None
            out.append(line)
            continue
        if fence is not None:
            out.append(line)
            continue
        parts, last = [], 0
        for c in INLINE_CODE.finditer(line):
            parts.append(STRAY_TAG.sub("&lt;", line[last:c.start()]))
            parts.append(c.group(0))
            last = c.end()
        parts.append(STRAY_TAG.sub("&lt;", line[last:]))
        out.append("".join(parts))
    return "\n".join(out)


def _meta(md: str):
    head = "\n".join(md.split("\n")[:15])
    updated = UPDATED.search(head)
    commit = COMMIT.search(head)
    return (updated.group(1) if updated else None,
            commit.group(1)[:10] if commit else None)


def _auto_summary(md: str) -> str:
    for line in md.split("\n")[1:]:
        s = line.strip()
        if not s or s[0] in "#>|-`*<" or s[:2].rstrip(".").isdigit() or s.startswith("---"):
            continue
        s = re.sub(r"[*`]", "", s)
        return s[:80] + ("…" if len(s) > 80 else "")
    return ""


def _updated_html(updated, commit, cls="doc-updated", tag="div"):
    if not updated:
        return f'<{tag} class="{cls}" data-updated="">最后更新：未注明</{tag}>'
    extra = f" · 基于 vLLM 代码 <code>{commit}</code>" if commit else ""
    return f'<{tag} class="{cls}" data-updated="{updated}">最后更新：{updated}{extra}</{tag}>'


def _demote_extra_h1(lines):
    # Material's TOC only lists the children of the first H1, so documents with
    # several H1 parts ("# 第一部分") get every later heading pushed down a level.
    fence, h1 = None, 0
    marks = []
    for i, line in enumerate(lines):
        m = re.match(r"(`{3,}|~{3,})", line.lstrip())
        if m:
            fence = None if fence and line.lstrip().startswith(fence) else (fence or m.group(1))
            continue
        if fence is None and re.match(r"#{1,6} ", line):
            marks.append(i)
            h1 += line.startswith("# ")
    if h1 > 1:
        for i in marks[1:]:
            if not lines[i].startswith("###### "):
                lines[i] = "#" + lines[i]
    return lines


def _tutorial_page(md: str, updated, commit) -> str:
    lines = _demote_extra_h1(md.split("\n"))
    # Keep the metadata lines of the header blockquote on separate lines.
    for i in range(min(15, len(lines) - 1)):
        if lines[i].startswith("> ") and lines[i + 1].startswith(">") and not lines[i].endswith("  "):
            lines[i] = lines[i].rstrip() + "  "
    banner = _updated_html(updated, commit)
    for i, line in enumerate(lines):
        if line.startswith("# "):
            lines[i + 1:i + 1] = ["", banner, ""]
            break
    else:
        lines[0:0] = [banner, ""]
    return _escape_stray_tags("\n".join(lines))


def _build_section(key, catalog):
    title, src_dir, intro = SECTIONS[key]
    out_dir = DOCS / key
    sources = {p.name: p for p in sorted(src_dir.glob("*.md"))}
    groups = []  # (group, [(file, title, summary, updated)])
    listed = set()
    for g in catalog.get(key) or []:
        docs = []
        for d in g["docs"]:
            if d["file"] not in sources:
                print(f"WARNING: catalog.yml lists missing file {key}/{d['file']}")
                continue
            docs.append((d["file"], d["title"], d["summary"]))
            listed.add(d["file"])
        groups.append((g["group"], docs))
    extra = []
    for name, path in sources.items():
        if name not in listed:
            print(f"WARNING: {key}/{name} is not in website/catalog.yml; shown under {OTHER_GROUP}")
            md = path.read_text(encoding="utf-8")
            first = next((l[2:].strip() for l in md.split("\n") if l.startswith("# ")), name)
            extra.append((name, first, _auto_summary(md)))
    if extra:
        groups.append((OTHER_GROUP, extra))

    # Remove generated pages whose source is gone.
    if out_dir.exists():
        for old in out_dir.glob("*.md"):
            if old.name != "index.md" and old.name not in sources:
                old.unlink()

    nav, cards = [], [f"# {title}", "", intro, "",
                      "每篇文档开头都注明了最后更新时间。vLLM 的代码一直在变化，距今超过 3 个月的文档会额外标出提醒，阅读时请以最新代码为准。", ""]
    for group, docs in groups:
        cards += [f"## {group}", "", '<div class="grid cards" markdown>', ""]
        group_nav = []
        for file, card_title, summary in docs:
            md = sources[file].read_text(encoding="utf-8")
            updated, commit = _meta(md)
            _write(out_dir / file, _tutorial_page(md, updated, commit))
            group_nav.append({card_title: f"{key}/{file}"})
            cards += [f"-   **[{card_title}]({file})**", "", "    ---", "", f"    {summary}", "",
                      "    " + _updated_html(updated, None, "card-updated", "span"), ""]
        cards += ["</div>", ""]
        nav.append({group: group_nav})
    _write(out_dir / "index.md", "\n".join(cards))
    return {title: [f"{key}/index.md", *nav]}


def _rewrite_readme_link(m):
    text, target = m.group(1), m.group(2)
    if target.startswith("./docs/workflow/"):
        return f"[{text}](assets/images/{Path(target).name})"
    for key, (_, src_dir, _) in SECTIONS.items():
        rel = "./" + str(src_dir.relative_to(REPO)) + "/"
        if target.startswith(rel):
            return f"[{text}]({key}/{target[len(rel):]})"
    if target.startswith("./"):
        kind = "blob" if Path(target.split("#")[0]).suffix else "tree"
        return f"[{text}]({GITHUB}/{kind}/master/{target[2:]})"
    if target.startswith("#-skill"):
        return f"[{text}](#skills)"
    return m.group(0)


def _build_home():
    readme = (REPO / "README_zh.md").read_text(encoding="utf-8")
    shutil.copyfile(REPO / "docs/workflow/workflow_zh.svg", DOCS / "assets/images/workflow_zh.svg")
    sections = re.split(r"(?m)^(?=## )", readme)
    keep = {}
    for s in sections:
        head = s.split("\n", 1)[0]
        for name in ("项目简介", "Skill 列表", "AI-native 工作流", "安装配置"):
            if head.startswith("## ") and name in head:
                keep[name] = s.rstrip() + "\n"
    keep["Skill 列表"] = keep["Skill 列表"].replace("## 🚀 Skill 列表", "## 🚀 Skill 列表 { #skills }", 1)
    counts = {k: len(list(SECTIONS[k][1].glob("*.md"))) for k in SECTIONS}
    hero = f"""---
hide:
  - navigation
---

# vLLM Dev Skills

<div class="grid cards" markdown>

-   :material-book-open-page-variant: **[vLLM Feature Tutorial](feature/index.md)**

    ---

    {counts["feature"]} 篇 vLLM 特性与模块的代码走读文档。

-   :material-cube-outline: **[vLLM Model Tutorial](model/index.md)**

    ---

    {counts["model"]} 篇 vLLM 中具体模型的结构与实现解析。

-   :fontawesome-brands-github: **[GitHub 仓库]({GITHUB})**

    ---

    全部 Skill 的定义、示例输出和安装方式。

</div>
"""
    body = "\n".join(keep[n] for n in ("项目简介", "AI-native 工作流", "Skill 列表", "安装配置"))
    # Drop README lines that point readers to this very site.
    body = "\n".join(l for l in body.split("\n") if "shen-shanshan.github.io/vllm-dev-skills" not in l)
    body = re.sub(r"\[([^\]]*)\]\(([^)\s]+)\)", _rewrite_readme_link, body)
    _write(DOCS / "index.md", hero + "\n" + body)


def on_config(config):
    catalog = _load_catalog()
    (DOCS / "assets/images").mkdir(parents=True, exist_ok=True)
    _build_home()
    nav = [{"主页": "index.md"}]
    for key in SECTIONS:
        nav.append(_build_section(key, catalog))
    config["nav"] = nav
    return config
