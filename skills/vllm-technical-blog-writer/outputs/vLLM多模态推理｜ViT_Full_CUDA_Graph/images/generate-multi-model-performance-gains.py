import csv
import xml.etree.ElementTree as ET
from collections import OrderedDict
from pathlib import Path

HERE = Path(__file__).resolve().parent
CSV_PATH = HERE / "multi-model-performance-gains.csv"
SVG_PATH = HERE / "multi-model-performance-gains.svg"

WIDTH = 1600
HEADER_HEIGHT = 150
ROW_HEIGHT = 40
GROUP_GAP = 14
FOOTER_HEIGHT = 92
ZERO_X = 880
PLOT_RIGHT = 1520
SCALE = (PLOT_RIGHT - ZERO_X) / 90
SVG_NS = "http://www.w3.org/2000/svg"
REQUIRED_COLUMNS = {
    "group", "model", "hardware_scenario", "metric", "metric_type",
    "before", "after", "unit", "relative_gain_percent", "semantic",
    "comparison", "source_url", "note",
}

ET.register_namespace("", SVG_NS)


def tag(name):
    return f"{{{SVG_NS}}}{name}"


def add(parent, name, attrs=None, text=None):
    element = ET.SubElement(parent, tag(name), attrs or {})
    if text is not None:
        element.text = text
    return element


def compact_label(row):
    scenario = row["hardware_scenario"].replace("1x", "1×").replace("2x", "2×").replace("4x", "4×")
    metric = row["metric"].replace(" Encoder Forward", "")
    return f"{scenario} · {metric}"


def validate(rows, fieldnames):
    missing = REQUIRED_COLUMNS - set(fieldnames or [])
    if missing:
        raise ValueError(f"CSV 缺少字段: {', '.join(sorted(missing))}")
    for line_number, row in enumerate(rows, start=2):
        if not row["source_url"] or not row["comparison"]:
            raise ValueError(f"CSV 第 {line_number} 行缺少 source_url 或 comparison")
        gain = float(row["relative_gain_percent"])
        if row["semantic"] not in {"收益", "回退"}:
            raise ValueError(f"CSV 第 {line_number} 行 semantic 无效")
        if (gain >= 0) != (row["semantic"] == "收益"):
            raise ValueError(f"CSV 第 {line_number} 行百分比符号与 semantic 不一致")
        if row["metric_type"] in {"latency", "throughput"}:
            before, after = float(row["before"]), float(row["after"])
            calculated = (
                (before - after) / before if row["metric_type"] == "latency"
                else (after - before) / before
            ) * 100
            if abs(calculated - gain) > 0.2:
                raise ValueError(
                    f"CSV 第 {line_number} 行百分比不匹配: {gain} vs {calculated:.2f}"
                )


with CSV_PATH.open(encoding="utf-8-sig", newline="") as file:
    reader = csv.DictReader(file)
    rows = list(reader)
    validate(rows, reader.fieldnames)

groups = OrderedDict()
for row in rows:
    groups.setdefault(row["group"], []).append(row)

plot_bottom = HEADER_HEIGHT + len(rows) * ROW_HEIGHT + (len(groups) - 1) * GROUP_GAP
height = plot_bottom + FOOTER_HEIGHT
svg = ET.Element(
    tag("svg"),
    {
        "width": str(WIDTH),
        "height": str(height),
        "viewBox": f"0 0 {WIDTH} {height}",
        "role": "img",
        "aria-labelledby": "title desc",
    },
)
add(svg, "title", {"id": "title"}, "多模型 ViT Encoder CUDA Graph 相对收益")
add(
    svg,
    "desc",
    {"id": "desc"},
    "按模型和场景分组的水平条形图，展示 Eager（CG off）到 Encoder CUDA Graph（CG on）的相对收益。"
    "▲与蓝色正值表示收益，▼与橙色负值表示回退；不同硬件、负载和指标不能横向排名。",
)
add(
    svg,
    "style",
    text="""
text { font-family: "Microsoft YaHei", "PingFang SC", "Noto Sans CJK SC", Arial, sans-serif; fill: #172033; }
.title { font-size: 28px; font-weight: 700; }
.subtitle { font-size: 17px; fill: #4b5568; }
.axis { font-size: 14px; fill: #596579; }
.group { font-size: 18px; font-weight: 700; }
.comparison { font-size: 13px; fill: #667085; }
.metric { font-size: 14px; }
.value-positive { font-size: 14px; font-weight: 700; fill: #075985; }
.value-negative { font-size: 14px; font-weight: 700; fill: #9a3412; }
.value-inside { font-size: 14px; font-weight: 700; fill: #ffffff; }
.note { font-size: 15px; fill: #4b5568; }
""",
)
add(svg, "rect", {"width": str(WIDTH), "height": str(height), "fill": "#ffffff"})
add(svg, "text", {"x": "800", "y": "40", "text-anchor": "middle", "class": "title"},
    "多模型 ViT Encoder CUDA Graph 相对收益")
add(svg, "text", {"x": "800", "y": "70", "text-anchor": "middle", "class": "subtitle"},
    "仅比较同一行 Before / After；硬件、负载与指标口径不同，不能用于模型横向排名")
add(svg, "text", {"x": "650", "y": "104", "class": "value-positive"}, "▲ +x% 收益")
add(svg, "text", {"x": "810", "y": "104", "class": "value-negative"}, "▼ -x% 回退")

for tick in range(-30, 91, 10):
    x = ZERO_X + tick * SCALE
    if tick != 0:
        add(svg, "line", {
            "x1": f"{x:.1f}", "y1": "128", "x2": f"{x:.1f}", "y2": str(plot_bottom),
            "stroke": "#d9e0e8", "stroke-width": "1",
        })
    add(svg, "text", {
        "x": f"{x:.1f}", "y": "124", "text-anchor": "middle", "class": "axis",
        "font-weight": "700" if tick == 0 else "400",
    }, "0" if tick == 0 else f"{tick:+d}%")
add(svg, "line", {
    "x1": str(ZERO_X), "y1": "128", "x2": str(ZERO_X), "y2": str(plot_bottom),
    "stroke": "#344054", "stroke-width": "2.5",
})

cursor_y = HEADER_HEIGHT
for group_index, (group_name, group_rows) in enumerate(groups.items()):
    group_height = len(group_rows) * ROW_HEIGHT
    add(svg, "rect", {
        "x": "20", "y": str(cursor_y - 4), "width": "1560", "height": str(group_height + 8),
        "rx": "8", "fill": "#f8fafc" if group_index % 2 == 0 else "#ffffff",
    })
    comparisons = " / ".join(dict.fromkeys(row["comparison"] for row in group_rows))
    center = cursor_y + group_height / 2
    add(svg, "text", {"x": "36", "y": str(int(center - 5)), "class": "group"}, group_name)
    add(svg, "text", {"x": "36", "y": str(int(center + 18)), "class": "comparison"}, comparisons)

    for row_index, row in enumerate(group_rows):
        center_y = cursor_y + row_index * ROW_HEIGHT + ROW_HEIGHT / 2
        add(svg, "text", {"x": "250", "y": str(int(center_y + 5)), "class": "metric"},
            compact_label(row))
        gain = float(row["relative_gain_percent"])
        endpoint = ZERO_X + gain * SCALE
        bar_x = min(ZERO_X, endpoint)
        bar_width = abs(endpoint - ZERO_X)
        positive = gain >= 0
        add(svg, "rect", {
            "x": f"{bar_x:.1f}", "y": str(center_y - 12), "width": f"{bar_width:.1f}",
            "height": "24", "rx": "4", "fill": "#38a7d8" if positive else "#e57a44",
        })
        value_text = f"{'▲' if positive else '▼'} {gain:+.2f}% {'收益' if positive else '回退'}"
        if positive and endpoint > 1400:
            attrs = {
                "x": f"{endpoint - 10:.1f}", "y": str(int(center_y + 5)),
                "text-anchor": "end", "class": "value-inside",
            }
        elif positive:
            attrs = {
                "x": f"{endpoint + 10:.1f}", "y": str(int(center_y + 5)),
                "class": "value-positive",
            }
        else:
            attrs = {
                "x": f"{endpoint - 10:.1f}", "y": str(int(center_y + 5)),
                "text-anchor": "end", "class": "value-negative",
            }
        add(svg, "text", attrs, value_text)
    cursor_y += group_height + GROUP_GAP

footer_y = plot_bottom + 22
add(svg, "line", {"x1": "20", "y1": str(footer_y - 10), "x2": "1580",
                  "y2": str(footer_y - 10), "stroke": "#d9e0e8"})
add(svg, "text", {"x": "30", "y": str(footer_y + 16), "class": "note"},
    "计算口径：延迟类 = (Before − After) / Before；吞吐类 = (After − Before) / Before。")
add(svg, "text", {"x": "30", "y": str(footer_y + 44), "class": "note"},
    "数据来源：vLLM 官方文档与各模型 PR。DeepSeek-OCR 原 PR 仅提供相对改善值；其他百分比由同一行 Before / After 校验。")

tree = ET.ElementTree(svg)
ET.indent(tree, space="  ")
tree.write(SVG_PATH, encoding="utf-8", xml_declaration=True)
