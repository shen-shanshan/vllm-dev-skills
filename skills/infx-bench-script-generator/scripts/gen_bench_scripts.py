#!/usr/bin/env python3
"""Generate local (non-Slurm) benchmark scripts from an InferenceX srt-slurm recipe.

Reads a recipe YAML (base + override variants), merges the selected variant,
and renders three bash scripts (docker run / server start / client replay)
plus variant.json and variant.env for inspection. Requires PyYAML.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

try:
    import yaml
except ImportError as exc:  # pragma: no cover
    sys.exit(
        "PyYAML is required: python3 -m pip install pyyaml "
        f"(original error: {exc})"
    )

TEMPLATE_DIR = Path(__file__).resolve().parent.parent / "assets" / "templates"

MODEL_PREFIX_RULES = [
    ("deepseek-v4.1-flash", "dsv41flash"),
    ("deepseek-v4", "dsv4"),
    ("minimax-m3", "minimaxm3"),
    ("qwen3.5", "qwen3.5"),
    ("glm-5.2", "glm5.2"),
    ("glm-5.3", "glm5.3"),
    ("kimi-k3", "kimik3"),
]

PROFILER_CONFIGS = {
    "prefill": (
        '{"profiler":"torch","torch_profiler_dir":"/logs/traces",'
        '"torch_profiler_activities":["CPU","CUDA"],'
        '"delay_iterations":0,"max_iterations":1,"ignore_frontend":true}'
    ),
    "decode": (
        '{"profiler":"torch","torch_profiler_dir":"/logs/traces",'
        '"torch_profiler_activities":["CPU","CUDA"],'
        '"delay_iterations":1,"max_iterations":1,"ignore_frontend":true}'
    ),
}

SKIP_SERVER_ARGS = {"revision", "tokenizer-revision"}


def load_recipe(path: str) -> dict:
    raw = yaml.safe_load(Path(path).read_text())
    if not isinstance(raw, dict) or "base" not in raw:
        raise ValueError("recipe must be an override-format YAML with a 'base' section")
    return raw


def deep_merge(base: dict, override: dict) -> dict:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def select_variant(raw: dict, override: str) -> dict:
    if override == "base":
        return dict(raw["base"])
    if override not in raw:
        raise ValueError(
            f"override {override!r} not found in recipe; available: "
            + ", ".join(sorted(k for k in raw if k.startswith("override_")))
        )
    return deep_merge(raw["base"], raw[override])


def derive_model_prefix(model_path: str) -> str:
    lowered = model_path.lower()
    for needle, prefix in MODEL_PREFIX_RULES:
        if needle in lowered:
            return prefix
    fallback = model_path.rsplit("/", 1)[-1].lower()
    return "".join(ch if ch.isalnum() else "-" for ch in fallback).strip("-")


def engine_type(recipe: dict) -> str:
    engine = recipe.get("engine", {})
    if isinstance(engine, dict):
        engine = engine.get("type", "")
    return str(engine or "")


def quote(value: str) -> str:
    return "'" + value.replace("'", "'\\''") + "'"


def flatten_args(args: dict, skip: set[str] | None = None) -> list[str]:
    skip = skip or set()
    flattened = []
    for key, value in args.items():
        if key in skip:
            continue
        flag = f"--{key}"
        if isinstance(value, bool):
            if value:
                flattened.append(flag)
        elif value is None:
            continue
        elif isinstance(value, (int, float)):
            flattened.append(f"{flag} {value}")
        elif isinstance(value, str):
            flattened.append(f"{flag} {quote(value)}")
        else:
            raise ValueError(f"unsupported arg type for {key}: {type(value).__name__}")
    return flattened


def args_block(lines: list[str]) -> str:
    if not lines:
        return ""
    return "  " + " \\\n  ".join(lines)


def env_exports(env: dict, skip_keys: set[str] | None = None) -> list[str]:
    skip_keys = skip_keys or set()
    return [f"export {key}={quote(str(value))}" for key, value in env.items() if key not in skip_keys]


def collect_setup_packages(recipe: dict) -> str:
    packages = []
    for section in (recipe.get("roles", {}).get("agg", {}), recipe.get("frontend", {})):
        env = section.get("env", {}) if isinstance(section, dict) else {}
        if isinstance(env, dict) and env.get("SETUP_PIP_PACKAGES"):
            packages.append(str(env["SETUP_PIP_PACKAGES"]))
    return " ".join(packages)


def render(template_name: str, tokens: dict[str, str]) -> str:
    content = (TEMPLATE_DIR / template_name).read_text()
    for token, value in tokens.items():
        content = content.replace(f"%%{token}%%", value)
    leftover = sorted(set(re.findall(r"%%[A-Z_]+%%", content)))
    if leftover:
        raise ValueError(f"template {template_name} has unresolved placeholders: {leftover}")
    return content


def build_tokens(recipe_path: str, override: str, args: argparse.Namespace) -> dict[str, str]:
    raw = load_recipe(recipe_path)
    recipe = select_variant(raw, override)

    model = recipe["model"]
    roles = recipe.get("roles", {})
    agg = roles.get("agg", {})
    agg_args = agg.get("args", {}) if isinstance(agg, dict) else {}
    agg_env = agg.get("env", {}) if isinstance(agg, dict) else {}
    frontend = recipe.get("frontend", {}) if isinstance(recipe.get("frontend"), dict) else {}
    benchmark = recipe.get("benchmark", {}) if isinstance(recipe.get("benchmark"), dict) else {}
    benchmark_env = benchmark.get("env", {}) if isinstance(benchmark, dict) else {}
    resources = recipe.get("resources", {}) if isinstance(recipe.get("resources"), dict) else {}

    framework = engine_type(recipe)
    if framework != "vllm":
        raise ValueError(f"unsupported engine {framework!r}; only vllm recipes are supported")

    model_id = str(model["path"]).removeprefix("hf:")
    model_prefix = derive_model_prefix(model_id)
    precision = str(model["precision"])
    gpu_type = str(resources.get("gpu_type", "gpu"))
    conc = str(benchmark_env.get("CONC", ""))
    if not conc:
        raise ValueError("selected variant has no benchmark.env.CONC")

    served_model_name = str(agg_args.get("served-model-name", model_id))
    router = frontend.get("type") == "vllm-router"
    profile = args.profile
    if profile != "off" and profile not in PROFILER_CONFIGS:
        raise ValueError(f"--profile must be one of off, {', '.join(sorted(PROFILER_CONFIGS))}")

    revision = str(agg_args.get("revision", ""))
    marker = "srt-slurm-recipes/"
    sanitized_recipe = recipe_path.split(marker, 1)[1] if marker in recipe_path else recipe_path
    sanitized_recipe = sanitized_recipe.removesuffix(".yaml").replace("/", "_")
    out_dir = Path(args.out_root) / f"{sanitized_recipe}_{override}"
    out_dir.mkdir(parents=True, exist_ok=True)

    container_name = f"infx-bench-{model_prefix}-{override.removeprefix('override_')}"
    result_filename = f"local_{model_prefix}_{precision}_{gpu_type}_{framework}_agentic_{override}"

    server_args = flatten_args(agg_args, skip=SKIP_SERVER_ARGS)
    if profile != "off":
        server_args.append(f"--profiler-config {quote(PROFILER_CONFIGS[profile])}")
    dp = int(agg_args.get("data-parallel-size", 1))
    router_args = [
        "--worker-urls http://localhost:8000",
        f"--intra-node-data-parallel-size {dp}",
        "--host 0.0.0.0",
        "--port 30000",
        *flatten_args(frontend.get("args", {})),
    ]

    engine_env = agg_env | (frontend.get("env", {}) if isinstance(frontend.get("env"), dict) else {})
    setup_packages = collect_setup_packages(recipe)
    pip_block = ""
    if setup_packages:
        pip_block = (
            f"echo 'Installing recipe runtime deps: {setup_packages}'\n"
            f"python3 -m pip install --no-cache-dir {setup_packages}"
        )

    script_dir_var = 'SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"'
    if args.model_path:
        model_mount = f'  -v "{os.path.expanduser(args.model_path)}:/model:ro" \\'
        model_note = "host path mounted read-only at /model"
    else:
        model_mount = f'  -v "${{CONTAINER_NAME}}-model:/model" \\'
        model_note = "named volume; 02_start_server.sh downloads weights once"
    if args.repo_path:
        # writable: uv/pip editable installs may write build artifacts into the checkout
        repo_mount = f'  -v "{os.path.expanduser(args.repo_path)}:/infmax-workspace" \\'
        repo_note = "host checkout mounted read-write at /infmax-workspace"
    else:
        repo_mount = f'  -v "${{CONTAINER_NAME}}-repo:/infmax-workspace" \\'
        repo_note = "named volume; 03_run_client.sh clones InferenceX once"
    if args.trace_path:
        trace_mount = f'  -v "{os.path.expanduser(args.trace_path)}:/hf-cache" \\'
        trace_note = "host dataset/token cache mounted at /hf-cache"
    else:
        trace_mount = f'  -v "$HOME/.cache/huggingface:/hf-cache" \\'
        trace_note = "reuses host HF cache + token; client downloads the trace dataset if missing"
    if args.results_dir:
        results_mount = f'  -v "{os.path.expanduser(args.results_dir)}:/logs" \\'
        results_note = "host results dir mounted at /logs"
    else:
        results_mount = f'  -v "${{CONTAINER_NAME}}-logs:/logs" \\'
        results_note = "named volume; InferenceX default RESULT_DIR=/logs/agentic"
    mount_args = "\n".join(
        [model_mount, repo_mount, trace_mount, results_mount, f'  -v "$SCRIPT_DIR:/infx-scripts:ro"']
    )

    if router:
        router_block = f"""# ---- vllm-router frontend (dep variant) ----
ROUTER_PORT=30000
echo "Starting vllm-router ... (log: $LOGS_DIR/router.log)"
nohup vllm-router \\
{args_block(router_args)} \\
  > "$LOGS_DIR/router.log" 2>&1 &
echo "vllm-router started (pid $!)"
router_deadline=$(( $(date +%s) + 600 ))
until curl -fsS "http://localhost:${{ROUTER_PORT}}/health" >/dev/null 2>&1; do
    if (( $(date +%s) > router_deadline )); then
        echo "ERROR: vllm-router not ready within 600s; see $LOGS_DIR/router.log" >&2
        tail -50 "$LOGS_DIR/router.log" >&2 || true
        exit 1
    fi
    sleep 5
done
echo "Router ready at http://localhost:${{ROUTER_PORT}}\""""
    else:
        router_block = ""

    client_port = "30000" if router else "8000"
    profile_step_label = {"prefill": "prefill", "decode": "decode"}.get(profile, "")
    profile_delay = {"prefill": "0", "decode": "1"}.get(profile, "0")
    if profile != "off":
        profiler_start_block = f"""# ---- torch profiler: capture one {profile_step_label} step (delay_iterations={profile_delay}) ----
echo "Starting engine profiler (capture one {profile_step_label} step)..."
curl -fsS -X POST "http://localhost:8000/start_profile" \\
    || echo "WARN: /start_profile failed; the server must be launched with --profiler-config" >&2"""
        profiler_stop_block = """# ---- stop profiler and flush traces ----
curl -fsS -X POST "http://localhost:8000/stop_profile" || true
echo "Profiler traces: /logs/traces/ (see 02_start_server.sh for the captured step type)\""""
    else:
        profiler_start_block = "# torch profiler: disabled (rerun the generator with --profile prefill|decode to enable)"
        profiler_stop_block = ""

    tokens = {
        "RECIPE_NAME": sanitized_recipe,
        "OVERRIDE": override,
        "CONTAINER_NAME": container_name,
        "IMAGE": str(model["container"]),
        "MODEL_ID": model_id,
        "MODEL_PREFIX": model_prefix,
        "FRAMEWORK": framework,
        "PRECISION": precision,
        "GPU_TYPE": gpu_type,
        "CONC": conc,
        "SERVED_MODEL_NAME": served_model_name,
        "RESULT_FILENAME": result_filename,
        "MODEL_REVISION": revision,
        "SCRIPT_DIR_VAR": script_dir_var,
        "MOUNT_ARGS": mount_args,
        "MOUNT_NOTES": "\n".join(
            [f"#   model:   {model_note}", f"#   repo:    {repo_note}", f"#   dataset: {trace_note}", f"#   results: {results_note}"]
        ),
        "ENGINE_ENV_EXPORTS": "\n".join(env_exports(engine_env, skip_keys={"SETUP_PIP_PACKAGES"})),
        "PIP_INSTALL_BLOCK": pip_block,
        "SERVER_ARGS": args_block(server_args),
        "ROUTER_BLOCK": router_block,
        "CLIENT_PORT": client_port,
        "BENCHMARK_ENV_EXPORTS": "\n".join(env_exports(benchmark_env, skip_keys={"MODEL", "CONC"})),
        "PROFILER_START_BLOCK": profiler_start_block,
        "PROFILER_STOP_BLOCK": profiler_stop_block,
    }
    return tokens, recipe, out_dir, result_filename


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", required=True, help="path to the srt-slurm recipe YAML")
    parser.add_argument("--override", required=True, help="override variant name, e.g. override_tp8_c56")
    parser.add_argument("--model-path", default="", help="host dir with model weights (else server script downloads)")
    parser.add_argument("--repo-path", default="", help="host InferenceX checkout (else client script clones)")
    parser.add_argument("--trace-path", default="", help="host dir for the trace dataset/HF cache (else ~/.cache/huggingface)")
    parser.add_argument("--results-dir", default="", help="host dir for results (else named volume at /logs)")
    parser.add_argument("--profile", default="off", choices=["off", "prefill", "decode"])
    parser.add_argument("--out-root", required=True, help="outputs root directory")
    args = parser.parse_args()

    tokens, recipe, out_dir, result_filename = build_tokens(args.recipe, args.override, args)

    for template, filename in [
        ("01_docker_run.sh.tmpl", "01_docker_run.sh"),
        ("02_start_server.sh.tmpl", "02_start_server.sh"),
        ("03_run_client.sh.tmpl", "03_run_client.sh"),
    ]:
        (out_dir / filename).write_text(render(template, tokens))

    variant = {
        "recipe": args.recipe,
        "override": args.override,
        "profile": args.profile,
        "derived": {
            "model_id": tokens["MODEL_ID"],
            "model_prefix": tokens["MODEL_PREFIX"],
            "framework": tokens["FRAMEWORK"],
            "precision": tokens["PRECISION"],
            "gpu_type": tokens["GPU_TYPE"],
            "conc": tokens["CONC"],
            "served_model_name": tokens["SERVED_MODEL_NAME"],
            "result_filename": tokens["RESULT_FILENAME"],
        },
        "merged": recipe,
    }
    (out_dir / "variant.json").write_text(json.dumps(variant, indent=2, sort_keys=True) + "\n")

    env_lines = []
    for section in (recipe.get("roles", {}).get("agg", {}), recipe.get("frontend", {}), recipe.get("benchmark", {})):
        if isinstance(section, dict):
            env_lines.extend(env_exports(section.get("env", {})))
    env_lines += [
        f"export MODEL={tokens['MODEL_ID']}",
        f"export MODEL_PREFIX={tokens['MODEL_PREFIX']}",
        f"export FRAMEWORK={tokens['FRAMEWORK']}",
        f"export PRECISION={tokens['PRECISION']}",
        f"export CONC={tokens['CONC']}",
        f"export SERVED_MODEL_NAME={tokens['SERVED_MODEL_NAME']}",
        f"export RESULT_FILENAME={tokens['RESULT_FILENAME']}",
    ]
    (out_dir / "variant.env").write_text("\n".join(env_lines) + "\n")

    print(f"generated {out_dir}:")
    for name in ("01_docker_run.sh", "02_start_server.sh", "03_run_client.sh", "variant.json", "variant.env"):
        print(f"  {out_dir / name}")
    print(f"aggregate result: /logs/{result_filename}.json ; raw artifacts: /logs/agentic/")


if __name__ == "__main__":
    main()
