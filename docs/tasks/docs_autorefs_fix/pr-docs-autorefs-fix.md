# [Doc] Fix strict docs build broken by unresolved autorefs

## Overview

The strict MkDocs build (Read the Docs, `fail_on_warning: true`) has been failing on `main` since 2026-11-05 with `Aborted with 12 warnings in strict mode!`. Every PR's docs check fails with the same warnings, e.g. builds [35688234](https://app.readthedocs.org/projects/vllm/builds/35688234/) (`main` at `e336407a3d`) and [35688313](https://app.readthedocs.org/projects/vllm/builds/35688313/). This PR fixes the docstrings behind those warnings. No code behavior changes.

## Claims

- The strict docs build passes again. All 12 `mkdocs_autorefs: Could not find cross-reference target` warnings are gone.
- The only changes are docstrings. No code changes.

## Validation

Ran the same command as Read the Docs, with `requirements/docs.txt` installed in a clean venv:

```bash
uv venv -p 3.12 /tmp/docs-venv
VIRTUAL_ENV=/tmp/docs-venv uv pip install -r requirements/docs.txt
/tmp/docs-venv/bin/python -m mkdocs build --clean --site-dir /tmp/docs-site --config-file mkdocs.yaml --strict
```

| Tree | Result |
|---|---|
| `main` (`e336407a3d`) | Exit 1, `Aborted with 12 warnings in strict mode!`, the same 12 warnings as Read the Docs build 35688234 |
| This PR | Exit 0, `Documentation built in 178.18 seconds`, no warnings |

## Details

There are two root causes.

**1. MiniMax-M3.1 drafter docstrings (11 warnings), from #65386.**
- Some docstrings write array shapes as consecutive brackets, such as `red[8 t + wave][row][token]`, `x [16][K]`, `x_in [rows][6144]`, `x8[t][16 tid ..]`, `xs8[t][tid / 2]` and `o [112][128]`.
- Markdown parses `[a][b]` as a reference-style link, so autorefs tries to resolve `b` (`row`, `K`, `6144`, `16 tid ..`, `tid / 2`, `128`) and fails.
- Fix: wrap these shapes in code spans in `drafter/device.py` (`red_store`, `xq_chunk`, `row_x`, `quant_store`) and `drafter/attention.py` (`store_partial`). Single brackets such as `[1, rank, site]` are not affected, so they are left as is.

**2. `TrtLlmMxfp4ExpertsModular` lost its docstring (1 warning), from #54109.**
- #54109 renamed the documented class to `TrtLlmMxfp4ExpertsModularBase`, then added new `TrtLlmMxfp4ExpertsModular` and `TrtLlmMxfp4Mxfp8ExpertsModular` subclasses without docstrings.
- mkdocstrings skips objects without a docstring by default, so the API page has no anchor for `TrtLlmMxfp4ExpertsModular`. The link in `docs/design/moe_kernel_features.md` then cannot resolve.
- Fix: give both subclasses a one-line docstring that names their activation quantization, so both get API anchors. The docs link stays as it is, because it names the concrete experts class.

---

<details>
<summary> Pull Request Checklist </summary>

- [ ] I used vLLM's `/pr-checklist` skill. (Mandatory for agents, optional for humans).
- [x] AI assistance was used during the creation of this PR.

- [x] **Design Fit:** Minimizes impact on core components, reuses existing functionality, and justifies added complexity.
- [x] **Testing and Validation:** Validates the change and ensures any added tests are meaningful and reliable, with CI coverage or documented CI resource constraints and validation performed outside CI.
- [x] **Code Quality and Style:** Keeps code and comments clear and concise, and updates relevant documentation and examples.
- [x] **Pull Request Contents:** Includes a brief summary and relevant links, supports claims with evidence, explains root causes and implementation trade-offs, and follows the contributing guide.
</details>

🤖 Generated with [Claude Code](https://claude.com/claude-code)
