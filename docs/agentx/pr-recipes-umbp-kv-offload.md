# Add UMBP (MORI) KV offload option for AMD

Repository: `vllm-project/recipes` · branch `amd-dsv41-flash-umbp` (fork `shen-shanshan/recipes`) · commit `Add UMBP (MORI) KV offload option for AMD`

## Summary

Adds `umbp` to the composing KV Offload options in `taxonomy.yaml`, so recipes can offer vLLM's `UMBPStoreConnector` (MORI's unified memory buffer pool, standalone mode) the way they offer Simple, LMCache or IAXL.

- `--kv-transfer-config` uses `UMBPStoreConnector` in `standalone` mode with `kv_load_failure_policy: recompute`, so a block evicted between lookup and load is recomputed instead of failing the request.
- A **UMBP Server** companion tab starts MORI's `umbp_standalone_server` before `vllm serve`. The server owns one node-local host-DRAM pool that every engine and DP rank on the host shares. KV identical on every TP rank, such as MLA latents, is stored once.
- Scope: `brands: [AMD]` (the vLLM ROCm images ship MORI built with UMBP), single-node strategies only (the server is node-local), and `requires_opt_in: true`. A recipe offers it only after recording an end-to-end run as `kv_offload_support: { umbp: verified }`, as with the Offloading tiers and IAXL.
- The gRPC and fd sockets live under `/dev/shm/umbp/`. A pip install and the builder's ROCm container (`--ipc=host`) both see that path, so the server and the engine find each other however each is started. A private `/tmp` does not work across containers: the first InferenceX sweep of this configuration failed on exactly that.
- SCHEMA.md and CLAUDE.md now list UMBP among the fail-closed and composing options.

The first recipe to use it is DeepSeek-V4.1-Flash on MI355X (vllm-project/vllm#XXXXX). That PR's check renders against this repo's `main`, so **this PR needs to merge first**.

## Validation

- `node scripts/build-recipes-api.mjs`: `✓ JSON API: 228 models …`, and `public/taxonomy.json` carries the new option.
- `pnpm test:schema` (5/5), `pnpm test:kv-offload` (11/11), `pnpm test:presets` (15/15) pass.
- With `sources.local.yaml` pointing `vllm` at the companion vLLM branch, `node scripts/validate-source.mjs --source vllm --dir <vllm>/recipes` reports the recipes valid. The DeepSeek-V4.1-Flash `mi355x_agentx_dp2_umbp` preset renders the `UMBPStoreConnector` config and the UMBP Server tab with matching socket addresses.
- `pnpm test:render` has one failing case (`resolveSelection renders every preset point identically`) for MiniMax-M3.1-Flash `mi355x_agentx_pd` and Mistral-Large-4 `h200_dcp8`. It fails the same way on `main` without this change.
- The server command was run locally on MI355X: it binds both sockets under `/dev/shm/umbp/`, and it rebinds over stale socket files left by an earlier server.

<details>
<summary>AI model disclosure</summary>

- Model/version: `claude-opus-5-5[1m]` (Claude Opus 5.5, 1M context), via Claude Code.
- Role: read the taxonomy, schema and synthesis code; drafted the `umbp` option, the doc updates and this description; ran the build and tests. A human chose the scope (AMD, single-node, opt-in) and reviewed the change.

</details>

🤖 Generated with [Claude Code](https://claude.com/claude-code)
