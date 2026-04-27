# Qwen3.6-27B on a single RTX 5090

**A validated recipe for serving Qwen3.6-27B on a single 32 GB RTX 5090** — full OpenAI API, vision, tool calling, streaming, speculative decoding, all verified end-to-end via `scripts/verify-full.sh`.

Based on [`Lorbus/Qwen3.6-27B-int4-AutoRound`](https://huggingface.co/Lorbus/Qwen3.6-27B-int4-AutoRound) via vLLM with MTP speculative decoding + fp8_e4m3 KV cache. Built on [`Sandermage/genesis-vllm-patches`](https://github.com/Sandermage/genesis-vllm-patches) + a CUDA graph capture fix that ships in this repo.

> 📖 **Write-up:** *[Qwen3.6-27B on a single RTX 5090 — the recipe](https://medium.com/)*
> 🐛 **Upstream bug reports:** [vllm-project/vllm#40807](https://github.com/vllm-project/vllm/issues/40807) (CUDA graph crash — worked around locally) · [vllm-project/vllm#40831](https://github.com/vllm-project/vllm/issues/40831) (TurboQuant × spec-decode output-quality, isolated to cudagraph capture; root cause TBD)

---

## About the 125K context headline

The [original write-up](https://medium.com/) reported 85–106 TPS at 125K context using TurboQuant KV. Under broader functional testing since publication we found that the originally-shipped 125K config produces degenerate token loops on tool calls, long-context recall, and occasionally streaming. An eight-probe investigation traced the failure to **vLLM's CUDA graph capture/replay machinery for spec-decode + TurboQuant** — not the kernels, not torch.compile inductor output, not attention math. Filed upstream as [#40831](https://github.com/vllm-project/vllm/issues/40831); root cause within the cudagraph layer is still TBD upstream. Full probe ladder + analysis in the [Technical background](#technical-background--why-the-long-ctx-config-disables-cudagraph) section below.

The correct workaround: `--compilation-config '{"cudagraph_mode":"NONE"}'` (disables CUDA graph capture, keeps torch.compile inductor on). Cost: ~60% TPS (85 → 33 narrative). Restores correctness across tool calls, recall, and streaming. When upstream lands a fix, that flag can be dropped and TPS recovers.

**The default config (`docker-compose.yml`) is unaffected** — MTP n=3 + fp8_e4m3 KV + vision at 20K, ~85 TPS peak, no workaround needed (fp8 KV doesn't go through TurboQuant's custom backend).

---

## Evidence matrix — what works

Measured on 1× RTX 5090, vLLM image pinned to tested digest, `scripts/verify-full.sh`:

| Test | Default (MTP + fp8_e4m3 + vision, 20K) |
|---|---|
| Server + Genesis patches | ✅ |
| Basic completion (Paris) | ✅ |
| **Tool calling** | **✅** |
| **Streaming (SSE)** | **✅** clean output |
| Thinking / reasoning | ✅ |
| **Long-context recall** (10K) | **✅** |
| Short-prompt TPS (narrative) | 65.9 |
| Peak TPS | 85 |
| Max context | 20K |
| Vision | ✅ |
| VRAM | 22.8 GB |

---

## Production numbers — default config

```
  Qwen3.6-27B on 1× RTX 5090 (32 GB, default config)
  ────────────────────────────────────────────────────────────
  Throughput      66 TPS (narrative)  /  84 TPS (code, peak 85)
  Context          20 K tokens
  Vision           Enabled (MoonViT BF16)
  VRAM            22.8 / 32 GB
  Server          vLLM · full OpenAI API
  Tools           ✅ working   Streaming ✅   Thinking ✅
  Spec-decode    MTP n=3 · AL 2.87–3.39 · accept 94/81/64%
```

Beats [Lorbus card's RTX 5090 baseline](https://huggingface.co/Lorbus/Qwen3.6-27B-int4-AutoRound) (~60 TPS) on consumer Ampere hardware. All functionality verified.

---

## Requirements

- **GPU:** 1× NVIDIA RTX 5090 (32 GB, Blackwell GB202).
- **Driver:** 580.x or newer (for CUDA 13 runtime in the pinned vLLM image).
- **Chat template:** `~/ai/qwen3.5-enhanced.jinja` must exist on the host — the compose file bind-mounts it at startup. Obtain it from the repo or provide your own Qwen3-compatible template.
- **Disk:** ~20 GB free for model weights.
- **Software:**
  - Docker with NVIDIA Container Toolkit
  - `git`, `curl`, `sha256sum` (setup script uses them)
  - `hf` CLI *or* `huggingface-cli` (install: `pip install 'huggingface-hub[hf_transfer]'`)

No system Python required.

---

## Quick start

```bash
# 1. Clone this repo
git clone https://github.com/CobraPhil/qwen36-27b-single-5090.git
cd qwen36-27b-single-5090

# 2. Fetch Genesis patches + download + SHA-verify the model (~20 GB, 10-30 min)
bash scripts/setup.sh

# 3. Start the server
cd compose && docker compose up -d

# 4. Watch it come up (~2 min for cold compile)
docker logs -f vllm-qwen36-27b
# Wait for "Application startup complete"

# 5. Sanity test
curl -sf http://localhost:8020/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3.6-27b-autoround",
       "messages":[{"role":"user","content":"Capital of France?"}],
       "max_tokens":30}'

# 6. Run the canonical benchmark
cd .. && bash scripts/bench.sh
```

That's it. The stack serves on `http://localhost:8020/v1/*` as a drop-in OpenAI-compatible endpoint — point any OpenAI SDK, Open WebUI, LM Studio, or Cline at it.

---

## Why this works where other recipes don't

Three hurdles had to be cleared for this config to run on a single consumer 24 GB card:

### 1. The published int4-AutoRound quant preserves `mtp.fc` at full precision

A vanilla `auto-round` run on Qwen3.6-27B packs the MTP fusion layer (`mtp.fc`) as INT4. In that form, vLLM's `Qwen3_5MTP` loader silently skips loading it (param name mismatch: expects `fc.weight`, finds `fc.qweight`). Result: MTP "loads" with zero parameters and produces **0% draft acceptance**.

Both [`Lorbus/Qwen3.6-27B-int4-AutoRound`](https://huggingface.co/Lorbus/Qwen3.6-27B-int4-AutoRound) and [`Intel/Qwen3.6-27B-int4-AutoRound`](https://huggingface.co/Intel/Qwen3.6-27B-int4-AutoRound) work around this — they ship `mtp.fc.weight` as a plain unquantized BF16 tensor. Lorbus does it implicitly (the `.weight` tensor is in the file with no explicit `extra_config` entry); Intel adds an explicit `mtp.fc: {bits: 16, data_type: fp}` to `extra_config`. Functionally identical: same 18 GB on disk, same 2013 tensors, same architecture, same INT4/group_size=128/auto_round packing. We use Lorbus because it's what we tested end-to-end; Intel's variant should be a drop-in if you prefer that source.

Quick check that whichever quant you use has the fix: look for `mtp.fc.weight` (not `mtp.fc.qweight`) in the safetensors index.

### 2. Genesis patches bypass the TurboQuant hybrid gate

`Qwen3.6-27B` is a Qwen3-Next hybrid model: interleaved DeltaNet (Gated Linear Attention) + standard attention layers. vLLM's TurboQuant KV cache refuses to initialize on hybrid models:

```
NotImplementedError: TurboQuant KV cache is not supported for hybrid
(attention + Mamba) models. Boundary layer protection requires uniform
attention layers.
```

[Sandermage's Genesis patches](https://github.com/Sandermage/genesis-vllm-patches) are a 20-patch runtime monkey-patcher that, among other things, rewrites the hybrid gate to compute boundary protection only over attention layers. Works on Ampere SM 80–86.

### 3. Our `patch_tolist_cudagraph.py` fixes CUDA graph capture

Even with the hybrid gate bypassed, vLLM still crashed during engine warmup:

```
turboquant_attn.py:570  qsl = query_start_loc.tolist()
RuntimeError: Cannot copy between CPU and CUDA tensors during CUDA graph
capture unless the CPU tensor is pinned.
```

The continuation-prefill branch of `_prefill_attention` forces a GPU→CPU sync via `.tolist()`, which is illegal during CUDA graph capture. This trips when `--speculative-config` + `--enable-chunked-prefill` + `turboquant_*` KV are combined (vLLM PR #40092 — merged 2026-04-23 — fixed the fast path but left this continuation branch untouched).

Our patch (`patches/patch_tolist_cudagraph.py`) is a disk-edit that wraps both `.tolist()` sites with `torch.cuda.is_current_stream_capturing()` guards. During capture, fall back to the graph-safe fast path; at inference, run the original slow path unchanged. Safe because `unified_attention_with_output` is in vLLM V1's `splitting_ops` list — attention outputs during capture are only consulted for memory profiling, not graph content.

Without this patch, the documented workaround is `--compilation-config.cudagraph_mode=none`, which costs **−55% short-prompt TPS** and makes the whole setup net-negative vs plain fp8 KV.

---

## Configuration notes

### Speculative decoding

`num_speculative_tokens=3` (MTP) is the sweet spot on this model:

| n | Narr TPS | Code TPS | Mean AL | Position-wise accept |
|---|---|---|---|---|
| 1 | 55 | 59 | 1.9 | 96% |
| 2 | 61 | 70 | 2.4 | 82/56% |
| **3 ⭐** | **64** | **80** | **3.4** | **92/81/67%** |
| 4 | 63 | 82 | 3.0 | 83/55/36/**21%** |

n=4 barely beats n=3 on code peak but the position-4 draft accept collapses to 21% — wasted work. Don't go higher.

### KV cache

`turboquant_3bit_nc` is the smallest preset that boots cleanly. `turboquant_4bit_nc` and `turboquant_k8v4` also work with the patches but give less context:

| Preset | Bits | Per-token bytes | Single-card ceiling |
|---|---|---|---|
| default (BF16) | 16 | ~55 KB | ~8K |
| `fp8_e4m3` / `fp8_e5m2` | 8 | ~28 KB | ~32K |
| `turboquant_k8v4` | 8+4 avg 6 | ~28 KB | ~40K |
| `turboquant_4bit_nc` | 4+4 avg 4 | ~23 KB | ~84K |
| **`turboquant_3bit_nc`** ⭐ | **3+3** | **~17 KB** | **~125K** |

### Context ceiling

`--max-model-len 125000` is the largest value vLLM's `_check_enough_kv_cache_memory` pre-check accepts with our config. The KV pool itself (198K tokens) is larger, but the extra capacity is used for Mamba/DeltaNet recurrent state + prefix cache + spec-decode scratch blocks — **don't bypass the pre-check**, those tokens are load-bearing.

### Power cap

Production runs at 230W per card (quiet, cool, stable). For ~+10% mean TPS during heavy sessions:

```bash
sudo nvidia-smi -pl 330 -i 0   # replace 0 with your GPU index
```

Past the knee: diminishing returns as SM clocks saturate.

---

## Benchmarking

```bash
bash scripts/bench.sh
```

Runs 3 warmup + 3 narrative (800-word essay, 1000 tokens) + 2 code (quicksort, 800 tokens) against the canonical prompts used throughout this repo. Reports wall time, completion tokens, TPS per request, plus GPU state and the last 3 SpecDecoding metrics lines (mean AL + per-position accept rates).

Expected numbers on an RTX 5090:

| Run | Wall | TPS |
|---|---|---|
| warmup 1 (cold) | 12–15 s | 70–80 |
| warmup 2+3 (warm) | 10–11 s | 90–100 |
| narrative (warmed) | 10–16 s | 60–105 |
| code (warmed) | 8–12 s | 60–100 |

A 125K turboquant config (not shipped here) runs at a more uniform ~33 TPS because it disables CUDA graph capture (`cudagraph_mode=NONE`); spec-decode acceptance dips don't compound with cudagraph variance. See [Technical background](#technical-background--why-the-long-ctx-config-disables-cudagraph) for detail.

---

## Compose config

This repo ships a single compose file: `docker-compose.yml` — MTP n=3 + fp8_e4m3 KV + vision at 20K ctx, ~85 TPS peak. It is the fully validated default.

The server binds to port 8020. `docker compose down` before restarting with different settings.

The compose is pinned to a specific vLLM image digest (`bbac761a...`), uses the Genesis patches fetched by `setup.sh`, and applies `patch_tolist_cudagraph.py` at container startup alongside the Genesis patcher.

---

## Technical background — why the long-ctx config disables cudagraph

**TurboQuant KV is frontier-level.** It landed in vLLM mainline only weeks before this repo was published and is still under active development. The spec-decode × TurboQuant interaction we hit is one of several compatibility edges upstream is still working through (see vLLM's tracking issue [#40069](https://github.com/vllm-project/vllm/issues/40069) — "Speculative decoding / Eagle" and "Hybrid attention models" both unchecked).

We isolated the bug through six probes:

| # | turboquant | spec-dec | cudagraph | torch.compile | result | TPS |
|---|---|---|---|---|---|---|
| 1 | ✅ | off | ✅ | ✅ | ✅ all tests pass | 40 |
| 2 | ✅ | ngram n=3 | ✅ | ✅ | ✗ same loops as MTP | -- |
| 3 (MiMo dense) | ✅ | MTP n=1 | ✅ | ✅ | ✗ first-token collapse | -- |
| 4 | ✅ | MTP n=3 | ✅ | + `_CONTINUATION_DECODE_THRESHOLD=0` | ✗ | -- |
| 5 | ✅ | MTP n=3 | ❌ | ❌ | ✅ all tests pass | 23 |
| **6** | ✅ | MTP n=3 | **❌** | ✅ | **✅ all tests pass** | **33** |

**What this isolates:**

- Probe 1 → TurboQuant alone is fine.
- Probes 2-3 → bug isn't MTP-specific; isn't hybrid-attention-specific.
- Probe 4 → bug isn't in the within-batch `_prefill_attention` decode-fast-path routing (paper-backed bias-compounding hypothesis was wrong).
- Probe 5 → disabling **both** torch.compile and cudagraph fixes the bug — compilation machinery is the culprit.
- Probe 6 → disabling **only** cudagraph (keeping torch.compile inductor on) also fixes the bug — **isolating the bug to CUDA graph capture/replay specifically**.

**The Triton kernels are correct when invoked dynamically. torch.compile inductor output is correct.** What corrupts the output is how the captured CUDA graph handles spec-decode's runtime shapes vs warmup-shape capture for the TurboQuant attention path. Specific root cause is still **TBD upstream**.

We initially hypothesized — and Sander independently flagged — that [PR #40798](https://github.com/vllm-project/vllm/pull/40798) ("Share decode scratch workspace across layers") was the structural fix, because it moves `_tq_mid_o_buf` / `_tq_output_buf` / `_tq_lse_buf` from per-layer `register_buffer(B=max_num_seqs=1)` to `WorkspaceManager.get_simultaneous()` (persistent base-buffer with a stable `data_ptr`). Probe 8 backported the PR onto our pinned nightly digest and tested against the originally-failing config: **bug persists** with all of #40798's structural changes applied (verified live in the running container, with TPS at ~96 confirming cudagraph + compile genuinely engaged). So either #40798 is necessary but not sufficient, or there's a companion change in `main` we haven't backported, or the bug is in a different code path than the per-layer scratch buffers entirely. Full probe-8 data: [#40831 follow-up](https://github.com/vllm-project/vllm/issues/40831#issuecomment-4317503179).

The validated workaround for a 125K config is `--compilation-config '{"cudagraph_mode":"NONE"}'`. Cost: ~60% TPS (85 → 33 narrative). Drop the flag once upstream lands a fix.

**What we're doing about it:**

- **[#40807](https://github.com/vllm-project/vllm/issues/40807)** — CUDA graph crash workaround via `patches/patch_tolist_cudagraph.py` (separate from #40831).
- **[#40831](https://github.com/vllm-project/vllm/issues/40831)** — output-quality bug, six-probe isolation in the issue. Cross-references adjacent PRs ([#40074](https://github.com/vllm-project/vllm/pull/40074), [#40122](https://github.com/vllm-project/vllm/pull/40122), [#40706](https://github.com/vllm-project/vllm/pull/40706), [#40798](https://github.com/vllm-project/vllm/pull/40798)).
- The default config stays on fp8_e4m3 (no cudagraph workaround needed) at 20K ctx, ~85 TPS.

When upstream resolves the cudagraph capture issue, the long-ctx variant drops `cudagraph_mode=NONE` and TPS recovers to the original ~85+.

---

## Troubleshooting

### `Cannot copy between CPU and CUDA tensors during CUDA graph capture`

The `patch_tolist_cudagraph.py` didn't apply. Check the container logs for:

```
[tolist_cudagraph_fix] Patched ... Site A: ok, Site B: ok
```

If not present, the anchor text may have drifted in a newer vLLM image. The compose is already pinned to a tested digest — if you changed it, revert to the pinned digest in `docker-compose.yml`, or open an issue here.

### `NotImplementedError: TurboQuant KV cache is not supported for hybrid`

Genesis patches didn't apply. Check logs for `INFO:genesis_patch:` lines. Re-run `bash scripts/setup.sh` to ensure `patches/genesis/` exists, then restart the container.

### Model load OOMs

- Too little free VRAM at launch. Close other GPU processes.
- If you have multiple GPUs, set `CUDA_VISIBLE_DEVICES=N` in `compose/docker-compose.yml` (uncomment the line).

### `block_size (4128) must be <= max_num_batched_tokens (2048)`

You edited `--max-num-batched-tokens`. Keep it ≥ 4128 for this context length — Qwen3-Next's Mamba block_size scales with `max-model-len`.

### Short-prompt TPS stuck at ~30

If you're seeing ~30 TPS on the default config, something is wrong — check that `patch_tolist_cudagraph.py` applied (`docker logs ... | grep tolist_cudagraph_fix`). Expected is ~65–85 TPS.

### Tool calls return `<tool_call>{...}</tool_call>` as plain text (tool extraction doesn't fire)

Check container logs for:

```
[11/17] Qwen3 <tool_call> implicit reasoning end (PR #35687)...
  [FAILED] Qwen3 tool_call fix
```

If you see `[FAILED]`, your vLLM image drifted past the anchor Genesis Patch 12 expects. The compose file is already pinned to a tested digest — revert `docker-compose.yml` to the pinned `image:` line if you changed it. On the pinned digest (vLLM `0.19.2rc1.dev21+g893611813`), all four Qwen3 tool-call sub-patches apply cleanly — look for `[OK] Qwen3 tool_call fix`.

---

## Repo layout

```
qwen36-27b-single-5090/
├── README.md                                   (this file)
├── LICENSE                                     Apache-2.0
├── .gitignore
├── patches/
│   ├── genesis_shim.py                         copies genesis/_genesis into vLLM at startup;
│   │                                            mounted as /patches/patch_genesis_unified.py
│   ├── patch_tolist_cudagraph.py               CUDA graph capture crash fix (#40807)
│   ├── patch_pr40798_workspace.py              research artifact — backports vllm#40798;
│   │                                            does NOT fix #40831 (probe 8); kept for
│   │                                            reproducibility of the negative result
│   └── genesis/                                (gitignored; fetched by setup.sh)
│       └── vllm/_genesis/                      the actual patch code used at runtime
├── compose/
│   └── docker-compose.yml                      MTP n=3 + fp8_e4m3 KV + vision, 20K, ~85 TPS
└── scripts/
    ├── setup.sh                                clone Genesis + download model + SHA verify
    ├── verify.sh                               quick smoke test (~10 sec)
    ├── verify-full.sh                          full functional test — streaming, thinking, needle (~3 min)
    └── bench.sh                                canonical TPS bench
```

**Host requirement:** `~/ai/qwen3.5-enhanced.jinja` must exist before running `docker compose up`. This file is bind-mounted as the chat template. If missing, the container exits immediately with a mount error.

---

## What this is NOT

- A vLLM fork — `patch_tolist_cudagraph.py` is a disk-edit applied at container startup, not a fork. When upstream merges the fix, this patch becomes a no-op (anchor won't match, script prints a warning and exits cleanly).
- A quantization recipe — we use Lorbus's INT4 quant as-is. The recipe for producing future `mtp.fc`-preserved quants is in [Lorbus's model card](https://huggingface.co/Lorbus/Qwen3.6-27B-int4-AutoRound#reproduction).
- A benchmark rig — included `bench.sh` is the minimum needed to verify your setup matches ours. For rigorous A/B comparisons use something like [`vllm-project/bench`](https://github.com/vllm-project/bench).

---

## Upstream status

- **[#40069](https://github.com/vllm-project/vllm/issues/40069)** — TurboQuant/HIGGS follow-ups tracker (upstream). Lists "Speculative decoding / Eagle" and "Hybrid attention models" as unchecked.
- **[#40807](https://github.com/vllm-project/vllm/issues/40807)** — our CUDA graph `.tolist()` crash; worked around locally via `patch_tolist_cudagraph.py`. Sandermage's [v7.10 Genesis tree](https://github.com/Sandermage/genesis-vllm-patches) reaches the same end state via pre-allocation (Patches 23 + 44).
- **[#40831](https://github.com/vllm-project/vllm/issues/40831)** — our TurboQuant × spec-decode output-quality bug. Eight-probe ladder + Sander's independent confirmation isolate it to **CUDA graph capture/replay** (probe 6: cudagraph off, torch.compile on → all 9 prompts pass at 33 TPS, including Sander's `tool_call_simple` / `code_quicksort` / `structured_xml` failure cases). Workaround: `--compilation-config '{"cudagraph_mode":"NONE"}'`. Root cause within the cudagraph layer: still TBD upstream — see #40798 below.
- **[PR #40798](https://github.com/vllm-project/vllm/pull/40798)** — *initially hypothesized fix; tested via probe 8 backport, **bug persists**.* Moves `_tq_mid_o_buf` / `_tq_output_buf` / `_tq_lse_buf` from per-layer `register_buffer(B=max_num_seqs)` to `WorkspaceManager.get_simultaneous()`. Sander and I both expected this would close the pointer-drift between warmup-shape capture and runtime-shape replay. We applied the PR's full diff to the pinned nightly via [`patches/patch_pr40798_workspace.py`](./patches/patch_pr40798_workspace.py) (research artifact, not shipped) and ran verify-full.sh + the 9-prompt Layer-2 probe against the cudagraph-on config. Same degenerate loops as before. Either #40798 is necessary but not sufficient, or a companion change in `main` we haven't backported is also required, or the bug is in a different code path than the per-layer scratch buffers entirely.
- **Sandermage's [P56](https://github.com/Sandermage/genesis-vllm-patches/blob/main/vllm/_genesis/wiring/patch_56_spec_decode_decode_path_guard.py)** — routing-layer workaround (architecturally equivalent to our Probe 4 patch). Marked superseded by our `cudagraph_mode=NONE` workaround since it only addresses the catastrophic surface.
- Sandermage Genesis: we may contribute `patch_tolist_cudagraph.py` to their unified script. They have offered to extract Patches 23 + 44 to upstream.

Until upstream lands a fix: fp8_e4m3 + MTP at 20K is the shipped config — fully functional at ~85 TPS. The 125K turboquant path works correctly only with `cudagraph_mode=NONE` (verified via probe 6); that workaround costs ~60% TPS and would be a candidate for a future extended-context variant once the upstream bug is resolved.

---

## Credits

- **Qwen team** (@Alibaba_Qwen) — for the base model and a usable MTP head architecture
- **Lorbus** — for the AutoRound INT4 quant with preserved BF16 `mtp.fc`
- **[Sandermage](https://github.com/Sandermage/genesis-vllm-patches)** — for the Genesis patch set that made TurboQuant work on hybrid models on consumer Ampere; for independently reproducing #40831 on a different rig (2× A5000 + Qwen3-Next-35B-A3B-FP8 + ngram), confirming the cudagraph-off workaround, and engaging honestly with each negative result during the probe ladder
- **[vibhavagarwal5](https://github.com/vllm-project/vllm/pull/38479)** — for the original TurboQuant landing PR and the [tracking issue #40069](https://github.com/vllm-project/vllm/issues/40069) that made the spec-decode-unverified status visible upfront
- **vLLM project** — for shipping TurboQuant and actively maintaining the backend
- **Intel AutoRound** — for the quantization framework

Our contribution here is `patch_tolist_cudagraph.py`, the original write-up linking it all together, and this reproducible recipe. Everything else is brilliant work by people we stand on the shoulders of.

---

## License

Apache 2.0. Do what you want with it. If you get better numbers, please open an issue — we'd love to see it.
