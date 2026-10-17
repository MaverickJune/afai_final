# Training environment setup (Yale Bouchet)

Set up 2026-10-11/12. Two environments:

| | What | Where |
|---|---|---|
| slime (training/rollout) | Official image `slimerl/slime`, Megatron-LM trainer + SGLang rollout | `~/containers/slime_v0.4.0-cu129.sif` |
| afai_env (analysis/eval) | conda prefix env | `~/envs/afai_env` |

## 1. slime container

**Image:** `docker://slimerl/slime:v0.4.0-cu129` (same as tag `v0.4.0`), pushed 2026-10-09
**Pinned digest:** `sha256:626fbff9c062bc3723ee617a6930ff96913750f6847a8df9e2319d84f8169c45`
**SIF:** `~/containers/slime_v0.4.0-cu129.sif`, 18 GB,
sha256 `1877674763701cc4a373d793c89332f31e9b76536f93bcecb61ede7bd2c5d665`
**Built by:** `jobs/pull_slime.sbatch` (`day`, 8 CPU, 32 GB). Layer cache: `~/scratch_pi_lz484/$USER/apptainer/cache`.

### Versions inside the image

| Component | Version |
|---|---|
| slime | `0b0c277d5b4c` (= v0.4.0 tag), at `/root/slime` (editable install) |
| Megatron-LM | `4b4acac9a1d2` (megatron.core 0.19.2), at `/root/Megatron-LM` |
| SGLang | 0.5.15.post1 (`0b3bb0cbe318`, slime-patched), at `/sgl-workspace/sglang` |
| torch | 2.11.0+cu129, CUDA 12.9, cuDNN 9.17.1 (pip) |
| TransformerEngine | 2.18.0 |
| flash-attn | 2.8.3 |
| NCCL | 2.30.7 (pip) |
| Ray | 2.59.0 |
| transformers | 5.12.1 |
| Python | 3.12.3 |

torch arch list: `sm_75 sm_80 sm_86 sm_90 sm_100 sm_120`. **Both H200 (sm_90) and B200 (sm_100) verified on hardware.**
Node driver on both gpu_devel H200 and B200: 580.178.04.

### `apptainer exec` template

All container jobs use `jobs/slime_common.sh`:

```bash
source jobs/slime_common.sh          # sets SLIME_SIF, SCR, HF_HOME, slime_exec()
slime_exec python my_script.py       # or: slime_exec bash my_inner_script.sh
```

which expands to:

```bash
apptainer exec --nv --cleanenv --writable-tmpfs \
    --bind /nfs/roberts/scratch \
    --env HF_HOME=$HOME/scratch_pi_lz484/$USER/hf_cache \
    --env PYTHONNOUSERSITE=1 --env PYTHONUNBUFFERED=1 \
    --env PYTHONPATH=/root/Megatron-LM \
    --env CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
    --env LD_LIBRARY_PATH=/usr/local/lib/python3.12/dist-packages/nvidia/cudnn/lib:/usr/local/lib/python3.12/dist-packages/nvidia/nccl/lib:/usr/local/cuda/lib64:/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/.singularity.d/libs \
    --env CUDA_DEVICE_MAX_CONNECTIONS=1 \
    --env SLURM_JOB_ID=$SLURM_JOB_ID \
    ~/containers/slime_v0.4.0-cu129.sif <cmd>
```

- `$HOME` is bound by default; `~/scratch_pi_lz484` is a symlink into `/nfs/roberts/scratch`, hence that bind.
- The image is read-only. slime's scripts use relative paths (`source scripts/models/...`), so `cd /root/slime`
  inside the container; write all outputs to scratch.
- Do **not** load conda or CUDA modules in container jobs. `--cleanenv` keeps host module variables out.

### Ray inside a Slurm job

The pattern used in `jobs/lib/router_freeze_test.sh` (all inside one `slime_exec bash ...`):

```bash
RAY_PORT=$((20000 + SLURM_JOB_ID % 5000)); DASH_PORT=$((RAY_PORT + 1))
ray start --head --node-ip-address 127.0.0.1 --port $RAY_PORT --dashboard-port $DASH_PORT \
    --num-gpus <N> --disable-usage-stats --temp-dir /tmp/ray_${SLURM_JOB_ID}
trap 'ray stop --force || true' EXIT
ray job submit --address=http://127.0.0.1:$DASH_PORT --runtime-env-json='{"env_vars": {"PYTHONPATH": "/root/Megatron-LM/", "CUDA_DEVICE_MAX_CONNECTIONS": "1"}}' \
    -- python3 train.py ...
```

**For real runs also pass** `--num-cpus $SLURM_CPUS_PER_TASK --object-store-memory <bytes>`: Ray auto-detected
all 48 node CPUs and a 200 GB object store in a job that had 8 CPUs / 64 GB.

## 2. afai_env (analysis / evaluation)

- Path: `~/envs/afai_env` (prefix env, 7.1 GB), Python 3.12 from conda-forge, packages from pip.
- Built by `jobs/build_afai_env.sbatch`; full pin list in `logs/afai_env_freeze.txt`.
- torch 2.11.0+cu128 (pip CUDA wheels, no CUDA module), transformers 5.19.0, safetensors 0.8.0, accelerate 1.15.0,
  numpy 2.5.4, pandas 3.0.6, matplotlib 3.11.2.
- Use in jobs:

```bash
module reset && module load miniconda && conda activate ~/envs/afai_env
export PYTHONNOUSERSITE=1 HF_HOME=$HOME/scratch_pi_lz484/$USER/hf_cache
```

Verified on H200 (`verify_afai_env` 29248780): CUDA available, arch list includes `sm_90` and `sm_100`; both
slime-converted HF checkpoints (Qwen3-0.6B round trip and the tiny MoE after frozen-router SFT) load with
`AutoModelForCausalLM` and generate. Loading 30B checkpoints: `device_map="auto"` (accelerate installed).

## 3. Verification results

All on `gpu_devel`, 1 GPU each. Scripts in `jobs/`, logs in `logs/`.

| # | Test | Job | Result |
|---|---|---|---|
| 1 | Imports, versions, CUDA, arch list, bf16 matmul, TE linear, flash-attn, cuDNN conv — H200 | `verify_container` 29247877 | PASS (with cuDNN fix, see problems) |
| 1b | Same on B200 | `verify_container_b200` 29247940 | PASS |
| 2 | SGLang serves Qwen3-0.6B, one chat + one `/generate` completion | `verify_sglang` 29248140 | PASS ("17 * 23 = 391"; "… Paris.") |
| 3 | Qwen3-30B-A3B configs, SFT support | source inspection | see §4 |
| 4a | Qwen3-0.6B HF → torch_dist → HF | `verify_convert` 29248396 | PASS: 310/310 tensors bit-identical, logits diff 0 (needs wrapper, see problems) |
| 4b | Tiny Qwen3-MoE (2 layers, 8 experts) HF → torch_dist → HF | `verify_router_freeze` 29248553 | PASS: 69/69 tensors bit-identical |
| 5 | Router freeze, end-to-end slime SFT (4 steps, lr 1e-3) on tiny MoE | `verify_router_freeze` 29248553 | PASS: frozen run router bit-identical, 67 other tensors changed; control run router changed; loss 11.96 → 7.3 in both |
| 6 | afai_env on GPU, loads slime-converted HF checkpoints | `verify_afai_env` 29248780 | PASS |

## 4. slime capabilities (v0.4.0)

**Qwen3-30B-A3B:** supported.
- Model config: `/root/slime/scripts/models/qwen3-30B-A3B.sh` (48 layers, 128 experts, top-8, softmax router).
  Same architecture for `-Base`.
- GRPO example: `/root/slime/scripts/run-qwen3-30B-A3B.sh`, 1 node × 8 GPUs, TP4 / EP8 / PP1, `--colocate`,
  `--rollout-num-gpus-per-engine 8`. Docs: `/root/slime/docs/en/examples/qwen3-30B-A3B.md`.
- Conversion of the 30B model per the docs uses `torchrun --nproc-per-node 8 tools/convert_hf_to_torch_dist.py`.
- Moonlight-16B-A3B fallback: `scripts/models/moonlight.sh`, `scripts/run-moonlight-16B-A3B.sh`.

**SFT:** supported, same trainer: `--loss-type sft_loss --rollout-function-path slime.rollout.sft_rollout.generate_rollout
--prompt-data <parquet with messages column> --input-key messages --calculate-per-token-loss
--disable-compute-advantages-and-returns --debug-train-only` (no SGLang engines started).
Examples: `run-qwen3-4B-base-sft.sh`, `run-qwen3-235B-A22B-sft.sh`, `run-qwen3.5-35B-A3B-sft.sh`.
No ready-made Qwen3-30B-A3B SFT script; combine the 30B model/parallel args with the SFT args above
(verified working on the tiny MoE in `jobs/lib/router_freeze_test.sh`).

**HF ↔ torch_dist:**
- HF → torch_dist: `cd /root/slime && source scripts/models/<model>.sh && python tools/convert_hf_to_torch_dist.py ${MODEL_ARGS[@]} --hf-checkpoint HF --save OUT`
  (writes `OUT/release/`).
- torch_dist → HF: **use `jobs/lib/convert_torch_dist_to_hf.py`** (same CLI as slime's tool):
  `python $PROJ/jobs/lib/convert_torch_dist_to_hf.py --input-dir CKPT/iter_0000003 --output-dir OUT_HF --origin-hf-dir ORIG_HF --vocab-size 151936`
- Alternative for training runs: `--save-hf <path with {rollout_id}>` makes slime write HF checkpoints during training (not yet tested here).

**Router freezing:** `--freeze-params-name-list 'mlp\.router\.'`
- slime regex-matches (`re.search`) Megatron parameter names and sets `requires_grad=False` before the optimizer is built
  (`slime/backends/megatron_utils/model_provider.py::freeze_model_params`).
- Megatron router names: `decoder.layers.N.mlp.router.weight` (Qwen3-MoE has no router bias / expert_bias).
  In HF they are `model.layers.N.mlp.gate.weight`.
- The inverse option `--only-train-params-name-list` also exists (mutually exclusive).
- Only the trainer side is affected; SGLang rollout routing still uses the (unchanged) router weights it is synced.

## 4b. Qwen3-30B-A3B-Base: data, checkpoints, launcher

| What | Path (scratch, purged after 30 days) | Made by |
|---|---|---|
| HF base model (57 GB, 16 shards) | `~/scratch_pi_lz484/$USER/models/Qwen3-30B-A3B-Base` | `jobs/download_30b.sbatch` |
| GRPO data / eval | `.../data/dapo-math-17k/dapo-math-17k.jsonl`, `.../data/aime-2024/aime-2024.jsonl` | same |
| Megatron torch_dist (`--ref-load`) | `~/scratch_pi_lz484/$USER/ckpt/Qwen3-30B-A3B-Base_torch_dist` (57 GB) | `jobs/convert_30b.sbatch` |

Conversion of the 30B model on **one** H200 (job 29263777, 18 min): HF → torch_dist → HF round trip, 18867/18867
tensors bit-identical, logits diff 0. Download needs `--mem=64G` (16G was OOM-killed by hf_xet).

**Launcher:** `jobs/lib/run_30b.sh {sft|grpo} RUN_DIR [extra train.py args]`, run inside the container. Settings via
env vars (passed with `slime_exec env VAR=... bash jobs/lib/run_30b.sh ...` because of `--cleanenv`):
`FREEZE_ROUTER=1`, `PROMPT_DATA`, `NUM_ROLLOUT`, `NUM_EPOCH`, `ROLLOUT_BS`, `N_SAMPLES`, `GLOBAL_BS`, `MAX_RESP_LEN`,
`SAVE_INTERVAL`, `LR`, `MAX_TOKENS_PER_GPU`, `RAY_CPUS`, `RAY_OBJ_STORE_GB`. Parallelism follows slime's example
(TP4/EP8 on 8 GPUs) and scales EP to the GPU count. Example job: `jobs/smoke_30b.sbatch`
(`-p gpu_h200,gpu_b200 --gpus=8 -c 48 --mem=1000G`; GPU partitions allow at most 8 CPUs per GPU).

## 4c. 30B full-node smoke test (job 29263778, 8× H200, 2026-10-13, ~45 min)

`jobs/smoke_30b.sbatch`, router frozen, on Qwen3-30B-A3B-Base:

| Part | Result |
|---|---|
| SFT, 2 steps (TP4/EP8, CPU-offloaded Adam) | PASS: trains (loss 1.47 → 1.03); all 48 router weights bit-identical to base, attention changed; `--save-hf` export bit-identical to torch_dist → HF conversion (18,867/18,867 tensors) |
| GRPO, 2 rollouts (SGLang colocated, 8 × 4 samples, ≤1024 tokens) | Mechanics PASS (rollout, train step, save, `--save-hf` identical), **but every reward was 0**, so advantages were 0 and the weights did not change |

**Why the rewards were 0:** `--rm-type deepscaler` (slime's example default, used in `run_30b.sh grpo`) returns 0 unless the
response contains `</think>`. A **base** model does not emit `<think>…</think>`, so GRPO on the raw base model with this
reward learns nothing. For GRPO either start from a checkpoint that produces `<think>` (e.g. after the OpenR1 SFT), or use a
reward that does not need it: `--rm-type math`, `dapo`, or `boxed_math` (grade only the `\boxed{}` answer).

## 5. Problems found and fixes

1. **Image pull stalled when `APPTAINER_TMPDIR` was on NFS scratch** (job 29229347: >50 min extracting, ~1 MB/s).
   Fix: `APPTAINER_TMPDIR=/tmp/apptainer_tmp_$SLURM_JOB_ID` (node-local, 1.6 TB), cache still on scratch.
   Pull then took ~47 min total. The stalled job was cancelled and its temp dir removed.
2. **cuDNN version mismatch inside the image.** The image contains an old system cuDNN 9.10.2
   (`/usr/lib/x86_64-linux-gnu/libcudnn.so.9.10.2`) next to torch's pip cuDNN 9.17.1. After TE etc. are imported
   the system copy can win and `torch.backends.cudnn` raises "PyTorch was compiled against (9, 17, 1) but found
   runtime version (9, 10, 2)". Not caused by Apptainer (`--nv` injects no cuDNN). Fix: pip cuDNN first on
   `LD_LIBRARY_PATH` (built into `slime_exec`). `jobs/lib/gpu_check.py` fails loudly if this regresses.
3. **slime's `tools/convert_torch_dist_to_hf.py` is broken with this image's Megatron.** Megatron `4b4acac` stores
   the checkpoint's common state (incl. `args`) inside the `.distcp` files ("common_state" ShardedObject) and no
   longer writes `common.pt`; the slime tool (v0.4.0 and `main` as of 2026-10-12) still reads `common.pt` →
   `FileNotFoundError`. Fix: `jobs/lib/convert_torch_dist_to_hf.py` wraps the unmodified tool and reads the embedded
   common state when `common.pt` is absent. `tools/convert_torch_dist_to_hf_parallel.py` has the same bug
   (not wrapped yet). Not reported upstream.
4. **slime example run scripts begin with `pkill -9 sglang/ray/python`.** Never copy that preamble on shared nodes.
5. **Ray resource auto-detection** sees the whole node (see §1 "Ray inside a Slurm job").
6. **`module load miniconda`** creates `~/.conda/envs -> /nfs/roberts/project/pi_lz484/wb379/ycrc_conda/envs` (full
   quota); since that mkdir fails, the module warns on every load that `~/.conda/envs` is in home. Harmless while
   it stays empty: always use `--prefix` envs. `~/.conda/pkgs` points at scratch (fine).
7. Harmless noise: `SyntaxWarning: invalid escape sequence` from Megatron/torchao, modelopt warning about
   transformers 5.12.1, "GPTModel IS DEPRECATED".
8. Files written from inside the container show group `nogroup` in container `ls` (unmapped GID display).
9. **`ray stop --force` can crash inside the container** (psutil `os.pidfd_open`: `OSError: [Errno 22]`) before it
   stops the GCS server. Seen after a GRPO run that stopped for resume, when slime keeps the SGLang serving processes
   alive (job 29877033). Harmless for a Slurm job that runs Ray once (Slurm kills leftovers at job end). A second
   `ray start` in the same job reuses the port and attaches to the old GCS ("Session name ... does not match
   persisted value"). Fix: run each Ray session in its own PID namespace, `SLIME_APPTAINER_FLAGS=--pid slime_exec ...`
   (used by `jobs/test_grpo_ckpt_tiny.sbatch`).

## 6. Storage used

- Home (125 GB quota, backed up): SIF 18 GB + afai_env 7.1 GB.
- Scratch (purged after 30 days): `apptainer/cache` (image layers, ~22 GB, can be deleted any time; only needed to
  re-pull), `hf_cache`, `models/Qwen3-0.6B`, `verify/` (test outputs, all disposable), `pip_cache`, `conda_pkgs`.

## 7. Reproducing this setup (another account on Bouchet)

Everything is per-user: scripts use `$HOME` and `SCR=${SCR:-$HOME/scratch_pi_lz484/$USER}`. If your scratch lives
elsewhere (e.g. a different PI group), submit with `SCR=/your/scratch/dir sbatch ...` (sbatch exports it).

1. Get the code into your home (code lives in home; it is backed up): `git clone <repo URL> ~/afai_project && cd ~/afai_project`
2. Container (identical image, pinned by digest; ~50 min): `sbatch jobs/pull_slime.sbatch`
   → `~/containers/slime_v0.4.0-cu129.sif`. The file's sha256 can differ from the one above (SIF build metadata);
   the image contents are what the digest pins. Check with step 4.
3. afai_env (exact pins from `envs/afai_env.requirements.txt`; ~15 min): `sbatch jobs/build_afai_env.sbatch`
   → `~/envs/afai_env`. Never `conda create -n` (see problem 6).
4. Verify (1 GPU each, a few minutes): `sbatch jobs/verify_container.sbatch`, then `verify_sglang`, `verify_convert`,
   `verify_router_freeze`, `verify_afai_env` (the last one reads outputs of `verify_convert`/`verify_router_freeze`).
   Compare the versions/commits printed by `verify_container` with §1.
5. 30B model and data: `D=$(sbatch --parsable jobs/download_30b.sbatch); sbatch --dependency=afterok:$D jobs/convert_30b.sbatch`
   (~25 min total). If you are in the same group, you can instead point runs at an existing copy with
   `HF_CKPT=... REF_LOAD=...` (group scratch dirs are group-readable).
6. Runs: see `jobs/smoke_30b.sbatch` and `jobs/lib/run_30b.sh`.
