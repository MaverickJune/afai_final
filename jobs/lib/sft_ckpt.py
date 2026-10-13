"""Checkpoint schedule for long slime SFT runs that span several Slurm jobs.

Replaces slime's `--save-interval` schedule (which also forces a full save at every epoch boundary) with:

  * N weights-only checkpoints in HF format (bf16 safetensors, no optimizer state) at log-spaced optimizer
    steps over the WHOLE run: round(T ** (i / (N - 1))) for i = 0..N-1, T = total steps. More saves early.
  * ONE full resumable Megatron checkpoint (model + optimizer + scheduler + RNG + data cursor), written
      - when the Slurm job is close to its time limit (then training stops cleanly; resubmit to resume), or
      - at the final step.
    After a full save succeeds, older full checkpoints of the run are deleted, so only one exists.

Two entry points, both inside the slime container:

  Driver:  TRAIN_ENTRY=<this file>  (run_30b.sh runs `python3 $TRAIN_ENTRY <train.py args>` from /root/slime)
  Actors:  --custom-megatron-init-path sft_ckpt.actor_init
           Replaces the module-level `save` / `save_hf_model_to_path` used by MegatronTrainRayActor.save_model.
           (Patching the method itself is unreliable: Ray binds actor methods when the class is loaded.)

The driver writes $SFT_CKPT_DIR/save_requests/<rollout_id>.json before every save; all actor ranks read the same
file, so collective saves stay consistent across ranks.

Environment (forward to Ray with PASS_ENV):
  SFT_CKPT_DIR           run directory for requests, hf_ckpts/, ckpt_log.jsonl, STATUS        (required)
  SFT_N_LOG_CKPTS        number of weights-only checkpoints (default 10)
  JOB_END_EPOCH          unix time when the Slurm job is killed (time-based full save off if unset)
  SFT_SAVE_MARGIN_MIN    minutes reserved for the full save + shutdown (default 45)
  SFT_STOP_AFTER_STEP    testing only: force full save + stop after this optimizer step
"""

import json
import os
import shutil
import sys
import time
from pathlib import Path


class StopForResume(Exception):
    """Raised after the time-limit full checkpoint, to end this job cleanly."""


def log_spaced_steps(total_steps: int, n: int) -> list[int]:
    """n distinct 1-based optimizer steps in [1, total_steps], geometrically spaced, always incl. the last."""
    if total_steps <= n:
        return list(range(1, total_steps + 1))
    steps = {max(1, round(total_steps ** (i / (n - 1)))) for i in range(n)}
    k = 1
    while len(steps) < n:  # rounding collisions at the low end: fill with the smallest unused steps
        steps.add(k)
        k += 1
    return sorted(steps)


def _ckpt_dir() -> Path:
    return Path(os.environ["SFT_CKPT_DIR"])


def _request_path(rollout_id: int) -> Path:
    return _ckpt_dir() / "save_requests" / f"{rollout_id}.json"


def _read_request(rollout_id: int) -> dict:
    return json.loads(_request_path(rollout_id).read_text())


# ----------------------------------------------------------------------------------------------- actor side
def actor_init(args):
    """--custom-megatron-init-path hook, runs once in every training actor process."""
    from slime.backends.megatron_utils import actor as actor_mod

    if getattr(actor_mod, "_sft_ckpt_patched", False):
        return
    orig_save = actor_mod.save
    orig_save_hf = actor_mod.save_hf_model_to_path

    def save(iteration, model, optimizer, opt_param_scheduler):
        if _read_request(iteration)["full"]:
            orig_save(iteration, model, optimizer, opt_param_scheduler)

    def save_hf_model_to_path(args, output_dir, model, **kwargs):
        hf_dir = _read_request(_current_rollout_id(output_dir))["hf_dir"]
        if hf_dir:
            orig_save_hf(args, Path(hf_dir), model, **kwargs)

    actor_mod.save = save
    actor_mod.save_hf_model_to_path = save_hf_model_to_path
    actor_mod._sft_ckpt_patched = True


def _current_rollout_id(output_dir) -> int:
    # save_model formats --save-hf with rollout_id; we set --save-hf to ".../_rollout_{rollout_id}".
    return int(str(output_dir).rsplit("_rollout_", 1)[1])


# ---------------------------------------------------------------------------------------------- driver side
class _Schedule:
    def __init__(self):
        self.n_log = int(os.environ.get("SFT_N_LOG_CKPTS", "10"))
        self.job_end = float(os.environ["JOB_END_EPOCH"]) if os.environ.get("JOB_END_EPOCH") else None
        self.margin = 60 * float(os.environ.get("SFT_SAVE_MARGIN_MIN", "45"))
        self.stop_after = int(os.environ["SFT_STOP_AFTER_STEP"]) if os.environ.get("SFT_STOP_AFTER_STEP") else None
        self.log_steps = None
        self.last_time = time.time()
        self.max_step_time = 0.0
        self.full_save_time = None

    def _log(self, record):
        record["time"] = time.strftime("%Y-%m-%d %H:%M:%S")
        print("[sft_ckpt]", json.dumps(record), flush=True)
        with open(_ckpt_dir() / "ckpt_log.jsonl", "a") as f:
            f.write(json.dumps(record) + "\n")

    def decide(self, args, rollout_id):
        if self.log_steps is None:
            self.log_steps = log_spaced_steps(args.num_rollout, self.n_log)
            self._log({"event": "schedule", "total_steps": args.num_rollout, "log_steps": self.log_steps,
                       "start_rollout_id": args.start_rollout_id, "job_end_epoch": self.job_end})
        now = time.time()
        self.max_step_time = max(self.max_step_time, now - self.last_time)
        self.last_time = now
        step = rollout_id + 1
        last = rollout_id == args.num_rollout - 1
        want_hf = step in self.log_steps
        time_up = False
        if self.job_end is not None:
            # Room for: up to two more steps, the full save itself, and shutdown margin.
            needed = 2 * self.max_step_time + (self.full_save_time or 0) + self.margin
            time_up = self.job_end - now < needed
        if self.stop_after is not None and step >= self.stop_after:
            time_up = True
        return step, want_hf, last or time_up, (time_up and not last)


def _prune_full_checkpoints(save_dir: Path, keep_rollout_id: int):
    keep = f"iter_{keep_rollout_id:07d}"
    for d in save_dir.glob("iter_*"):
        if d.is_dir() and d.name != keep:
            shutil.rmtree(d)
    for f in (save_dir / "rollout").glob("global_dataset_state_dict_*.pt"):
        if f.name != f"global_dataset_state_dict_{keep_rollout_id}.pt":
            f.unlink()


def main():
    sys.path.insert(0, "/root/slime")
    import train as slime_train
    from slime.observability.logging_utils import configure_logger
    from slime.utils.arguments import parse_args

    args, restore_plan = parse_args(return_restore_plan=True)
    configure_logger()
    assert args.eval_interval is None, "sft_ckpt drives saving through the save hook; eval is not supported"
    assert args.save_interval is not None and args.save_hf and "_rollout_{rollout_id}" in args.save_hf
    assert args.custom_megatron_init_path == "sft_ckpt.actor_init"
    (_ckpt_dir() / "save_requests").mkdir(parents=True, exist_ok=True)
    (_ckpt_dir() / "hf_ckpts").mkdir(parents=True, exist_ok=True)

    sched = _Schedule()
    orig_save_checkpoint = slime_train.save_checkpoint

    # train() asks should_run_periodic_action(rollout_id, args.save_interval, ...) whether to save; answer yes
    # every step and let save_checkpoint below decide. (eval_interval is None, so eval stays off.)
    slime_train.should_run_periodic_action = lambda rollout_id, interval, *a, **k: interval is not None

    def save_checkpoint(args, rollout_id, actor_model, critic_model, rollout_manager, **kwargs):
        step, want_hf, want_full, stop = sched.decide(args, rollout_id)
        if not (want_hf or want_full):
            return
        hf_dir = str(_ckpt_dir() / "hf_ckpts" / f"step_{step:05d}") if want_hf else None
        _request_path(rollout_id).write_text(json.dumps({"step": step, "hf_dir": hf_dir, "full": want_full}))
        t0 = time.time()
        orig_save_checkpoint(args, rollout_id, actor_model, critic_model, rollout_manager, **kwargs)
        elapsed = time.time() - t0
        if want_full:
            sched.full_save_time = elapsed
            _prune_full_checkpoints(Path(args.save), rollout_id)
        sched._log({"event": "save", "step": step, "rollout_id": rollout_id, "hf_dir": hf_dir,
                    "full": want_full, "full_dir": str(Path(args.save) / f"iter_{rollout_id:07d}") if want_full else None,
                    "seconds": round(elapsed, 1), "max_step_seconds": round(sched.max_step_time, 1)})
        sched.last_time = time.time()  # do not count save time as step time
        if stop:
            (_ckpt_dir() / "STATUS").write_text(f"needs_resume after step {step}\n")
            raise StopForResume(step)

    slime_train.save_checkpoint = save_checkpoint
    try:
        slime_train.main(args, restore_plan)
    except StopForResume as e:
        print(f"[sft_ckpt] stopped after step {e.args[0]} with a full checkpoint; resubmit the same job to resume",
              flush=True)
        return
    (_ckpt_dir() / "STATUS").write_text(f"done ({args.num_rollout} steps)\n")
    print("[sft_ckpt] training finished", flush=True)


if __name__ == "__main__":
    main()
