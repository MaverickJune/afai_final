#!/bin/bash
# jobstat.sh -- track this project's Slurm jobs. Run from anywhere on Bouchet (login node is fine: read-only queries).
#
#   ./jobstat.sh                 list my jobs: id, name, what it is, state, partition, GPUs, elapsed / limit,
#                                time left (running) or estimated start (pending), wait reason, log file
#   ./jobstat.sh start           Slurm's estimated start times of pending jobs
#   ./jobstat.sh info  JOBID     full details of one job (description, dependencies, resources, log path)
#   ./jobstat.sh log   JOBID     last 40 lines of the job's log      (./jobstat.sh log JOBID -f   to follow it)
#   ./jobstat.sh hist  [DAYS]    finished/failed/cancelled jobs of the last DAYS days (default 2), with exit codes
#   ./jobstat.sh eff   JOBID     CPU / memory efficiency of a finished job (seff)
#   ./jobstat.sh sft             progress of the OpenR1 SFT run: steps done / 976, loss, step time, ETA, checkpoints
#   ./jobstat.sh watch [SEC]     refresh the job list every SEC seconds (default 60; Ctrl-C to stop)
#
# Cancel a job yourself with:  scancel JOBID

PROJ=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
LOGS=$PROJ/logs
SCR=${SCR:-$HOME/scratch_pi_lz484/$USER}
SFT_RUN=${SFT_RUN:-$SCR/runs/sft_openr1math_idx3k1_qwen3_30b_a3b_base}
SFT_TOTAL_STEPS=976

# One-line description of a job: first comment line after the #SBATCH header of its script.
describe() {   # $1 = command path from squeue/sacct, $2 = job name
    local cmd=$1 name=$2 f
    if [ "$name" = interactive ]; then echo "interactive session (salloc / VS Code)"; return; fi
    for f in "$cmd" "$PROJ/jobs/$name.sbatch" "$PROJ/jobs/${name%_b200}.sbatch"; do
        if [ -f "$f" ] && [[ "$f" == *.sbatch ]]; then
            grep -m1 -E '^# [^ ]' "$f" | sed 's/^# //' | cut -c1-100
            return
        fi
    done
    echo "-"
}

logfile() {    # $1 = job id
    ls "$LOGS"/*_"$1".out 2>/dev/null | head -1
}

human_reason() {
    case "$1" in
        None|"") echo "" ;;
        Priority) echo "waiting: other jobs have higher priority" ;;
        Resources) echo "waiting: next in line, resources being freed" ;;
        Dependency) echo "waiting for the job it depends on" ;;
        DependencyNeverSatisfied) echo "dependency FAILED -> will never run (Slurm removes it)" ;;
        QOSMaxCpuPerUserLimit) echo "waiting: your other jobs use your CPU allowance" ;;
        QOSMaxGRESPerUser|QOSMaxGRESPerUserLimit) echo "waiting: your other jobs use your GPU allowance" ;;
        Nodes*DOWN*|*DRAINED*|*reserved*) echo "waiting: suitable nodes busy/reserved" ;;
        *) echo "$1" ;;
    esac
}

list() {
    local now; now=$(date +%s)
    printf "%-9s %-20s %-8s %-11s %-5s %-25s %-17s %s\n" JOBID NAME STATE PARTITION GPUS "ELAPSED / LIMIT" "LEFT | EST.START" "WHAT / WHY WAITING / LOG"
    squeue --me -h -o "%i|%j|%T|%P|%M|%l|%L|%S|%r|%o" | sort -t'|' -k1,1n | while IFS='|' read -r id name st part el lim left start reason cmd; do
        local gpus; gpus=$(squeue -h -j "$id" -O tres-alloc:300 | grep -oE 'gres/gpu=[0-9]+' | head -1 | cut -d= -f2)
        gpus=${gpus:-0}
        local when
        if [ "$st" = RUNNING ]; then when="left $left"
        elif [ "$start" != N/A ] && [ -n "$start" ]; then when="start ${start/T/ }"; when=${when:0:22}
        elif [[ "$reason" == Dependency* ]]; then when="after dependency"
        else when="start ?"; fi
        printf "%-9s %-20s %-8s %-11s %-5s %-25s %-17s %s\n" "$id" "${name:0:20}" "${st:0:8}" "${part:0:11}" "$gpus" "$el / $lim" "$when" "$(describe "$cmd" "$name")"
        local why; why=$(human_reason "$reason")
        [ -n "$why" ] && printf "%100s %s\n" "" "-> $why"
        local lf; lf=$(logfile "$id"); [ -n "$lf" ] && printf "%100s %s\n" "" "log: ${lf#$PROJ/}"
    done
    return 0
}

case "${1:-list}" in
    list) list ;;
    start) squeue --me --start -o "%.10i %.22j %.12P %.20S %.10l %r" ;;
    info)
        id=${2:?usage: jobstat.sh info JOBID}
        cmd=$(squeue -h -j "$id" -o "%o" 2>/dev/null); name=$(squeue -h -j "$id" -o "%j" 2>/dev/null)
        [ -z "$name" ] && { name=$(sacct -j "$id" -X -n -P -o JobName | head -1); cmd=""; }
        echo "what: $(describe "$cmd" "$name")"
        echo "log:  $(logfile "$id")"
        scontrol show job "$id" 2>/dev/null | grep -E "JobName|JobState|Dependency|RunTime|TimeLimit|StartTime|EndTime|Partition|NodeList|TRES|Command" \
            || sacct -j "$id" -o JobID,JobName,Partition,State,Elapsed,Start,End,ExitCode,MaxRSS ;;
    log)
        id=${2:?usage: jobstat.sh log JOBID [-f]}; f=$(logfile "$id")
        [ -z "$f" ] && { echo "no log for $id in $LOGS (not started yet?)"; exit 1; }
        echo "== $f"
        if [ "${3:-}" = -f ]; then tail -n 40 -f "$f"; else tail -n 40 "$f"; fi ;;
    hist)
        days=${2:-2}
        sacct -u "$USER" -S "now-${days}days" -X -o JobID%10,JobName%22,Partition%12,State%22,Elapsed%11,Start%20,End%20,ExitCode%8 ;;
    eff) seff "${2:?usage: jobstat.sh eff JOBID}" ;;
    sft)
        echo "run dir: $SFT_RUN"
        echo "STATUS:  $(cat "$SFT_RUN/STATUS" 2>/dev/null || echo 'not finished / not started')"
        [ -f "$SFT_RUN/jobs.log" ] && { echo "--- Slurm jobs of this run"; cat "$SFT_RUN/jobs.log"; }
        f=$(ls -t "$LOGS"/sft_openr1_30b_*.out 2>/dev/null | head -1)
        if [ -n "$f" ]; then
            echo "--- latest log: ${f#$PROJ/}"
            grep -oE "\[[0-9-]+ [0-9:]+\] model.py:[0-9]+ - step [0-9]+: \{'train/loss': [0-9.e+-]+" "$f" | \
            sed -E "s/\[([0-9-]+ [0-9:]+)\].*step ([0-9]+): \{'train\/loss': ([0-9.e+-]+)/\1|\2|\3/" | \
            awk -F'|' -v total=$SFT_TOTAL_STEPS '
                { cmd = "date -d \"" $1 "\" +%s"; cmd | getline t; close(cmd)
                  n++; ts[n] = t; st[n] = $2; loss[n] = $3 }
                END {
                    if (n == 0) { print "no training steps logged yet"; exit }
                    k = (n > 20 ? 20 : n - 1)
                    printf "steps done: %d / %d   last loss: %.4f   (logged at step %d)\n", st[n] + 1, total, loss[n], st[n]
                    printf "recent losses:"; for (i = (n > 5 ? n - 4 : 1); i <= n; i++) printf " %.4f", loss[i]; print ""
                    if (k > 0) {
                        dt = (ts[n] - ts[n - k]) / k; left = (total - st[n] - 1) * dt
                        printf "avg step time (last %d): %.0f s   ETA for remaining %d steps: %.1f h (excl. saves/requeue)\n", k, dt, total - st[n] - 1, left / 3600
                    }
                }'
        else
            echo "no sft_openr1_30b log yet"
        fi
        echo "--- checkpoints"
        ls "$SFT_RUN/hf_ckpts" 2>/dev/null | sed 's/^/  HF weights-only: /'
        ls -d "$SFT_RUN"/ckpt/iter_* 2>/dev/null | sed 's/^/  full resumable:  /'
        [ -f "$SFT_RUN/ckpt_log.jsonl" ] && { echo "--- last save events"; tail -n 3 "$SFT_RUN/ckpt_log.jsonl"; }; true ;;
    watch)
        while true; do clear; date; list; sleep "${2:-60}"; done ;;
    -h|--help|help) sed -n '2,15p' "${BASH_SOURCE[0]}" ;;
    *) echo "unknown command '$1'"; sed -n '2,15p' "${BASH_SOURCE[0]}"; exit 1 ;;
esac
