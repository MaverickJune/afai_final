#!/bin/bash
# vscode_node.sh -- get a small compute node for VS Code Remote-SSH (code editing only; no GPU).
#
#   ./vscode_node.sh            start (or reuse) a node for 4 h and print its name
#   ./vscode_node.sh 8          same, for 8 hours
#   ./vscode_node.sh status     show the current VS Code node, if any
#   ./vscode_node.sh stop       release it (scancel)
#
# How it works: submits a batch job that only sleeps, so the allocation stays alive without keeping a terminal
# open (unlike salloc). Then connect VS Code (Remote-SSH) to the printed node; you can ssh to a node while you
# have a job on it. The connection ends when the job ends (time limit or `./vscode_node.sh stop`).
#
# Size: 2 CPUs / 8 GB is plenty for the editor + language servers + git. It also leaves room in your per-user
# CPU allowance on `devel` for CPU jobs such as jobs/prep_openr1_sft.sbatch (devel limits CPUs per user).
# Do not run training or model loading on this node; submit those with sbatch to GPU partitions.

PARTITION=devel
CPUS=2
MEM=8G
NAME=vscode
PROJ=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

current() {   # job id of a running/pending VS Code job, if any
    squeue --me -h -n "$NAME" -o "%i" | head -1
}

print_node() {   # $1 = job id; waits until running, then prints the node
    local id=$1 state node reason
    echo "job $id: waiting for a node ..."
    while true; do
        read -r state node reason <<< "$(squeue -h -j "$id" -o "%T %N %r")"
        [ -z "$state" ] && { echo "job $id is gone (ended or cancelled)."; exit 1; }
        if [ "$state" = RUNNING ]; then break; fi
        echo "  $(date +%H:%M:%S) $state ($reason)"
        sleep 10
    done
    local left; left=$(squeue -h -j "$id" -o "%L")
    echo
    echo "VS Code node: $node     (job $id, time left $left)"
    echo
    echo "In VS Code: Remote-SSH -> Connect to Host -> $node"
    echo "  (needs an ~/.ssh/config entry on your laptop that reaches compute nodes through the login node, e.g."
    echo "     Host ${node%%n*}*"
    echo "         ProxyJump <your Bouchet login host>"
    echo "         User $USER"
    echo "   )"
    echo "Release it when done:  $0 stop"
}

case "${1:-}" in
    status)
        id=$(current)
        if [ -z "$id" ]; then echo "no VS Code node job"; exit 0; fi
        squeue -h -j "$id" -o "job %i  %T  node %N  elapsed %M / %l  left %L  %r" ;;
    stop)
        id=$(current)
        if [ -z "$id" ]; then echo "no VS Code node job"; exit 0; fi
        scancel "$id" && echo "released job $id" ;;
    ""|[0-9]*)
        hours=${1:-4}
        id=$(current)
        if [ -n "$id" ]; then
            echo "reusing existing VS Code job $id"
        else
            mkdir -p "$PROJ/logs"
            if ! id=$(sbatch --parsable -p "$PARTITION" -c "$CPUS" --mem="$MEM" -t "${hours}:00:00" -J "$NAME" \
                --output="$PROJ/logs/vscode_%j.out" --wrap "echo \"VS Code node: \$(hostname)\"; sleep infinity" 2>&1); then
                echo "sbatch failed: $id"
                if [[ "$id" == *QOS* ]]; then
                    echo "-> $PARTITION limits jobs/CPUs per user. Your current $PARTITION jobs:"
                    squeue --me -h -p "$PARTITION" -o "   %i %20j %C CPUs %T"
                    echo "   Wait for one to finish, or end one (e.g. an old interactive session) with scancel JOBID."
                fi
                exit 1
            fi
            echo "submitted job $id ($PARTITION, $CPUS CPUs, $MEM, ${hours} h)"
        fi
        print_node "$id" ;;
    -h|--help|help) sed -n '2,16p' "$0" ;;
    *) echo "unknown argument '$1'"; sed -n '2,16p' "$0"; exit 1 ;;
esac
