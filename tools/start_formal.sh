#!/bin/zsh
# Start (or resume) the formal runs of Step 5 from this folder, kept awake, with a log.
# Run from the root of the frozen step5 worktree:  zsh tools/start_formal.sh
# The exit status is the launcher's, not the log's: PIPE_FAIL makes a failed launcher fail the whole pipeline.
# tee -i ignores Ctrl-C, so the log keeps what the launcher writes while it stops its workers.
setopt PIPE_FAIL
cd "$(dirname "$0")/.." || exit 1
log="${FORMAL_LOG:-outputs/search/formal/launch.log}"
mkdir -p "$(dirname "$log")" || exit 1
caffeinate -dims /opt/miniconda3/envs/gphh/bin/python -u tools/run_formal.py "$@" 2>&1 | tee -i -a "$log"
