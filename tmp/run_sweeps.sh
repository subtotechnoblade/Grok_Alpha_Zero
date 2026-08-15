#!/usr/bin/env bash
# Runs the sweeps back to back, restarting any that the WSL dxgkrnl driver kills
# (dxgvmbus.c:3095 field-spanning-write oops). Each script resumes from its CSV, so a
# restart only reruns the arm that was in flight.
#
# no `set -u`: this env's activate.d/activate-binutils_linux-64.sh reads ADDR2LINE
# before setting it, so nounset aborts conda activate itself.
source ~/miniconda3/etc/profile.d/conda.sh
conda activate WSL_tensorflow_general
cd /mnt/c/Users/TryHarder/Desktop/Pythoncode/Grok_Zero

for s in "$@"; do
    for i in 1 2 3 4 5 6; do
        echo "=== $s attempt $i ==="
        if stdbuf -oL -eL python "tmp/$s.py" >> "tmp/$s.log" 2>&1; then
            echo "=== $s complete ==="
            break
        fi
        echo "=== $s died, resuming ==="
    done
done
echo ALLDONE
