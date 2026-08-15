#!/usr/bin/env bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate WSL_tensorflow_general
cd /mnt/c/Users/TryHarder/Desktop/Pythoncode/Grok_Zero
for g in 1 2 3; do
    echo "########## model gen $g on its own held-out games ##########"
    python tmp/analyze_horizon.py "$g" "$g" 2>&1 |
        grep -vE 'cuda|cuFFT|cuDNN|cuBLAS|absl|oneDNN|gpu_device|XLA|device_compiler|TF-TRT|Assert'
done
