set -e
cd /mnt/c/Users/TryHarder/Desktop/Pythoncode/Grok_Zero
source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null
conda activate WSL_tensorflow_general
while pgrep -f step2_reachfit.py > /dev/null; do sleep 20; done
echo "=== step2 finished, starting step3 weight ==="
python -u tmp/converge_audit/step3_arms.py run2 12 weight  > tmp/converge_audit/step3_weight_run2.log 2>&1
echo "=== step3 weight done ==="
python -u tmp/converge_audit/step3_arms.py run2 12 window  > tmp/converge_audit/step3_window_run2.log 2>&1
echo "=== step3 window done ==="
