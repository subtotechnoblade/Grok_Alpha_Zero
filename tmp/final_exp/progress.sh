#!/bin/sh
# compact arm progress: generations completed + last self-play progress line
cd "$(dirname "$0")/../.." || exit 1
for a in "$@"; do
  d="tmp/final_exp/$a/Grok_Zero_Train"
  [ -d "$d" ] || { echo "$a: not started"; continue; }
  gens=$(ls "$d" 2>/dev/null | grep -E '^[0-9]+$' | sort -n | tr '\n' ' ')
  data=$(ls "$d"/*/Self_Play_Data.h5 2>/dev/null | wc -l)
  last=$(tr '\r' '\n' < "tmp/final_exp/$a.log" 2>/dev/null | grep -E 'Generating self play games|Epoch [0-9]+/|Generation:' | tail -1)
  sz=$(du -sh "tmp/final_exp/$a.log" 2>/dev/null | cut -f1)
  echo "$a | gen dirs: $gens | h5 files: $data | log $sz"
  echo "    $last"
done
