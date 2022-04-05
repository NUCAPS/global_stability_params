#!/usr/bin/env bash

declare -a dates=$(seq 20210815 20210823)
dates+=$(seq 20210917 20210924)

declare -a casenames=("tc-henri-m01" "tc-henri-m02" "tc-odette-m01" "tc-odette-m02")

for casename in ${casenames[@]}; do
# create intermediate file dirs
  cd /home/rebekah/stability/derived
  mkdir $casename
  cd /home/rebekah/stability/gridded
  mkdir $casename
  cd /home/rebekah/stability

  # Calc derived fields
  for date in ${dates[@]}; do
    python demo_files.py --date "${date}" --casename "${casename}" --indir 'data_single'
  done
done


# Calc gridded files
for casename in ${casenames[@]}; do
  for date in ${dates[@]}; do
    echo python gridding.py --date "${date}" --casename "${casename}"
  done
done

# ignore this for now....
# casename="flights"
# python demo_files_matchups_snd.py --casename "${casename}" --indir 'data_single'
# demo_files_matchups_snd.py --casename "flights" --indir 'data_single'
# demo_files_matchups.py --casename "flights" --indir 'data_single'
