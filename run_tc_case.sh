#!/usr/bin/env bash

declare -a dates=$(seq 20220501 20220630)
# dates+=$(seq 20210917 20210924)

declare -a casenames=("hwt_2022")

for casename in ${casenames[@]}; do
# create intermediate file dirs
  cd /home/rebekah/stability/sfc_vals
  mkdir $casename
  cd /home/rebekah/stability/gridded
  mkdir $casename
  cd /home/rebekah/stability

  # Calc derived fields
  for date in ${dates[@]}; do
    # python demo_files.py --date "${date}" --casename "${casename}" --indir '/mnt/nucaps-s3/stability/' --outdir 'sfc_vals'
    python surface_temps.py --date "${date}" --casename "${casename}" --indir '/mnt/nucaps-s3/stability/' --outdir 'sfc_vals'
  done
done

# Calc gridded files
# for casename in ${casenames[@]}; do
#   for date in ${dates[@]}; do
#     echo python gridding.py --date "${date}" --casename "${casename}"
#   done
# done

# ignore this for now....
# casename="flights"
# python demo_files_matchups_snd.py --casename "${casename}" --indir 'data_single'
# demo_files_matchups_snd.py --casename "flights" --indir 'data_single'
# demo_files_matchups.py --casename "flights" --indir 'data_single'
