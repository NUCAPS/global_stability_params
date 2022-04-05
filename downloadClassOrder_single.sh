#!/usr/bin/env bash

#declare -a orders=("3384362325" "3384362335" "3384362345")
declare -a orders=("8177234890")
savloc=/home/rebekah/stability/data_single/henri/

cd ${savloc}

for order in "${orders[@]}"; do
  wget --no-clobber ftp://ftp.avl.class.noaa.gov/${order}/001/* .
done

for f in *.tar; do
 tar -xvf $f
done

#rm ${savloc}/*.tar
#rm ${savloc}/*.gz

cd -
