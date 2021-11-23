#!/usr/bin/env bash

# find your/folder -type f -mtime +3 -exec rm {} \;

declare -a orders=("sub/rebekah.esm/55400")
savloc=/home/rebekah/stability/data

cd ${savloc}

for order in "${orders[@]}"; do
  wget --no-clobber ftp://ftp.avl.class.noaa.gov/${order}/* .
done

for f in *.tar; do
 tar -xvf $f
done

# rm ${savloc}/*.tar

# cd -
